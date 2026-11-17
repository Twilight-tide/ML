# -*- coding: utf-8 -*-
"""对联生成 Web 服务。零依赖，只用标准库 + torch。

用法：
    python couplet/webapp/server.py
    浏览器打开 http://127.0.0.1:8765
"""

import os
import sys
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "src"))

from dataset import load_stoi
from model import (Seq2SeqTransformer, make_pad_mask, make_causal_mask,
                   SOS_ID, EOS_ID, PAD_ID)
from tone import load_table

PORT = 8765
CKPT = os.path.join(BASE, "results", "run2", "best.pt")
PUNCT = set("，、。？！；：（）《》「」『』【】…—〇·-")

_device = None
_model = None
_itos = None
_stoi = None
_tone = None


def init():
    global _device, _model, _itos, _stoi, _tone
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _stoi, _itos = load_stoi(os.path.join(BASE, "data", "vocab.json"))
    _tone = load_table()

    ck = torch.load(CKPT, map_location=_device, weights_only=False)
    _model = Seq2SeqTransformer(**ck["config"]).to(_device)
    _model.load_state_dict(ck["model_state"])
    _model.eval()

    print("词表 %d 字 / 平仄表 %d 字" % (len(_itos), len(_tone)))
    print("模型 %s  step %d  val %.4f  %s"
          % (os.path.basename(CKPT), ck.get("step", -1),
             ck.get("val_loss", float("nan")), _device))


# ---------------------------------------------------------------- 生成
@torch.no_grad()
def _decode(enc_out, src_mask, max_len=40, temperature=None, top_k=20, ban_dup=True):
    """temperature=None 时走贪心，否则采样。"""
    ys = torch.full((1, 1), SOS_ID, dtype=torch.long, device=_device)
    used = set()

    for _ in range(max_len):
        tm = make_causal_mask(ys.size(1), _device)
        dec_out = _model.decode(ys, enc_out, src_mask, tm)[0]
        logits = _model.head(dec_out[:, -1]).clone()

        if ban_dup:
            for cid in used:
                logits[0, cid] = float("-inf")

        if temperature is None:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            lg = logits / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg[lg < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(torch.softmax(lg, -1), 1)

        ys = torch.cat([ys, nxt], 1)
        if nxt.item() == EOS_ID:
            break
        if _itos[nxt.item()] not in PUNCT:
            used.add(nxt.item())

    return "".join(_itos[x] for x in ys[0].tolist()[1:]
                   if x not in (EOS_ID, PAD_ID))


# ---------------------------------------------------------------- 规则
def strip_punct(text):
    return [c for c in text if c not in PUNCT]


def check(up, down):
    """三条可自动检验的规则。"""
    u, d = strip_punct(up), strip_punct(down)

    len_ok = len(u) == len(d)

    c = Counter(d)
    dup = {k: v for k, v in c.items() if v >= 2}

    tone_ok = None
    if u and d:
        tu, td = _tone.get(u[-1]), _tone.get(d[-1])
        if tu and td:
            tone_ok = (tu == "ze" and td == "ping")

    return {
        "len": len_ok,
        "dup": len(dup) == 0,
        "dup_chars": "".join(dup.keys()),
        "tone": tone_ok,
        "tone_detail": "%s/%s" % (_tone.get(u[-1], "?") if u else "?",
                                  _tone.get(d[-1], "?") if d else "?"),
    }


def generate(up, n=4, temperature=0.9, top_k=20, ban_dup=True):
    ids = [_stoi[c] for c in up if c in _stoi]
    if not ids:
        return {"up": up, "results": [], "error": "没有可编码的字"}

    src = torch.tensor([ids], device=_device)
    src_mask = make_pad_mask(src)
    enc_out = _model.encode(src, src_mask)[0]

    t0 = time.time()
    outs = []

    # 第一个走贪心：最稳，可复现
    outs.append(_decode(enc_out, src_mask, ban_dup=ban_dup))

    # 其余采样：多样性
    tries = 0
    while len(outs) < n and tries < n * 4:
        tries += 1
        s = _decode(enc_out, src_mask, temperature=temperature,
                    top_k=top_k, ban_dup=ban_dup)
        if s and s not in outs:
            outs.append(s)

    elapsed = (time.time() - t0) * 1000

    results = []
    for i, down in enumerate(outs):
        chk = check(up, down)
        results.append({
            "down": down,
            "mode": "贪心" if i == 0 else "采样",
            "checks": chk,
            "score": sum(1 for v in (chk["len"], chk["dup"], chk["tone"]) if v is True),
        })

    return {
        "up": up,
        "results": results,
        "ms": round(elapsed, 1),
        "ban_dup": ban_dup,
    }


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send_file(os.path.join(HERE, "index.html"))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            up = (body.get("text") or "").strip()
            if not up:
                self._send_json({"error": "请输入上联"}, 400)
                return
            resp = generate(
                up,
                n=int(body.get("n", 4)),
                temperature=float(body.get("temperature", 0.9)),
                top_k=int(body.get("top_k", 20)),
                ban_dup=bool(body.get("ban_dup", True)),
            )
            self._send_json(resp)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _send_file(self, path):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    init()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("服务已启动： http://127.0.0.1:%d  （Ctrl+C 停止）" % PORT)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
