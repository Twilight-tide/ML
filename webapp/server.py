# -*- coding: utf-8 -*-
"""实时预测分布可视化 —— 零依赖本地服务。

用法（在项目根目录、已激活 .venv 的终端里）：
    python webapp/server.py

然后浏览器打开 http://127.0.0.1:7860

它只做一件事：给定一段文本，返回模型对「下一个字符」的原始分数（logits），
以及这段文本的困惑度。前端负责把分数变成条形图，并用温度滑块重新归一化。
"""

import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
from model import TinyTransformer  # noqa: E402

CKPT = os.path.join(ROOT, "results", "model.pt")
VOCAB = os.path.join(ROOT, "data", "vocab.json")
INDEX = os.path.join(HERE, "index.html")

TOP_K = 20
PORT = 7860

model = None
itos = {}
stoi = {}
device = "cpu"
max_len = 256
n_params = 0


def load():
    global model, itos, stoi, device, max_len, n_params

    if not os.path.exists(CKPT):
        raise SystemExit("找不到 %s，请先运行 python src/train.py" % CKPT)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = TinyTransformer(c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
                        c["n_layers"], c["max_len"], c["dropout"])
    m.load_state_dict(ck["model_state"])
    m.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(device)

    v = json.load(open(VOCAB, encoding="utf-8"))
    itos = {int(k): ch for k, ch in v["itos"].items()}
    stoi = {ch: i for i, ch in itos.items()}

    max_len = c["max_len"]
    n_params = sum(p.numel() for p in m.parameters())
    model = m


def pretty(ch):
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    if ch == " ":
        return "␣"
    return ch


@torch.no_grad()
def predict(text):
    ids, skipped = [], []
    for ch in text:
        if ch in stoi:
            ids.append(stoi[ch])
        else:
            skipped.append(ch)

    if not ids:
        return {"ok": False, "reason": "empty" if not text else "out_of_vocab",
                "skipped": skipped[:20]}

    ids = ids[-max_len:]
    x = torch.tensor([ids], device=device)
    logits = model(x)[0]

    ppl = None
    if len(ids) > 1:
        lp = torch.log_softmax(logits, dim=-1)
        tgt = torch.tensor(ids[1:], device=device).unsqueeze(1)
        nll = -lp[:-1].gather(1, tgt).sum().item()
        ppl = math.exp(nll / (len(ids) - 1))

    last = logits[-1]
    probs = torch.softmax(last, dim=-1)
    ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum().item()

    top = torch.topk(last, TOP_K)
    return {
        "ok": True,
        "n_tokens": len(ids),
        "skipped": skipped[:20],
        "ppl": ppl,
        "entropy": ent,
        "topk": [
            {"ch": pretty(itos[int(i)]), "logit": float(v)}
            for v, i in zip(top.values, top.indices)
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def _write(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(INDEX, "rb") as f:
                self._write(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._write(200, json.dumps({
                "params": n_params,
                "device": device,
                "vocab": len(itos),
                "max_len": max_len,
            }), "application/json; charset=utf-8")
        else:
            self._write(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/predict":
            self._write(404, "not found", "text/plain; charset=utf-8")
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            payload = {}
        out = predict(payload.get("text") or "")
        self._write(200, json.dumps(out, ensure_ascii=False),
                    "application/json; charset=utf-8")

    def log_message(self, *args):
        pass


def main():
    load()
    print("=" * 58)
    print("实时预测分布可视化")
    print("=" * 58)
    print("  参数量   : %s" % format(n_params, ","))
    print("  设备     : %s" % device)
    print("  词表     : %d 个字符" % len(itos))
    print("  地址     : http://127.0.0.1:%d" % PORT)
    print("=" * 58)
    print("  在浏览器里打开上面的地址，然后开始打字。Ctrl+C 结束。")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
