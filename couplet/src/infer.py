# -*- coding: utf-8 -*-
"""推理：给定上联，生成下联。

用法：
    python couplet/src/infer.py --ckpt results/run2/best.pt --text "春回大地千山秀"
    python couplet/src/infer.py --ckpt results/run2/best.pt --n 5 --temperature 0.9
"""

import os
import sys
import json
import hashlib
import argparse

import torch

from dataset import load_stoi
from model import (Seq2SeqTransformer, make_pad_mask, make_causal_mask,
                   SOS_ID, EOS_ID, PAD_ID)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def vocab_fingerprint(itos):
    s = json.dumps({str(k): v for k, v in sorted(itos.items())}, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Seq2SeqTransformer(**ck["config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


@torch.no_grad()
def greedy_decode(model, src, max_len=40):
    """每步取概率最大的字。稳定，但缺乏多样性。"""
    src_mask = make_pad_mask(src)
    enc_out, _ = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), SOS_ID, dtype=torch.long, device=src.device)

    for _ in range(max_len):
        tgt_mask = make_causal_mask(ys.size(1), src.device)
        dec_out, _, _ = model.decode(ys, enc_out, src_mask, tgt_mask)
        logits = model.head(dec_out[:, -1])          # 只取最后一步的输出
        nxt = logits.argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        if (nxt == EOS_ID).all():
            break
    return ys


@torch.no_grad()
def sample_decode(model, src, max_len=40, temperature=0.9, top_k=20, seed=None):
    """按概率采样，能出多个不同候选。"""
    if seed is not None:
        torch.manual_seed(seed)
    src_mask = make_pad_mask(src)
    enc_out, _ = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), SOS_ID, dtype=torch.long, device=src.device)

    for _ in range(max_len):
        tgt_mask = make_causal_mask(ys.size(1), src.device)
        dec_out, _, _ = model.decode(ys, enc_out, src_mask, tgt_mask)
        logits = model.head(dec_out[:, -1]) / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ys = torch.cat([ys, nxt], dim=1)
        if (nxt == EOS_ID).all():
            break
    return ys


def to_text(ids, itos):
    out = []
    for x in ids:
        if x == EOS_ID:
            break
        if x in (SOS_ID, PAD_ID):
            continue
        out.append(itos[x])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "results", "run1", "best.pt"))
    ap.add_argument("--text", default="")
    ap.add_argument("--n", type=int, default=1, help="生成几个候选")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-len", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stoi, itos = load_stoi(os.path.join(BASE, "data", "vocab.json"))

    model, ck = load_model(args.ckpt, device)

    # 词表校验：防止用错词表导致静默乱码
    if "vocab_fp" in ck and ck["vocab_fp"] != vocab_fingerprint(itos):
        print("ERROR: checkpoint 的词表指纹与当前 vocab.json 不一致")
        print("  ckpt : %s" % ck["vocab_fp"])
        print("  当前 : %s" % vocab_fingerprint(itos))
        return 1

    print("模型：%s   step %d   val %.4f   %s"
          % (os.path.basename(args.ckpt), ck.get("step", -1),
             ck.get("val_loss", float("nan")), device))

    text = args.text or input("上联：").strip()
    ids = [stoi[c] for c in text if c in stoi]
    if len(ids) != len(text):
        miss = [c for c in text if c not in stoi]
        print("警告：这些字不在词表里，已跳过：%s" % "".join(miss))
    src = torch.tensor([ids], device=device)

    print("上联：%s（%d 字）" % (text, len(ids)))
    if args.greedy or args.n == 1:
        out = greedy_decode(model, src, args.max_len)
        print("下联：%s" % to_text(out[0].tolist(), itos))
    else:
        for k in range(args.n):
            out = sample_decode(model, src, args.max_len, args.temperature,
                                args.top_k, seed=None)
            print("候选%d：%s" % (k + 1, to_text(out[0].tolist(), itos)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
