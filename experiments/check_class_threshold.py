# -*- coding: utf-8 -*-
"""专项检验：class 频率跨过阈值后，生成格式是否改善。

对照技术报告 §4.4 的预测：
    旧语料 `class ` 出现 2,149 次 → 写不出 `class Foo:`
    新语料 `class ` 出现 5,285 次 → 预期能写出

用法：
    python experiments/check_class_threshold.py
"""

import os
import re
import sys
import json

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
os.chdir(BASE)

from model import TinyTransformer  # noqa: E402

CLS = re.compile(r"^class\s+([A-Za-z_]\w*)\s*(\([^)]*\))?\s*:\s*$")


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = TinyTransformer(c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
                        c["n_layers"], c["max_len"], c["dropout"])
    m.load_state_dict(ck["model_state"])
    m.eval().to(device)
    vpath = ck.get("vocab_file") or "data/vocab.json"
    v = json.load(open(vpath, encoding="utf-8"))
    itos = {int(k): ch for k, ch in v["itos"].items()}
    return m, itos, {ch: i for i, ch in itos.items()}, device


@torch.no_grad()
def test(ckpt, n=30, tokens=90, temperature=0.6, top_k=40):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, itos, stoi, dev = load(ckpt, device)
    ids = [stoi[ch] for ch in "class " if ch in stoi]
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    ok = 0
    samples = []
    for i in range(n):
        torch.manual_seed(2000 + i)
        out = m.generate(x, tokens, temperature=temperature, top_k=top_k)
        text = "".join(itos[j] for j in out[0].tolist())
        first = text.split("\n")[0]
        if CLS.match(first):
            ok += 1
            if len(samples) < 2:
                samples.append(first)
    return ok / n * 100, samples


def main():
    print("=" * 72)
    print("专项检验：`class ` 开头的生成格式正确率（每模型 30 个样本）")
    print("=" * 72)
    cases = [
        ("results/model.pt", "旧语料 630万", "class 出现 2,149 次"),
        ("results/model_big.pt", "新语料 2097万", "class 出现 5,285 次"),
    ]
    for ck, tag, note in cases:
        if not os.path.exists(ck):
            print("跳过", ck)
            continue
        pct, samples = test(ck)
        print("%-16s %s" % (tag, note))
        print("  格式正确率: %.0f%%" % pct)
        for s in samples:
            print("    例: %s" % s)
        print()
    print("判读：若新语料明显更高，说明「频率阈值」假说成立；")
    print("      若两者接近，说明阈值不是主因，需另找解释。")


if __name__ == "__main__":
    main()
