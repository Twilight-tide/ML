# -*- coding: utf-8 -*-
"""生成质量的量化评估：用可自动判定的规则给样本打分。

为什么需要这个：
    肉眼对比 3-4 个样本的方差极大，容易得出相反结论。
    这里用 100 个样本统计「函数签名是否合法」等可判定指标。

指标（都只针对以 "def " 开头的生成）：
    sig     首行是合法的 `def 名字(参数):`（正则 + 标识符校验）
    nodup   参数列表里没有重复的参数名（self, self 这种明显退化）
    body    第一行函数体有缩进（不是空行、不是顶格）
    quote   三引号出现偶数次（docstring 没写一半就断）

用法：
    python experiments/eval_generation.py
    python experiments/eval_generation.py --samples 20 --tokens 120
"""

import os
import re
import sys
import json
import argparse

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
os.chdir(BASE)

from model import TinyTransformer  # noqa: E402

IDENT = re.compile(r"^[A-Za-z_]\w*$")
SIG = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:\s*$")


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = TinyTransformer(c["vocab_size"], c["d_model"], c["n_heads"],
                            c["d_ff"], c["n_layers"], c["max_len"], c["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    vpath = ckpt.get("vocab_file") or "data/vocab.json"
    v = json.load(open(vpath, encoding="utf-8"))
    itos = {int(k): ch for k, ch in v["itos"].items()}
    stoi = {ch: i for i, ch in itos.items()}
    return model, c, itos, stoi, device, vpath


def analyze(text):
    lines = text.split("\n")
    first = lines[0] if lines else ""
    r = {"sig": False, "nodup": False, "body": False, "quote": False}

    m = SIG.match(first)
    if m:
        r["sig"] = True
        raw = m.group(2)
        names = []
        ok = True
        for part in raw.split(","):
            p = part.split("=")[0].split(":")[0].strip().lstrip("*")
            if not p:
                continue
            if not IDENT.match(p):
                ok = False
                break
            names.append(p)
        r["nodup"] = ok and len(names) == len(set(names))

    body = [l for l in lines[1:] if l.strip()]
    if body:
        r["body"] = body[0][:1] in (" ", "\t")

    r["quote"] = text.count('"""') % 2 == 0
    return r


@torch.no_grad()
def run(model, itos, stoi, device, samples, tokens, temperature, top_k, seed):
    prompts = ["def ", "    def ", "    def __init__(self"]
    stats = {k: 0 for k in ("sig", "nodup", "body", "quote")}
    total = 0
    examples = []

    ids_space = stoi.get(" ", 0)
    for pi, prompt in enumerate(prompts):
        ids = [stoi[ch] for ch in prompt if ch in stoi] or [ids_space]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        for i in range(samples):
            torch.manual_seed(seed + pi * 1000 + i)
            out = model.generate(x, tokens, temperature=temperature, top_k=top_k)
            text = "".join(itos[j] for j in out[0].tolist())
            # 去掉 prompt 自身回显，只看模型续写；但签名判断要带 prompt
            a = analyze(text)
            for k in stats:
                stats[k] += int(a[k])
            total += 1
            if len(examples) < 3:
                examples.append(text)

    return {k: v / total * 100 for k, v in stats.items()}, total, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=["results/model.pt", "results/model_big.pt"])
    ap.add_argument("--samples", type=int, default=20, help="每个 prompt 生成几个")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("生成质量量化评估  (temperature=%.2f, top-k=%d, 每条 %d 字符)"
          % (args.temperature, args.top_k, args.tokens))
    print("=" * 78)
    print("%-26s %8s %8s %8s %8s %6s" % (
        "模型", "签名合法", "参数不重复", "有函数体", "引号成对", "样本"))
    print("-" * 78)

    for ck in args.ckpt:
        if not os.path.exists(ck):
            print("跳过（不存在）:", ck)
            continue
        model, c, itos, stoi, dev, vpath = load_model(ck, device)
        pct, n, _ = run(model, itos, stoi, dev, args.samples, args.tokens,
                        args.temperature, args.top_k, args.seed)
        name = os.path.basename(ck)
        print("%-26s %7.1f%% %7.1f%% %7.1f%% %7.1f%% %6d" % (
            name, pct["sig"], pct["nodup"], pct["body"], pct["quote"], n))

    print("-" * 78)
    print("说明：签名合法 = 首行是 `def 名字(参数):` 且参数都是合法标识符；")
    print("      参数不重复 = 排除 `self, self` 这类退化；")
    print("      有函数体 = 第一行非空内容有缩进。")


if __name__ == "__main__":
    main()
