# -*- coding: utf-8 -*-
"""跨语料公平评测：让两个不同语料的模型在同一份「都没见过」的数据上比困惑度。

为什么需要这个脚本：
    换语料后，各自的验证集不同，val_loss 无法直接比较。
    必须找一个两版语料都没收录的第三方测试集。

测试集来源：
    标准库的 Lib/test 目录——prepare_data.py 与 prepare_data_big.py
    都把 "test" 放进了目录黑名单，所以这里的代码两版都没见过。

用法：
    python experiments/eval_cross.py
    python experiments/eval_cross.py --ckpt-a results/model.pt --ckpt-b results/model_big.pt
"""

import os
import sys
import json
import glob
import argparse
import random

import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
os.chdir(BASE)

from model import TinyTransformer  # noqa: E402


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = TinyTransformer(
        c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
        c["n_layers"], c["max_len"], c["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, c, ckpt.get("n_params")


def build_test_text(stdlib, n_files=40, min_kb=8, seed=7):
    """从 stdlib/test 采样若干文件，拼成测试文本。"""
    root = os.path.join(stdlib, "test")
    files = []
    for r, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in names:
            if f.endswith(".py"):
                p = os.path.join(r, f)
                try:
                    if os.path.getsize(p) >= min_kb * 1024:
                        files.append(p)
                except OSError:
                    continue
    if not files:
        raise SystemExit("测试集为空，检查路径: %s" % root)
    random.Random(seed).shuffle(files)
    parts = []
    for p in files[:n_files]:
        try:
            t = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        parts.append(t)
    return "\n\n".join(parts), len(files)


@torch.no_grad()
def evaluate(model, text, stoi, device, seq_len=64, max_chars=400_000):
    """滑动窗口算平均交叉熵，返回 (loss, ppl, 覆盖率, 实际评估字符数)。"""
    text = text[:max_chars]
    ids = [stoi[ch] for ch in text if ch in stoi]
    covered = len(ids)
    if covered < seq_len * 2:
        return float("nan"), float("nan"), 0.0, 0

    data = torch.tensor(ids, dtype=torch.long, device=device)
    lossf = nn.CrossEntropyLoss()
    total, count = 0.0, 0

    # 不重叠地切窗，避免重复计算
    for i in range(0, len(data) - seq_len - 1, seq_len):
        x = data[i:i + seq_len].unsqueeze(0)
        y = data[i + 1:i + seq_len + 1].unsqueeze(0)
        logits = model(x)
        loss = lossf(logits.view(-1, logits.size(-1)), y.view(-1))
        total += loss.item() * seq_len
        count += seq_len

    avg = total / count
    return avg, float(torch.exp(torch.tensor(avg))), covered / len(text), covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", default="results/model.pt",
                    help="模型 A（旧语料 567 万）")
    ap.add_argument("--ckpt-b", default="results/model_big.pt",
                    help="模型 B（新语料 2097 万）")
    ap.add_argument("--vocab-a", default="data/vocab.json")
    ap.add_argument("--vocab-b", default="data/vocab_big.json")
    ap.add_argument("--files", type=int, default=40, help="采样多少个测试文件")
    ap.add_argument("--max-chars", type=int, default=400_000,
                    help="最多评估多少字符")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 定位标准库 test 目录
    import sysconfig
    stdlib = sysconfig.get_paths()["stdlib"]

    print("=" * 72)
    print("跨语料公平评测")
    print("=" * 72)
    print("测试集: %s\\test（两版语料的目录黑名单都含 test）" % stdlib)

    text, n_avail = build_test_text(stdlib, n_files=args.files)
    print("  采样 %d 个文件（共 %d 个可选），%s 字符"
          % (args.files, n_avail, format(len(text), ",")))
    print()

    rows = []
    for tag, ckpt, vpath in [("A 旧语料 567万", args.ckpt_a, args.vocab_a),
                             ("B 新语料 2097万", args.ckpt_b, args.vocab_b)]:
        if not os.path.exists(ckpt):
            print("跳过 %s：找不到 %s" % (tag, ckpt))
            continue
        model, c, n_params = load_model(ckpt, device)
        v = json.load(open(vpath, encoding="utf-8"))
        stoi = {ch: int(i) for i, ch in v["itos"].items()}

        loss, ppl, cov, n_eval = evaluate(
            model, text, stoi, device, seq_len=c["seq_len"],
            max_chars=args.max_chars)
        rows.append((tag, n_params, c["vocab_size"], loss, ppl, cov, n_eval))

    if not rows:
        raise SystemExit("没有可评测的模型")

    print("%-18s %10s %7s %9s %9s %10s" % (
        "模型", "参数量", "词表", "loss", "ppl", "词表覆盖"))
    print("-" * 72)
    for tag, n, vs, loss, ppl, cov, _ in rows:
        print("%-18s %10s %7d %9.4f %9.3f %9.3f%%" % (
            tag, format(n, ",") if n else "?", vs, loss, ppl, cov * 100))
    print("-" * 72)

    if len(rows) == 2:
        la, lb = rows[0][3], rows[1][3]
        print()
        print("B 相对 A：loss %+.2f%%   ppl %+.2f%%"
              % ((lb - la) / la * 100, (rows[1][4] - rows[0][4]) / rows[0][4] * 100))
        print("（数值越低越好；负数表示 B 更好）")
        print()
        print("注意：两个模型的词表覆盖率不同，覆盖率低的会「少算」那些字符，")
        print("      理论上对 ppl 略有偏袒。看覆盖率那一列判断偏差大小。")


if __name__ == "__main__":
    main()
