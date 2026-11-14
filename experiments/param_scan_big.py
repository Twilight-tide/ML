# -*- coding: utf-8 -*-
"""在大语料（2097 万字符）上重跑参数扫描。

背景：§2 的扫描是在 630 万字符上做的，结论是「50 万最优，94 万过拟合」。
现在语料翻了 3.3 倍，最优参数量应当上移。本脚本用同一份大语料、
同 seed、同步数重测，看结论是否改变。

用法：
    python experiments/param_scan_big.py
    python experiments/param_scan_big.py --steps 30000
"""

import os
import sys
import math
import time
import argparse

import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
os.chdir(BASE)

from config import Config            # noqa: E402
from model import TinyTransformer, count_parameters  # noqa: E402
from train import load_data, get_batch, lr_lambda_factory, evaluate  # noqa: E402

CONFIGS = [
    # (标签, d_model, n_heads, d_ff, n_layers)
    ("50 万 (旧基线)", 128, 8, 680, 2),
    ("94 万",          128, 8, 640, 4),
]


def run(tag, d_model, n_heads, d_ff, n_layers, train_data, val_data,
        cfg, steps, eval_batches):
    c = Config().check()
    c.d_model = d_model
    c.n_heads = n_heads
    c.d_ff = d_ff
    c.n_layers = n_layers
    c.check()

    torch.manual_seed(c.seed)
    model = TinyTransformer(c.vocab_size, c.d_model, c.n_heads, c.d_ff,
                            c.n_layers, c.max_len, c.dropout).to(c.device)
    n_params = count_parameters(model)

    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(c))
    lossf = nn.CrossEntropyLoss()
    c.eval_batches = eval_batches

    best_v, best_step = float("inf"), 0
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, c)
        logits = model(x)
        loss = lossf(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        opt.step()
        sched.step()

        if step % c.eval_every == 0:
            vl, _ = evaluate(model, val_data, c, lossf)
            if vl < best_v:
                best_v, best_step = vl, step
            print("    %-12s step %6d  val %.4f  (最优 %.4f @ %d)  %.0fs"
                  % (tag, step, vl, best_v, best_step, time.time() - t0),
                  flush=True)

    return n_params, best_v, best_step, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval-batches", type=int, default=100)
    args = ap.parse_args()

    cfg = Config().check()
    cfg.corpus_file = "data/corpus_big.txt"
    cfg.vocab_file = "data/vocab_big.json"

    print("=" * 76)
    print("大语料上的参数扫描   (语料 %s)" % cfg.corpus_file)
    print("=" * 76)

    train_data, val_data, itos, stoi = load_data(cfg)
    cfg.vocab_size = len(itos)
    print("训练/验证字符: %s / %s   步数 %d   eval_batches %d"
          % (format(len(train_data), ","), format(len(val_data), ","),
             args.steps, args.eval_batches))
    print()

    results = []
    for tag, dm, nh, dff, nl in CONFIGS:
        print("--- %s: d_model=%d h=%d d_ff=%d layers=%d ---" % (tag, dm, nh, dff, nl))
        n, v, s, sec = run(tag, dm, nh, dff, nl, train_data, val_data,
                           cfg, args.steps, args.eval_batches)
        results.append((tag, n, v, math.exp(v), s, sec))
        print()

    print("=" * 76)
    print("%-18s %12s %10s %8s %8s %8s" % ("配置", "参数量", "最优val", "ppl", "步数", "用时"))
    print("-" * 76)
    for tag, n, v, ppl, s, sec in results:
        print("%-18s %12s %10.4f %8.2f %8d %7.0fs" % (
            tag, format(n, ","), v, ppl, s, sec))
    print("-" * 76)

    if len(results) == 2:
        a, b = results
        print("参数量 %.1f 倍  →  val 变化 %+.2f%%"
              % (b[1] / a[1], (b[2] - a[2]) / a[2] * 100))
        print()
        print("判读：若 94 万仍明显更好且未过拟合，说明最优点上移，")
        print("      §2「50 万最优」的结论需要在新语料下重写。")


if __name__ == "__main__":
    main()
