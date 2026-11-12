# -*- coding: utf-8 -*-
"""参数规模扫描实验（技术报告 §2）。

固定数据、随机种子、批大小、序列长度、训练步数，只改变模型尺寸，
观察验证损失随参数量的变化，用于确定当前语料下的最优参数规模。

用法（在项目根目录）：
    python experiments/param_scan.py

四组全跑约 6 分钟（RTX 5060 Laptop）。
"""
import sys, os, time, math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

import torch
import torch.nn as nn
from config import Config
from model import TinyTransformer
from train import load_data, get_batch, lr_lambda_factory, evaluate

# 四组配置：(标签, d_model, n_heads, d_ff, n_layers)
GROUPS = [
    ("A 基准 10 万", 64, 4, 232, 2),
    ("B 放大 30 万", 96, 6, 560, 2),
    ("D 放大 50 万", 128, 8, 680, 2),
    ("C 放大 94 万", 128, 8, 640, 4),
]

TOTAL_STEPS = 12000


def build(d_model, n_heads, d_ff, n_layers, device):
    c = Config()
    c.d_model, c.n_heads, c.d_ff, c.n_layers = d_model, n_heads, d_ff, n_layers
    c.device = device
    c.check()
    m = TinyTransformer(c.vocab_size, c.d_model, c.n_heads, c.d_ff,
                        c.n_layers, c.max_len, c.dropout).to(device)
    n = sum(p.numel() for p in m.parameters())
    return m, n


def run(tag, d_model, n_heads, d_ff, n_layers, tr, va, cfg):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    model, n_param = build(d_model, n_heads, d_ff, n_layers, cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg))
    lossf = nn.CrossEntropyLoss()

    print(f"\n=== {tag} ===  d_model={d_model} heads={n_heads} "
          f"d_ff={d_ff} layers={n_layers}  参数={n_param:,}")
    print(f"{'step':>6} {'epoch':>6} {'train':>8} {'val':>8} {'ppl':>7} {'sec':>5}")
    print("-" * 48)

    t0 = time.time()
    run_loss, best = [], (9e9, 0)
    last_train = 0.0
    for step in range(1, TOTAL_STEPS + 1):
        x, y = get_batch(tr, cfg)
        logits = model(x)
        loss = lossf(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        run_loss.append(loss.item())

        if step % 1000 == 0:
            vl, vppl = evaluate(model, va, cfg, lossf)
            avg_train = sum(run_loss) / len(run_loss)
            last_train = avg_train
            if vl < best[0]:
                best = (vl, step)
            ep = step * cfg.batch_size * cfg.seq_len / 5_668_894
            print(f"{step:>6} {ep:>6.2f} {avg_train:>8.4f} "
                  f"{vl:>8.4f} {vppl:>7.2f} {time.time()-t0:>5.0f}", flush=True)
            run_loss = []

    print(f"最优 val {best[0]:.4f} @ {best[1]} 步   "
          f"ppl {math.exp(best[0]):.2f}   用时 {time.time()-t0:.0f}s")
    return n_param, best[0], best[1], time.time() - t0, last_train


def main():
    cfg = Config().check()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {cfg.device}")
    if cfg.device == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")

    tr, va, itos, stoi = load_data(cfg)
    print(f"训练集 {len(tr):,} token   验证集 {len(va):,} token")
    print(f"固定条件: seed={cfg.seed}  batch={cfg.batch_size}  "
          f"seq_len={cfg.seq_len}  步数={TOTAL_STEPS}")

    rows = []
    for tag, dm, nh, dff, nl in GROUPS:
        n, v, s, t, trl = run(tag, dm, nh, dff, nl, tr, va, cfg)
        rows.append((tag, n, v, s, t, trl))

    print("\n" + "=" * 76)
    print(f"{'配置':<16}{'参数量':>12}{'最优val':>10}{'ppl':>8}"
          f"{'步数':>8}{'耗时':>8}{'train/val差':>12}")
    print("-" * 76)
    for tag, n, v, s, t, trl in rows:
        print(f"{tag:<16}{n:>12,}{v:>10.4f}{math.exp(v):>8.2f}"
              f"{s:>8}{t:>7.0f}s{v-trl:>12.2f}")
    print("-" * 76)

    base = rows[0][2]
    prev = None
    for tag, n, v, s, t, trl in rows:
        d_base = (base - v) / base * 100
        d_prev = "" if prev is None else f"   比上档 {(prev - v) / prev * 100:+.1f}%"
        print(f"{tag:<16}相对基准 {d_base:+5.1f}%{d_prev}")
        prev = v


if __name__ == "__main__":
    main()
