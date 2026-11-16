# -*- coding: utf-8 -*-
"""训练脚本。

用法：
    python couplet/src/train.py --steps 3000
    python couplet/src/train.py --steps 30000 --eval-every 500 --out results/run2
"""

import os
import sys
import time
import math
import json
import hashlib
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_split, collate_batch, load_stoi, PAD_ID
from model import (Seq2SeqTransformer, make_pad_mask, make_causal_mask,
                   VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS_ENC,
                   N_LAYERS_DEC, MAX_LEN, DROPOUT)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


class NoamScheduler:
    """论文式(3)：

        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    前半段线性上升（warmup），后半段按 step^(-0.5) 衰减。
    用 min 把两段拼在一起：step < warmup 时右边小，之后左边小。
    """

    def __init__(self, optimizer, d_model, warmup):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.step_num = 0
        self.last_lr = 0.0

    def step(self):
        self.step_num += 1
        s = self.step_num
        lr = self.d_model ** -0.5 * min(s ** -0.5, s * self.warmup ** -1.5)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.last_lr = lr
        return lr


def vocab_fingerprint(itos):
    """词表的指纹，存进 checkpoint 防止用错词表（踩过这个坑）。"""
    s = json.dumps({str(k): v for k, v in sorted(itos.items())}, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


@torch.no_grad()
def evaluate(model, dl, lossf, device):
    """在验证集上算平均 loss（按 token 加权）。"""
    model.eval()
    tot, n_tok = 0.0, 0
    for src, dec, tgt in dl:
        src, dec, tgt = src.to(device), dec.to(device), tgt.to(device)
        src_mask = make_pad_mask(src)
        tgt_mask = make_causal_mask(dec.size(1), device) & make_pad_mask(dec)
        logits = model(src, dec, src_mask, tgt_mask)
        loss = lossf(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        k = (tgt != PAD_ID).sum().item()
        tot += loss.item() * k
        n_tok += k
    model.train()
    return tot / n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(BASE, "results", "run1"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # --out 允许写相对路径：一律相对 couplet/ 解析。
    # 否则 `--out results/run2` 会落到执行命令的目录（仓库根），
    # 与代码生成项目的 results/ 混在一起。
    if not os.path.isabs(args.out):
        args.out = os.path.join(BASE, args.out)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stoi, itos = load_stoi(os.path.join(BASE, "data", "vocab.json"))
    RAW = os.path.join(BASE, "data", "raw", "couplet")

    t0 = time.time()
    train_ds, tr = build_split(os.path.join(RAW, "train"), stoi)
    test_ds, te = build_split(os.path.join(RAW, "test"), stoi)
    print("训练集 %s 条（丢 %d 条坏字）/ 验证集 %s 条，构建 %.1f 秒"
          % (format(len(train_ds), ","), tr["n_oov"],
             format(len(test_ds), ","), time.time() - t0))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_batch, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_batch)

    torch.manual_seed(args.seed)
    model = Seq2SeqTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF,
                               N_LAYERS_ENC, N_LAYERS_DEC, MAX_LEN,
                               DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # 初始 lr 设成 1.0 无所谓——每一步都会被 NoamScheduler 覆盖
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                 betas=(0.9, 0.98), eps=1e-9)
    sched = NoamScheduler(optimizer, D_MODEL, args.warmup)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    print("参数量 %s / 设备 %s / 总步数 %d" % (format(n_params, ","), device, args.steps))
    print()
    print("%7s %10s %9s %9s %7s %6s" % ("step", "lr", "train", "val", "ppl", "秒"))
    print("-" * 56)

    log_path = os.path.join(args.out, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("step,lr,train_loss,val_loss,val_ppl,elapsed\n")

    model.train()
    t0 = time.time()
    run, n_run = 0.0, 0
    best_val = float("inf")

    for step, (src, dec, tgt) in enumerate(train_dl, 1):
        src, dec, tgt = src.to(device), dec.to(device), tgt.to(device)
        src_mask = make_pad_mask(src)
        tgt_mask = make_causal_mask(dec.size(1), device) & make_pad_mask(dec)

        logits = model(src, dec, src_mask, tgt_mask)
        loss = lossf(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sched.step()          # 必须在 opt.step() 之前，否则用旧 lr 更新
        optimizer.step()

        run += loss.item()
        n_run += 1

        if step % args.eval_every == 0:
            val = evaluate(model, test_dl, lossf, device)
            print("%7d %10.3e %9.4f %9.4f %7.3f %6.1f"
                  % (step, sched.last_lr, run / n_run, val,
                     math.exp(val), time.time() - t0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("%d,%.6e,%.4f,%.4f,%.4f,%.1f\n"
                        % (step, sched.last_lr, run / n_run, val,
                           math.exp(val), time.time() - t0))
            if val < best_val:
                best_val = val
                torch.save({
                    "model_state": model.state_dict(),
                    "config": dict(vocab_size=VOCAB_SIZE, d_model=D_MODEL,
                                   n_heads=N_HEADS, d_ff=D_FF,
                                   n_layers_enc=N_LAYERS_ENC,
                                   n_layers_dec=N_LAYERS_DEC,
                                   max_len=MAX_LEN, dropout=DROPOUT),
                    "n_params": n_params,
                    "step": step,
                    "val_loss": val,
                    "vocab_fp": vocab_fingerprint(itos),
                }, os.path.join(args.out, "best.pt"))
            run, n_run = 0.0, 0

        if step >= args.steps:
            break

    print()
    print("完成 %d 步，耗时 %.1f 秒，最优 val %.4f（ppl %.3f）"
          % (args.steps, time.time() - t0, best_val, math.exp(best_val)))
    print("产物：%s" % args.out)


if __name__ == "__main__":
    main()
