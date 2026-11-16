# -*- coding: utf-8 -*-
"""M3-1 验证：数据管道"""
import os
import sys
import time
import tracemalloc

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "src"))

from dataset import build_split, collate_batch, load_stoi, PAD_ID, SOS_ID, EOS_ID

stoi, itos = load_stoi(os.path.join(BASE, "data", "vocab.json"))
RAW = os.path.join(BASE, "data", "raw", "couplet")

print("=" * 64)
print("1. 构建两个划分")
print("=" * 64)
tracemalloc.start()
t0 = time.time()
train_ds, tr = build_split(os.path.join(RAW, "train"), stoi)
test_ds, te = build_split(os.path.join(RAW, "test"), stoi)
dt = time.time() - t0
cur, peak = tracemalloc.get_traced_memory()

for name, st in (("train", tr), ("test", te)):
    print("  %-6s 原始 %-8s 过滤短联 -%-6s 丢弃坏字 -%-3d 最终 %s"
          % (name, format(st["n_raw"], ","), format(st["n_short"], ","),
             st["n_oov"], format(st["n_keep"], ",")))
print("  耗时 %.1f 秒   当前内存 %.0f MB   峰值 %.0f MB"
      % (dt, cur / 1e6, peak / 1e6))

print()
print("=" * 64)
print("2. 取一条样本，还原成文本")
print("=" * 64)
s, d, t = train_ds[12345]
print("  src ids :", s.tolist())
print("  src 文本:", "".join(itos[x] for x in s.tolist()))
print("  tgt 文本:", "".join(itos[x] for x in t.tolist()[:-1]), "+ </s>")
print("  dec 首位是 <s>:", d[0].item() == SOS_ID,
      "   tgt 末位是 </s>:", t[-1].item() == EOS_ID)
print("  长度 src %d / dec %d / tgt %d （dec 与 tgt 必须相等）"
      % (len(s), len(d), len(t)))

print()
print("=" * 64)
print("3. 一个 batch 的 padding")
print("=" * 64)
dl = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate_batch)
src, dec, tgt = next(iter(dl))
print("  src", tuple(src.shape), " dec", tuple(dec.shape), " tgt", tuple(tgt.shape))
print("  每行有效长度 src:", (src != PAD_ID).sum(1).tolist())
print("  每行有效长度 tgt:", (tgt != PAD_ID).sum(1).tolist())
print()
for r in range(2):
    print("  src[%d] 还原: %s" % (r, "".join(itos[x] for x in src[r].tolist() if x != PAD_ID)))
    print("  tgt[%d] 还原: %s" % (r, "".join(itos[x] for x in tgt[r].tolist() if x != PAD_ID)))

print()
print("=" * 64)
print("4. 吞吐")
print("=" * 64)
dl2 = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
t0 = time.time()
n = 0
for _ in dl2:
    n += 1
    if n >= 200:
        break
speed = 200 * 64 / (time.time() - t0)
print("  200 个 batch 耗时 %.2f 秒 → %.0f 条/秒" % (time.time() - t0, speed))
print("  一个 epoch %s 条 ≈ %.1f 秒" % (format(len(train_ds), ","), len(train_ds) / speed))
print("  总 batch 数 %s（batch=64）" % format(len(train_ds) // 64, ","))
