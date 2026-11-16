# -*- coding: utf-8 -*-
"""M2-2 验证：PositionalEncoding"""
import os
import sys
import math

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from model import PositionalEncoding

D_MODEL, MAX_LEN = 64, 40
pos = PositionalEncoding(D_MODEL, MAX_LEN, dropout=0.0)

# ---- 1. 第 0 个位置 ----
print("=" * 60)
print("1. 第 0 个位置：sin(0)=0, cos(0)=1")
print("=" * 60)
print("  pe[0][:8]      =", [round(v, 4) for v in pos.pe[0, 0, :8].tolist()])
print("  64 维的取值集合 =", sorted(set(round(v, 4) for v in pos.pe[0, 0].tolist())))

# ---- 2. 波长表 ----
print()
print("=" * 60)
print("2. 每一对的波长（波长 = 2π / div_term）")
print("=" * 60)
div = torch.exp(torch.arange(0, D_MODEL, 2, dtype=torch.float)
                * (-math.log(10000.0) / D_MODEL))
for i in (0, 1, 5, 15, 31):
    print("  第 %2d 对（第 %2d/%2d 维）  波长 = %10.2f"
          % (i, 2 * i, 2 * i + 1, 2 * math.pi / div[i].item()))

# ---- 3. 关键：40 个位置内，各维的变化跨度 ----
print()
print("=" * 60)
print("3. max_len=40，但最长波长 47117 —— 哪些维度还能区分位置？")
print("=" * 60)
for col in (0, 1, 30, 31, 62, 63):
    v = pos.pe[0, :MAX_LEN, col]
    span = (v.max() - v.min()).item()
    flag = "  ← 几乎不变，等于没用" if span < 0.01 else ""
    print("  第 %2d 维  跨度 %.6f%s" % (col, span, flag))

# ---- 4. 加性 + 形状 ----
print()
print("=" * 60)
print("4. 加性：形状不变")
print("=" * 60)
x = torch.zeros(2, 10, D_MODEL)
y = pos(x)
print("  输入", tuple(x.shape), "→ 输出", tuple(y.shape))
print("  全 0 输入时，输出应等于 PE 本身")
print("  y[0,3,:6] =", [round(v, 4) for v in y[0, 3, :6].tolist()])
print("  pe[3,:6]  =", [round(v, 4) for v in pos.pe[0, 3, :6].tolist()])

# ---- 5. register_buffer 生效了吗 ----
print()
print("=" * 60)
print("5. register_buffer 检查")
print("=" * 60)
print("  named_buffers:", [n for n, _ in pos.named_buffers()])
print("  named_parameters:", [n for n, _ in pos.named_parameters()], " ← 应为空")
print("  可训练参数个数:", sum(p.numel() for p in pos.parameters()), " ← 应为 0")
