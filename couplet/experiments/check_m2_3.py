# -*- coding: utf-8 -*-
"""M2-3 验证：PositionwiseFeedForward"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from model import PositionwiseFeedForward, D_MODEL, D_FF

ff = PositionwiseFeedForward(D_MODEL, D_FF, dropout=0.0)

print("=" * 58)
print("1. 参数量明细")
print("=" * 58)
for n, p in ff.named_parameters():
    print("  %-10s %-16s %s" % (n, tuple(p.shape), format(p.numel(), ",")))
total = sum(p.numel() for p in ff.parameters())
print("  " + "-" * 44)
print("  合计 %s   （手算 64*256+256 + 256*64+64 = 33,088）" % format(total, ","))
print("  占 810,496 的 %.1f%%；编码器+解码器共 4 层 = %s"
      % (total / 810496 * 100, format(total * 4, ",")))

print()
print("=" * 58)
print("2. 形状不变")
print("=" * 58)
x = torch.randn(2, 10, D_MODEL)
print("  ", tuple(x.shape), "→", tuple(ff(x).shape))

print()
print("=" * 58)
print("3. ReLU 在起作用")
print("=" * 58)
probe = torch.tensor([[[-2.0, -1.0, 0.0, 1.0, 2.0]]])
print("  输入      :", probe.tolist()[0][0])
print("  relu 之后 :", F.relu(probe).tolist()[0][0])

print()
print("=" * 58)
print("4. 逐位置独立：改一个位置，其他位置必须完全不变")
print("=" * 58)
x1 = torch.randn(1, 5, D_MODEL)
x2 = x1.clone()
x2[0, 0] = torch.randn(D_MODEL)          # 只动第 0 个位置
y1, y2 = ff(x1), ff(x2)
print("  位置 1-4 的输出完全不变吗:", torch.equal(y1[0, 1:], y2[0, 1:]))
print("  位置 0 自己的变化量: %.4f" % torch.norm(y1[0, 0] - y2[0, 0]).item())

