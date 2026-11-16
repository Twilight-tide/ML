# -*- coding: utf-8 -*-
"""M2-5 验证：EncoderLayer"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from model import EncoderLayer, D_MODEL, N_HEADS, D_FF

# dropout=0.0 保证结果确定（否则每次运行数字都不同）
enc = EncoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)

print("=" * 60)
print("1. 参数量")
print("=" * 60)
groups = {"self_attn": 0, "ff": 0, "norm": 0}
for n, p in enc.named_parameters():
    key = "norm" if n.startswith("norm") else ("ff" if n.startswith("ff") else "self_attn")
    groups[key] += p.numel()
for k, v in groups.items():
    print("  %-10s %s" % (k, format(v, ",")))
print("  " + "-" * 40)
print("  单层合计 %s" % format(sum(p.numel() for p in enc.parameters()), ","))

print()
print("=" * 60)
print("2. 形状不变")
print("=" * 60)
x = torch.randn(2, 12, D_MODEL)
y, w = enc(x)
print("  (2,12,64) →", tuple(y.shape), " attn", tuple(w.shape))

print()
print("=" * 60)
print("3. 双向性：改最后一个位置，第 0 个位置必须跟着变")
print("=" * 60)
x1 = torch.randn(1, 6, D_MODEL)
x2 = x1.clone()
x2[0, -1] = torch.randn(D_MODEL)
y1, _ = enc(x1)
y2, _ = enc(x2)
print("  第 0 位输出变化量: %.4f   → 应 > 0（双向）"
      % torch.norm(y1[0, 0] - y2[0, 0]).item())

print()
print("=" * 60)
print("4. 对照：如果误加了因果掩蔽")
print("=" * 60)
causal = torch.tril(torch.ones(6, 6, dtype=torch.bool))
y1c, _ = enc(x1, mask=causal)
y2c, _ = enc(x2, mask=causal)
print("  第 0 位输出变化量: %.6f   → 应 == 0（退化成解码器）"
      % torch.norm(y1c[0, 0] - y2c[0, 0]).item())

print()
print("=" * 60)
print("5. LayerNorm 之后每个位置的统计量")
print("=" * 60)
print("  均值（应≈0）:", [round(v, 6) for v in y[0].mean(-1).tolist()[:4]])
print("  标准差（应≈1）:", [round(v, 4) for v in y[0].std(-1, unbiased=False).tolist()[:4]])

print()
print("=" * 60)
print("6. 残差的意义：让「什么都不做」成为可能")
print("=" * 60)
torch.manual_seed(0)
xs = torch.randn(1, 4, D_MODEL)
base = nn.LayerNorm(D_MODEL)
print("  子层正常时，输出与 LN(x) 的距离: %.4f"
      % torch.norm(enc(xs)[0] - base(xs)).item())

with torch.no_grad():
    enc.self_attn.w_o.weight.zero_()
    enc.self_attn.w_o.bias.zero_()
    enc.ff.w_2.weight.zero_()
    enc.ff.w_2.bias.zero_()
    ya = enc(xs)[0]
print("  把两个子层的输出投影置零后:")
print("    输出 == LN(x) 吗:", torch.allclose(ya, base(xs), atol=1e-6))
print("    最大差异: %.2e" % (ya - base(xs)).abs().max().item())
