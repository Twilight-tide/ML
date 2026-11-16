# -*- coding: utf-8 -*-
"""M2-6 验证：DecoderLayer（含交叉注意力）"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from model import DecoderLayer, D_MODEL, N_HEADS, D_FF

dec = DecoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)

print("=" * 60)
print("1. 参数量：比编码器多一个交叉注意力")
print("=" * 60)
groups = {}
for n, p in dec.named_parameters():
    key = n.split(".")[0]
    groups[key] = groups.get(key, 0) + p.numel()
for k, v in groups.items():
    print("  %-12s %s" % (k, format(v, ",")))
total = sum(p.numel() for p in dec.parameters())
print("  " + "-" * 42)
print("  单层合计 %s   （编码器 49,984 + 16,640 + 128 = 66,752）"
      % format(total, ","))

print()
print("=" * 60)
print("2. 交叉注意力：T_tgt 和 T_src 不等长")
print("=" * 60)
B, T_tgt, T_src = 2, 7, 5
x = torch.randn(B, T_tgt, D_MODEL)
enc_out = torch.randn(B, T_src, D_MODEL)
causal = torch.tril(torch.ones(T_tgt, T_tgt, dtype=torch.bool))
y, w_self, w_cross = dec(x, enc_out, tgt_mask=causal)
print("  x (2,7,64)  enc_out (2,5,64)")
print("  输出          ", tuple(y.shape))
print("  自注意力权重  ", tuple(w_self.shape), "  ← (T_tgt, T_tgt)")
print("  交叉注意力权重", tuple(w_cross.shape), "  ← (T_tgt, T_src)")

print()
print("=" * 60)
print("3. 交叉注意力权重每行和为 1（在 src 上做 softmax）")
print("=" * 60)
print("  ", [round(v, 4) for v in w_cross[0, 0].sum(-1).tolist()])

print()
print("=" * 60)
print("4. 因果性：改 tgt 末位，前面位置必须不变")
print("=" * 60)
x2 = x.clone()
x2[:, -1] = torch.randn(D_MODEL)
y2, _, _ = dec(x2, enc_out, tgt_mask=causal)
print("  位置 0..5 输出完全不变:", torch.equal(y[0, :-1], y2[0, :-1]))

print()
print("=" * 60)
print("5. 交叉注意力是连通的：改 src，输出必须变")
print("=" * 60)
enc2 = enc_out.clone()
enc2[0, 0] = torch.randn(D_MODEL)
y3, _, _ = dec(x, enc2, tgt_mask=causal)
print("  输出是否完全相同:", torch.equal(y, y3), "  （应为 False）")
print("  变化量: %.4f" % torch.norm(y - y3).item())

print()
print("=" * 60)
print("6. 反例：如果忘了传 enc_out（Q=K=V 全用解码器自己）")
print("=" * 60)
a_self, w_wrong = dec.self_attn(x, mask=causal)
print("  这样做的权重形状:", tuple(w_wrong.shape),
      " ← 本该是 (2,8,7,5)，实际是 (2,8,7,7)")
print("  形状对不上但不会报错，因为 T 恰好也能自洽")
