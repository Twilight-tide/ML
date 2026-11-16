# -*- coding: utf-8 -*-
"""M2-4 验证：MultiHeadAttention"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from model import MultiHeadAttention, D_MODEL, N_HEADS

mha = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0)

print("=" * 58)
print("1. 参数量（4 个 d_model×d_model 的线性层）")
print("=" * 58)
print("  合计 %s   （手算 4 × (64*64+64) = %s）"
      % (format(sum(p.numel() for p in mha.parameters()), ","),
         format(4 * (64 * 64 + 64), ",")))

print()
print("=" * 58)
print("2. 自注意力")
print("=" * 58)
x = torch.randn(2, 10, D_MODEL)
out, attn = mha(x)
print("  x (2,10,64) → out", tuple(out.shape), " attn", tuple(attn.shape))

print()
print("=" * 58)
print("3. 交叉注意力（Q 和 K/V 长度不同）")
print("=" * 58)
q = torch.randn(2, 7, D_MODEL)
kv = torch.randn(2, 13, D_MODEL)
out2, attn2 = mha(q, kv)
print("  q (2,7,64)  kv (2,13,64) → out", tuple(out2.shape), " attn", tuple(attn2.shape))
print("  输出长度跟 Q 走:", out2.shape[1] == 7)

print()
print("=" * 58)
print("4. 因果掩蔽：改末位输入，前面位置的输出必须不变")
print("=" * 58)
T = 5
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
x1 = torch.randn(1, T, D_MODEL)
x2 = x1.clone()
x2[0, T - 1] = torch.randn(D_MODEL)          # 只改最后一个位置
y1, _ = mha(x1, mask=causal)
y2, _ = mha(x2, mask=causal)
print("  位置 0..%d 输出完全不变:" % (T - 2), torch.equal(y1[0, :-1], y2[0, :-1]))
print("  最后一个位置变化量: %.4f" % torch.norm(y1[0, -1] - y2[0, -1]).item())

print()
print("=" * 58)
print("5. 交叉注意力是连通的：改 kv，输出必须变")
print("=" * 58)
kv2 = kv.clone()
kv2[0, 0] = torch.randn(D_MODEL)
o_a, _ = mha(q, kv)
o_b, _ = mha(q, kv2)
print("  改 kv 第 0 个位置后，输出是否完全相同:", torch.equal(o_a, o_b), "（应为 False）")
print("  输出变化量: %.4f" % torch.norm(o_a - o_b).item())

print()
print("=" * 58)
print("6. 拆头正确性：第 i 个头拿到的应是第 i 段 d_k 维")
print("=" * 58)
with torch.no_grad():
    mha.w_q.weight.copy_(torch.eye(D_MODEL))
    mha.w_q.bias.zero_()
probe = torch.randn(1, 3, D_MODEL)
q_proj = mha.w_q(probe)
q_heads = q_proj.view(1, 3, N_HEADS, mha.d_k).transpose(1, 2)
print("  第 0 个头 == 原向量第 0-7 维 :",
      torch.allclose(q_heads[0, 0], probe[0, :, 0:8]))
print("  第 3 个头 == 原向量第 24-31 维:",
      torch.allclose(q_heads[0, 3], probe[0, :, 24:32]))
