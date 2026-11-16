# -*- coding: utf-8 -*-
"""M2-1 验证：ScaledDotProductAttention

三组检查：手算对照 / mask 是否失效 / d_k 用的是哪一维。
"""
import os
import sys
import math

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from model import ScaledDotProductAttention


attn = ScaledDotProductAttention()

# ---- 验证 1：手算一个 1×2 的例子 ----
print("=" * 58)
print("验证 1：手算对照")
print("=" * 58)
q = torch.tensor([[[[1.0, 0.0]]]])                  # (1,1,1,2)
k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])       # (1,1,2,2)
v = torch.tensor([[[[10.0, 0.0], [0.0, 10.0]]]])     # (1,1,2,2)

out, w = attn(q, k, v)
print("scores = q @ k^T / sqrt(2) = [[1, 0] / 1.414] = [[0.7071, 0]]")
print("权重 w   =", w.tolist())
print("输出 out =", out.tolist())
print()
print("手算：softmax([0.7071, 0]) = [0.6698, 0.3302]")
print("      out = 0.6698×[10,0] + 0.3302×[0,10] = [6.6976, 3.3024]")

# ---- 验证 2：mask 用 0 会失效 ----
print()
print("=" * 58)
print("验证 2：mask 为什么必须用 -inf")
print("=" * 58)
q2 = torch.tensor([[[[1.0, 0.0]]]])
k2 = torch.tensor([[[[1.0, 0.0], [2.0, 0.0]]]])      # 第二个 key 与 q 更相似
v2 = torch.tensor([[[[10.0, 0.0], [0.0, 10.0]]]])
mask = torch.tensor([[True, False]])                  # 屏蔽第二个位置

scores = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(2)
print("原始 scores      :", scores.tolist()[0][0])
print("不 mask          →", F.softmax(scores, dim=-1).tolist()[0][0])
print("mask 用 0 填充   →", F.softmax(scores.masked_fill(~mask, 0.0), dim=-1).tolist()[0][0])
print("mask 用 -inf     →", F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1).tolist()[0][0])

out2, w2 = attn(q2, k2, v2, mask)
print("实际调用结果 w2  =", w2.tolist())
print("实际调用结果 out2=", out2.tolist())

# ---- 验证 3：T_q 和 T_k 不等长 ----
print()
print("=" * 58)
print("验证 3：Q 和 K 长度不等（交叉注意力就是这样）")
print("=" * 58)
q3 = torch.randn(2, 8, 7, 8)      # 7 个查询
k3 = torch.randn(2, 8, 13, 8)     # 13 个位置
v3 = torch.randn(2, 8, 13, 8)
o3, a3 = attn(q3, k3, v3)
print("q (2,8,7,8)  k (2,8,13,8)")
print("  out  shape:", tuple(o3.shape), "  ← 跟 Q 走，7")
print("  attn shape:", tuple(a3.shape), "  ← 7 个查询 × 13 个位置")
print("  每行权重和:", a3.sum(-1)[0, 0].tolist(), "  ← 应全是 1.0")
