# -*- coding: utf-8 -*-
"""M2-7 验证：总装"""
import os
import sys
import json
import math

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "src"))

from model import (Seq2SeqTransformer, make_causal_mask, make_pad_mask,
                   VOCAB_SIZE, D_MODEL, N_HEADS, D_FF,
                   N_LAYERS_ENC, N_LAYERS_DEC, MAX_LEN, DROPOUT, PAD_ID,
                   SOS_ID, EOS_ID)

torch.manual_seed(42)
model = Seq2SeqTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF,
                           N_LAYERS_ENC, N_LAYERS_DEC, MAX_LEN, DROPOUT, PAD_ID)
model.eval()          # ← 关键：关掉 dropout，否则两次前向结果不同

print("=" * 62)
print("1. 参数量分解")
print("=" * 62)
groups = {}
for n, p in model.named_parameters():
    groups[n.split(".")[0]] = groups.get(n.split(".")[0], 0) + p.numel()
for k, v in groups.items():
    print("  %-12s %s" % (k, format(v, ",")))
print("  " + "-" * 44)
print("  合计 %s   （预算 810,496）"
      % format(sum(p.numel() for p in model.parameters()), ","))

print()
print("=" * 62)
print("2. 权重共享")
print("=" * 62)
print("  head.weight is embed.weight:", model.head.weight is model.embed.weight)
print("  未共享的话总量会多 %s" % format(VOCAB_SIZE * D_MODEL, ","))

print()
print("=" * 62)
print("3. 前向")
print("=" * 62)
B, T_src, T_tgt = 2, 7, 6
src = torch.randint(3, VOCAB_SIZE, (B, T_src))
tgt = torch.randint(3, VOCAB_SIZE, (B, T_tgt))
src_mask = make_pad_mask(src)
tgt_mask = make_causal_mask(T_tgt) & make_pad_mask(tgt)
logits = model(src, tgt, src_mask, tgt_mask)
print("  logits", tuple(logits.shape))
print("  概率和:", [round(v, 4) for v in torch.softmax(logits, -1)[0].sum(-1).tolist()[:4]])

print()
print("=" * 62)
print("4. 因果性：改 tgt 末位，前面 logits 必须不变")
print("=" * 62)
tgt2 = tgt.clone()
tgt2[:, -1] = torch.randint(3, VOCAB_SIZE, (B,))
logits2 = model(src, tgt2, src_mask, tgt_mask)
print("  位置 0..%d 完全不变:" % (T_tgt - 2), torch.equal(logits[0, :-1], logits2[0, :-1]))

print()
print("=" * 62)
print("5. 真实对联数据 + 初始 loss")
print("=" * 62)
vocab = {int(k): v for k, v in json.load(
    open(os.path.join(BASE, "data", "vocab.json"), encoding="utf-8"))["itos"].items()}
stoi = {c: i for i, c in vocab.items()}

with open(os.path.join(BASE, "data", "raw", "couplet", "train", "in.txt"), encoding="utf-8") as f:
    up = f.readline().split()
with open(os.path.join(BASE, "data", "raw", "couplet", "train", "out.txt"), encoding="utf-8") as f:
    down = f.readline().split()

s = torch.tensor([[stoi[c] for c in up]])
t = torch.tensor([[SOS_ID] + [stoi[c] for c in down]])
y = torch.tensor([[stoi[c] for c in down] + [EOS_ID]])

with torch.no_grad():
    out = model(s, t, make_pad_mask(s), make_causal_mask(t.size(1)) & make_pad_mask(t))
    loss = F.cross_entropy(out.reshape(-1, VOCAB_SIZE), y.reshape(-1))

print("  上联:", "".join(up))
print("  下联:", "".join(down))
print("  logits", tuple(out.shape))
print("  初始 loss = %.4f   （理论值 ln(9014) = %.4f）"
      % (loss.item(), math.log(VOCAB_SIZE)))
