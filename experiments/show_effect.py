# -*- coding: utf-8 -*-
"""生成效果展示 + 语料频率对照（技术报告 §4）。

对若干起始关键字各生成一段，同时统计该关键字在语料中的出现次数，
用于观察「模式频次 → 生成质量」的对应关系。

用法（在项目根目录）：
    python experiments/show_effect.py
"""
import sys, os, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

import torch
from model import TinyTransformer

CKPT = "results/model.pt"
NEW_TOKENS = 60
TEMPS = (0.3, 0.7)
PROMPTS = ["if ", "def ", "return ", "import ", "class "]

# ---------- 语料频率 ----------
corpus = open("data/corpus.txt", encoding="utf-8").read()
freq = {p: corpus.count(p) for p in PROMPTS}
freq["self."] = corpus.count("self.")

# ---------- 加载模型 ----------
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
c = ck["config"]
m = TinyTransformer(c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
                    c["n_layers"], c["max_len"], c["dropout"])
m.load_state_dict(ck["model_state"])
m.eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(dev)

v = json.load(open("data/vocab.json", encoding="utf-8"))
itos = {int(k): ch for k, ch in v["itos"].items()}
stoi = {ch: i for i, ch in itos.items()}

n_param = sum(p.numel() for p in m.parameters())
print("=" * 78)
print(f"模型 {n_param:,} 参数   d_model={c['d_model']}  "
      f"头={c['n_heads']}  层={c['n_layers']}  设备={dev}")
print(f"语料 {len(corpus):,} 字符")
print("=" * 78)
print("\n语料中各模式的出现次数：")
for k in ["self."] + PROMPTS:
    print(f"  {k!r:<12}{freq[k]:>8,} 次")

print("\n" + "=" * 78)
print("生成样例（| 之后为模型续写部分）")
print("=" * 78)

for p in PROMPTS:
    ids = [stoi[ch] for ch in p if ch in stoi]
    print(f"\n【{p!r}】  语料出现 {freq[p]:,} 次")
    print("-" * 78)
    for temp in TEMPS:
        torch.manual_seed(42)
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        out = m.generate(x, NEW_TOKENS, temperature=temp, top_k=10)
        full = "".join(itos[int(i)] for i in out[0])
        print(f"  T={temp}  {full[:len(p)]}|{full[len(p):]}")
