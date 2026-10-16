# -*- coding: utf-8 -*-
"""注意力可视化。

对应论文附录 “Attention Visualizations”（图 3、图 4、图 5）：
论文用热力图展示注意力头学到了什么，这里做同一件事。

产物：
    results/attention_heads.png   各注意力头的权重矩阵（对应论文图 3/图 5）
    results/attention_layer_avg.png  按层平均的注意力
"""

import os
import sys
import json

import torch
import matplotlib
matplotlib.use("Agg")            # 无界面后端，适合脚本运行
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyTransformer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

# 中文字体设置，避免图上中文显示为方块
for f in ["Microsoft YaHei", "SimHei", "DengXian"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    m = TinyTransformer(c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
                        c["n_layers"], c["max_len"], c["dropout"])
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, c


def main():
    ckpt_path = "results/model.pt"
    if not os.path.exists(ckpt_path):
        raise SystemExit("找不到 %s，请先训练" % ckpt_path)

    model, c = load_model(ckpt_path)
    v = json.load(open("data/vocab.json", encoding="utf-8"))
    itos = {int(k): ch for k, ch in v["itos"].items()}
    stoi = {ch: i for i, ch in itos.items()}

    # 用一段有结构的代码作为输入，便于观察注意力模式
    prompt = "def __init__(self, config):\n    self.config = config\n"
    ids = [stoi[ch] for ch in prompt if ch in stoi]
    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        _, attns = model(x, return_attn=True)

    # attns: list[n_layers] of (1, h, T, T)
    n_layers = len(attns)
    n_heads = attns[0].size(1)
    T = attns[0].size(-1)
    labels = [itos[i].replace("\n", "\\n") for i in ids]

    print("=" * 74)
    print("注意力可视化")
    print("=" * 74)
    print("  层数: %d   每层头数: %d   序列长度: %d" % (n_layers, n_heads, T))
    print("  输入: %r" % prompt)

    # ---------------- 图 1：每层每个头的注意力矩阵 ----------------
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(2.6 * n_heads, 2.6 * n_layers),
                             squeeze=False)
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi]
            w = attns[li][0, hi].numpy()
            ax.imshow(w, cmap="viridis", aspect="auto", vmin=0, vmax=w.max())
            ax.set_title("层%d 头%d" % (li + 1, hi + 1), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("各注意力头的权重矩阵（对应论文附录 图3/图5）", fontsize=12)
    fig.tight_layout()
    p1 = "results/attention_heads.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  已保存: %s" % p1)

    # ---------------- 图 2：按层平均 + 因果掩蔽检查 ----------------
    fig, axes = plt.subplots(1, n_layers, figsize=(5.2 * n_layers, 4.6),
                             squeeze=False)
    for li in range(n_layers):
        ax = axes[0][li]
        w = attns[li][0].mean(0).numpy()        # 对该层所有头取平均
        im = ax.imshow(w, cmap="magma", aspect="auto")
        ax.set_title("第 %d 层（%d 个头平均）" % (li + 1, n_heads), fontsize=11)
        ax.set_xlabel("被注意的位置 (key)")
        ax.set_ylabel("当前位置 (query)")
        step = max(1, T // 12)
        ax.set_xticks(range(0, T, step))
        ax.set_xticklabels([labels[i] for i in range(0, T, step)],
                           rotation=90, fontsize=7)
        ax.set_yticks(range(0, T, step))
        ax.set_yticklabels([labels[i] for i in range(0, T, step)], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("按层平均的注意力（对应论文附录）", fontsize=12)
    fig.tight_layout()
    p2 = "results/attention_layer_avg.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  已保存: %s" % p2)

    # ---------------- 数值检查 ----------------
    print()
    print("数值检查:")
    for li in range(n_layers):
        w = attns[li][0]                       # (h, T, T)
        rowsum = w.sum(-1)
        # 因果掩蔽：上三角（未来位置）应恒为 0
        future = torch.triu(torch.ones(T, T), diagonal=1).bool()
        max_future = w[:, future].abs().max().item() if T > 1 else 0.0
        print("  层%d: 每行和∈[%.6f, %.6f]  未来位置最大权重=%.2e %s" % (
            li + 1, rowsum.min().item(), rowsum.max().item(), max_future,
            "（因果掩蔽正确）" if max_future < 1e-8 else "（异常！）"))

    # 注意力集中度：熵越低说明越集中
    print()
    print("各头注意力集中度（熵，越低越集中）:")
    for li in range(n_layers):
        w = attns[li][0].clamp_min(1e-9)
        ent = -(w * w.log()).sum(-1).mean(-1)          # (h,)
        print("  层%d: %s" % (li + 1, "  ".join(
            "头%d=%.2f" % (h + 1, ent[h].item()) for h in range(n_heads))))


if __name__ == "__main__":
    main()
