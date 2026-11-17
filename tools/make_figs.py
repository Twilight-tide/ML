# -*- coding: utf-8 -*-
"""为技术报告生成三张图"""
import os
import sys
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 由本文件位置推导：tools/ -> 项目根 -> couplet/
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "couplet")
OUT = os.path.join(BASE, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# 中文字体
for name in ("Microsoft YaHei", "SimHei", "SimSun"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

INK = "#2b2620"
RED = "#a63a2e"
GREEN = "#4a7c59"
GRAY = "#9a9086"
LINE = "#d9d2c5"

# ---------------------------------------------------------------- 图 1 训练曲线
steps, vals = [], []
with open(os.path.join(BASE, "results", "run2", "train_log.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        steps.append(int(row["step"]))
        vals.append(float(row["val_loss"]))

fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(steps, vals, color=RED, lw=2, marker="o", ms=3.5, zorder=3)
ax.fill_between(steps, vals, 4.0, color=RED, alpha=0.06)

ax.axvline(400, color=GRAY, ls="--", lw=1, zorder=2)
ax.annotate("warmup 结束\nlr 达峰值 6.25e-3", xy=(400, vals[0]),
            xytext=(400, vals[0] + 0.13), ha="center", fontsize=8.5, color=GRAY,
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))

ax.scatter([steps[-1]], [vals[-1]], color=GREEN, s=60, zorder=5)
ax.annotate("12,000 步  val 4.2385\nppl 69.3", xy=(steps[-1], vals[-1]),
            xytext=(steps[-1] - 4200, vals[-1] + 0.18), fontsize=9, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9))

ax.set_xlabel("训练步数", fontsize=10, color=INK)
ax.set_ylabel("验证损失（nats/字符）", fontsize=10, color=INK)
ax.set_title("训练曲线：12,000 步内单调下降，无过拟合迹象", fontsize=11.5, color=INK, pad=12)
ax.grid(alpha=0.25, ls=":", color=LINE)
ax.set_xlim(0, steps[-1] * 1.02)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(LINE)
ax.tick_params(colors=GRAY, labelsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "train_curve.png"), bbox_inches="tight")
plt.close(fig)
print("图1 训练曲线 -> train_curve.png")

# ---------------------------------------------------------------- 图 2 平仄双峰
PUNCT = set("，、。？！；：（）《》「」『』【】…—〇·-")
RAW = os.path.join(BASE, "data", "raw", "couplet")


def lasts(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            cs = [c for c in line.split() if c not in PUNCT]
            if cs:
                out.append(cs[-1])
    return out


cu = Counter(lasts(os.path.join(RAW, "train", "in.txt")))
cd = Counter(lasts(os.path.join(RAW, "train", "out.txt")))

ratios = []
for ch in set(cu) | set(cd):
    d, u = cd.get(ch, 0), cu.get(ch, 0)
    if d + u >= 20:
        ratios.append(d / (d + u))

fig, ax = plt.subplots(figsize=(7.2, 3.8))
n, bins, patches = ax.hist(ratios, bins=40, range=(0, 1),
                           color=GRAY, edgecolor="white", lw=0.6, zorder=3)
for p, left in zip(patches, bins[:-1]):
    if left <= 0.3 - 1e-9:
        p.set_facecolor(GREEN)
    elif left >= 0.7 - 1e-9:
        p.set_facecolor(RED)

n_ze = sum(1 for r in ratios if r <= 0.3)
n_ping = sum(1 for r in ratios if r >= 0.7)
n_mid = len(ratios) - n_ze - n_ping

ax.annotate("%d 字\n判为仄声" % n_ze, xy=(0.17, max(n) * 0.78), fontsize=9.5,
            color=GREEN, ha="center", fontweight="bold")
ax.annotate("%d 字\n判为平声" % n_ping, xy=(0.83, max(n) * 0.78), fontsize=9.5,
            color=RED, ha="center", fontweight="bold")
ax.annotate("%d 字\n判不出 (4%%)" % n_mid, xy=(0.5, max(n) * 0.30), fontsize=9,
            color=GRAY, ha="center")

ax.set_xlabel("该字作为「下联末字」的比例", fontsize=10, color=INK)
ax.set_ylabel("字数", fontsize=10, color=INK)
ax.set_title("从语料反推平仄：分布是双峰的，说明「上仄下平」真实存在于数据中",
             fontsize=11, color=INK, pad=12)
ax.grid(alpha=0.25, ls=":", axis="y", color=LINE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(LINE)
ax.tick_params(colors=GRAY, labelsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "tone_distribution.png"), bbox_inches="tight")
plt.close(fig)
print("图2 平仄分布 -> tone_distribution.png  (仄 %d / 平 %d / 中间 %d)" % (n_ze, n_ping, n_mid))

# ---------------------------------------------------------------- 图 3 三类规则
rules = ["字数相等", "无不规则重字", "上仄下平"]
ref = [100.0, 83.5, 97.4]
model = [98.0, 46.5, 97.9]
fixed = [97.5, 100.0, 97.9]

x = range(len(rules))
w = 0.26

fig, ax = plt.subplots(figsize=(7.2, 3.8))
b1 = ax.bar([i - w for i in x], ref, w, label="参考答案", color=GRAY, zorder=3)
b2 = ax.bar(list(x), model, w, label="模型（普通贪心）", color=RED, zorder=3)
b3 = ax.bar([i + w for i in x], fixed, w, label="模型 + 禁止重字约束", color=GREEN, zorder=3)

for bars in (b1, b2, b3):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5,
                "%.1f" % r.get_height(), ha="center", fontsize=8.5, color=INK)

ax.set_xticks(list(x))
ax.set_xticklabels(rules, fontsize=10.5, color=INK)
ax.set_ylabel("通过率 (%)", fontsize=10, color=INK)
ax.set_ylim(0, 112)
ax.set_title("三类规则：局部可判的能学会，需要全局记忆的靠约束解决", fontsize=11, color=INK, pad=12)
ax.legend(fontsize=9, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.13), ncol=3)
ax.grid(alpha=0.25, ls=":", axis="y", color=LINE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(LINE)
ax.tick_params(colors=GRAY, labelsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "rule_comparison.png"), bbox_inches="tight")
plt.close(fig)
print("图3 规则对比 -> rule_comparison.png")

print()
print("输出目录:", OUT)
for f in sorted(os.listdir(OUT)):
    print("  %-26s %.1f KB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1024))
