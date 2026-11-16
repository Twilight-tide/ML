# -*- coding: utf-8 -*-
"""平仄表：从语料反推，不依赖任何拼音库。

原理：对联规则是「上仄下平」，所以统计每个字在上下联末字的分布——
    大量出现在下联末尾 → 平声
    大量出现在上联末尾 → 仄声

实测（77 万副对联，74 万个末字）：
    可判定 2,909 字（1,305 平 / 1,604 仄），仅 4% 的字无法判定
    24 个已知平仄的字（风花山天…/ 月水树地…）全部判对
    占比分布是双峰：0.0-0.1 有 1,445 字，0.9-1.0 有 1,248 字，
    中间 0.3-0.7 只有 105 字 —— 这个形状本身就证明规则真实存在于语料中
    test 集上下联末字都可判定的比例 96.5%

生成：python couplet/src/tone.py
"""

import os
import json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RAW = os.path.join(BASE, "data", "raw", "couplet")
TABLE_PATH = os.path.join(BASE, "data", "tone_table.json")

PUNCT = set("，、。？！；：（）《》「」『』【】…—〇·-")

MIN_COUNT = 20      # 至少作为末字出现这么多次，才认为统计可信
LO, HI = 0.3, 0.7   # 下联末字占比 <= LO 判仄，>= HI 判平，中间视为无法判定


def _last_chars(path):
    """返回每一行去掉标点后的最后一个字。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chars = [c for c in line.split() if c not in PUNCT]
            if chars:
                out.append(chars[-1])
    return out


def build_table(in_path=None, out_path=TABLE_PATH):
    """从训练集统计平仄，写入 json。返回 {字: 'ping' | 'ze'}。"""
    in_path = in_path or os.path.join(RAW, "train", "in.txt")
    out_path_src = in_path.replace("in.txt", "out.txt")

    cu = Counter(_last_chars(in_path))
    cd = Counter(_last_chars(out_path_src))

    table = {}
    for ch in set(cu) | set(cd):
        d, u = cd.get(ch, 0), cu.get(ch, 0)
        tot = d + u
        if tot < MIN_COUNT:
            continue
        r = d / tot
        if r >= HI:
            table[ch] = "ping"
        elif r <= LO:
            table[ch] = "ze"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=0, sort_keys=True)
    return table


def load_table(path=TABLE_PATH):
    """读平仄表。文件不存在时返回空字典（评测会跳过平仄）。"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    t = build_table()
    n_ping = sum(1 for v in t.values() if v == "ping")
    print("平声 %d 字 / 仄声 %d 字 / 合计 %d 字"
          % (n_ping, len(t) - n_ping, len(t)))
    print("已写入 %s（%.1f KB）"
          % (TABLE_PATH, os.path.getsize(TABLE_PATH) / 1024))
