# -*- coding: utf-8 -*-
"""M1 验收：数据管道的五项检查。

用法：python couplet/experiments/check_data.py
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)              # couplet/
sys.path.insert(0, os.path.join(BASE, "src"))

from prepare_data import load_file, filter_pairs, build_vocab, encode_pair, decode

VOCAB_PATH = os.path.join(BASE, "data", "vocab.json")

# 期望值（已实测，见 docs/01-技术方案.md §4）
EXPECT_RAW = 770491
EXPECT_FILTERED = 755993
EXPECT_VOCAB = 9014


def check(name, actual, expected):
    """打印一项检查结果，返回是否通过。"""
    ok = actual == expected
    mark = "OK  " if ok else "FAIL"
    print("  [%s] %-24s %s" % (mark, name, format(actual, ",") if isinstance(actual, int) else actual))
    if not ok:
        print("         期望 %s" % expected)
    return ok


def main():
    print("=" * 64)
    print("M1 验收：数据管道")
    print("=" * 64)
    print()

    ups_raw = load_file("train/in.txt")
    downs_raw = load_file("train/out.txt")

    # ---- 检查 1：原始条数 + 脏数据 ----
    print("1. 原始数据")
    ok1 = check("训练集条数", len(ups_raw), EXPECT_RAW)
    ok1 &= check("上联条数 == 下联条数", len(ups_raw) == len(downs_raw), True)

    dirty = [(i, u, d) for i, (u, d) in enumerate(zip(ups_raw, downs_raw))
             if len(u) != len(d)]
    ok1 &= check("脏数据（字数不等）", len(dirty), 0)
    if dirty:
        print("         前 3 条异常：")
        for i, u, d in dirty[:3]:
            print("           第 %d 行：%s / %s" % (i, "".join(u), "".join(d)))
    print()

    # ---- 检查 2：过滤 ----
    print("2. 过滤 1-2 字短联")
    ups, downs, dropped = filter_pairs(ups_raw, downs_raw)
    ok2 = check("过滤后条数", len(ups), EXPECT_FILTERED)
    ok2 &= check("丢弃条数", dropped, EXPECT_RAW - EXPECT_FILTERED)
    print()

    # ---- 检查 3：词表 ----
    print("3. 词表")
    itos, stoi = build_vocab(ups, downs)
    ok3 = check("词表大小", len(itos), EXPECT_VOCAB)
    ok3 &= check("id 0 = <pad>", itos[0], "<pad>")
    ok3 &= check("id 1 = <s>", itos[1], "<s>")
    ok3 &= check("id 2 = </s>", itos[2], "</s>")
    print()

    # ---- 检查 4：与已保存的文件一致（可复现性）----
    print("4. 与 data/vocab.json 比对")
    if not os.path.exists(VOCAB_PATH):
        print("  [FAIL] 找不到 vocab.json，先运行 prepare_data.py")
        ok4 = False
    else:
        saved = {int(k): v for k, v in
                 json.load(open(VOCAB_PATH, encoding="utf-8"))["itos"].items()}
        ok4 = check("重新生成的词表 == 已保存的", saved == itos, True)
        if not ok4:
            diff = [(i, saved.get(i), itos.get(i))
                    for i in range(max(len(saved), len(itos)))
                    if saved.get(i) != itos.get(i)]
            print("         前 5 个差异（id, 文件里, 重新生成）:")
            for row in diff[:5]:
                print("           ", row)
    print()

    # ---- 检查 5：编码 / 解码来回转换 ----
    print("5. id → 文本 → id 来回转换")
    ok5 = True
    for idx in (0, 1000, len(ups) - 1):
        up, down = ups[idx], downs[idx]
        enc, dec_in, dec_tgt = encode_pair(up, down, stoi)

        back_up = decode(enc, itos)
        back_down = decode(dec_in[1:], itos)      # 跳过开头的 <s>

        good = (back_up == "".join(up)
                and back_down == "".join(down)
                and len(dec_in) == len(dec_tgt) == len(down) + 1
                and dec_in[0] == stoi["<s>"]
                and dec_tgt[-1] == stoi["</s>"])
        ok5 &= good
        print("  [%s] 第 %-7d 条  %s" % ("OK  " if good else "FAIL", idx, "".join(up)))
        if not good:
            print("         上联还原: %s" % back_up)
            print("         下联还原: %s" % back_down)
            print("         dec_in 长度 %d, dec_tgt 长度 %d, 期望 %d"
                  % (len(dec_in), len(dec_tgt), len(down) + 1))
    print()

    # ---- 汇总 ----
    print("=" * 64)
    total = all([ok1, ok2, ok3, ok4, ok5])
    if total:
        print("全部通过 —— M1 完成")
    else:
        print("有检查未通过，见上方 FAIL 项")
    print("=" * 64)


if __name__ == "__main__":
    main()
