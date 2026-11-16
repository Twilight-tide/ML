# -*- coding: utf-8 -*-
import os
from collections import Counter


HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "data", "raw", "couplet")

# 常见中文标点（保留）
KEEP_PUNCT = set("，、。？！；：（）《》「」『』【】…—〇·")
# 语料里用 - 表示破折号，保留
KEEP_EXTRA = set("-")


def load_file(name):
    """读一个文件，返回每行的字列表。

    name 形如 "train/in.txt"
    """
    out = []
    with open(os.path.join(RAW, name), encoding="utf-8") as f:
        for line in f:
            out.append(line.split())  # ← 注意是 split()，不是 list()
    return out

def filter_pairs(ups, downs, min_len=3):
    """丢掉上联短于 min_len 的。返回（保留的 ups, downs, 丢弃数）。"""
    keep_u, keep_d = [], []
    for u, d in zip(ups, downs):
        if len(u) >= min_len:
            keep_u.append(u)
            keep_d.append(d)
    return keep_u, keep_d, len(ups) - len(keep_u)


def keep_char(ch):
    """字符是否保留：汉字 + 常见标点 + 连字符。

    丢弃的：私用区字符（字体渲染残留的坏字）、
            注音符号/希腊字母等明显是坏字的（如 ㄚ 本该是 丫）
    """
    if "\u4e00" <= ch <= "\u9fff":  # 汉字
        return True
    if ch in KEEP_PUNCT or ch in KEEP_EXTRA:
        return True
    return False


def build_vocab(ups, downs):
    counter = Counter()
    for seqs in (ups, downs):
        for s in seqs:
            counter.update(c for c in s if keep_char(c))    # ← 只统计保留的

    chars = ["<pad>", "<s>", "</s>"]
    rest = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    chars += [c for c, _ in rest]

    itos = {i: c for i, c in enumerate(chars)}
    stoi = {c: i for i, c in itos.items()}
    return itos, stoi



def encode_pair(up, down, stoi):
    """把一副对联编码成三个序列。

    enc     编码器输入 = 上联
    dec_in  解码器输入 = [<s>] + 下联
    dec_tgt 解码器目标 = 下联 + [</s>]
    """
    enc = [stoi[c] for c in up]
    dec_in = [stoi["<s>"]] + [stoi[c] for c in down]
    dec_tgt = [stoi[c] for c in down] + [stoi["</s>"]]
    return enc, dec_in, dec_tgt


def decode(ids, itos):
    """把 id 序列还原成字符串（用于验证）。"""
    return "".join(itos[i] for i in ids)

def save_vocab(itos, path):
    """存成 {"itos": {"0": "<pad>", ...}} 格式（和原项目一致）。"""
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"itos": {str(k): v for k, v in itos.items()}},
                  f, ensure_ascii=False, indent=1)




if __name__ == "__main__":
    ups = load_file("train/in.txt")
    print("总条数:", len(ups))
    print("第 1 条:", ups[0])
    print("它的字数:", len(ups[0]))

    downs = load_file("train/out.txt")
    ups, downs, dropped = filter_pairs(ups, downs)
    print("过滤后:", len(ups), " 丢弃:", dropped)

    itos, stoi = build_vocab(ups, downs)
    print("词表大小:", len(itos))
    print("前 3 个:", [itos[i] for i in range(3)])
    print("高频前 10:", "".join(itos[i] for i in range(3, 13)))
    # 验证：挑第一条，看能不能来回转换
    up, down = ups[0], downs[0]
    enc, dec_in, dec_tgt = encode_pair(up, down, stoi)

    print()
    print("上联原文 :", "".join(up))
    print("enc      :", enc)
    print("还原     :", decode(enc, itos))
    print("一致吗?  ", decode(enc, itos) == "".join(up))
    print()
    print("下联原文 :", "".join(down))
    print("dec_in   :", dec_in, "  长度", len(dec_in))
    print("dec_tgt  :", dec_tgt, "  长度", len(dec_tgt))

    save_vocab(itos, os.path.join(os.path.dirname(HERE), "data", "vocab.json"))
    print("词表已保存")
