# -*- coding: utf-8 -*-
"""数据准备：从本地 Python 标准库源码构建字符级语料。

为什么用标准库源码作语料：
  1. 完全本地，不依赖网络（本项目所在网络环境下 GitHub 等源不稳定）
  2. 规模合适（约 10 MB），足以让 10 万参数的模型学到结构
  3. 生成结果肉眼可判：模型输出的是否像 Python 代码，一眼就能看出来
"""

import os
import sys
import json
import sysconfig
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config

# 跳过这些目录：测试代码冗长重复，且含大量无意义的字符串
SKIP_DIRS = {"test", "tests", "__pycache__", "idlelib", "lib2to3",
             "site-packages", "turtledemo", "distutils"}

MAX_TOTAL_BYTES = 6 * 1024 * 1024   # 取前 6 MB，避免语料过大拖慢训练


def collect_corpus(root: str, max_bytes: int = MAX_TOTAL_BYTES):
    """遍历标准库目录，收集 .py 源码并拼成一个字符串。"""
    chunks, total = [], 0
    files = []
    for r, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for n in sorted(names):
            if n.endswith(".py"):
                files.append(os.path.join(r, n))

    for p in files:
        try:
            if os.path.getsize(p) < 500:
                continue
            t = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        chunks.append(t)
        total += len(t)
        if total >= max_bytes:
            break

    return "\n\n".join(chunks)


def build_vocab(text: str, vocab_size: int):
    """按字符频率取前 vocab_size 个，构成字符表。"""
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    # 高频在前，保证常见字符一定被收录
    ordered = sorted(freq.items(), key=lambda kv: -kv[1])
    chars = [c for c, _ in ordered[:vocab_size]]
    itos = {i: c for i, c in enumerate(chars)}
    stoi = {c: i for i, c in itos.items()}
    return itos, stoi


def main():
    cfg = Config()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base)

    stdlib = sysconfig.get_paths()["stdlib"]
    print("标准库目录:", stdlib)

    print("正在收集语料 ...")
    text = collect_corpus(stdlib)
    print("语料字符数: %s (%.2f MB)" % (format(len(text), ","), len(text) / 1024 / 1024))
    if len(text) < 100000:
        raise SystemExit("语料过小，请检查标准库路径")

    itos, stoi = build_vocab(text, cfg.vocab_size)

    # 统计词表覆盖率（未收录字符会被丢弃）
    covered = sum(1 for ch in text if ch in stoi)
    print("词表大小: %d" % len(itos))
    print("字符覆盖率: %.3f%%" % (100 * covered / len(text)))

    os.makedirs("data", exist_ok=True)
    with open(cfg.corpus_file, "w", encoding="utf-8") as f:
        f.write(text)
    with open(cfg.vocab_file, "w", encoding="utf-8") as f:
        json.dump({"itos": {str(k): v for k, v in itos.items()}}, f,
                  ensure_ascii=False, indent=1)

    print()
    print("已写入:")
    print("  %s (%.2f MB)" % (cfg.corpus_file, os.path.getsize(cfg.corpus_file) / 1024 / 1024))
    print("  %s" % cfg.vocab_file)
    print()
    print("词表中最常见的 30 个字符:")
    print("  " + " ".join(repr(itos[i]) for i in range(min(30, len(itos)))))


if __name__ == "__main__":
    main()
