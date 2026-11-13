# -*- coding: utf-8 -*-
"""n-gram 基线：用一个「数数」的模型，给 Transformer 的困惑度找一个参照系。

为什么要这个脚本
----------------
Transformer 的 ppl 是 4.46。这个数字本身没有意义——它可能是「很好」，
也可能是「字符级任务本来就该这么低」。唯一能判断的办法是：
让一个完全不同的、笨得多的模型做同一道题，看它能拿多少分。

n-gram 就是那个笨模型。它的全部本事只有一句：**数一数历史上「片段 A 后面
跟着字符 c」出现过多少次**，然后按比例押注。没有注意力、没有梯度、没有
反向传播，甚至连神经网络都不是。

它笨，但它是语言建模的经典强基线（2000 年代统计机器翻译的主力）。
如果 50 万参数的 Transformer 打不过它，那说明这一堆矩阵乘法白算了。

模型定义（插值绝对折扣）
------------------------
    p(c | h) = max(count(h,c) - D, 0) / count(h)          ← 见过的，打个折
             + D * U(h) / count(h) * p(c | h')            ← 剩下的概率质量

其中 h' 是 h 去掉最左边一个字符（退一步看更短的上下文），U(h) 是 h 后面
出现过多少种不同字符。递归到一元模型为止。

- 减 D：叫「绝对折扣」。因为见过一次不代表下次一定见，抠掉一点分给没见过的。
  这是 Kneser-Ney 平滑的简化版（不做 continuation count）。
- 插值而不是回退：哪怕 h 见过，也要分一点概率给短上下文，避免「见过一次就
  押 100%」的自信错误。

公平性设计（重要）
------------------
1. 训练语料与 Transformer 完全相同（data/corpus_big.txt）。
2. 测试集直接复用 experiments/eval_cross.py 的 build_test_text()，保证是
   同一批文件、同一个随机种子、同一段文本。
3. 只用词表内的 96 个字符，映射成和 Transformer 完全一样的 id 序列，
   所以两边预测的是同一个字符序列，loss 可以直接比。
4. 上下文每 64 个字符清零一次，与 Transformer 的 64 长度窗口一致。
   （另给一组「不清零」的结果作为 n-gram 的能力上限。）

输出：nats/字符 与 ppl，和 eval_cross.py 的口径完全一致。

用法：
    python experiments/ngram_baseline.py
    python experiments/ngram_baseline.py --corpus data/corpus.txt --vocab data/vocab.json
    python experiments/ngram_baseline.py --max-order 5 --window 64 --save results/ngram_baseline.csv
"""

import os
import sys
import json
import time
import argparse

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "experiments"))
sys.path.insert(0, os.path.join(BASE, "src"))
os.chdir(BASE)

# 直接复用跨语料评测的测试集构造函数，保证两边看到的是同一段文本
from eval_cross import build_test_text  # noqa: E402


# ---------------------------------------------------------------- n-gram 模型

class NgramLM:
    """字符级 n-gram，插值绝对折扣。

    内部用「整数打包」表示 n-gram：把 n 个字符 id 拼成一个整数
    （base = 97，多出来的一路留给 PAD）。这样就能把整条语料的 n-gram
    放进一个 int64 数组，交给 numpy 排序计数，不必用 Python 字典。
    """

    def __init__(self, vocab_size, max_order=5, discount=0.75):
        self.V = vocab_size
        self.PAD = vocab_size          # 一个词表外的特殊符号，只用来标记「没有上下文」
        self.base = vocab_size + 1     # 打包进制
        self.N = max_order
        self.D = discount
        self.counts = {}               # order -> (uniq_keys, counts)
        self.ctx = {}                  # order -> (uniq_ctx, totals, n_unique_follow)

    # ---------------- 打包 ----------------
    def _prepare(self, ids, block_len):
        """在每段开头插入 N-1 个 PAD，返回 (打包用的数组, 有效位置掩码)。

        block_len=None 表示整条序列一段（上下文只在开头被截断）。
        否则每 block_len 个字符重新开始一段，模拟 Transformer 的定长窗口。
        """
        n_pad = self.N - 1
        if block_len is None:
            blocks = [ids]
        else:
            blocks = [ids[i:i + block_len] for i in range(0, len(ids), block_len)]

        pad = np.full(n_pad, self.PAD, dtype=np.int64)
        parts, valid = [], []
        for b in blocks:
            parts.append(np.concatenate([pad, b]))
            # 每段的第一个真实字符是「无上下文」位置，Transformer 也不会这样预测
            valid.append(np.concatenate([np.zeros(n_pad + 1, dtype=bool),
                                         np.ones(len(b) - 1, dtype=bool)]))
        return np.concatenate(parts), np.concatenate(valid)

    def _keys(self, arr, order):
        """返回每个位置「长度为 order 的 n-gram」的打包整数，无效位置为负。"""
        keys = arr
        for _ in range(order - 1):
            nxt = np.empty_like(arr)
            nxt[0] = -1
            nxt[1:] = keys[:-1] * self.base + arr[1:]
            keys = nxt
        keys[keys < 0] = -1
        return keys

    # ---------------- 训练（这里只是数数） ----------------
    def train(self, ids):
        t0 = time.time()
        arr, _ = self._prepare(ids, block_len=None)
        total_entries = 0
        for n in range(1, self.N + 1):
            keys = self._keys(arr, n)
            keys = keys[keys >= 0]
            uniq, cnt = np.unique(keys, return_counts=True)
            self.counts[n] = (uniq, cnt.astype(np.int64))
            total_entries += len(uniq)

            # 按上下文（去掉最后一位）汇总：总次数 与 后继种类数
            ctx_keys = uniq // self.base          # 已升序
            starts = np.flatnonzero(np.diff(ctx_keys)) + 1
            starts = np.concatenate([[0], starts])
            totals = np.add.reduceat(cnt, starts).astype(np.int64)
            n_follow = np.diff(np.concatenate([starts, [len(uniq)]])).astype(np.int64)
            self.ctx[n] = (ctx_keys[starts], totals, n_follow)

            print("  %d-gram: %10s 条  (%.1fs)"
                  % (n, format(len(uniq), ","), time.time() - t0))
        self.total_entries = total_entries
        print("  计数表共 %s 条，用时 %.1fs"
              % (format(total_entries, ","), time.time() - t0))

    # ---------------- 查表 ----------------
    @staticmethod
    def _lookup(keys_sorted, vals, q):
        idx = np.searchsorted(keys_sorted, q)
        idx = np.clip(idx, 0, len(keys_sorted) - 1)
        hit = keys_sorted[idx] == q
        return np.where(hit, vals[idx], 0)

    def logprob(self, ids, block_len=64, upto=None):
        """返回 (平均负对数似然, ppl, 评估字符数, 全阶上下文未见过比例)。

        upto 用来只算到第 k 阶，方便画出「阶数 → ppl」的曲线。
        """
        upto = upto or self.N
        arr, valid = self._prepare(ids, block_len)
        pos = np.flatnonzero(valid)

        p = np.full(len(pos), 1.0 / self.V)   # 0 阶：均匀分布兜底
        unseen_ctx_rate = None

        for n in range(1, upto + 1):
            keys = self._keys(arr, n)[pos]
            uniq, cnt = self.counts[n]
            c_hc = self._lookup(uniq, cnt, keys)

            ctx_keys, totals, n_follow = self.ctx[n]
            c_h = self._lookup(ctx_keys, totals, keys // self.base)
            u_h = self._lookup(ctx_keys, n_follow, keys // self.base)

            if n == upto:
                unseen_ctx_rate = float(np.mean(c_h == 0))

            seen = c_h > 0
            safe_tot = np.where(seen, c_h, 1)
            p_n = (np.maximum(c_hc - self.D, 0) + self.D * u_h * p) / safe_tot
            p = np.where(seen, p_n, p)

        p = np.maximum(p, 1e-12)
        nll = float(-np.mean(np.log(p)))
        return nll, float(np.exp(nll)), len(pos), unseen_ctx_rate

    # ---------------- 生成 ----------------
    def next_probs(self, ctx, upto=None):
        """给定上下文（id 列表），返回词表 96 个字符各自的插值概率。

        评测时只需要「真实那个字符」的概率，生成时需要全部候选，所以单独写一份。
        """
        upto = upto or self.N
        ctx = list(ctx[-(self.N - 1):])
        while len(ctx) < self.N - 1:
            ctx.insert(0, self.PAD)

        cand = np.arange(self.V, dtype=np.int64)
        p = np.full(self.V, 1.0 / self.V)
        for n in range(1, upto + 1):
            h = ctx[-(n - 1):] if n > 1 else []
            key_h = 0
            for s in h:
                key_h = key_h * self.base + s
            keys = key_h * self.base + cand

            uniq, cnt = self.counts[n]
            c_hc = self._lookup(uniq, cnt, keys)
            ctx_keys, totals, n_follow = self.ctx[n]
            q = np.full(self.V, key_h, dtype=np.int64)
            c_h = self._lookup(ctx_keys, totals, q)
            u_h = self._lookup(ctx_keys, n_follow, q)

            seen = c_h > 0
            safe = np.where(seen, c_h, 1)
            p = np.where(seen, (np.maximum(c_hc - self.D, 0) + self.D * u_h * p) / safe, p)
        return p / p.sum()

    def sample(self, prompt, n_chars, stoi, itos, temperature=0.8, seed=0):
        rng = np.random.default_rng(seed)
        ctx = [stoi[c] for c in prompt if c in stoi]
        out = [prompt]
        for _ in range(n_chars):
            p = self.next_probs(ctx)
            if temperature != 1.0:
                p = p ** (1.0 / temperature)
                p = p / p.sum()
            nxt = int(rng.choice(self.V, p=p))
            out.append(itos[nxt])
            ctx.append(nxt)
        return "".join(out)


# ---------------------------------------------------------------- 评测流程

def load_vocab(path):
    v = json.load(open(path, encoding="utf-8"))
    stoi = {ch: int(i) for i, ch in v["itos"].items()}
    itos = {int(i): ch for i, ch in v["itos"].items()}
    return stoi, itos, len(v["itos"])


def to_ids(text, stoi):
    """与 eval_cross.py 完全一致：丢掉词表外字符，只保留 id。"""
    return np.array([stoi[ch] for ch in text if ch in stoi], dtype=np.int64)


def tune(corpus, stoi, V, orders, discounts, dev_chars=200_000, window=64):
    """在语料的后 10%（和 Transformer 的验证集同一段）上剪折扣系数 D。

    计数与 D 无关，所以只数一遍，之后换 D 只是重新插值，代价可以忽略。
    """
    ids = to_ids(corpus, stoi)
    cut = int(len(ids) * 0.9)
    train_ids, dev_ids = ids[:cut], ids[cut:cut + dev_chars]

    print("=" * 74)
    print("调参：在验证集上剪折扣系数 D（不碰测试集）")
    print("=" * 74)
    print("训练段 %s 字符 / 验证段 %s 字符（语料后 10%%，与 Transformer 的 val 同一段）"
          % (format(len(train_ids), ","), format(len(dev_ids), ",")))
    print()

    lm = NgramLM(V, max_order=max(orders), discount=discounts[0])
    lm.train(train_ids)

    print("\n%-8s" % "阶数", end="")
    for d in discounts:
        print("%10s" % ("D=%.2f" % d), end="")
    print()
    print("-" * 74)
    best = (None, None, 1e9)
    for n in orders:
        print("%-8s" % ("%d-gram" % n), end="")
        for d in discounts:
            lm.D = d
            _, ppl, _, _ = lm.logprob(dev_ids, block_len=window, upto=n)
            print("%10.3f" % ppl, end="")
            if ppl < best[2]:
                best = (n, d, ppl)
        print()
    print("-" * 74)
    print("最优：%d-gram，D=%.2f，验证 ppl %.3f" % best)
    lm.D = best[1]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_big.txt")
    ap.add_argument("--vocab", default="data/vocab_big.json")
    ap.add_argument("--max-order", type=int, default=5)
    ap.add_argument("--window", type=int, default=64,
                    help="上下文清零的窗口长度，与 Transformer 的 seq_len 对齐")
    ap.add_argument("--discount", type=float, default=0.75)
    ap.add_argument("--files", type=int, default=40, help="测试集采样文件数")
    ap.add_argument("--max-chars", type=int, default=400_000,
                    help="最多评测多少字符（与 eval_cross.py 默认一致）")
    ap.add_argument("--tag", default="", help="结果表里的名字")
    ap.add_argument("--save", default="", help="把结果追加写入这个 csv")
    ap.add_argument("--tune", action="store_true",
                    help="先在验证集上剪折扣系数 D，再用最优值训练")
    ap.add_argument("--sample", nargs="*", default=None,
                    help="给定开头，用 n-gram 续写，例如 --sample 'def ' 'class '")
    ap.add_argument("--sample-chars", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    corpus = open(args.corpus, encoding="utf-8").read()
    stoi, itos, V = load_vocab(args.vocab)
    train_ids = to_ids(corpus, stoi)

    import sysconfig
    stdlib = sysconfig.get_paths()["stdlib"]
    text, n_avail = build_test_text(stdlib, n_files=args.files)
    text = text[:args.max_chars]          # 与 eval_cross.py 取同一段前缀
    test_ids = to_ids(text, stoi)

    print("=" * 74)
    print("n-gram 基线")
    print("=" * 74)
    print("语料      : %s（%s 字符，词表内 %s）"
          % (args.corpus, format(len(corpus), ","), format(len(train_ids), ",")))
    print("词表      : %s（%d 个字符）" % (args.vocab, V))
    print("测试集    : %s\\test 的 %d 个文件，%s 字符（词表内 %s）"
          % (stdlib, args.files, format(len(text), ","), format(len(test_ids), ",")))
    print("平滑      : 插值绝对折扣 D=%.2f，最高 %d 阶" % (args.discount, args.max_order))
    print()

    if args.tune:
        best = tune(corpus, stoi, V, orders=list(range(2, args.max_order + 1)),
                    discounts=[0.25, 0.5, 0.75, 0.9], window=args.window)
        args.discount = best[1]
        print("\n用 D=%.2f 在全部语料上重新数数。" % args.discount)
        print()

    print("训练（数数）：")
    lm = NgramLM(V, max_order=args.max_order, discount=args.discount)
    lm.train(train_ids)
    print()

    rows = []
    print("评测（上下文每 %d 字符清零，与 Transformer 的 seq_len 对齐）：" % args.window)
    print("%-8s %9s %9s %11s %9s"
          % ("阶数", "loss", "ppl", "评估字符", "上文未见"))
    print("-" * 74)
    for n in range(1, args.max_order + 1):
        nll, ppl, n_eval, unseen = lm.logprob(test_ids, block_len=args.window, upto=n)
        rows.append((str(n), nll, ppl, n_eval, unseen, args.window))
        print("%-8s %9.4f %9.3f %11s %8.1f%%"
              % ("%d-gram" % n, nll, ppl, format(n_eval, ","), unseen * 100))

    nll, ppl, n_eval, unseen = lm.logprob(test_ids, block_len=None)
    rows.append(("full", nll, ppl, n_eval, unseen, "full"))
    print("-" * 74)
    print("%-8s %9.4f %9.3f %11s %8.1f%%   ← 不清零上下文（上限）"
          % ("%d-gram" % args.max_order, nll, ppl, format(n_eval, ","),
             unseen * 100))
    print("-" * 74)
    print("loss 单位 nats/字符，与 eval_cross.py 同一口径；越低越好。")
    print("「该阶上下文未见过」= 这一阶上文在训练集里从未出现，只能退到短上下文猜。")

    if args.sample:
        print()
        print("=" * 74)
        print("用同一个 n-gram 续写（temperature=%.2f，每段 %d 字符）"
              % (args.temperature, args.sample_chars))
        print("=" * 74)
        for i, prompt in enumerate(args.sample):
            out = lm.sample(prompt, args.sample_chars, stoi, itos,
                            temperature=args.temperature, seed=1000 + i)
            print("── %r" % prompt)
            print(out)
            print()

    if args.save:
        new = not os.path.exists(args.save)
        with open(args.save, "a", encoding="utf-8") as f:
            if new:
                f.write("tag,corpus,vocab,order,window,loss,ppl,n_eval,"
                        "unseen_ctx_rate,table_entries\n")
            for order, nll, ppl, n_eval, unseen, win in rows:
                f.write("%s,%s,%s,%s,%s,%.4f,%.3f,%d,%.4f,%d\n"
                        % (args.tag or "ngram", os.path.basename(args.corpus),
                           os.path.basename(args.vocab), order,
                           win, nll, ppl, n_eval, unseen, lm.total_entries))
        print("\n已追加到 %s" % args.save)


if __name__ == "__main__":
    main()
