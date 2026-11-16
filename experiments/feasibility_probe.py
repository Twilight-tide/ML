# -*- coding: utf-8 -*-
"""可行性探针：量化「改造成代码补全助手」的几项关键指标。

【只读测量】不修改 src/ 下任何文件，不训练，不覆盖任何已有产物。

测什么、为什么：

A. 下一字符准确率
   补全的本质就是「猜下一个字符」。这个数字是整个可行性的天花板。
   - top-1 准确率 = 直接采纳时对的比例
   - top-3 / top-5 = 给候选列表时的可用率

B. 按字符类别拆开看
   补全体验由「空格、换行、字母」这几类决定，平均值会骗人。

C. 上下文长度的影响
   模型训练时只见过 64 字符窗口（config.seq_len=64），
   但 position encoding 支持到 256。测 1..256 各位置的准确率，
   看「给更长的上文」到底有没有用（即能否外推）。

D. 多字符补全质量
   贪心续写 K 个字符，与真实续写比对：
   - 逐字符命中率
   - 最长正确前缀（能连续写对几个字符才出错）← 决定单次补全的价值

E. 延迟
   现有 generate() 每生成一个字符都要把整段重算一遍（没有 KV cache）。
   实测单字符耗时，判断能否做到交互级（<200ms）。

用法：
    python experiments/feasibility_probe.py
    python experiments/feasibility_probe.py --ckpt results/model.pt
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "experiments"))
sys.path.insert(0, os.path.join(BASE, "src"))
os.chdir(BASE)

from model import TinyTransformer            # noqa: E402
from eval_cross import build_test_text       # noqa: E402  复用同一测试集


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = TinyTransformer(c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
                        c["n_layers"], c["max_len"], c["dropout"])
    m.load_state_dict(ck["model_state"])
    m.eval()

    vpath = ck.get("vocab_file") or "data/vocab.json"
    v = json.load(open(vpath, encoding="utf-8"))
    itos = {int(k): ch for k, ch in v["itos"].items()}
    stoi = {ch: i for i, ch in itos.items()}
    return m, c, itos, stoi, ck.get("n_params")


def char_class(ch):
    if ch == "\n":
        return "换行"
    if ch == " ":
        return "空格"
    if ch == "\t":
        return "缩进"
    if ch.isalpha():
        return "字母"
    if ch.isdigit():
        return "数字"
    return "符号"


@torch.no_grad()
def measure_accuracy(model, ids, device, win, batch=64, max_windows=400):
    """在长窗口上一次性拿到所有位置的下一字符预测。

    因为模型是因果的，logits[t] 只依赖 <=t，所以一次 forward
    就能同时得到窗口内每个位置的预测——效率比逐位置跑高几百倍。

    返回 (top1命中, top3命中, top5命中, 目标类别列表, 位置列表)
    """
    n_win = min(max_windows, len(ids) // win)
    starts = np.arange(n_win) * win
    hit1 = np.zeros((n_win, win), dtype=bool)
    hit3 = np.zeros((n_win, win), dtype=bool)
    hit5 = np.zeros((n_win, win), dtype=bool)

    for b0 in range(0, n_win, batch):
        blk = starts[b0:b0 + batch]
        X = np.stack([ids[s:s + win] for s in blk])
        x = torch.tensor(X, dtype=torch.long, device=device)
        logits = model(x)                      # (B, win, V)

        # 位置 t 的预测由 logits[t-1] 给出，目标是 ids[t]
        pred = logits[:, :-1, :]               # (B, win-1, V)
        tgt = x[:, 1:]                         # (B, win-1)
        topk = torch.topk(pred, 5, dim=-1).indices
        for k, arr in ((1, hit1), (3, hit3), (5, hit5)):
            arr[b0:b0 + len(blk), 1:] = (topk[:, :, :k] == tgt.unsqueeze(-1)) \
                .any(-1).cpu().numpy()
    return hit1, hit3, hit5


@torch.no_grad()
def greedy_continuation(model, ids, device, ctx=64, K=24, n=300, seed=0):
    """贪心续写 K 个字符，与真实续写比对。

    返回 (逐字符命中率, 每样本最长正确前缀的数组)
    """
    rng = np.random.default_rng(seed)
    pos = rng.integers(ctx, len(ids) - K - 1, size=n)

    total_hit, total_cnt = 0, 0
    prefixes = []
    for p in pos:
        prompt = torch.tensor([ids[p - ctx:p].tolist()], device=device)
        truth = ids[p:p + K].tolist()
        cur = prompt
        best = 0
        for j in range(K):
            logits = model(cur)[:, -1, :]
            nxt = int(logits.argmax(-1).item())
            if j == 0 or best == j:
                if nxt == truth[j]:
                    best = j + 1
            total_hit += int(nxt == truth[j])
            total_cnt += 1
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        prefixes.append(best)
    return total_hit / total_cnt, np.array(prefixes)


@torch.no_grad()
def measure_latency(model, stoi, itos, device, prompt="def ", n_chars=50, reps=3):
    """实测：现有 generate() 这种「每步重算整段」的写法，单字符多久。"""
    ids = [stoi[c] for c in prompt if c in stoi]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # 预热
    model.generate(x, 5, temperature=1.0, top_k=None)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        model.generate(x, n_chars, temperature=1.0, top_k=None)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    return dt, dt / n_chars * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/model_big.pt")
    ap.add_argument("--max-chars", type=int, default=160_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, c, itos, stoi, n_params = load(args.ckpt)
    model = model.to(device)

    import sysconfig
    stdlib = sysconfig.get_paths()["stdlib"]
    text, _ = build_test_text(stdlib, n_files=40)
    text = text[:args.max_chars]
    ids = np.array([stoi[ch] for ch in text if ch in stoi], dtype=np.int64)
    id_list = torch.tensor(ids, dtype=torch.long)

    print("=" * 76)
    print("代码补全可行性探针（只读测量）")
    print("=" * 76)
    print("检查点    : %s" % args.ckpt)
    print("参数量    : %s" % format(n_params, ","))
    print("结构      : d_model=%d  n_heads=%d  n_layers=%d  d_ff=%d"
          % (c["d_model"], c["n_heads"], c["n_layers"], c["d_ff"]))
    print("训练窗口  : seq_len=%d  |  PE 上限 max_len=%d" % (c["seq_len"], c["max_len"]))
    print("设备      : %s" % device)
    print("测试集    : 标准库 Lib/test，%s 字符（训练时被排除，未见过）"
          % format(len(ids), ","))
    print()

    # ---------------- A + B ----------------
    print("=" * 76)
    print("A. 下一字符准确率（在训练窗口 64 之内）")
    print("=" * 76)
    win = c["max_len"]
    hit1, hit3, hit5 = measure_accuracy(model, ids, device, win)
    # 只看位置 1..63，即训练时见过的上下文长度范围
    m_tr = slice(1, min(64, win))
    a1 = hit1[:, m_tr].mean()
    a3 = hit3[:, m_tr].mean()
    a5 = hit5[:, m_tr].mean()
    print("  top-1  %.2f%%     ← 直接采纳的准确率" % (a1 * 100))
    print("  top-3  %.2f%%" % (a3 * 100))
    print("  top-5  %.2f%%     ← 给候选列表的可用率" % (a5 * 100))
    print()

    print("  按目标字符类别拆开（top-1）：")
    cls_stat = {}
    for t in range(1, min(64, win)):
        for w in range(hit1.shape[0]):
            ch = itos[int(ids[w * win + t])]
            k = char_class(ch)
            cls_stat.setdefault(k, [0, 0])
            cls_stat[k][0] += int(hit1[w, t])
            cls_stat[k][1] += 1
    print("  %-8s %8s %10s" % ("类别", "占比", "top-1"))
    print("  " + "-" * 30)
    total = sum(v[1] for v in cls_stat.values())
    for k, (h, n) in sorted(cls_stat.items(), key=lambda kv: -kv[1][1]):
        print("  %-8s %7.1f%% %9.2f%%" % (k, n / total * 100, h / n * 100))
    print()

    # ---------------- C ----------------
    print("=" * 76)
    print("C. 上下文长度的影响（位置 1..%d 的准确率）" % win)
    print("=" * 76)
    bands = [(1, 8), (8, 16), (16, 32), (32, 64), (64, 96), (96, 128),
             (128, 192), (192, win)]
    print("  %-14s %10s %10s" % ("上下文长度", "top-1", "样本数"))
    print("  " + "-" * 38)
    for lo, hi in bands:
        if hi > win:
            continue
        acc = hit1[:, lo:hi].mean()
        print("  %-14s %9.2f%% %10s"
              % ("%d-%d" % (lo, hi), acc * 100, format(hit1.shape[0] * (hi - lo), ",")))
    print()
    print("  注：位置 >64 的部分模型训练时从未见过（seq_len=64），")
    print("      这里测的是位置编码能否外推，而非「长上下文能力」。")
    print()

    # ---------------- D ----------------
    print("=" * 76)
    print("D. 多字符补全质量（贪心续写，看能连续写对几个字符）")
    print("=" * 76)
    for K in (8, 16, 24):
        acc, pref = greedy_continuation(model, id_list, device, ctx=64, K=K,
                                        n=300, seed=args.seed)
        print("  续写 %2d 字符：逐字符命中 %5.1f%%   最长正确前缀 中位 %2d / 均值 %4.1f"
              % (K, acc * 100, int(np.median(pref)), pref.mean()))
    print()
    print("  说明：最长正确前缀 = 从起点开始连续写对多少个字符才第一次出错。")
    print("        这个值决定「一次补全能帮上多少忙」。")
    print()

    # ---------------- E ----------------
    print("=" * 76)
    print("E. 延迟（现有 generate 写法：每步重算整段，无 KV cache）")
    print("=" * 76)
    dt, ms = measure_latency(model, stoi, itos, device)
    print("  单字符耗时 : %.1f ms" % ms)
    print("  一次 20 字符补全 ≈ %.0f ms" % (ms * 20))
    print("  交互体验参考：<100ms 无感，<300ms 可接受，>500ms 明显卡顿")
    print()
    print("=" * 76)
    print("测量完成。（本脚本不修改任何文件，也不训练）")
    print("=" * 76)


if __name__ == "__main__":
    main()
