# -*- coding: utf-8 -*-
"""对联规则评价：四条可自动检验的规则。

用法：
    python couplet/src/evaluate.py --ckpt results/run2/best.pt --n 200
    python couplet/src/evaluate.py --ckpt results/run2/best.pt --n 200 --ref-only
"""

import os
import sys
import random
import argparse
from collections import Counter

import torch

from dataset import load_stoi
from model import (Seq2SeqTransformer, make_pad_mask, make_causal_mask,
                   SOS_ID, EOS_ID, PAD_ID)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

PUNCT = set("，、。？！；：（）《》「」『』【】…—〇·-")

from tone import load_table

_TONE = {}


def strip_punct(text):
    """去掉标点，只留汉字。字数、重字都按这个算。"""
    return [c for c in text if c not in PUNCT]


def rule_length(up, down):
    """规则 1：上下联字数相等。"""
    return len(strip_punct(up)) == len(strip_punct(down))


def rule_dup(down):
    """规则 2：无不规则重字。返回重复的 {字: 次数}，空字典表示通过。"""
    c = Counter(strip_punct(down))
    return {ch: n for ch, n in c.items() if n >= 2}


def tone_of(ch):
    """返回 'ping' / 'ze' / None。

    查从语料反推的平仄表（见 tone.py），不依赖拼音库。
    """
    return _TONE.get(ch)


def rule_tone(up, down):
    """规则 3：上仄下平。上联末字仄声，下联末字平声。
    无法判定时返回 None（不参与统计）。
    """
    u, d = strip_punct(up), strip_punct(down)
    if not u or not d:
        return None
    tu, td = tone_of(u[-1]), tone_of(d[-1])
    if tu is None or td is None:
        return None
    return tu == "ze" and td == "ping"


@torch.no_grad()
def greedy(model, ids, device, max_len=40):
    src = torch.tensor([ids], device=device)
    src_mask = make_pad_mask(src)
    enc_out, _ = model.encode(src, src_mask)
    ys = torch.full((1, 1), SOS_ID, dtype=torch.long, device=device)
    for _ in range(max_len):
        tm = make_causal_mask(ys.size(1), device)
        out, _, _ = model.decode(ys, enc_out, src_mask, tm)
        nxt = model.head(out[:, -1]).argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], 1)
        if nxt.item() == EOS_ID:
            break
    return "".join(itos_get(x) for x in ys[0].tolist()[1:] if x not in (EOS_ID, PAD_ID))


_ITOS = None


def itos_get(i):
    return _ITOS[i]


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = Seq2SeqTransformer(**ck["config"]).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, ck


def report(name, pairs, stoi):
    """统计一批 (上联, 下联) 的四条规则通过率。"""
    n = len(pairs)
    ok_len = ok_dup = 0
    tone_total = tone_ok = 0
    dup_examples = []

    for up, down in pairs:
        if rule_length(up, down):
            ok_len += 1
        d = rule_dup(down)
        if not d:
            ok_dup += 1
        elif len(dup_examples) < 2:
            dup_examples.append((up, down, d))
        t = rule_tone(up, down)
        if t is not None:
            tone_total += 1
            tone_ok += int(t)

    print("=" * 64)
    print(name)
    print("=" * 64)
    print("  样本数        : %d" % n)
    print("  字数相等      : %d/%d = %5.1f%%" % (ok_len, n, ok_len / n * 100))
    print("  无重字        : %d/%d = %5.1f%%" % (ok_dup, n, ok_dup / n * 100))
    if tone_total:
        print("  上仄下平      : %d/%d = %5.1f%%  （可判定 %d 条）"
              % (tone_ok, tone_total, tone_ok / tone_total * 100, tone_total))
    else:
        print("  上仄下平      : 跳过（平仄表为空，先运行 tone.py 生成）")
    for up, down, d in dup_examples:
        print("    重字例: %s → %s  %s" % (up, down, d))
    print()


def main():
    global _ITOS, _TONE
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "results", "run2", "best.pt"))
    ap.add_argument("--n", type=int, default=200, help="抽取测试样本数")
    ap.add_argument("--m", type=int, default=2000, help="抽样池上限")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stoi, _ITOS = load_stoi(os.path.join(BASE, "data", "vocab.json"))
    _TONE = load_table()
    print("平仄表 %d 字" % len(_TONE))

    RAW = os.path.join(BASE, "data", "raw", "couplet")
    with open(os.path.join(RAW, "test", "in.txt"), encoding="utf-8") as f:
        te_in = [l.split() for l in f]
    with open(os.path.join(RAW, "test", "out.txt"), encoding="utf-8") as f:
        te_out = [l.split() for l in f]

    pool = []
    for i in range(min(args.m, len(te_in))):
        if any(c not in stoi for c in te_in[i] + te_out[i]):
            continue
        if not (3 <= len(te_in[i]) <= 25):
            continue
        pool.append(i)
    random.seed(args.seed)
    idx = random.sample(pool, min(args.n, len(pool)))

    # 1) 参考答案作为判据校准
    ref_pairs = [("".join(te_in[i]), "".join(te_out[i])) for i in idx]
    report("参考答案（判据校准：这里应该接近 100%）", ref_pairs, stoi)

    # 2) 模型生成
    model, ck = load_model(args.ckpt, device)
    print("模型 %s  step %d  val %.4f"
          % (os.path.basename(os.path.dirname(args.ckpt)), ck.get("step", -1),
             ck.get("val_loss", float("nan"))))
    print()

    gen_pairs = []
    for i in idx:
        up = te_in[i]
        gen = greedy(model, [stoi[c] for c in up], device)
        gen_pairs.append(("".join(up), gen))
    report("模型生成", gen_pairs, stoi)


if __name__ == "__main__":
    main()
