# -*- coding: utf-8 -*-
"""文本生成应用。

加载训练好的模型，按自回归方式逐字符生成。这是本项目最直观的验收方式：
看模型写出的 Python 代码像不像。

用法：
    python src/generate.py                        # 用默认提示词
    python src/generate.py --prompt "def "        # 指定开头
    python src/generate.py --prompt "import " --tokens 300 --temperature 0.8
"""

import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from model import TinyTransformer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)


def load_model(ckpt_path):
    """返回 (模型, 配置字典, 设备)。配置从检查点里读，保证结构一致。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = TinyTransformer(
        c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
        c["n_layers"], c["max_len"], c["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    return model, c, dev, ckpt


def load_vocab(ckpt, override=None):
    """定位词表：命令行 > 检查点里记录的路径 > 默认路径。

    为什么需要这个：换语料后词表 id 映射会变（新旧两版 96 个位置只有 20 个相同），
    用错词表生成出来就是完全乱码，而且不会报错。
    """
    for cand in (override, ckpt.get("vocab_file"), "data/vocab.json"):
        if cand and os.path.exists(cand):
            v = json.load(open(cand, encoding="utf-8"))
            return {int(k): ch for k, ch in v["itos"].items()}, cand
    raise SystemExit("找不到词表文件，请用 --vocab 指定")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/model.pt")
    ap.add_argument("--prompt", default="def ")
    ap.add_argument("--tokens", type=int, default=200, help="生成多少个字符")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="越低越保守，越高越随机")
    ap.add_argument("--top-k", type=int, default=40, help="只从概率最高的 k 个中采样")
    ap.add_argument("--num", type=int, default=3, help="生成几个样本")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--vocab", default=None,
                    help="词表文件；默认用检查点里记录的，旧检查点回退 data/vocab.json")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit("找不到模型文件 %s，请先运行 python src/train.py" % args.ckpt)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, c, dev, ckpt = load_model(args.ckpt)
    itos, vocab_path = load_vocab(ckpt, args.vocab)
    stoi = {ch: i for i, ch in itos.items()}
    if len(itos) != c["vocab_size"]:
        raise SystemExit("词表大小 %d 与模型 %d 不符，请检查 --vocab"
                         % (len(itos), c["vocab_size"]))

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 74)
    print("文本生成")
    print("=" * 74)
    print("  模型参数: %s   层数: %d   头数: %d   d_model: %d" % (
        format(n_params, ","), c["n_layers"], c["n_heads"], c["d_model"]))
    print("  设备    : %s" % dev)
    print("  词表    : %s (%d 字符)" % (vocab_path, len(itos)))
    print("  提示词  : %r" % args.prompt)
    print("  温度    : %.2f   top-k: %s   生成长度: %d 字符" % (
        args.temperature, args.top_k, args.tokens))
    print()

    # 词表外的字符直接跳过
    ids = [stoi[ch] for ch in args.prompt if ch in stoi]
    if not ids:
        ids = [stoi.get(" ", 0)]
    x = torch.tensor([ids], dtype=torch.long, device=dev)

    for i in range(args.num):
        out = model.generate(x, args.tokens, temperature=args.temperature,
                             top_k=args.top_k)
        text = "".join(itos[j] for j in out[0].tolist())
        print("-" * 74)
        print("样本 %d" % (i + 1))
        print("-" * 74)
        print(text)
        print()


if __name__ == "__main__":
    main()
