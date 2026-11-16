# -*- coding: utf-8 -*-
"""数据管道：把对联语料变成可批处理的张量。

为什么不用 list of list：
  75 万条 × 3 个 Python 列表 ≈ 2 GB，而 Windows 的 DataLoader 多进程是
  spawn（不是 fork），每个 worker 会复制整个数据集 —— 直接爆内存。
  扁平 numpy int16 + 偏移索引只占 64 MB，复制 8 份也无所谓。
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from prepare_data import load_file, filter_pairs

PAD_ID, SOS_ID, EOS_ID = 0, 1, 2


class CoupletDataset(Dataset):
    """扁平数组 + 偏移索引。

    src_flat / dec_flat / tgt_flat  : 所有序列首尾相接拼成的一维 int16 数组
    *_off                           : 长度 n+1 的偏移，第 i 条是 flat[off[i]:off[i+1]]
    """

    def __init__(self, src_flat, src_off, dec_flat, dec_off, tgt_flat, tgt_off):
        self.src_flat, self.src_off = src_flat, src_off
        self.dec_flat, self.dec_off = dec_flat, dec_off
        self.tgt_flat, self.tgt_off = tgt_flat, tgt_off

    def __len__(self):
        return len(self.src_off) - 1

    def __getitem__(self, i):
        s = self.src_flat[self.src_off[i]:self.src_off[i + 1]].astype(np.int64)
        d = self.dec_flat[self.dec_off[i]:self.dec_off[i + 1]].astype(np.int64)
        t = self.tgt_flat[self.tgt_off[i]:self.tgt_off[i + 1]].astype(np.int64)
        return torch.from_numpy(s), torch.from_numpy(d), torch.from_numpy(t)


def collate_batch(batch, pad_id=PAD_ID):
    """一批不等长样本 → 补齐成 (B, T_max)。

    注意：dec 和 tgt 的长度永远相等（都是 len(下联)+1），
    所以两者用同一个 T_tgt。
    """
    srcs, decs, tgts = zip(*batch)
    B = len(batch)
    T_src = max(len(x) for x in srcs)
    T_tgt = max(len(x) for x in decs)

    src = torch.full((B, T_src), pad_id, dtype=torch.long)
    dec = torch.full((B, T_tgt), pad_id, dtype=torch.long)
    tgt = torch.full((B, T_tgt), pad_id, dtype=torch.long)

    for i, (s, d, t) in enumerate(batch):
        src[i, :len(s)] = s
        dec[i, :len(d)] = d
        tgt[i, :len(t)] = t

    return src, dec, tgt


def pack(seqs):
    """list of (list of int) → (扁平 int16 数组, int64 偏移数组)"""
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int32, count=len(seqs))
    off = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    flat = np.empty(int(off[-1]), dtype=np.int16)
    for i, s in enumerate(seqs):
        flat[off[i]:off[i + 1]] = s
    return flat, off


def build_split(prefix, stoi, min_len=3):
    """读一个划分（train / test），过滤，编码，打包。

    prefix: '.../couplet/train' 或 '.../couplet/test'
    返回 (Dataset, 统计字典)
    """
    ups = load_file(os.path.join(prefix, "in.txt"))
    downs = load_file(os.path.join(prefix, "out.txt"))
    n_raw = len(ups)

    ups, downs, n_short = filter_pairs(ups, downs, min_len)

    src_list, dec_list, tgt_list = [], [], []
    n_oov = 0
    for u, d in zip(ups, downs):
        # 含被清洗掉的坏字 → 整条丢弃（无法编码）
        if any(c not in stoi for c in u) or any(c not in stoi for c in d):
            n_oov += 1
            continue
        ids = [stoi[c] for c in d]
        src_list.append([stoi[c] for c in u])
        dec_list.append([SOS_ID] + ids)
        tgt_list.append(ids + [EOS_ID])

    src_flat, src_off = pack(src_list)
    dec_flat, dec_off = pack(dec_list)
    tgt_flat, tgt_off = pack(tgt_list)

    ds = CoupletDataset(src_flat, src_off, dec_flat, dec_off, tgt_flat, tgt_off)
    stats = dict(n_raw=n_raw, n_short=n_short, n_oov=n_oov, n_keep=len(ds))
    return ds, stats


def load_stoi(vocab_path):
    """读 vocab.json，返回 stoi 和 itos。"""
    import json
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)["itos"]
    itos = {int(k): v for k, v in raw.items()}
    return {c: i for i, c in itos.items()}, itos
