# -*- coding: utf-8 -*-
"""超参数配置。

参数量目标：约 10 万（论文原版是 6500 万，这里等比缩小约 650 倍）。
缩小比例的分配见 README.md。
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- 数据 ----------------
    vocab_size: int = 96        # 字符表大小（取语料中出现频率最高的 N 个字符）
    seq_len: int = 64           # 训练时的上下文长度
    max_len: int = 256          # 位置编码预计算的最大长度（生成时可超出 seq_len）

    # ---------------- 模型 ----------------
    # 括号内为论文 base 模型的取值，便于对照
    d_model: int = 64           # 论文 512
    n_heads: int = 4            # 论文 8
    d_ff: int = 232             # 论文 2048；此处取值使总参数量恰好落在 10 万附近
    n_layers: int = 2           # 论文 6
    dropout: float = 0.1        # 论文 5.4 节 P_drop = 0.1

    # ---------------- 训练 ----------------
    batch_size: int = 32
    steps: int = 3000
    warmup_steps: int = 200     # 论文 5.3 节的学习率预热步数
    lr_scale: float = 1.0       # 在论文公式基础上再乘一个系数，便于微调
    eval_every: int = 250
    eval_batches: int = 20
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"

    # ---------------- 路径 ----------------
    corpus_file: str = "data/corpus.txt"
    vocab_file: str = "data/vocab.json"
    ckpt_file: str = "results/model.pt"
    log_file: str = "results/train_log.csv"

    @property
    def d_k(self) -> int:
        """每个注意力头的维度 d_k = d_model / h（论文 3.2.2 节）。"""
        return self.d_model // self.n_heads

    def check(self):
        assert self.d_model % self.n_heads == 0, \
            "d_model 必须能被 n_heads 整除（论文 3.2.2 节要求）"
        assert self.d_k > 0
        return self
