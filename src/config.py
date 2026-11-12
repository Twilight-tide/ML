# -*- coding: utf-8 -*-
"""超参数配置。

参数量：约 50 万（论文原版 6500 万，等比缩小约 130 倍）。
取值依据是 2026-11-12 的参数量扫描实验，结果见下方「模型」段的注释。
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
    #
    # 2026-11-12 参数量扫描（同 seed / batch / seq_len / 12000 步，语料 567 万字符）：
    #   10 万 → val 1.9039 (ppl 6.71)   78 秒
    #   30 万 → val 1.7683 (ppl 5.86)   80 秒
    #   50 万 → val 1.7270 (ppl 5.62)   77 秒   ← 取这一档
    #   94 万 → val 1.7048 (ppl 5.50)  133 秒   （9000 步后过拟合）
    # 94 万只比 50 万好 1.3%，参数却翻倍、训练慢 73%，且 train/val 差距从 0.32 扩大到 0.46。
    # 另外注意：深度（层数）比宽度贵得多——宽度是并行，层数是串行。
    d_model: int = 128          # 论文 512
    n_heads: int = 8            # 论文 8；d_k = 128/8 = 16，与原配置一致
    d_ff: int = 680             # 论文 2048；此处取值使总参数量落在 50 万附近
    n_layers: int = 2           # 论文 6
    dropout: float = 0.1        # 论文 5.4 节 P_drop = 0.1

    # ---------------- 训练 ----------------
    batch_size: int = 32
    steps: int = 12000          # 3 个 epoch 左右；实验显示此步数下 val 仍在下降
    warmup_steps: int = 200     # 论文 5.3 节的学习率预热步数
    lr_scale: float = 1.0       # 在论文公式基础上再乘一个系数，便于微调
    eval_every: int = 1000      # 总步数变多后，降低评测频次以省时间
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
