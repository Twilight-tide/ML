# -*- coding: utf-8 -*-
"""seq2seq Transformer —— 按论文《Attention Is All You Need》实现。

模块清单（按依赖顺序，一个一个来）：
    1. ScaledDotProductAttention    式(1)
    2. PositionalEncoding           §3.5
    3. PositionwiseFeedForward      式(2)
    4. MultiHeadAttention           §3.2.2
    5. EncoderLayer                 §3.1
    6. DecoderLayer                 §3.1（含交叉注意力）
    7. Seq2SeqTransformer           总装
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 配置（与 docs/01-技术方案.md §2.3 一致）----
VOCAB_SIZE = 9014
D_MODEL = 64
N_HEADS = 8            # d_k = 64 / 8 = 8
D_FF = 256             # = 4 × d_model，与论文 2048/512 同比例
N_LAYERS_ENC = 2
N_LAYERS_DEC = 2
DROPOUT = 0.1
MAX_LEN = 40
PAD_ID, SOS_ID, EOS_ID = 0, 1, 2


class ScaledDotProductAttention(nn.Module):
    """式(1)：Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V

    q : (B, h, T_q, d_k)
    k : (B, h, T_k, d_k)
    v : (B, h, T_k, d_k)
    mask : (T_q, T_k)，True 表示允许注意

    返回 (out, attn)
        out  (B, h, T_q, d_k)      加权求和的结果
        attn (B, h, T_q, T_k)      注意力权重，M5 画图要用
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)                                   # 最后一维，不是 d_model

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        return out, attn

class PositionalEncoding(nn.Module):
    """§3.5 位置编码。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    返回 x + PE —— 加性，不是拼接，所以形状不变。
    """

    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)

        # 用 exp(log) 形式算 10000^(-2i/d_model)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))                # (d_model/2,)

        pe[:, 0::2] = torch.sin(pos * div)      # 偶数维用 sin
        pe[:, 1::2] = torch.cos(pos * div)      # 奇数维用 cos
        pe = pe.unsqueeze(0)                    # (1, max_len, d_model)

        # buffer 不是可训练参数，但会跟着 .to(device) 走、也会进 state_dict
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)，只取前 T 个位置
        return self.dropout(x + self.pe[:, :x.size(1)])

class PositionwiseFeedForward(nn.Module):
    """式(2)：FFN(x) = max(0, x W1 + b1) W2 + b2

    "position-wise" 的意思是：每个位置独立、共用同一组权重做变换。
    位置之间不交换信息——那是注意力层的事。
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)     # 升维：64 → 256
        self.w_2 = nn.Linear(d_ff, d_model)     # 降回：256 → 64
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MultiHeadAttention(nn.Module):
    """§3.2.2
        MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O
        head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    forward(q_in, kv_in=None, mask=None)
        kv_in=None    → 自注意力（Q=K=V=q_in）
        kv_in 有值    → 交叉注意力（Q 来自解码器，K/V 来自编码器输出）
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.h = n_heads
        self.d_k = d_model // n_heads

        # 对应论文里的 W_i^Q、W_i^K、W_i^V。用一个大矩阵一次算完所有头，
        # 之后再 view 拆开——数学上等价于 h 个小矩阵，但快得多。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)          # W^O

        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, q_in, kv_in=None, mask=None):
        if kv_in is None:
            kv_in = q_in

        B, T_q, _ = q_in.shape
        T_k = kv_in.size(1)          # ← 注意：和 T_q 不一定相等

        # (B, T, d_model) -> (B, h, T, d_k)
        q = self.w_q(q_in).view(B, T_q, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(kv_in).view(B, T_k, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(kv_in).view(B, T_k, self.h, self.d_k).transpose(1, 2)

        out, attn = self.attn(q, k, v, mask)

        # (B, h, T_q, d_k) -> (B, T_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.w_o(out), attn

class EncoderLayer(nn.Module):
    """§3.1 编码器层：两个子层，每个都是 LayerNorm(x + Sublayer(x))。

    子层 1：多头自注意力（双向，没有因果掩蔽）
    子层 2：逐位置前馈

    顺序按论文 §5.4：dropout 加在子层输出上，
    在「与输入相加、归一化」之前。
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T_src, d_model)
        attn_out, attn_w = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x, attn_w

class DecoderLayer(nn.Module):
    """§3.1 解码器层：三个子层。

    子层 1：掩蔽自注意力（Q=K=V=解码器自己，因果掩蔽）
    子层 2：交叉注意力（Q 来自本层，K/V 来自编码器输出）—— §3.2.3
    子层 3：逐位置前馈
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)   # ← 新增
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)          # 交叉注意力专用
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        """
        x       : (B, T_tgt, d_model)   解码器当前位置的表示
        enc_out : (B, T_src, d_model)   编码器输出
        src_mask: (T_tgt, T_src)        交叉注意力用
        tgt_mask: (T_tgt, T_tgt)        自注意力用（因果）
        """
        # 子层 1：掩蔽自注意力
        a1, w_self = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(a1))

        # 子层 2：交叉注意力。Q 是本层输出，K/V 是编码器输出
        a2, w_cross = self.cross_attn(x, enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout2(a2))

        # 子层 3：前馈
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x, w_self, w_cross


def make_pad_mask(seq, pad_id=PAD_ID):
    """(B, T) → (B, 1, 1, T)，True 表示"这个位置不是 pad"

    多出来的两个 1 是为了广播到 (B, h, T_q, T_k)。
    """
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(T, device=None):
    """(T, T) 下三角，True 表示允许注意。第 i 行只能看 0..i。"""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

class Seq2SeqTransformer(nn.Module):
    """完整的编码器-解码器 Transformer。"""

    def __init__(self, vocab_size, d_model, n_heads, d_ff,
                 n_layers_enc, n_layers_dec,
                 max_len=40, dropout=0.1, pad_id=PAD_ID):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers_enc)])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers_dec)])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight        # 权重共享，必须在初始化之前

        # 论文 §5.4 附近的惯例：Xavier 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """上联 → 编码器输出。返回 (enc_out, 每层的注意力权重)。"""
        x = self.pos(self.embed(src) * math.sqrt(self.d_model))
        attns = []
        for layer in self.encoder:
            x, a = layer(x, mask=src_mask)
            attns.append(a)
        return x, attns

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        """下联（右移） + 编码器输出 → 解码器输出。"""
        x = self.pos(self.embed(tgt) * math.sqrt(self.d_model))
        self_attns, cross_attns = [], []
        for layer in self.decoder:
            x, a1, a2 = layer(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
            self_attns.append(a1)
            cross_attns.append(a2)
        return self.norm(x), self_attns, cross_attns

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out, _ = self.encode(src, src_mask)
        dec_out, _, _ = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.head(dec_out)          # (B, T_tgt, vocab_size) 的 logits
