# -*- coding: utf-8 -*-
"""Transformer 模型实现。

严格按论文《Attention Is All You Need》的结构编写，每个类都注明对应章节与公式编号，
便于你对照译文阅读：

    §3.2.1  式(1)  ScaledDotProductAttention
    §3.2.2         MultiHeadAttention
    §3.3    式(2)  PositionwiseFeedForward
    §3.5           PositionalEncoding
    §3.1           EncoderLayer（残差 + 层归一化）
    §3.4           权重共享 + 嵌入缩放

与论文的差异（因本应用体量小、任务是纯解码器语言建模）：
  - 论文是 编码器-解码器（为翻译任务）；这里只用解码器，做自回归语言建模。
    解码器内部结构（掩蔽自注意力 + 残差 + 层归一化 + 前馈）与论文完全一致。
  - 论文的编码器-解码器交叉注意力层在此不适用，故未实现。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# §3.2.1 缩放点积注意力  —— 式(1)
#     Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
# =============================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q: (B, h, T_q, d_k)
        k: (B, h, T_k, d_k)
        v: (B, h, T_k, d_v)
        mask: (T_q, T_k)，True 表示允许注意，False 表示掩蔽
        """
        d_k = q.size(-1)

        # 式(1) 分子：Q K^T。除以 sqrt(d_k) 是防止 d_k 大时点积过大、
        # 把 softmax 推入梯度极小的区域（论文 §3.2.1 正文与脚注 4 的论证）。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # §3.2.3：解码器需阻止信息向左流动，把非法连接设为 -inf
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 式(1) 后半：加权求和
        out = torch.matmul(attn, v)
        return out, attn


# =============================================================================
# §3.2.2 多头注意力
#     MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
#     其中 head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
# =============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.h = n_heads
        self.d_k = d_model // n_heads   # 论文：d_k = d_v = d_model / h = 64

        # 论文中的投影矩阵 W_i^Q, W_i^K, W_i^V（用一个大矩阵一次算完所有头）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 输出投影 W^O ∈ R^{h·d_v × d_model}
        self.w_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        # 线性投影后拆分成 h 个头：(B, T, d_model) -> (B, h, T, d_k)
        q = self.w_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        out, attn = self.attn(q, k, v, mask)

        # 拼接各头后过 W^O
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(out), attn


# =============================================================================
# §3.3 逐位置前馈网络 —— 式(2)
#     FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
# =============================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # max(0, ...) 即 ReLU
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# =============================================================================
# §3.5 位置编码
#     PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#     PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# 论文选择正弦函数而非可学习嵌入，是为了能外推到比训练更长的序列。
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe[pos, i] 按上式预计算
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 10000^(2i/d_model) 在对数域计算，避免溢出
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)   # 偶数维用 sin
        pe[:, 1::2] = torch.cos(pos * div)   # 奇数维用 cos
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # §3.5：位置编码与嵌入直接相加（二者维度相同 d_model）
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# §3.1 解码器层
#     每个子层的输出为 LayerNorm(x + Sublayer(x))
#     自注意力子层加因果掩蔽（§3.2.3）
# =============================================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)   # §5.4 残差 dropout
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- 子层 1：掩蔽多头自注意力 ---
        a, attn = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(a))     # 残差连接 + 层归一化

        # --- 子层 2：逐位置前馈网络 ---
        f = self.ffn(x)
        x = self.norm2(x + self.dropout2(f))
        return x, attn


# =============================================================================
# 完整模型
# =============================================================================
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 max_len, dropout=0.1, tie_weights=True):
        super().__init__()
        self.d_model = d_model

        # §3.4 嵌入与 Softmax
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # §3.4：论文在两个嵌入层与 softmax 前的线性变换之间共享权重矩阵，
        # 并在嵌入层把权重乘以 sqrt(d_model)。
        if tie_weights:
            self.head.weight = self.embed.weight
        self.scale = math.sqrt(d_model)   # §3.4 的 sqrt(d_model) 缩放
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, idx, return_attn=False):
        """
        idx: (B, T) 的 token 索引
        返回 logits (B, T, vocab) 及（可选）各层注意力权重
        """
        B, T = idx.shape

        # 嵌入乘以 sqrt(d_model)（论文 §3.4）
        x = self.embed(idx) * self.scale
        x = self.pos(x)

        # §3.2.3 因果掩蔽：位置 i 只能看到 <= i 的位置
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))

        attns = []
        for layer in self.layers:
            x, a = layer(x, mask)
            attns.append(a)

        x = self.norm_out(x)
        logits = self.head(x)

        if return_attn:
            return logits, attns
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成（对应论文 §3 描述的 auto-regressive 过程）。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.pos.pe.size(1):]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def count_parameters(model):
    """统计参数量。共享权重只算一次（PyTorch 的 parameters() 会去重）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
