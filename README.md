# 用 10 万参数的 Transformer 写 Python 代码

一个完整可跑的小项目：**按论文《Attention Is All You Need》的架构实现 Transformer，
参数量精确控制在 10 万，训练它生成 Python 代码。**

论文原版是 6500 万参数、8 块 P100 训练 12 小时。这里等比缩小约 650 倍，
在一台笔记本的 RTX 5060（8 GB 显存）上 **约 25 秒**训练完。

---

## 快速开始

```powershell
# 1. 准备语料（从本机 Python 标准库源码提取，不需要联网）
python src/prepare_data.py

# 2. 训练（约 25 秒）
python src/train.py

# 3. 生成代码
python src/generate.py --prompt "def " --tokens 300

# 4. 看注意力头学到了什么
python src/visualize_attention.py
```

---

## 模型与论文的对应关系

代码里每个类都标了论文的章节号，方便对照译文阅读：

| 论文位置 | 内容 | 代码 |
|---|---|---|
| §3.2.1 式(1) | `Attention(Q,K,V) = softmax(QK^T/√d_k)V` | `ScaledDotProductAttention` |
| §3.2.2 | 多头注意力 `Concat(head_1..head_h)W^O` | `MultiHeadAttention` |
| §3.3 式(2) | `FFN(x) = max(0, xW_1+b_1)W_2+b_2` | `PositionwiseFeedForward` |
| §3.1 | 残差连接 + 层归一化 | `DecoderLayer` |
| §3.2.3 | 因果掩蔽（阻止看到未来位置） | `forward` 中的 `torch.tril` 掩码 |
| §3.4 | 嵌入与输出层权重共享、乘 `√d_model` | `TinyTransformer` 的 `tie_weights` 与 `scale` |
| §3.5 | 正弦位置编码 | `PositionalEncoding` |
| §5.3 式(3) | 学习率预热与衰减 | `train.py` 的 `lr_lambda_factory` |
| §5.4 | 残差 dropout `P_drop=0.1` | 各处 `nn.Dropout` |

### 与论文的差异（有意为之）

- **只实现了解码器。** 论文用于翻译，是编码器-解码器结构；本任务是自回归语言建模，
  只需解码器。解码器内部结构（掩蔽自注意力 + 残差 + 层归一化 + 前馈）与论文完全一致。
- **未实现编码器-解码器交叉注意力**，因为该任务没有独立的编码器输入。
- **层数减为 2**（论文 6），符合 10 万参数的规模。

---

## 参数量构成

```
d_model=64   n_heads=4   d_k=16   d_ff=232   n_layers=2   vocab=96

词嵌入（含输出层，权重共享）      6,144
2 个解码器层                    93,776
输出层归一化                       128
----------------------------------------
合计                          100,048     ← 恰好 10 万
```

论文 base 模型：`d_model=512, n_heads=8, d_ff=2048, n_layers=6, vocab≈37000`，共 6500 万。

**为什么 `d_ff=232` 这么奇怪的数字？** 为了让总量精确落在 10 万。
这是反推出来的：先固定 `d_model=64, n_heads=4, n_layers=2, vocab=96`，再解出使总参数量等于 100,000 的 `d_ff`。

---

## 文件说明

```
src/
  config.py               超参数配置
  prepare_data.py         从标准库源码构建字符级语料
  model.py                Transformer 实现（含论文章节注释）
  train.py                训练循环 + 论文式(3) 学习率调度
  generate.py             文本生成
  visualize_attention.py  注意力热力图（对应论文附录 图3/图5）

data/                     语料与词表（可由脚本重新生成）
results/                  模型、训练日志、注意力图
notebooks/                留给你做实验
```

---

## 语料说明

用 **本机 Python 标准库源码**（约 6 MB，96 个字符的词表）。选择它的原因：

1. **零网络依赖** —— 本项目所在网络环境下 GitHub 等源不稳定，标准库必然存在
2. **规模合适** —— 6 MB 足以让 10 万参数的模型学到语法结构
3. **结果可判断** —— 模型写出的东西像不像 Python 代码，肉眼一看便知

---

## 一个值得记录的 bug：学习率小了 1000 倍

第一版训练时 loss 从 4.57 降到 3.71 就几乎不动了，生成的文本是乱码：

```
def  ,
dcul" ac #nok)n
 ll.,ol)n.=s da&t pp)
```

**症状像是模型容量不足，但真正的原因是学习率。**

论文式(3) 在步数 3000 时应为 `2.28e-03`，而实际只用了 `2.28e-06`：

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)     # ← 问题在这
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
```

`LambdaLR` 的实际学习率是 **`optimizer 的 base_lr` × `lambda(step)`**。
我误以为 lambda 的返回值就是最终学习率，于是它被 `1e-3` 又缩小了 1000 倍。

修正方式是让 lambda 直接返回论文公式的值，并把 `base_lr` 设为 1.0：

```python
opt = torch.optim.Adam(model.parameters(), lr=1.0)      # ← 修正
```

**这个 bug 的教训**：loss 不降有很多种原因，不要一上来就怀疑模型太小或数据太少。
先把论文里的关键数值**算出来，和实际用的对一遍**——这次一算就发现差了三个数量级。

---

## 硬件与环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop（8 GB，sm_120） |
| torch | 2.11.0+cu128 |
| Python | 3.14.0 |

环境配置的坑（必须用 cu128 的原因、pip 假死排查）见项目根目录的 `环境说明.md`。

---

## 可以自己试的实验

这个项目的价值在于**可以改**。几个直接可做的：

1. **改 `n_heads`**：`n_heads=1` 会怎样？论文表 3 的 (A) 行做过这个消融，
   结论是单头比最佳设置差 0.9 BLEU。你能在小模型上重现类似现象吗？
2. **改 `d_k`**：论文 (B) 行发现减小 `d_k` 会损害质量，为什么？
3. **去掉位置编码**：模型还能学会顺序吗？（论文 §3.5 讨论过这个）
4. **换语料**：换成中文文本、或你自己的笔记，看生成结果怎么变
5. **调温度**：`temperature=0.3` 和 `1.5` 的输出差别说明了什么？

跑第 1 或第 2 个实验时，把结果和论文表 3 对照——**这是从小模型实验理解大模型结论的练习**，
也是科研里常见的思维方式。
