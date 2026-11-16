# 从 numpy 到 PyTorch 训练循环

写给已经会用 numpy 手写训练循环、但没写过 PyTorch 的人。

**前提**：你写过 `src/最小闭环.py` 和 `src/0301逻辑回归.py`——
那里面有完整的前向、损失、梯度、更新四步。这份材料只做一件事：
**把同一个循环换成 PyTorch 的写法**，然后一步步走到能训 Transformer。

---

## 怎么用这份材料

五个练习，**必须按顺序做**。每个都是"你先自己写，卡住再看答案"。

```
练习 1  PyTorch 的基本单位：张量        （15 分钟）
练习 2  自动求导：backward() 到底做了什么  （20 分钟）
练习 3  最小训练循环：线性回归            （30 分钟）
练习 4  把模型装进 nn.Module            （30 分钟）
练习 5  加一层非线性：手写 MLP            （30 分钟）
```

做完练习 5，你就具备了写 `train.py` 的全部知识——
剩下的只是"多几层"和"数据怎么切"，不是新概念。

---

## 练习 1 · 张量

### 概念

PyTorch 的 `Tensor` 就是 numpy 的 `ndarray`，多了两件事：
1. 能放到 GPU 上
2. **能记住自己是怎么算出来的**（这个叫计算图，练习 2 用）

### 先动手

在 `couplet/experiments/` 下建 `p1_tensor.py`，逐行敲（不要复制）：

```python
import numpy as np
import torch

# 1. 创建
a_np = np.array([1.0, 2.0, 3.0])
a_pt = torch.tensor([1.0, 2.0, 3.0])
print(a_np, a_pt)

# 2. 两者运算方式几乎一样
print(a_np * 2, a_pt * 2)
print(a_np.sum(), a_pt.sum())
print(a_np.mean(), a_pt.mean())

# 3. 矩阵乘法：注意符号不同
X = torch.randn(4, 3)
w = torch.randn(3)
print("X @ w =", X @ w)      # numpy 也是 @
print("X.matmul(w) =", X.matmul(w))   # 等价写法

# 4. 互相转换
print(type(a_pt.numpy()))          # tensor -> ndarray
print(type(torch.from_numpy(a_np)))  # ndarray -> tensor

# 5. 形状操作（和 numpy 一样）
t = torch.randn(2, 3, 4)
print(t.shape, t.view(2, 12).shape, t.transpose(0, 1).shape)
```

### 要搞清楚的三件事

**（1）`torch.tensor` vs `torch.Tensor`** —— 用小写 `torch.tensor(...)`，
它是"从数据构造一个张量"。大写那个是历史遗留，别用。

**（2）dtype 很重要。** 试这个：

```python
x = torch.tensor([1, 2, 3])        # 整数！
print(x.dtype)                     # torch.int64
y = torch.tensor([1.0, 2.0, 3.0])
print(y.dtype)                     # torch.float32
```

**在神经网络里几乎永远用 float32**。整型张量不能做梯度（练习 2 会看到报错）。

**（3）`@` 和 `*` 完全不同。**

```python
A = torch.randn(3, 3)
print((A * A).shape)    # 逐元素相乘，还是 (3,3)
print((A @ A).shape)    # 矩阵乘法
```

numpy 里你知道这个区别。PyTorch 里同样，但**错误更难发现**——
有时 shape 恰好对得上，结果错了却不报错。

### ✅ 自检

- [ ] `torch.randn(4,3) @ torch.randn(3)` 得到什么形状？为什么？
- [ ] 整数张量做 `.mean()` 会怎样？（试试看）
- [ ] `t.view()` 和 `t.reshape()` 有什么区别？（提示：一个要求内存连续）

---

## 练习 2 · 自动求导

### 这是 PyTorch 唯一真正新的东西

你在 numpy 里手推过：

```python
grad = 2 * X.T @ (pred - y) / n     # 均方误差的梯度
```

这个公式是你从链式法则推出来的。现在试试让 PyTorch 自己算。

### 先动手

建 `p2_autograd.py`：

```python
import torch

# ---------- 实验 A：一维，肉眼可验证 ----------
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2 + 2 * x + 1        # 数学上：y = (x+1)^2，dy/dx = 2x+2
y.backward()                   # 反向传播
print("x =", x.item(), " y =", y.item(), " dy/dx =", x.grad.item())
# 手算：dy/dx = 2*3+2 = 8，对上了吗？
```

**关键点**：`requires_grad=True` 是在说"这个张量需要算梯度"。
只有它参与的计算才会被记录下来。

### 实验 B：和你手推的公式对照

```python
import torch

torch.manual_seed(0)
n, d = 200, 3
X = torch.randn(n, d)
w_true = torch.tensor([2.0, -1.0, 0.5])
y = X @ w_true + 0.1 * torch.randn(n)

w = torch.zeros(d, requires_grad=True)

pred = X @ w
loss = ((pred - y) ** 2).mean()
loss.backward()                     # ← 一行代替你的推导

# 左边：PyTorch 自动算的；右边：你手推的公式
grad_auto = w.grad
grad_manual = 2 * X.T @ (pred.detach() - y) / n

print("自动:", grad_auto)
print("手推:", grad_manual)
print("最大差值:", (grad_auto - grad_manual).abs().max().item())
```

**这个练习的意义**：证明 `backward()` 算的就是你手推的那个东西。
看到"最大差值 ≈ 1e-7"（浮点误差），你就能信任它了。

### 实验 C：梯度会累加（重要！）

```python
w = torch.tensor([1.0, 2.0], requires_grad=True)

loss1 = (w ** 2).sum()
loss1.backward()
print("第一次:", w.grad)          # [2, 4]

loss2 = (w ** 2).sum()
loss2.backward()
print("第二次:", w.grad)          # [4, 8] ← 累加了！

w.grad.zero_()                    # 手动清零
print("清零后:", w.grad)
```

**这就是训练循环里必须有 `opt.zero_grad()` 的原因。**
不清零，梯度会一轮一轮累加，参数直接跑飞。

### 一个必须知道的开关

```python
with torch.no_grad():
    y = x ** 2
    # 这里面的计算不记录梯度（省内存、更快）

# 或者：
y = x.detach()      # 从计算图里"摘"出来
```

**什么时候用**：做验证/推理时（不需要梯度）、想拿张量的值但不要梯度时。

⚠️ 一个常见错误：

```python
pred = X @ w
loss = ((pred - y) ** 2).mean()
grad_manual = 2 * X.T @ (pred - y) / n   # ← 这里！
```

如果 `pred` 带梯度，这个"手推版本"会也被记进计算图，
然后你调 `backward()` 就会报错（试图对已经是图的一部分的东西再求导）。
**所以要写 `pred.detach()`**，就像实验 B 里那样。

### ✅ 自检

- [ ] 实验 A 的 `dy/dx` 等于 8 吗？手算验证一下
- [ ] 实验 B 的最大差值是多少量级？为什么不是严格 0？
- [ ] 为什么训练循环里要 `zero_grad()`？不做会怎样？
- [ ] `with torch.no_grad()` 什么时候用？

---

## 练习 3 · 最小训练循环

### 目标：把 `最小闭环.py` 用 PyTorch 重写

你的 numpy 版是这样的（`src/最小闭环.py`）：

```python
w_gd = np.zeros(d)
for step in range(300):
    pred = X @ w_gd
    loss = ((pred - y) ** 2).mean()
    grad = 2 * X.T @ (pred - y) / n
    w_gd -= lr * grad
```

### 先动手

建 `p3_loop.py`，自己写。提示：

```python
import torch

# 数据（和张量一样，但要 float32）
X = torch.randn(200, 3)
y = X @ torch.tensor([2.0, -1.0, 0.5]) + 0.1 * torch.randn(200)

# 参数：要 requires_grad=True
w = torch.zeros(3, requires_grad=True)

lr = 0.1

for step in range(300):
    # ① 前向
    pred = ...
    # ② 损失
    loss = ...
    # ③ 清梯度（练习 2 实验 C 的原因）
    ...
    # ④ 反向
    ...
    # ⑤ 更新 —— 注意不能直接 w -= lr*w.grad
    with torch.no_grad():
        w -= lr * w.grad
        ...   # 还要记得清空 grad
```

### 卡住的地方在这里：为什么更新要 `with torch.no_grad()`

因为 `w` 本身带 `requires_grad=True`，
`w -= lr * w.grad` 这个操作**会被记进计算图**——
你不想"更新参数"这个动作本身再被求导。

`with torch.no_grad():` 就是关掉记录。

**这也是为什么后面要用 `optimizer`**：它把这几行封装了，
省得每次写 `no_grad` 和手动清零。

### 参考答案

写不出来再看：

```python
for step in range(300):
    pred = X @ w                       # ①
    loss = ((pred - y) ** 2).mean()    # ②

    w.grad = None                      # ③ 清梯度（也可以写成 w.grad.zero_()
    loss.backward()                    # ④ 自动求导

    with torch.no_grad():              # ⑤ 更新（关掉记录）
        w -= lr * w.grad

    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.6f}")

print("学到的 w:", w.detach())
# 应该接近 [2.0, -1.0, 0.5]
```

### ✅ 自检

- [ ] 学到的 `w` 接近 `[2.0, -1.0, 0.5]` 吗？
- [ ] 把 `w.grad = None` 那行注释掉，会发生什么？（自己试，别怕错）
- [ ] `loss.item()` 里的 `.item()` 是干什么的？不加会怎样？

---

## 练习 4 · nn.Module

### 为什么需要它

练习 3 里 `pred = X @ w` 只有一个参数。但神经网络有几十个参数，
总不能一个一个手写。

`nn.Module` 解决两件事：

1. **自动收集所有参数**（不用手写 `[w1, b1, w2, b2, ...]`）
2. **统一 forward 接口**

### 先动手

建 `p4_module.py`：

```python
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()                       # ← 这行必须有
        self.w = nn.Parameter(torch.zeros(in_features))   # ← 用 Parameter 包一下
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x @ self.w + self.b


model = LinearRegression(3)

print("模型参数:")
for name, p in model.named_parameters():     # ← 自动收集，不用手写
    print(" ", name, p.shape)

print("总参数量:", sum(p.numel() for p in model.parameters()))

# 用一下
X = torch.randn(5, 3)
print("输出形状:", model(X).shape)
```

### 三个必须理解的规则

**（1）`super().__init__()` 必须第一行。**
不写会报错，因为父类要做初始化（登记参数表等）。

**（2）参数要用 `nn.Parameter` 包。**

```python
self.w = torch.zeros(3)              # ✗ 不会被 model.parameters() 收集
self.w = nn.Parameter(torch.zeros(3))  # ✓
```

这就是 `model.parameters()` 能自动找到所有参数的原因——
它遍历的是被 `Parameter` 登记过的属性。

**（3）网络定义在 `__init__`，怎么算在 `forward`。**

```python
def forward(self, x):        # 名字必须是 forward
    return x @ self.w + self.b
```

**调用时要写 `model(x)` 而不是 `model.forward(x)`**——前者会自动处理一些钩子。
虽然两者结果一样，但养成习惯。

### 配上 optimizer

```python
model = LinearRegression(3)
opt = torch.optim.SGD(model.parameters(), lr=0.1)   # ← 把参数交给它

X = torch.randn(200, 3)
y = X @ torch.tensor([2.0, -1.0, 0.5]) + 0.1 * torch.randn(200)

for step in range(300):
    pred = model(X)
    loss = ((pred - y) ** 2).mean()

    opt.zero_grad()      # 清梯度（练习 2 实验 C 的坑，它帮你处理了）
    loss.backward()      # 自动求导
    opt.step()           # 更新参数（不用写 no_grad 了）

    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.6f}")

print("w =", model.w.detach())
print("b =", model.b.detach())
```

**对照练习 3**，`opt.zero_grad()` 代替了 `w.grad = None`，
`opt.step()` 代替了 `with torch.no_grad(): w -= lr*w.grad`。

**到这里，训练循环的骨架就齐了：**

```python
for x, y in data:
    pred = model(x)          # 前向
    loss = criterion(pred, y)  # 损失
    opt.zero_grad()          # 清梯度
    loss.backward()          # 反向
    opt.step()               # 更新
```

**这就是 `train.py` 的核心。剩下的都是配菜。**

### ✅ 自检

- [ ] 把 `nn.Parameter` 换成普通张量，`model.parameters()` 还能找到吗？
- [ ] `super().__init__()` 删掉会报什么错？
- [ ] 为什么调用用 `model(x)` 而不是 `model.forward(x)`？

---

## 练习 5 · 加非线性

### 目标：手写一个两层 MLP，解决 XOR

**为什么做这个**：线性模型分不开 XOR，必须要有非线性。
这正是你书里 `ch04` 讲的东西——现在亲手实现一遍。

### 先动手

建 `p5_mlp.py`：

```python
import torch
import torch.nn as nn

# XOR 数据：4 个点
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)     # ← nn.Linear 自带 w 和 b
        self.fc2 = nn.Linear(hidden, out_dim)
        self.relu = nn.ReLU()                     # ← 非线性

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)      # ← 没有这行，两层等价于一层
        x = self.fc2(x)
        return x


model = MLP(2, 8, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = nn.BCEWithLogitsLoss()    # 二分类损失（内部含 sigmoid）

for step in range(2000):
    pred = model(X)
    loss = criterion(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 400 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# 看预测
with torch.no_grad():
    out = torch.sigmoid(model(X))
    print("\n预测 (应该接近 0,1,1,0):")
    for x, o, t in zip(X.tolist(), out.tolist(), y.tolist()):
        print(f"  {x} -> {o[0]:.3f}  (目标 {t[0]})")
```

### 三个新东西

**（1）`nn.Linear` 已经封装好了 `x @ W.T + b`。**

注意它内部用的是 `x @ W.T`——PyTorch 的 `nn.Linear` 权重形状是 `(out, in)`，
和我们在练习 4 里写的 `(in,)` 相反。这是约定，记住就行。

**（2）`nn.ReLU()` 是非线性。**

**把它删掉试试**——两层线性叠起来还是线性，XOR 永远学不会。
这就是你书里讲的"深度需要非线性"。

**（3）`nn.BCEWithLogitsLoss` 而不是自己写 sigmoid + 交叉熵。**

PyTorch 提供了数值更稳定的版本。名字里的 "WithLogits" 意思是
"输入是没经过 sigmoid 的原始分数"，它内部会处理。

**类似的**：多分类用 `nn.CrossEntropyLoss`，它内部含 softmax。
**这就是 seq2seq 要用的那个损失函数。**

### ✅ 自检

- [ ] 预测值接近 `0, 1, 1, 0` 吗？
- [ ] 把 `self.relu` 那行删掉，会怎样？（一定要试）
- [ ] 把 hidden 从 8 改成 2，还能学会吗？（容量太小的表现）
- [ ] `nn.Linear(2, 8)` 有多少个参数？（提示：`2*8 + 8 = 24`）

---

## 做完五个练习，你就有了什么

对照对联项目的 `train.py` 需要什么：

| train.py 的组成 | 你在哪个练习学过 |
|---|---|
| 数据加载成张量 | 练习 1 |
| 模型定义（`nn.Module`） | 练习 4、5 |
| 前向 + 损失 | 练习 3、5 |
| `zero_grad / backward / step` | 练习 4 |
| 清梯度、`no_grad` 的时机 | 练习 2、3 |
| 验证时不更新参数 | 练习 2（`no_grad`） |
| 交叉熵损失 | 练习 5（`CrossEntropyLoss`） |

**seq2seq 的 train.py 多出来的只有：**
1. 模型更复杂（多个 `nn.Module` 拼起来）
2. 数据要按批取（不是一次性全喂）
3. 学习率调度（论文式(3)）
4. 两个输入（编码器和解码器各一个）

**这四件事都不是新概念**，是"你已经会的东西 + 更多细节"。

---

## 学习顺序建议

```
第 1 天（1 小时）   练习 1 + 练习 2
第 2 天（1 小时）   练习 3 + 练习 4
第 3 天（1 小时）   练习 5
第 4 天（1 小时）   读 src/train.py，对照本文档看每个部分对应哪个练习
第 5 天              开始写 couplet/src/train.py
```

**第 4 天那步别跳过**：读一遍你已经有的 `src/train.py`（代码生成项目的），
你会发现"原来每一行我都认识"。这个感觉很重要。

---

## 卡住的时候怎么办

**先看报错的第一行和最后一行。** PyTorch 的报错通常很长，但关键信息在：

- 第一行：什么错误
- 最后一行：错误发生在你代码的哪一行

常见的三类：

| 报错关键词 | 通常原因 |
|---|---|
| `requires_grad` / `does not require grad` | 参数忘了 `requires_grad=True`，或对不该求导的东西调了 `backward()` |
| `shape` / `size mismatch` | 维度对不上，打印 `.shape` 看 |
| `RuntimeError: Trying to backward through the graph a second time` | 忘了 `zero_grad()`，或忘了 `.detach()` |

把报错贴给我，我帮你定位。
