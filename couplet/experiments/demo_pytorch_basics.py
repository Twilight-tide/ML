# -*- coding: utf-8 -*-
"""PyTorch 训练循环演示 —— 配合 docs/04-PyTorch训练循环入门.md 使用。

【这是演示脚本，不是练习答案】
它把五个练习的核心现象一次跑完，让你先"看见"再"自己写"。
看完输出后，请对照文档自己动手写 p1~p5，不要跳过练习。

运行：
    python couplet/experiments/demo_pytorch_basics.py

输出分五段，对应文档的五个练习：
    A  张量：和 numpy 几乎一样，但多了 dtype
    B  自动求导：backward() 算的就是你手推的公式
    C  梯度累加：为什么必须 zero_grad()
    D  最小训练循环：用 PyTorch 重写「最小闭环.py」
    E  nn.Module + 非线性：手写 MLP 解决 XOR
"""

import torch
import torch.nn as nn

DIV = "=" * 72


def sec(title):
    print()
    print(DIV)
    print(title)
    print(DIV)


# =============================================================================
sec("A · 张量：和 numpy 几乎一样")

import numpy as np

a_np = np.array([1.0, 2.0, 3.0])
a_pt = torch.tensor([1.0, 2.0, 3.0])
print("numpy :", a_np, " 均值", a_np.mean())
print("torch :", a_pt, " 均值", a_pt.mean().item())

X = torch.randn(4, 3)
w = torch.randn(3)
print("\n矩阵乘 X @ w :", (X @ w).shape, "   （4x3 @ 3 -> 4）")

print("\n【关键差别 1】dtype：整数张量不能求梯度")
i = torch.tensor([1, 2, 3])
f = torch.tensor([1.0, 2.0, 3.0])
print("  torch.tensor([1,2,3])     dtype =", i.dtype)
print("  torch.tensor([1.,2.,3.])  dtype =", f.dtype)
print("  神经网络里几乎永远用 float32")

print("\n【关键差别 2】* 和 @ 不同")
A = torch.randn(3, 3)
print("  (A * A).shape =", tuple((A * A).shape), "  ← 逐元素")
print("  (A @ A).shape =", tuple((A @ A).shape), "  ← 矩阵乘法（恰好同形，不报错但意义不同）")


# =============================================================================
sec("B · 自动求导：backward() 到底做了什么")

print("手算题： y = (x+1)^2  = x^2 + 2x + 1， dy/dx = 2x + 2")
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print("  x = 3 时，PyTorch 算出的 dy/dx =", x.grad.item())
print("  手算 2*3+2 = 8  →", "对上" if abs(x.grad.item() - 8) < 1e-5 else "对不上")

print("\n更重要的：验证它算的就是你手推的均方误差梯度")
torch.manual_seed(0)
n, d = 200, 3
Xd = torch.randn(n, d)
w_true = torch.tensor([2.0, -1.0, 0.5])
yd = Xd @ w_true + 0.1 * torch.randn(n)

wv = torch.zeros(d, requires_grad=True)
pred = Xd @ wv
loss = ((pred - yd) ** 2).mean()
loss.backward()

grad_auto = wv.grad.clone()
grad_manual = 2 * Xd.T @ (pred.detach() - yd) / n   # 你手推的公式

print("  自动求导:", [round(v, 6) for v in grad_auto.tolist()])
print("  手推公式:", [round(v, 6) for v in grad_manual.tolist()])
print("  最大差值: %.2e   ← 浮点误差量级，说明两者等价"
      % (grad_auto - grad_manual).abs().max().item())


# =============================================================================
sec("C · 梯度累加：为什么必须 zero_grad()")

wz = torch.tensor([1.0, 2.0], requires_grad=True)
(wz ** 2).sum().backward()
print("  第 1 次 backward 后 grad =", wz.grad.tolist(), "  （数学上 = 2w = [2, 4]）")

(wz ** 2).sum().backward()
print("  第 2 次 backward 后 grad =", wz.grad.tolist(), "  ← 累加了！（本应还是 [2,4]）")

wz.grad.zero_()
print("  手动 zero_() 之后 grad =", wz.grad.tolist())
print()
print("  → 这就是训练循环里 opt.zero_grad() 存在的唯一原因。")
print("    不清零，梯度每轮翻倍累加，参数会迅速跑飞。")


# =============================================================================
sec("D · 最小训练循环：PyTorch 版「最小闭环.py」")

torch.manual_seed(0)
Xd = torch.randn(200, 3)
w_true = torch.tensor([2.0, -1.0, 0.5])
yd = Xd @ w_true + 0.1 * torch.randn(200)

wv = torch.zeros(3, requires_grad=True)
lr = 0.1

print("  手写版（对应文档练习 3）：")
for step in range(301):
    pred = Xd @ wv
    loss = ((pred - yd) ** 2).mean()

    wv.grad = None          # 清梯度
    loss.backward()         # 自动求导
    with torch.no_grad():   # 更新时不记录梯度
        wv -= lr * wv.grad

    if step % 100 == 0:
        print("    step %3d   loss %.6f" % (step, loss.item()))

print("  学到的 w :", [round(v, 4) for v in wv.detach().tolist()])
print("  真值 w   : [2.0, -1.0, 0.5]")
print("  → 注意：PyTorch 版和你的 numpy 版是同一个循环，只是梯度不用手推")


# =============================================================================
sec("E · nn.Module + 非线性：手写 MLP 解决 XOR")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


Xx = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
yx = torch.tensor([[0.], [1.], [1.], [0.]])

print("  参数量: %d  (fc1: 2*8+8=24, fc2: 8*1+1=9)"
      % sum(p.numel() for p in MLP(2, 8, 1).parameters()))

for tag, use_relu in [("有 ReLU", True), ("去掉 ReLU", False)]:
    torch.manual_seed(0)
    model = MLP(2, 8, 1)
    if not use_relu:
        model.relu = nn.Identity()          # 等价于删掉那行
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    crit = nn.BCEWithLogitsLoss()

    for _ in range(3000):
        loss = crit(model(Xx), yx)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = torch.sigmoid(model(Xx)).squeeze().tolist()
    print("\n  【%s】最终 loss %.4f" % (tag, loss.item()))
    print("    预测:", [round(v, 3) for v in out])
    print("    目标: [0.0, 1.0, 1.0, 0.0]")
    if use_relu:
        print("    → 学会了 XOR")
    else:
        print("    → 学不会！两层纯线性叠加还是线性，分不开 XOR")

print()
print("  → 这就是「深度需要非线性」的实验证据（对应书 ch04）")

print()
print(DIV)
print("演示结束。接下来请打开 docs/04-PyTorch训练循环入门.md，")
print("自己动手写 p1~p5，不要复制本文档的代码。")
print(DIV)
