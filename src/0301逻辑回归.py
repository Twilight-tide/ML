import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- 数据 ----------
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

# 标准化：让每个特征均值 0、方差 1。
# 不做的话梯度下降会因为特征尺度差异而震荡。
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

# ---------- 模型 ----------
def sigmoid(z):
    """数值稳定版。"""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out

w = np.zeros(X_tr.shape[1])
b = 0.0
lr = 0.1

# ---------- 训练 ----------
for step in range(2000):
    z = X_tr @ w + b
    p = sigmoid(z)

    # 交叉熵（加 1e-9 防止 log(0)）
    loss = -(y_tr * np.log(p + 1e-9) +
             (1 - y_tr) * np.log(1 - p + 1e-9)).mean()

    # 梯度：p - y 再乘 x。（3.4 节的结论）
    grad_w = X_tr.T @ (p - y_tr) / len(y_tr)
    grad_b = (p - y_tr).mean()

    w -= lr * grad_w
    b -= lr * grad_b

    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss:.4f}")

# ---------- 评估 ----------
acc = ((sigmoid(X_te @ w + b) > 0.5).astype(int) == y_te).mean()
print(f"测试集准确率: {acc:.4f}")