import numpy as np
np.random.seed(0)

# ---- 造数据：y = X · w_true + 噪声 ----
n, d = 200, 3
X = np.random.randn(n, d)
w_true = np.array([2.0, -1.0, 0.5])
y = X @ w_true + 0.1 * np.random.randn(n)

# ---- 解法一：正规方程（解析解）----
w_ls = np.linalg.solve(X.T @ X, X.T @ y)
# 用 solve 而不是 inv：数值上更稳，也更快

# ---- 解法二：梯度下降（迭代解）----
w_gd = np.zeros(d)
lr = 0.1
losses = []
for step in range(300):
    pred = X @ w_gd
    loss = ((pred - y) ** 2).mean()
    losses.append(loss)
    grad = 2 * X.T @ (pred - y) / n     # 就是上面推的那个公式
    w_gd -= lr * grad

print("真值  ", w_true)
print("解析解", w_ls)
print("梯度下降", w_gd)
print("最大差值", np.abs(w_ls - w_gd).max())