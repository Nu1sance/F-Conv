import numpy as np

# 创建示例数据
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

print("原始数据X (形状4×3):")
print(X)
print()

# ==================== 操作1：X.T @ X ====================
result1 = X.T @ X
print("操作1：X.T @ X 的结果（形状3×3）:")
print(result1)
print()

# ==================== 操作2：np.cov(X.T) ====================
# np.cov(X.T) 的含义：把X的转置作为输入
# 等价于 np.cov(X, rowvar=False)
result2 = np.cov(X.T)
print("操作2：np.cov(X.T) 的结果（形状3×3）:")
print(result2)
print()

# ==================== 操作3：手动计算协方差矩阵 ====================
X_centered = X - X.mean(axis=0)  # 先中心化
print(X_centered)
result3 = (X_centered.T @ X_centered) / (X.shape[0] - 1)
print("操作3：手动计算 (X_centered.T @ X_centered)/(n-1)（形状3×3）:")
print(result3)
print()

# ==================== 对比 ====================
print("=" * 60)
print("对比结果：")
print("=" * 60)
print("操作2 == 操作3？", np.allclose(result2, result3))
print("操作1 == 操作2？", np.allclose(result1, result2))
print()