# import numpy as np
#
# # 创建示例数据
# X = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
#     [10, 11, 12]
# ])
#
# print("原始数据X (形状4×3):")
# print(X)
# print()
#
# # ==================== 操作1：X.T @ X ====================
# result1 = X.T @ X
# print("操作1：X.T @ X 的结果（形状3×3）:")
# print(result1)
# print()
#
# # ==================== 操作2：np.cov(X.T) ====================
# # np.cov(X.T) 的含义：把X的转置作为输入
# # 等价于 np.cov(X, rowvar=False)
# result2 = np.cov(X.T)
# print("操作2：np.cov(X.T) 的结果（形状3×3）:")
# print(result2)
# print()
#
# # ==================== 操作3：手动计算协方差矩阵 ====================
# X_centered = X - X.mean(axis=0)  # 先中心化
# print(X_centered)
# result3 = (X_centered.T @ X_centered) / (X.shape[0] - 1)
# print("操作3：手动计算 (X_centered.T @ X_centered)/(n-1)（形状3×3）:")
# print(result3)
# print()
#
# # ==================== 对比 ====================
# print("=" * 60)
# print("对比结果：")
# print("=" * 60)
# print("操作2 == 操作3？", np.allclose(result2, result3))
# print("操作1 == 操作2？", np.allclose(result1, result2))
# print()

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
# Define the 2D high-frequency sine wave function
def high_freq_wave_2d(x, y, frequency=10):
    return np.sin(2 * np.pi * frequency * (x + y))


# Rotation function to apply a 2D rotation
def rotate_coords(x, y, angle):
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Apply rotation matrix to each coordinate
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot

# Create a meshgrid for the x and y coordinates
x_vals = np.linspace(0, 1, 1000)
y_vals = np.linspace(0, 1, 1000)
x, y = np.meshgrid(x_vals, y_vals)

# Original high-frequency wave (sin(x + y))
z_original = high_freq_wave_2d(x, y)

# Plot the original high-frequency wave
fig, axes = plt.subplots(1, 3, figsize=(24, 6))  # Create 4 subplots

# Plot original function
im0 = axes[0].imshow(z_original, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
axes[0].set_title("Original High-Frequency Wave")
axes[0].set_xlabel("x", style='italic', fontsize=14)
axes[0].set_ylabel("y", style='italic', fontsize=14)
# fig.colorbar(im0, ax=axes[0])

# Rotation angles
angles = [15,30]  # Rotation angles in degrees

# Plot the rotated functions
for i, angle in enumerate(angles):
    # Rotate the coordinates and calculate the rotated wave
    x_rot, y_rot = rotate_coords(x, y, angle)
    z_rot = high_freq_wave_2d(x_rot, y_rot)

    # Plot the rotated wave
    im = axes[i + 1].imshow(z_rot, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    axes[i + 1].set_title(f"Rotated by {angle}°")
    axes[i + 1].set_xlabel("x", style='italic', fontsize=14)
    axes[i + 1].set_ylabel("y", style='italic', fontsize=14)
    # fig.colorbar(im, ax=axes[i + 1])

# Display the plot
plt.tight_layout()
plt.show()
