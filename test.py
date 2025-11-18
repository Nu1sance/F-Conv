import torch
import numpy as np
from F_Conv import Fconv_PCA

# 创建卷积层
# Fconv_PCA中的inNum和outNum代表每个方向的通道数，要乘以tranNum才是总通道数
# conv = Fconv_PCA(
#     sizeP=3,
#     inNum=64 // 8,
#     outNum=64 // 8,
#     tranNum=8,
#     inP=3,
#     padding=1,
#     ifIni=0
# )
#
# print("=" * 60)
# print("参数详情：")
# print("=" * 60)
#
# # 基函数
# print(f"Basis shape: {conv.Basis.shape}")
# print(f"Basis 是否可训练: {conv.Basis.requires_grad}")
#
# # 权重
# print(f"\nWeights shape: {conv.weights.shape}")
# print(f"Weights 是否可训练: {conv.weights.requires_grad}")
# Rank = conv.weights.shape[-1]
# print(f"实际 Rank: {Rank}")
#
# # 偏置
# print(f"\nBias shape: {conv.c.shape}")
#
# # 总参数量
# total_params = conv.weights.numel() + conv.c.numel()
# print(f"\n可训练参数总量: {total_params:,}")
#
# # 前向传播测试
# x = torch.randn(2, 64, 32, 32)  # [batch, channel, H, W]
# y = conv(x)
# print(f"\n输入形状: {x.shape}")
# print(f"输出形状: {y.shape}")
# print(f"输出通道数: {y.shape[1]} = {conv.outNum} * {conv.tranNum}")

print("=" * 60)


def MaskC(SizeP, tranNum):
    p = (SizeP - 1) / 2
    x = np.arange(-p, p + 1) / p
    print(x)
    print("=" * 60)
    X, Y = np.meshgrid(x, x)
    print(X)
    print("=" * 60)
    print(Y)
    print("=" * 60)
    C = X ** 2 + Y ** 2
    # print(C)
    # print("=" * 60)
    if tranNum == 4:
        Mask = np.ones([SizeP, SizeP])
    else:
        if SizeP > 4:
            Mask = np.exp(-np.maximum(C - 1, 0) / 0.2)
        else:
            Mask = np.exp(-np.maximum(C - 1, 0) / 2)
    # print(Mask)
    # print("=" * 60)
    return X, Y, Mask

sizeP, tranNum = 7, 8
inP = sizeP
inX, inY, Mask = MaskC(sizeP, tranNum)
X0 = np.expand_dims(inX,2)
Y0 = np.expand_dims(inY,2)
Mask = np.expand_dims(Mask,2)
theta = np.arange(tranNum) / tranNum * 2 * np.pi
print(theta)
print( "=" * 60)
theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
print(theta)
print( "=" * 60)
X = np.cos(theta)*X0-np.sin(theta)*Y0
Y = np.cos(theta)*Y0+np.sin(theta)*X0
print(X.shape)
print("=" * 60)
print(Y)
print("=" * 60)
X = np.expand_dims(np.expand_dims(X, 3), 4)
Y = np.expand_dims(np.expand_dims(Y, 3), 4)
print(X.shape)
print("=" * 60)
print(Y)
print("=" * 60)
v = np.pi / inP * (inP - 1)
p = inP / 2
print(v)
print("=" * 60)
k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
l = np.reshape(np.arange(inP), [1, 1, 1, 1, inP])
print(k)
print("=" * 60)
BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
print(BasisC.shape)
print("=" * 60)
print(BasisS.shape)
print("=" * 60)
BasisC = np.reshape(BasisC, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)  # 掩码
BasisS = np.reshape(BasisS, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)

BasisC = np.reshape(BasisC, [sizeP * sizeP * tranNum, inP * inP])  # [7 * 7 * 8, 7 * 7]的一个矩阵
BasisS = np.reshape(BasisS, [sizeP * sizeP * tranNum, inP * inP])  # 每行代表一个采样坐标，该行有 7 * 7 个采样值

BasisR = np.concatenate((BasisC, BasisS), axis=1)  # [392, 98] 49个cosine + 49个sine

U, S, VT = np.linalg.svd(np.matmul(BasisR.T, BasisR))  # B^T \cdot B [98, 98] 基函数之间的内积（相似度）矩阵
print(U.shape)
print("=" * 60)
print(S.shape)
print("=" * 60)
print(VT.shape)
print("=" * 60)