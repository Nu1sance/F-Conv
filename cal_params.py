# F_Conv 的参数形状为 [outplane // tranNum, inplane // tranNum, tranNum, size * size], 总参数量为 outplane * inplane * size * size // tranNum
# E2CNN 的参数形状为目前可以理解为 [outplane // tranNum, inplane // tranNum, tranNum, 6 | 5 | 11 | 12]
# CNN 的参数为 [outplane, inplane, size, size]

import torch
import torch.nn as nn
from F_Conv import Fconv_PCA
from e2cnn.nn import R2Conv, GeometricTensor, FieldType
from e2cnn import gspaces

size = 6
inplane = 8
outplane = 8
tranNum = 8
bias = False

r2_act = gspaces.Rot2dOnR2(N=tranNum)
N = r2_act.fibergroup.order()
intype = FieldType(r2_act, [r2_act.regular_repr] * (inplane // N))
outtype = FieldType(r2_act, [r2_act.regular_repr] * (outplane // N))

F_Conv = Fconv_PCA(
    sizeP=size,
    inNum=inplane // tranNum,
    outNum=outplane // tranNum,
    tranNum=tranNum,
    padding=1,
    ifIni=0,
    bias=bias
)
E_Conv = R2Conv(
    kernel_size=size,
    in_type=intype,
    out_type=outtype,
    padding=1,
    bias=bias
)
Conv = nn.Conv2d(
    kernel_size=size,
    in_channels=inplane,
    out_channels=outplane,
    padding=1,
    bias=bias
)

Conv_params = sum(p.numel() for p in Conv.parameters() if p.requires_grad)
FConv_params = sum(p.numel() for p in F_Conv.parameters() if p.requires_grad)
EConv_params = sum(p.numel() for p in E_Conv.parameters() if p.requires_grad)
print(f"Total number of parameters in Conv2d layer: {Conv_params}")
print(f"Total number of parameters in FConv layer: {FConv_params}")
print(f"Total number of parameters in EConv layer: {EConv_params}")
