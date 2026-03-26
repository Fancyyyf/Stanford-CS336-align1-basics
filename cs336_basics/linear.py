import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """
    Linear Module
    用于固定乘法层，输入维度为 in_features，输出维度为 out_features。
     - 该模块包含一个权重矩阵 W，形状为 (out_features, in_features)，不包含偏置项。（确保行优先，提高性能）
     - 权重矩阵 W 的元素使用截断正态分布进行初始化，均值为 0，标准差为 sqrt(2 / (in_features + out_features))，截断范围为 [-3*std, 3*std]。
     - 前向传播时，输入 x 的最后一维大小为 in_features，输出为 x @ W^T，形状为 (..., out_features)。
     - 该模块不包含任何可学习的偏置项。
     - 该模块的参数 W 应该注册为 nn.Parameter，以便在训练过程中被优化器更新。
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # construct and store your parameter as W (not W^T) for memory ordering reasons.(out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # For initializations, use the settings from above along with trunc_normal_
        # 方差 sigma^2 = 2 / (d_in + d_out)
        std = math.sqrt(2.0 / (in_features + out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation to the input.
        # 因为输入 x 最后一维是 in_features，而我们存的是 W (out, in)
        # 所以必须转置权重，利用 Python 原生的 '@' 矩阵乘法算子
        return x @ self.weight.T