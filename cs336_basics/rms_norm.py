import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    用均方根 (Root Mean Square) 替代了传统方差，去除了减去均值的操作和偏置项。
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        # 防除零溢出的极小值 epsilon
        self.eps = eps

        # 权重参数 g_i，形状为 (d_model)，每个特征维度一个增益参数
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存原始的数据类型 （bf16 & f16）
        in_dtype = x.dtype

        # 升维到 f32, 防止下面计算平方时数值溢出
        x_f32 = x.to(torch.float32)

        # 计算 RMS 分母：求最后一个维度 (d_model) 的均方
        # keepdim=True 极其重要！它能让形状 (batch, seq, d_model) 变成 (batch, seq, 1)，防止计算完均值后降维

        mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)

        # 归一化， 用 torch.rsqrt(x) 代替 1.0 / torch.sqrt(x)
        # rsqrt 是底层 C++/CUDA 专门优化的倒数开方函数，速度更快且数值更稳定
        # eps防止除零
        x_normed = x_f32 * torch.rsqrt(mean_sq + self.eps)

        # 乘以可学习的权重
        result = x_normed * self.weight.to(torch.float32)

        # 将结果变回原本的类型（bf16 或 f16），以节省显存
        return result.to(in_dtype)