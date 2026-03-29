import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        构造 RoPE 模块。
        theta: 旋转基频 Theta (例如 10000.0)
        d_k: Query 和 Key 的维度大小
        max_seq_len: 最大支持序列长度
        """
        super().__init__()

        # 预计算频率因子 (inv_freq)
        # 公式: theta_{i,k} = i / (Theta ** ((2k-2)/d))
        # 这里用 arange(0, d_k, 2) 生成 2k-2 的等效序列
        # 使用 float32 计算以确保极小频率的数值稳定性
        powers = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        inv_freq = 1.0 / (theta ** powers)  # Shape: (d_k // 2,)

        # 生成绝对位置张量 i, Shape: (max_seq_len)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # 外积运算展开完整的二维相位矩阵 (m * theta)
        # freqs[i, j] = positions[i] * inv_freq[j]
        freqs = torch.outer(positions, inv_freq)  # Shape: (max_seq_len, d_k // 2)

        # 相邻配对维度的物理扩充
        # 为了与相邻维度 [x0, x1], [x2, x3] 配对，我们需要将每个频率复制两次
        # 结果变成 [freq0, freq0, freq1, freq1, ...]
        emb = torch.repeat_interleave(freqs, 2, dim=-1)  # Shape: (max_seq_len, d_k)

        # 注册为 Buffer，并设置 persistent=False 避免污染 state_dict
        # 固定cos&sin矩阵，多层共享
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        应用 RoPE 到输入张量。
        x: (..., seq_len, d_k) - 容忍任意前置 batch 维度
        token_positions: (..., seq_len) - 用于切片缓存的实际 token 位置 （用于后续推理时可变长引入新token时）
        """
        # 动态索引：利用 token_positions 抽取当前序列对应的 cos 和 sin 相位
        # PyTorch 的高级索引会自动处理 Broadcasting
        cos = self.cos_cached[token_positions]  # Shape: (..., seq_len, d_k)
        sin = self.sin_cached[token_positions]  # Shape: (..., seq_len, d_k)

        # 构造旋转向量 x_rotated

        # 方案A:相邻配对 (当前采用，对应文档公式 8)
        # 切分偶数索引 x0, x2... 和奇数索引 x1, x3...
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # 利用 stack（两个一维叠加升维） + flatten（拍扁降维） 重组为 [-x1, x0, -x3, x2, ...]
        x_rotated = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
        # -----------------------------------------------------------
        # 方案 B: LLaMA 对半拆分 (仅供参考)
        # x_low, x_high = x.chunk(2, dim=-1)
        # x_rotated = torch.cat([-x_high, x_low], dim=-1)
        # 注意: 若用此法，init 中的 emb 构造必须改成 torch.cat([freqs, freqs], dim=-1)

        # 复数乘法的实数域展开 (Euler's Formula)
        # 保证 x_rotated 和 x 参与计算时数据类型一致
        # 构造新向量相加而避免矩阵乘法（仅用于这种一个矩阵固定时的优化）
        out = (x * cos) + (x_rotated * sin)

        # 保持与原始输入相同的 dtype (防止 BFloat16/Float16 漂移)
        return out.to(x.dtype)