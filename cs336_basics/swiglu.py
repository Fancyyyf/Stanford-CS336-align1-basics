import torch
import torch.nn as nn
from cs336_basics import linear

class FFN(nn.Module):
    """
    这是 Transformer 存储知识的独立闭门记忆库, 在注意力机制完成后进行训练修改确保稳定。
    采用 LLaMA 同款的双流门控架构 (SwiGLU)。
    """

    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            # 理论的 8/3 膨胀维度d_ff
            hidden_dim = int(8 * d_model / 3)
            # 向上取整到 64 的倍数，防止 GPU 内存碎片化
            multiple_of = 64
            self.d_ff = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        else:
            self.d_ff = d_ff

        # SwiGLU 的三大核心矩阵 (摒弃bias， 提高性能)(使用自己构建的linear类)
        # 门控 (Gate)
        self.w1 = linear.Linear(d_model, self.d_ff, device=device, dtype=dtype)
        # 升维增大信息(Up)
        self.w3 = linear.Linear(d_model, self.d_ff, device=device, dtype=dtype)
        # 降维总结 (Down)
        self.w2 = linear.Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播查询知识库。
        x 形状: (batch_size, seq_len, d_model)
        """
        # Gate Projection
        gate = self.w1(x)
        # 使用防溢出的 torch.sigmoid 构建 SiLU 激活函数
        silu_gate = gate * torch.sigmoid(gate)

        # 升维 (Up Projection)
        up = self.w3(x)

        # 门控筛选 (Element-wise Multiplication)
        activated_hidden = silu_gate * up

        # 压缩降维，汇入主干道 (Down Projection)
        return self.w2(activated_hidden)