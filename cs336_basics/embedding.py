import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    负责将整数 Token IDs 映射为 d_model 维度的稠密向量。
    本质上是一个通过物理内存指针偏移进行 O(1) 查表的高性能哈希字典。
    """

    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        """
            vocab_size (int): 词表大小，即字典的行数。
            d_model (int): 特征维度，即每个词的语义长度,空间位置。
            device: 目标设备，直接在这里开辟物理内存，拒绝 CPU 缓存中转。
            dtype: 目标数据类型，通常为 bfloat16 以节省半数显存。
        """

        super().__init__()

        # 开辟物理内存，注册为可学习的参数 (Parameter)
        self.weight = nn.Parameter(
            torch.empty((vocab_size, d_model), device=device, dtype=dtype)
        )

        # 正态分布截断初始化[-3, 3]
        # 这些数字将作为模型最初的“随机记忆”。
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（查字典）。
        x (torch.Tensor): 包含整数 Token IDs 的张量。
                            形状可以是 (batch_size, sequence_length) 或者任意维度。
                            此处的 sequence_length 可以随生成过程动态增长(训练/生成)。

        Returns:
            torch.Tensor: 映射后的稠密浮点张量。
                         形状膨胀为 (batch_size, sequence_length, d_model)
        """
        # 利用 PyTorch C++ 底层的高级索引 (Advanced Indexing)
        # 底层直接通过 x 里的整数，计算出物理内存地址的指针偏移量，然后把那 d_model 个数字从显存拷贝出来。
        return self.weight[x]