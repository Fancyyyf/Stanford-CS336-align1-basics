import torch
import math


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    归一化函数
    x: 任意形状的输入张量
    dim: 需要进行概率归一化的维度
    """
    # 寻址最大值，必须保持维度 keepdim=True 以触发后续的 Broadcasting 对齐
    max_vals = torch.max(x, dim=dim, keepdim=True).values

    # 平移不变性：全员减去最大值，彻底消灭 inf 溢出可能
    # exp_x 的最大值被死死锁在 exp(0) = 1
    exp_x = torch.exp(x - max_vals)

    # 计算分母
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    # 概率收敛：得到严格总和为 1 的分布
    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    缩放点积注意力算子。
    query: (batch_size, ..., seq_len, d_k)
    key:   (batch_size, ..., seq_len, d_k)
    value: (batch_size, ..., seq_len, d_v)  维度不一定统一，但工程实现另d_k = d_v = d_model / h
    mask:  (seq_len, seq_len) 的布尔型矩阵，False 表示物理阻断（下三角以隔绝未来信息）
    """
    # 提取 d_k，用于后续的方差缩放
    d_k = query.size(-1)

    # 点积寻址 (Compute Affinity)
    # query 形状: (..., seq_len, d_k)
    # key 转置后形状: (..., d_k, seq_len)
    # scores 形状: (..., seq_len_Q, seq_len_K)
    scores = query @ key.transpose(-2, -1)

    # 方差缩放 (Scale)
    # 将方差强行拉回 1，拯救 Softmax 的梯度流
    scores = scores / math.sqrt(d_k)

    # 因果掩码防火墙 (Causal Masking)
    if mask is not None:
        # False 的位置必须被填入 -inf
        # 使用 ~mask (按位取反) 选中所有 False 的位置，利用广播机制覆盖 (..., seq_len, seq_len)
        scores = scores.masked_fill(~mask, float('-inf'))

    # 概率归一化 (Softmax)
    # 注意：注意力永远是在键（Key）的序列维度上做分布，即最后一个维度 dim=-1
    attention_weights = softmax(scores, dim=-1)

    # 语义聚合 (Aggregate)
    # attention_weights 形状: (..., seq_len, seq_len)
    # value 形状: (..., seq_len, d_v)
    # output 形状: (..., seq_len, d_v)
    output = attention_weights @ value

    return output