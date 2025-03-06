import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# used in computing the linear transformation
from kernel import act_quant, weight_dequant, fp8_gemm

world_size = 1  # 参与分布式训练的总进程数或设备数
rank = 0        # 当前进程（或 GPU）的编号，取值范围从 0 到 world_size-1
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4  # 16,384
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944  # Intermediate dimension for MLP layers.
    moe_inter_dim: int = 1408  # Intermediate dimension for MoE layers.
    n_layers: int = 27  # Number of transformer layers.
    n_dense_layers: int = 1  # Number of dense layers in the model.
    n_heads: int = 16  # Number of attention heads.

    # moe
    n_routed_experts: int = 64  # Number of routed experts for MoE layers.
    n_shared_experts: int = 2   # Number of shared experts for MoE layers.
    n_activated_experts: int = 6  # Number of activated experts in MoE layers.
    n_expert_groups: int = 1  # Number of expert groups.
    n_limited_groups: int = 1  # Number of limited groups for MoE routing.
    score_func: Literal["softmax", "sigmoid"] = "softmax"  # Scoring function for MoE routing.
    route_scale: float = 1  #  Scaling factor for routing scores.

    # mla
    q_lora_rank: int = 0  # LoRA rank for query projections.
    kv_lora_rank: int = 512  # LoRA rank for key,value projections.
    qk_nope_head_dim: int = 128  # Dimension for query-key projections without positional embeddings.
    qk_rope_head_dim: int = 64   # Dimension for query-key projections with rotary embeddings.
    v_head_dim: int = 128  # Dimension for value projections.

    # yarn
    original_seq_len: int = 4096  # Original sequence length.
    rope_theta: float = 10000.0   # Base for rotary positional encoding. 编码中的底数
    rope_factor: float = 40       # Scaling factor for extended sequence lengths.
    beta_fast: int = 32           # Fast beta correction factor.
    beta_slow: int = 1            # Slow beta correction factor.
    mscale: float = 1.            # Scaling factor for extended attention.


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.
    将词汇表均分到多个并行进程（或 GPU）上，各个进程只负责处理自己那一部分词汇，通过通信将最终的 embedding 结果合并起来

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"

        self.part_vocab_size = (vocab_size // world_size)   # 划分词表
        self.vocab_start_idx = rank * self.part_vocab_size  # 确定每个进程负责词表开始位置
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        # 每个进程只存储自己那部分词汇的 embedding 权重矩阵，其形状为 (局部词汇数, embedding 维度)
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: tensor.Torch) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            # mask 用于标记那些不属于当前进程负责范围内的 token 索引
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx  # 计算局部索引
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)  # 使用分布式通信操作 all_reduce 将所有进程的 embedding 结果求和
        return y

        

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    A linear transformation for incoming data x: y = xA^T + b
    Supports specialized implementations based on quantization and tensor formats.

    args:
        x: input tensor
        weight: A, may be quantized and requires dequantization for certain cases
        bias: b, default to None

    returns:
        torch.Tensor: the result of linear transformation, may involve quantization-aware computations

    notes:
        - if `weigth` is quantized, (e.g., `element_size() == 1`), a dequantized version is used
            torch.Tensor.element_size() -> Returns the size in bytes of an individual element.
        - if `gemm_impl == 'bf16'`, dequantization and a `bf16` GEMM operation are applied
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weigth.element_size() > 1:
        # i.e. not quantized
        return F.linear(x, weight, bias)
    elif gemm_impl == 'bf16':
        # apply dequantization and a `bf16` GEMM operation, 利用 bfloat16 的高性能计算
        # 对量化权重进行反量化
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # 采用 fp8 gemm
        # 对输入 x 进行量化, 通过 act_quant() 量化后的 x 及其缩放因子 scale
        x, scale = act_quant(x, block_size)  
        # 执行低精度（FP8）的矩阵乘法计算
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y

class Linear(nn.Module):
    """
    Customized Linear layer

    args:
        in_features: number of input features
        out_features: number of output features
        bias: bias term
        dtype: Data type for the layer. Defaults to `torch.bfloat16`
    """
    dtype = torch.bfloat16
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        
        if self.weight.element_size() == 1:
            # weight is quantized, 需要为权重附加一个缩放因子参数
            # 缩放因子的尺寸是根据 block_size 计算的，目的是对权重矩阵按块进行量化，从而在每个块内保持较好的数值精度
            # block_size = 128
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        eles:
             self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: tensor.Torch) -> torch.Torch:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    args:
        in_features: number of input features
        out_features: total number of output features
        bias: default -> False
        dtype: default -> `torch.bfloat16`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_feature, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, se;f.weight, self.bias)
        return y

class RowParallelLinear(Linear):
      """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Norm
    rms(x) = sqrt{1/n sum_i (a_i)^2 + eps}

    args:
        dim: input tensor dimension
        eps: epsilon value for numeriacal stability
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precomputed_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.
    预计算频率对应的复数指数值, 根据模型参数预先计算出用于旋转位置编码的复数值

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim  # 旋转位置编码作用的头部维度大小
    seqlen = args.max_seq_len
    # 计算位置校正的参数，当序列长度超过原始训练时的最大长度（args.original_seq_len）时需要做一些补偿
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta     # 编码中的底数，用于计算指数
    factor = args.rope_factor  # 用于对频率进行缩放调整的因子

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.
        根据旋转次数 num_rotations 计算“校正维度”。
        用于量化在更长序列中旋转角度变化的“幅度”，从而为后续平滑过渡提供依据。
        
        correction_dim = (dim * ln(max_seq_len/num_rotations * 2pi)) / 2 * ln(base)

        args:
            num_rotations: number of rotations to compute the corrections for
            dim: embedding space dimension
            base: base value for exponential computation
            max_seq_len: maximum sequence length

        returns:
            float: the corrections dimension based on the input parameters 
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.
        根据给定的旋转下界 low_rot 和上界 high_rot，计算出对应的校正维度范围（下界和上界），并对结果做有效性限制（下界至少为 0，上界不超过 dim-1）。
        用这个范围来决定在哪些维度上应用频率补偿。

        args:
            low_rot: lower bound for the number of rotations
            high_rot:  upper bound for the number of rotations
            dim: dimensionality of the embedding space
            base: base value for exponential computation
            max_seq_len: maximum sequence length
        return:
            Tuple[int, int]: the range of corrections dimensions(low, high), clamped to valid indices
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len)
        high = math.floor(find_correction_dim(high_rot, dim, base, max_seq_len)
        return max(low, 0), min(high, dim - 1)
    
    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.
        生成一个线性递增的斜坡函数，输出一个形状为 (dim,) 的张量，其数值从 0 到 1 线性插值，并在 [min, max] 区间内进行平滑过渡。
        
        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001  # 对 max 做微小调整，避免除 0 错误
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # 取 dim 中的偶数位置（即每对实部和虚部共享同一频率）
    # 得到的 freqs 张量形状为 (dim/2,)，每个元素代表对应维度上的旋转频率
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # 扩展序列长度时的频率调整
    if seqlen > args.original_seq_len:
        # 找到校正的维度范围
        low, high = find_correction_range(low_rot=beta_fast, fast_rot=beta_slow, dim=dim, base=base, max_seq_len=args.original_seq_len)
        # 计算出一个平滑因子 smooth（取值在 0 到 1 之间），使得在不同维度上能平滑过渡地调整频率。
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 构造位置角度矩阵并转换为复数表示
    t = torch.arange(seqlen)  # 构造时间步 t 的序列
    # 计算外积，得到形状为 (seqlen, len(freqs)) 的矩阵，
    # 每个位置的元素为时间步与对应频率的乘积，代表角度（弧度）。
    freqs = torch.outer(t, freqs)
    # 将这些角度转换为单位幅值的复数
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: tensor.Torch, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensors.

    args:
        x: input tensor with positional embeddings to be applied
        freqs_cis: precomputed complex exponential values for positional embeddings

    return:
        torch.Tensor: tensor with rotary embeddings applied
    """
    dtype = x.dtype
    # 先将 x 转换为 float 类型，再将最后一维重塑为形状 (-1, 2)，
    # 然后利用 torch.view_as_complex 将其视作复数张量
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # reshape，使其能够在 批次、序列长度和头数 等维度上与 x 对应，确保后续复数乘法时可以自动广播
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # 将输入 x 与旋转因子 freqs_cis 相乘, 此乘法相当于对输入向量进行旋转，嵌入了位置信息
    # 再通过 torch.view_as_real 转换回实数形式（这会将复数分解为实部和虚部，最后一维的大小变为2）
    # 使用 flatten(3) 将最后两维重新展平（将复数拆分的两个维度合并），以便保持与原始张量一致的维度结构
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Headed Laten Attention Layer

    Attributes:
        dim: input features dimension
        n_head: number of attention heads
        n_local_heads: attention heads per distributed system
        q_lora_rank: rank of low-rank query projection
        kv_lora_rank: rank of low-rank key-value projection
        qk_nope_head_dim: dimension of non-positional query/key projections
        qk_rope_head_dim: dimension of rotary-positional query/key projections
        qk_head_dim: total dimension of value projections = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim: dimension of value projections
        softmax_scale: scaling factor for softmax in attention computation
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # 对 q 权重矩阵
        if self.q_lora_rank == 0:
            # q_lora_rank = 0, 不进行低秩分解
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 进行低秩分解
            # 先降维 dim -> q_lora_rank
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            # 归一化后扩展 q_lora_rank -> n_heads * qk_head_dim
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # 对 kv 权重矩阵
        # 降维 dim -> kv_lora_rank + qk_rope_head_dim, 将旋转位置编码 与 低秩部分拼接
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # 归一化后扩展 kv_lora_rank -> n_heads * (qk_nope_head_dim + v_head_dim)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5  # 根号 d_k

        if args.max_seq_len > args.original_seq_len:
            # 如果实际序列长度 max_seq_len 大于原始训练时的 original_seq_len
            # 则通过参数计算一个额外的缩放因子（mscale），对 softmax 缩放因子进行调整
            # 为了在长序列下弥补由于位置编码扩展可能带来的数值偏差
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == 'naive':
            # 分布缓存 key 和 value
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            # 缓存合并的 kv 和 pe
            # 存储低秩和位置编码部分
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        # 计算 q
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        # 分隔 q -> 不带位置编码部分 和 用于旋转位置编码的部分
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # 添加旋转位置编码
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # 计算 k, v
        kv = self.wkv_a(x)  # 一个中间张量
        # 分割成 低秩映射 (kv_lora_rank) 和 应用于旋转位置编码的部分 (qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # 先通过 unsqueeze(2)增加一个维度（以便后续广播）再嵌入旋转位置编码
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        if atten_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)  # 拼接出完整的 查询向量 q
            kv = self.wkv_b(self.kv_norm(kv))  # 计算出 kv
            # reshape -> (batch_size, seq_len, n_local_heads, qk_nope_head_dim + v_head_dim)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            # 分割为 不带位置编码的键 和 值投影
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # 将 不带位置编码 和 带位置信息 的键 拼接
            # 对 k_pe 使用 expand 调整尺寸以匹配本地头数
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            # 缓存当前块的键和值到 k_cache 与 v_cache
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # 计算注意力得分
            score = torch.einsum("bshd, bthd -> bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # 对 self.wkv_b 的权重进行检查，如果有量化则调用 weight_dequant 得到反量化权重
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            # 然后重塑为 (n_local_heads, *, kv_lora_rank)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            # 将 q_nope 与反量化后的权重前部（对应 qk_nope_head_dim）相乘，得到新的查询表示
            q_nope = torch.einsum('bshd, hdb -> bshc', q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # 将经过 RMSNorm 的低秩部分 kv 缓存到 kv_cache
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            # 将旋转位置编码部分 k_pe（经 squeeze 调整维度后）缓存在 pe_cache 中
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            # 计算注意力得分, 包括两部分
            # (计算 q_nope 与 kv_cache 之间的点积 + 计算 q_pe 与 pe_cache 之间的点积) * self.softmax_scale
            scores = (
                torch.einsum("bshc, btc -> bshc", q_nope, self.kv_cache[:bsz, :end_pos]) + 
                torch.einsum("bshr, btr -> bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

    if mask is not None:
        # 提供 mask，将其 unsqueeze 后加到 scores 上
        scores += mask.unsqueeze(1)
    # 对 scores 在最后一个维度（代表不同时间步或位置）上做 softmax 操作，确保注意力权重归一化
    scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

    if atten_impl == 'naive':
        # 将归一化后的注意力得分与缓存的 v_cache 进行加权求和，得到最终的注意力输出
        x = torch.einsum("bsht, bthd -> bshd", scores, self.v_cache[:bsz, :end_pos])
    else:
        # 先对 kv_cache 计算加权结果
        x = torch.einsum("bsht, btc -> bshc", scores, self.kv_cache[:bsz, :end_pos])
        # 再用反量化权重后部（对应 v 部分）做一次线性映射
        x = torch.einsum("bshc, hdc -> bshd", x, wkv_b[:, -self.v_head_dim:])

    # 将多头输出（先将各头 flatten 成一维）送入 self.wo（RowParallelLinear 层）映射回原始维度，从而得到与输入相同形状的输出
    x = self.wo(x.flatten(2))
    return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron used as a feed-forward layer

    attributes:
        w1: linear layer for input-to-hiddden
        w2: linear layer for hidden-to-output
        w3: additional linear for feature transformation
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        initialize mlp

        args:
            dim: dimension of input/output
            inter_dim: dimension of hidden layer
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the mlp layer
        """

        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a moe model.l bias term for the gate
    """
    def __init__(self, args: ModelArgs):
        " init gate "
        super().__init__()
        self.dim = args.dim   # input feature dimension
        self.topk = args.n_activated_experts  # number of top experts activated for each input
        self.n_groups = args.n_expert_groups  # number of groupts for routing
        self.topk_groups = args.n_limited_groups  # number of groups to route inputs to
        self.score_func = args.score_func     # scoring function softmax/sigmoid
        self.route_scale = args.route_scale   # scaling factor for routing weights
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))  # learnable weights for the gate
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None  # optional bias term for the gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores

        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weigths = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices

class Expert(nn.Module):
    " Expert Layer " 
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    MoE module
    """
    def __init__(self, args: ModuleArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.expert_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i <= self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer Block with attention and FFN.

    Attributes, all nn.Module:
        attn: MLA 
        ffn: FFN (MLP or MoE)
        attn_norm: layer norm for attention layer
        ffn_norm: layer norm for FFN
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        # RMSNorm -> Attention -> ResConn
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        # RMSNorm -> FFN -> ResConn
        x = x + self.ffn(self.ffn_norm(x))
        return

class Transformer(nn.Module):
    """
    Transformer in Deepseek, with positional embedding, multiple layers, and output projection.
    
    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        # 初始化 分布式变量
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 1
        Linear.dtype = torch.float8_e4m3fn if args.dtype == 'fp8' else torch.bfloat16
        
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), presistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)  # get hidden state
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]  # 归一化
        logits = self.head(h)    # 计算 logits
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            # 合并所有 logits
            logits = torch.cat(all_logits, dim=-1)
        return logits