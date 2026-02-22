"""
================================================================================
DiT (Diffusion Transformer) - 基于PyTorch的简化实现
================================================================================

参考论文: "Scalable Diffusion Models with Transformers" by Peebles & Xie, 2023
论文链接: https://arxiv.org/abs/2212.09748

【核心思想】
将扩散模型中的U-Net替换为Transformer：
1. 把图像切成小块(patches)，像ViT一样
2. 用Transformer处理这些patch序列
3. 通过adaLN(自适应层归一化)注入条件信息(时间步、类别等)

【文件说明】
- dit.py: 模型定义（你正在阅读的文件）
- train.py: 训练脚本
- sample.py: 采样/生成脚本
- train_sr.py: 超分辨率训练脚本
- sample_sr.py: 超分辨率采样脚本
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# 第一部分：位置编码 (Positional Encoding)
# =============================================================================
# 位置编码告诉Transformer每个patch在图像中的位置
# 这里使用正弦-余弦编码，是Transformer的经典做法

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    生成2D正弦-余弦位置编码

    【原理】
    - 将2D位置(h, w)编码为向量
    - 使用不同频率的正弦和余弦函数
    - 这样模型可以学习到相对位置关系

    Args:
        embed_dim: 嵌入维度（每个位置编码成的向量长度）
        grid_size: 网格大小（图像切成grid_size × grid_size个patch）

    Returns:
        pos_embed: [grid_size * grid_size, embed_dim] 位置编码矩阵

    【示例】
    如果grid_size=14（28x28图像，patch_size=2），则生成14×14=196个位置编码
    每个位置编码的维度是embed_dim（如256）
    """
    # 生成网格坐标
    grid_h = torch.arange(grid_size, dtype=torch.float32)  # [0, 1, 2, ..., grid_size-1]
    grid_w = torch.arange(grid_size, dtype=torch.float32)

    # 创建2D网格
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')  # 两个[grid_size, grid_size]的矩阵
    grid = torch.stack(grid, dim=0)  # [2, grid_size, grid_size]
    grid = grid.reshape(2, 1, grid_size, grid_size)

    # 分别对h和w方向编码，然后拼接
    # embed_dim维度分成两半，一半给h方向，一半给w方向
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])  # [grid_size*grid_size, embed_dim//2]
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])  # [grid_size*grid_size, embed_dim//2]
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # [grid_size*grid_size, embed_dim]

    return pos_embed


def get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    生成1D正弦-余弦位置编码

    【公式】
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    其中d是embed_dim，i是维度索引
    """
    assert embed_dim % 2 == 0, "embed_dim必须是偶数"

    # 计算频率：不同维度使用不同频率
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / 10000 ** (omega / (embed_dim / 2))  # [embed_dim//2]

    # pos: [H, W] -> [H*W]
    pos = pos.reshape(-1)

    # 计算正弦和余弦
    out = torch.outer(pos, omega)  # [H*W, embed_dim//2]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [H*W, embed_dim]

    return emb


# =============================================================================
# 第二部分：时间步编码 (Timestep Embedding)
# =============================================================================
# 扩散模型中，不同时间步t对应不同程度的噪声
# 需要把标量t编码成向量，让模型知道当前是哪个时间步

class TimestepEmbedder(nn.Module):
    """
    时间步嵌入器

    【作用】
    将时间步t（一个整数，如500）编码成一个向量（如256维）
    这个向量会被用于调制Transformer的特征

    【结构】
    t(标量) -> 正弦编码(256维) -> MLP -> hidden_size维向量
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),  # Swish激活函数
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] 时间步，每个样本一个整数
        Returns:
            t_emb: [B, hidden_size] 时间步嵌入向量
        """
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    将时间步t编码为正弦向量（类似位置编码）
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    # t: [B] -> [B, 1], freqs: [half] -> [1, half]
    args = t[:, None].float() * freqs[None]  # [B, half]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]

    # 如果dim是奇数，补零
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# =============================================================================
# 第三部分：标签编码 (Label Embedding)
# =============================================================================
# 用于类别条件生成，比如指定生成数字"5"

class LabelEmbedder(nn.Module):
    """
    类别标签嵌入器

    【作用】
    将类别标签（如数字0-9）编码成向量

    【Classifier-Free Guidance (CFG)】
    训练时随机把标签替换成特殊token，这样模型学会：
    - 有条件生成：给定类别生成对应图像
    - 无条件生成：不指定类别也能生成
    采样时可以调节条件强度，提高生成质量
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        # 多一个特殊的"无条件"token用于CFG
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Args:
            labels: [B] 类别标签
            train: 是否在训练模式（训练时才做dropout）
        Returns:
            label_emb: [B, hidden_size] 标签嵌入向量
        """
        if train and self.dropout_prob > 0:
            # 随机把一些标签替换成"无条件"token
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.num_classes, labels)
        return self.embedding_table(labels)


# =============================================================================
# 第四部分：Patch嵌入 (Patch Embedding)
# =============================================================================
# 把图像切成小块，每个小块变成一个向量
# 这是Vision Transformer (ViT)的核心操作

class PatchEmbed(nn.Module):
    """
    图像到Patch嵌入

    【原理】
    原始图像 [B, C, H, W]
    -> 切成patches [B, C, H/P, W/P, P, P]
    -> 展平 [B, num_patches, P*P*C]
    -> 线性投影 [B, num_patches, embed_dim]

    【简化实现】
    用一个Conv2d(kernel_size=patch_size, stride=patch_size)一步到位
    这等价于：切patch + 展平 + 线性投影

    【示例】
    输入: [B, 1, 28, 28] (MNIST灰度图)
    patch_size=2
    输出: [B, 196, 256] (14*14=196个patch，每个patch编码为256维向量)
    """
    def __init__(
        self,
        img_size: int = 28,      # 图像大小
        patch_size: int = 2,     # patch大小
        in_chans: int = 1,       # 输入通道数（灰度图=1，RGB=3）
        embed_dim: int = 256,    # 嵌入维度
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # patch数量

        # 用Conv2d实现patch嵌入
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 输入图像
        Returns:
            x: [B, num_patches, embed_dim] patch嵌入序列
        """
        # Conv2d: [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        # 展平空间维度: [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches]
        x = x.flatten(2)
        # 转置: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        return x


# =============================================================================
# 第五部分：DiT Transformer块 (核心！)
# =============================================================================
# 这是DiT的核心创新：adaLN-Zero

class DiTBlock(nn.Module):
    """
    DiT Transformer块

    【标准Transformer块】
    x -> LayerNorm -> Self-Attention -> Add -> LayerNorm -> MLP -> Add -> output

    【DiT的改进：adaLN-Zero】
    1. adaLN (Adaptive Layer Norm): 用条件信息(时间步+类别)来调制归一化
       - 普通LayerNorm: 固定的scale和shift参数
       - adaLN: 从条件预测scale(γ)和shift(β)

    2. Zero: 用零初始化最后的门控参数
       - 训练初期，残差分支输出接近零
       - 相当于训练初期是恒等映射
       - 让训练更稳定

    【数据流】
    x ──→ LayerNorm ──→ ×γ1, +β1 ──→ Attention ──→ ×gate1 ──→ + ──→
    │                                                      ↑      │
    └──────────────────────────────────────────────────────────────┘

    x ──→ LayerNorm ──→ ×γ2, +β2 ──→ MLP ──→ ×gate2 ──→ + ──→ output
    │                                                   ↑      │
    └───────────────────────────────────────────────────────────┘

    其中 γ1, β1, gate1, γ2, β2, gate2 都是从条件c预测出来的
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        # LayerNorm（不学习参数，参数由adaLN提供）
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-Attention
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # MLP (Feed-Forward Network)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # adaLN调制：从条件c预测6个调制参数
        # (γ1, β1, gate1, γ2, β2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] patch序列特征
            c: [B, D] 条件向量（时间步+类别）
        Returns:
            x: [B, N, D] 处理后的特征
        """
        # 1. 从条件c预测调制参数
        modulation = self.adaLN_modulation(c)  # [B, 6*D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation.chunk(6, dim=1)  # 每个 [B, D]

        # 2. Self-Attention 分支
        # LayerNorm + 调制
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        # Attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        # 残差连接 + 门控
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 3. MLP 分支
        # LayerNorm + 调制
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        # MLP
        mlp_out = self.mlp(x_norm)
        # 残差连接 + 门控
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


# =============================================================================
# 第六部分：输出层 (Final Layer)
# =============================================================================

class FinalLayer(nn.Module):
    """
    最终输出层

    【作用】
    将Transformer的输出 [B, N, D] 转换回图像patch [B, N, P*P*C]

    【结构】
    x -> adaLN -> Linear -> 输出

    【注意】
    Linear层用零初始化，这样训练初期模型输出接近零
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        # adaLN：预测2个参数(shift, scale)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] Transformer输出
            c: [B, D] 条件向量
        Returns:
            x: [B, N, P*P*C] 预测的噪声（每个patch的像素值）
        """
        modulation = self.adaLN_modulation(c)
        shift, scale = modulation.chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


# =============================================================================
# 第七部分：完整的DiT模型
# =============================================================================

class DiT(nn.Module):
    """
    Diffusion Transformer 完整模型

    【整体流程】
    1. 输入: 噪声图像 x_t, 时间步 t, 类别标签 y
    2. Patch嵌入: 图像 -> patch序列
    3. 加位置编码
    4. 通过多个DiT Block
    5. 输出层: 预测噪声
    6. Unpatchify: 恢复图像形状

    【超分辨率模式】
    额外输入条件图像 cond
    - 噪声图像和条件图像分别patchify
    - 各自加不同的位置编码
    - 在embedding维度拼接
    - 投影回原始维度
    """
    def __init__(
        self,
        img_size: int = 28,        # 图像大小
        patch_size: int = 2,       # patch大小
        in_channels: int = 1,      # 输入通道数
        hidden_size: int = 256,    # 隐藏层维度
        depth: int = 4,            # Transformer层数
        num_heads: int = 4,        # 注意力头数
        mlp_ratio: float = 4.0,    # MLP扩展比例
        num_classes: int = 10,     # 类别数
        learn_sigma: bool = False, # 是否学习噪声方差
        cond_channels: int = 0,    # 条件图像通道数（超分辨率用）
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        self.learn_sigma = learn_sigma
        self.use_cond = cond_channels > 0

        # ============ 嵌入层 ============
        # 噪声图像的patch嵌入
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size)

        # 条件图像的嵌入（用于超分辨率）
        if self.use_cond:
            self.cond_embedder = PatchEmbed(img_size, patch_size, cond_channels, hidden_size)
            # 条件图像的位置编码（与噪声图像不同！）
            self.cond_pos_embedding = nn.Parameter(
                torch.zeros(1, self.num_patches, hidden_size)
            )
            # 拼接后投影回hidden_size
            self.cond_proj = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.cond_embedder = None
            self.cond_pos_embedding = None
            self.cond_proj = None

        # 时间步和标签嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # ============ 位置编码 ============
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        # ============ Transformer块 ============
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # ============ 输出层 ============
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 初始化噪声图像位置编码
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            self.img_size // self.patch_size
        )
        self.pos_embedding.data.copy_(pos_embed.unsqueeze(0))

        # 初始化条件图像位置编码（不同的编码）
        if self.cond_pos_embedding is not None:
            cond_pos_embed = get_2d_sincos_pos_embed(
                self.hidden_size,
                self.img_size // self.patch_size
            )
            # 加偏移量使其与噪声位置编码不同
            cond_pos_embed = cond_pos_embed + 0.5
            self.cond_pos_embedding.data.copy_(cond_pos_embed.unsqueeze(0))

        # Zero-init 最终层（让训练初期输出接近零）
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero-init 每个block的adaLN（让训练初期是恒等映射）
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将patch序列转换回图像

        【逆操作】
        [B, N, P*P*C] -> [B, C, H, W]

        其中 N = H/P * W/P
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p

        # [B, N, P*P*C] -> [B, H/P, W/P, P, P, C]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        # [B, H/P, W/P, P, P, C] -> [B, C, H/P, P, W/P, P]
        x = x.permute(0, 5, 1, 3, 2, 4)
        # [B, C, H/P, P, W/P, P] -> [B, C, H, W]
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,           # [B, C, H, W] 噪声图像
        t: torch.Tensor,           # [B] 时间步
        y: torch.Tensor,           # [B] 类别标签
        cond: torch.Tensor = None  # [B, cond_channels, H, W] 条件图像（超分辨率用）
    ) -> torch.Tensor:
        """
        前向传播

        【输入】
        x: 噪声图像（在时间步t加了噪声的图像）
        t: 时间步（表示噪声程度，0=干净，1000=纯噪声）
        y: 类别标签（如MNIST的0-9）
        cond: 条件图像（超分辨率时使用，如低分辨率图像）

        【输出】
        预测的噪声 [B, C, H, W]
        """

        # ============ Step 1: Patch嵌入 ============
        # 噪声图像 -> patch序列 + 位置编码
        x_embed = self.x_embedder(x) + self.pos_embedding  # [B, N, D]

        # 如果有条件图像（超分辨率模式）
        if cond is not None and self.use_cond:
            # 条件图像 -> patch序列 + 不同的位置编码
            cond_embed = self.cond_embedder(cond) + self.cond_pos_embedding  # [B, N, D]
            # 拼接 [B, N, 2D] -> 投影 [B, N, D]
            x = torch.cat([x_embed, cond_embed], dim=-1)
            x = self.cond_proj(x)
        else:
            x = x_embed

        # ============ Step 2: 条件嵌入 ============
        # 时间步嵌入 + 类别嵌入
        c = self.t_embedder(t) + self.y_embedder(y, self.training)  # [B, D]

        # ============ Step 3: Transformer块 ============
        for block in self.blocks:
            x = block(x, c)  # [B, N, D]

        # ============ Step 4: 输出层 ============
        x = self.final_layer(x, c)  # [B, N, P*P*C]

        # ============ Step 5: 恢复图像形状 ============
        x = self.unpatchify(x)  # [B, C, H, W]

        return x


# =============================================================================
# 第八部分：模型工厂函数
# =============================================================================

def DiT_XS(**kwargs):
    """
    Extra Small DiT - 适合MNIST的小模型

    【参数量】约500万
    【建议】用于快速实验和学习
    """
    return DiT(
        img_size=28,
        patch_size=2,
        in_channels=1,
        hidden_size=256,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        **kwargs
    )


def DiT_S(**kwargs):
    """
    Small DiT - 稍大一点的模型

    【参数量】约1200万
    """
    return DiT(
        img_size=28,
        patch_size=2,
        in_channels=1,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs
    )


def DiT_SR_XS(img_size: int = 28, scale_factor: int = 4, **kwargs):
    """
    Extra Small DiT for Super-Resolution

    【用途】图像超分辨率
    【输入】噪声图像 + 低分辨率条件图像
    【输出】高分辨率图像

    Args:
        img_size: 高分辨率图像大小
        scale_factor: 超分辨率倍数 (2, 4, 8)
    """
    return DiT(
        img_size=img_size,
        patch_size=2,
        in_channels=1,
        hidden_size=256,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        cond_channels=1,  # 条件图像通道
        **kwargs
    )


def DiT_SR_S(img_size: int = 28, scale_factor: int = 4, **kwargs):
    """
    Small DiT for Super-Resolution
    """
    return DiT(
        img_size=img_size,
        patch_size=2,
        in_channels=1,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        cond_channels=1,
        **kwargs
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DiT 模型测试")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"设备: {device}\n")

    # ---------- 测试基础模型 ----------
    print("【测试1】DiT_XS - 基础生成模型")
    print("-" * 40)

    model = DiT_XS().to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入
    x = torch.randn(2, 1, 28, 28).to(device)      # 噪声图像
    t = torch.randint(0, 1000, (2,)).to(device)   # 时间步
    y = torch.randint(0, 10, (2,)).to(device)     # 类别标签

    output = model(x, t, y)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✓ 测试通过!\n")

    # ---------- 测试超分辨率模型 ----------
    print("【测试2】DiT_SR_XS - 超分辨率模型")
    print("-" * 40)

    sr_model = DiT_SR_XS().to(device)
    print(f"参数量: {sum(p.numel() for p in sr_model.parameters()):,}")

    # 模拟输入
    x = torch.randn(2, 1, 28, 28).to(device)      # 噪声图像
    cond = torch.randn(2, 1, 28, 28).to(device)   # 条件图像（上采样后的LR）
    t = torch.randint(0, 1000, (2,)).to(device)   # 时间步
    y = torch.randint(0, 10, (2,)).to(device)     # 类别标签

    output = sr_model(x, t, y, cond=cond)
    print(f"噪声图像形状: {x.shape}")
    print(f"条件图像形状: {cond.shape}")
    print(f"输出形状: {output.shape}")
    print("✓ 测试通过!\n")

    print("=" * 60)
    print("所有测试通过!")
    print("=" * 60)
