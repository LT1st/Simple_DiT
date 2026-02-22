"""
DiT (Diffusion Transformer) - 基于PyTorch的简化实现
参考: "Scalable Diffusion Models with Transformers" by Peebles & Xie, 2023
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> torch.Tensor:
    """
    生成2D正弦-余弦位置编码

    Args:
        embed_dim: 嵌入维度
        grid_size: 网格大小（假设正方形）
        cls_token: 是否包含cls token

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] 或 [1+grid_size*grid_size, embed_dim]
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)  # [2, grid_size, grid_size]
    grid = grid.reshape(2, 1, grid_size, grid_size)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """
    从网格生成2D位置编码
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = torch.cat([emb_h, emb_w], dim=1)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    从1D网格生成正弦-余弦位置编码
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / 10000 ** (omega / (embed_dim / 2))

    pos = pos.reshape(-1)
    out = torch.outer(pos, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb


class TimestepEmbedder(nn.Module):
    """
    时间步嵌入：将标量时间步嵌入到向量空间
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    创建正弦时间步嵌入
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LabelEmbedder(nn.Module):
    """
    类别标签嵌入
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels: torch.Tensor, train: bool = True) -> torch.Tensor:
        if train and self.dropout_prob > 0:
            # Classifier-free guidance: 随机dropout标签
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.num_classes, labels)
        return self.embedding_table(labels)


class PatchEmbed(nn.Module):
    """
    图像到Patch嵌入
    """
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 2,
        in_chans: int = 1,
        embed_dim: int = 256,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, C, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


class adaLNModulation(nn.Module):
    """
    自适应层归一化调制
    """
    def __init__(self, hidden_size: int, out_features: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, out_features),
        )

    def forward(self, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.modulation(c) * x


class DiTBlock(nn.Module):
    """
    DiT Transformer块，使用adaLN-Zero调制
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # adaLN调制参数：每个norm输出6个参数 (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # 计算调制参数
        modulation = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        # 自注意力
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class FinalLayer(nn.Module):
    """
    最终输出层
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation = self.adaLN_modulation(c)
        shift, scale = modulation.chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer主模型
    """
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 2,
        in_channels: int = 1,
        hidden_size: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        learn_sigma: bool = False,
        cond_channels: int = 0,  # 条件图像通道数（用于超分辨率等任务）
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.total_in_channels = in_channels + cond_channels  # 总输入通道
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        self.learn_sigma = learn_sigma

        # Patch嵌入（支持条件输入）
        self.x_embedder = PatchEmbed(img_size, patch_size, self.total_in_channels, hidden_size)

        # 时间步和标签嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # 位置编码（可学习）
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        # Transformer块
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 最终输出层
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化位置编码
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            self.img_size // self.patch_size
        )
        self.pos_embedding.data.copy_(pos_embed.unsqueeze(0))

        # Zero-init 最终层的bias和adaLN参数
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero-init 每个block的adaLN
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将patch序列转换回图像
        x: [B, N, P*P*C]
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, H, P, W, P]
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        x: [B, C, H, W] 噪声图像
        t: [B] 时间步
        y: [B] 类别标签
        cond: [B, cond_channels, H, W] 条件图像（可选，用于超分辨率等任务）
        """
        # 如果有条件输入，拼接到输入通道
        if cond is not None:
            x = torch.cat([x, cond], dim=1)  # [B, C+cond_channels, H, W]

        # Patch嵌入 + 位置编码
        x = self.x_embedder(x) + self.pos_embedding

        # 时间步和类别嵌入
        c = self.t_embedder(t) + self.y_embedder(y, self.training)

        # Transformer块
        for block in self.blocks:
            x = block(x, c)

        # 输出
        x = self.final_layer(x, c)

        # 转回图像格式
        x = self.unpatchify(x)

        return x


def DiT_XS(**kwargs):
    """Extra Small DiT for MNIST"""
    return DiT(img_size=28, patch_size=2, in_channels=1, hidden_size=256,
               depth=4, num_heads=4, mlp_ratio=4.0, **kwargs)


def DiT_S(**kwargs):
    """Small DiT"""
    return DiT(img_size=28, patch_size=2, in_channels=1, hidden_size=384,
               depth=6, num_heads=6, mlp_ratio=4.0, **kwargs)


def DiT_SR_XS(img_size: int = 28, scale_factor: int = 4, **kwargs):
    """
    Extra Small DiT for Super-Resolution
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


if __name__ == "__main__":
    # 简单测试
    device = torch.device("cpu")

    # 测试基础DiT模型
    print("=" * 50)
    print("Testing DiT_XS (base model)")
    print("=" * 50)
    model = DiT_XS().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 1, 28, 28).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    y = torch.randint(0, 10, (2,)).to(device)

    output = model(x, t, y)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("DiT model test passed!")

    # 测试超分辨率模型
    print("\n" + "=" * 50)
    print("Testing DiT_SR_XS (super-resolution model)")
    print("=" * 50)
    sr_model = DiT_SR_XS().to(device)
    print(f"SR Model parameters: {sum(p.numel() for p in sr_model.parameters()):,}")

    x = torch.randn(2, 1, 28, 28).to(device)  # 噪声图像
    cond = torch.randn(2, 1, 28, 28).to(device)  # 上采样后的LR条件图像
    t = torch.randint(0, 1000, (2,)).to(device)
    y = torch.randint(0, 10, (2,)).to(device)

    output = sr_model(x, t, y, cond=cond)
    print(f"Noisy input shape: {x.shape}")
    print(f"Condition shape: {cond.shape}")
    print(f"Output shape: {output.shape}")
    print("DiT_SR model test passed!")
