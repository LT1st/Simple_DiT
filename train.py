"""
================================================================================
DiT 训练脚本 - MNIST数据集
================================================================================

【扩散模型训练原理】
1. 前向扩散：给干净图像逐步加噪声，直到变成纯噪声
   x_0 -> x_1 -> x_2 -> ... -> x_T (纯噪声)

2. 反向去噪：训练模型预测每一步添加的噪声
   给模型 (x_t, t)，让它预测噪声 ε

3. 训练目标：最小化预测噪声和真实噪声的MSE
   Loss = MSE(模型预测的噪声, 实际添加的噪声)

【文件说明】
- 本文件：基础生成模型训练
- train_sr.py：超分辨率模型训练

【使用方法】
python train.py                    # 使用默认参数
python train.py --epochs 50        # 训练50个epoch
python train.py --batch_size 128   # 更大的batch size
================================================================================
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dit import DiT_XS


# =============================================================================
# 第一部分：DDPM噪声调度
# =============================================================================
# DDPM (Denoising Diffusion Probabilistic Models) 的核心是控制噪声的添加

def setup_ddpm_schedule(num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    设置DDPM噪声调度（线性beta schedule）

    【核心公式】
    β_t：第t步添加的噪声强度
    α_t = 1 - β_t
    ᾱ_t = α_1 × α_2 × ... × α_t（累积乘积）

    【前向扩散】
    x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε

    其中 ε ~ N(0, I) 是标准高斯噪声

    【为什么用线性schedule？】
    - 简单有效
    - β从小到大，初期噪声少，后期噪声多
    - 这保证x_T接近纯噪声

    Args:
        num_timesteps: 扩散步数T（通常1000）
        beta_start: β的起始值
        beta_end: β的结束值

    Returns:
        dict: 包含各种调度参数
    """
    # β_t：线性从beta_start到beta_end
    betas = torch.linspace(beta_start, beta_end, num_timesteps)

    # α_t = 1 - β_t
    alphas = 1.0 - betas

    # ᾱ_t = ∏_{i=1}^{t} α_i（累积乘积）
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        'betas': betas,                              # [T] 每步的噪声强度
        'alphas': alphas,                            # [T] 1 - beta
        'alphas_cumprod': alphas_cumprod,            # [T] 累积乘积 ᾱ_t
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),              # √ᾱ_t
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod), # √(1-ᾱ_t)
    }


def q_sample(x_0: torch.Tensor, t: torch.Tensor, schedule: dict, noise: torch.Tensor = None):
    """
    前向扩散：在时间步t给干净图像x_0添加噪声

    【公式】
    x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε

    这个公式的好处：
    - 可以直接从x_0跳到任意x_t，不用一步步扩散
    - 训练时可以随机采样t，提高效率

    Args:
        x_0: [B, C, H, W] 干净图像
        t: [B] 时间步（每个样本可能不同）
        schedule: 噪声调度参数
        noise: 可选的噪声（如果不提供则随机生成）

    Returns:
        x_t: [B, C, H, W] 加噪后的图像
        noise: [B, C, H, W] 添加的噪声（用于计算损失）
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    # 获取调度参数，调整形状以便广播
    # [T] -> 索引 [B] -> [B, 1, 1, 1]
    sqrt_alpha = schedule['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1).to(x_0.device)
    sqrt_one_minus_alpha = schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1).to(x_0.device)

    # q(x_t | x_0) = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε
    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    return x_t, noise


# =============================================================================
# 第二部分：训练主函数
# =============================================================================

def train(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    num_timesteps: int = 1000,
    save_dir: str = "checkpoints",
    device: str = "cpu",
):
    """
    训练DiT模型

    【训练流程】
    for each batch:
        1. 加载干净图像 x_0
        2. 随机采样时间步 t
        3. 生成噪声 ε，计算加噪图像 x_t
        4. 模型预测噪声 ε_θ(x_t, t)
        5. 计算损失 L = MSE(ε, ε_θ)
        6. 反向传播，更新参数

    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        num_timesteps: 扩散步数
        save_dir: 模型保存目录
        device: 设备 (cpu/cuda)
    """
    # ============ 设置设备 ============
    device = torch.device(device)
    print(f"使用设备: {device}")

    # ============ 创建保存目录 ============
    os.makedirs(save_dir, exist_ok=True)

    # ============ 数据预处理 ============
    # ToTensor(): 把PIL图像转成tensor，值域[0,1]
    # Normalize((0.5,), (0.5,)): 归一化到[-1, 1]
    # 为什么归一化到[-1, 1]？
    #   - 扩散模型的噪声是对称的（高斯分布）
    #   - 输入范围对称可以让模型学习更稳定
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # ============ 加载数据集 ============
    # MNIST: 28×28 灰度手写数字，60000训练 + 10000测试
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # 打乱数据
        num_workers=0,     # 数据加载线程数（Windows建议0）
        pin_memory=False   # 是否锁页内存（GPU训练时建议True）
    )

    print(f"数据集大小: {len(train_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"每轮批次数: {len(train_loader)}")

    # ============ 创建模型 ============
    model = DiT_XS(num_classes=10).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ============ 设置噪声调度 ============
    schedule = setup_ddpm_schedule(num_timesteps)

    # ============ 优化器 ============
    # AdamW: Adam + 权重衰减（更好的正则化）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # ============ 损失函数 ============
    # MSE: 均方误差，衡量预测噪声和真实噪声的距离
    criterion = nn.MSELoss()

    # ============ 学习率调度器 ============
    # CosineAnnealing: 学习率按余弦曲线衰减
    # 训练初期学习率大，后期逐渐减小
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    # ============ 训练循环 ============
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, labels) in enumerate(pbar):
            # ---- 数据准备 ----
            images = images.to(device)  # [B, 1, 28, 28]
            labels = labels.to(device)  # [B]

            # ---- 随机采样时间步 ----
            # 每个样本随机一个t，均匀分布在[0, T)
            t = torch.randint(0, num_timesteps, (images.shape[0],), device=device)

            # ---- 前向扩散：添加噪声 ----
            noisy_images, noise = q_sample(images, t, schedule)

            # ---- 模型预测噪声 ----
            # 输入：加噪图像、时间步、类别标签
            model_output = model(noisy_images, t, labels)

            # ---- 计算损失 ----
            loss = criterion(model_output, noise)

            # ---- 反向传播 ----
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

            # ---- 记录 ----
            epoch_loss += loss.item()
            global_step += 1

            # 更新进度条
            pbar.set_postfix(loss=loss.item())

        # ---- 更新学习率 ----
        scheduler.step()

        # ---- 打印epoch统计 ----
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.6f}")

        # ---- 保存checkpoint ----
        # 每5个epoch和最后一个epoch保存
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f"dit_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'schedule': schedule,  # 保存调度参数，采样时要用
            }, checkpoint_path)
            print(f"保存checkpoint: {checkpoint_path}")

    # ============ 保存最终模型 ============
    final_path = os.path.join(save_dir, "dit_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'schedule': schedule,
    }, final_path)
    print(f"保存最终模型: {final_path}")

    print("训练完成!")


# =============================================================================
# 第三部分：命令行入口
# =============================================================================

if __name__ == "__main__":
    train(
        epochs=20,
        batch_size=64,
        learning_rate=1e-4,
        num_timesteps=1000,
        device="cpu"
    )
