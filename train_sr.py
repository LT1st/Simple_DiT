"""
================================================================================
DiT 超分辨率训练脚本
================================================================================

【超分辨率任务】
输入：低分辨率(LR)图像
输出：高分辨率(HR)图像

【条件扩散模型】
与普通生成模型不同，条件扩散模型需要额外输入条件信息。
这里我们把低分辨率图像作为条件，指导模型生成高分辨率图像。

【训练流程】
1. 加载高分辨率图像 x_hr
2. 下采样得到低分辨率图像 x_lr
3. 上采样回原尺寸作为条件 cond
4. 给 x_hr 加噪声得到 x_t
5. 模型预测噪声：ε_θ(x_t, t, cond)
6. 计算损失

【使用方法】
python train_sr.py --scale 4     # 4x超分辨率
python train_sr.py --scale 2     # 2x超分辨率
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dit import DiT_SR_XS


# =============================================================================
# 第一部分：DDPM噪声调度（与train.py相同）
# =============================================================================

def setup_ddpm_schedule(num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
    """设置DDPM噪声调度"""
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod),
    }


def q_sample(x_0: torch.Tensor, t: torch.Tensor, schedule: dict, noise: torch.Tensor = None):
    """前向扩散：在时间步t添加噪声"""
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha = schedule['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1).to(x_0.device)
    sqrt_one_minus_alpha = schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1).to(x_0.device)

    return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise


# =============================================================================
# 第二部分：低分辨率图像生成
# =============================================================================

def create_lr_image(hr_image: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
    """
    创建低分辨率图像

    【流程】
    HR图像 -> 下采样 -> LR图像 -> 上采样 -> cond（条件图像）

    为什么不直接用LR图像作为条件？
    - 模型输入需要和输出同样尺寸
    - 上采样后的图像保留了LR的特征，但尺寸匹配

    Args:
        hr_image: [B, C, H, W] 高分辨率图像
        scale_factor: 下采样倍数

    Returns:
        lr_upsampled: [B, C, H, W] 上采样后的低分辨率图像
    """
    b, c, h, w = hr_image.shape
    lr_h, lr_w = h // scale_factor, w // scale_factor

    # 下采样（使用area插值，抗混叠效果好）
    lr_image = F.interpolate(hr_image, size=(lr_h, lr_w), mode='area')

    # 上采样回原尺寸（使用双线性插值）
    lr_upsampled = F.interpolate(lr_image, size=(h, w), mode='bilinear', align_corners=False)

    return lr_upsampled


# =============================================================================
# 第三部分：超分辨率数据集
# =============================================================================

class SRDataset(torch.utils.data.Dataset):
    """
    超分辨率数据集包装器

    【作用】
    把普通图像数据集转换成(HR, LR, label)三元组
    """
    def __init__(self, dataset, scale_factor: int = 4):
        self.dataset = dataset
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr_image, label = self.dataset[idx]
        # 动态生成LR图像
        lr_image = create_lr_image(hr_image.unsqueeze(0), self.scale_factor).squeeze(0)
        return hr_image, lr_image, label


# =============================================================================
# 第四部分：训练主函数
# =============================================================================

def train(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    num_timesteps: int = 1000,
    scale_factor: int = 4,
    img_size: int = 28,
    save_dir: str = "checkpoints_sr",
    device: str = "cpu",
):
    """
    训练超分辨率DiT模型

    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        num_timesteps: 扩散步数
        scale_factor: 超分辨率倍数 (2, 4, 8)
        img_size: 高分辨率图像大小
        save_dir: 模型保存目录
        device: 设备
    """
    # ============ 设置 ============
    device = torch.device(device)
    print(f"使用设备: {device}")
    print(f"超分辨率倍数: {scale_factor}x")

    os.makedirs(save_dir, exist_ok=True)

    # ============ 数据集 ============
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    base_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 包装为超分辨率数据集
    train_dataset = SRDataset(base_dataset, scale_factor=scale_factor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    print(f"数据集大小: {len(train_dataset)}")
    print(f"批次大小: {batch_size}")

    # ============ 模型 ============
    model = DiT_SR_XS(img_size=img_size, scale_factor=scale_factor).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    schedule = setup_ddpm_schedule(num_timesteps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ============ 训练循环 ============
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (hr_images, lr_images, labels) in enumerate(pbar):
            hr_images = hr_images.to(device)   # 高分辨率图像（目标）
            lr_images = lr_images.to(device)   # 低分辨率图像（条件）
            labels = labels.to(device)

            # 随机时间步
            t = torch.randint(0, num_timesteps, (hr_images.shape[0],), device=device)

            # 给HR图像加噪声
            noisy_images, noise = q_sample(hr_images, t, schedule)

            # 模型预测噪声（使用LR图像作为条件）
            model_output = model(noisy_images, t, labels, cond=lr_images)

            # 计算损失
            loss = criterion(model_output, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.6f}")

        # 保存checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f"dit_sr_{scale_factor}x_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'schedule': schedule,
                'scale_factor': scale_factor,
                'img_size': img_size,
            }, checkpoint_path)
            print(f"保存checkpoint: {checkpoint_path}")

    # 保存最终模型
    final_path = os.path.join(save_dir, f"dit_sr_{scale_factor}x_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'schedule': schedule,
        'scale_factor': scale_factor,
        'img_size': img_size,
    }, final_path)
    print(f"保存最终模型: {final_path}")

    print("训练完成!")


# =============================================================================
# 第五部分：命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练DiT超分辨率模型")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--scale", type=int, default=4, help="超分辨率倍数 (2, 4, 8)")
    parser.add_argument("--img_size", type=int, default=28, help="高分辨率图像大小")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        scale_factor=args.scale,
        img_size=args.img_size,
        device=args.device
    )
