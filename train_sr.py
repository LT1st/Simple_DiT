"""
DiT超分辨率训练脚本 - MNIST数据集
支持2x、4x等倍数的超分辨率
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dit import DiT_SR_XS


def setup_ddpm_schedule(num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    设置DDPM噪声调度（线性beta schedule）
    """
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
    """
    前向扩散过程：在时间步t添加噪声
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha = schedule['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1).to(x_0.device)
    sqrt_one_minus_alpha = schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1).to(x_0.device)

    return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise


def create_lr_image(hr_image: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
    """
    创建低分辨率图像
    hr_image: [B, C, H, W] 高分辨率图像
    返回: [B, C, H, W] 下采样后上采样回原尺寸的图像
    """
    b, c, h, w = hr_image.shape
    lr_h, lr_w = h // scale_factor, w // scale_factor

    # 下采样
    lr_image = F.interpolate(hr_image, size=(lr_h, lr_w), mode='area')

    # 上采样回原尺寸（作为条件）
    lr_upsampled = F.interpolate(lr_image, size=(h, w), mode='bilinear', align_corners=False)

    return lr_upsampled


class SRDataset(torch.utils.data.Dataset):
    """
    超分辨率数据集包装器
    """
    def __init__(self, dataset, scale_factor: int = 4):
        self.dataset = dataset
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hr_image, label = self.dataset[idx]
        lr_image = create_lr_image(hr_image.unsqueeze(0), self.scale_factor).squeeze(0)
        return hr_image, lr_image, label


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
    # 设置设备
    device = torch.device(device)
    print(f"Using device: {device}")
    print(f"Super-resolution scale: {scale_factor}x")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 归一化到[-1, 1]
    ])

    # 加载MNIST数据集
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

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(train_loader)}")

    # 创建模型
    model = DiT_SR_XS(img_size=img_size, scale_factor=scale_factor).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 设置噪声调度
    schedule = setup_ddpm_schedule(num_timesteps)

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (hr_images, lr_images, labels) in enumerate(pbar):
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)  # 条件图像（上采样后的LR）
            labels = labels.to(device)

            # 随机采样时间步
            t = torch.randint(0, num_timesteps, (hr_images.shape[0],), device=device)

            # 前向扩散：给HR图像添加噪声
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
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

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
            print(f"Saved checkpoint: {checkpoint_path}")

    # 保存最终模型
    final_path = os.path.join(save_dir, f"dit_sr_{scale_factor}x_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'schedule': schedule,
        'scale_factor': scale_factor,
        'img_size': img_size,
    }, final_path)
    print(f"Saved final model: {final_path}")

    print("Training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DiT for Super-Resolution")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor (2, 4, 8)")
    parser.add_argument("--img_size", type=int, default=28, help="High-resolution image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        scale_factor=args.scale,
        img_size=args.img_size,
        device=args.device
    )
