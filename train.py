"""
DiT训练脚本 - MNIST数据集
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dit import DiT_XS


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
    q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha = schedule['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1).to(x_0.device)
    sqrt_one_minus_alpha = schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1).to(x_0.device)

    return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise


def train(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    num_timesteps: int = 1000,
    save_dir: str = "checkpoints",
    device: str = "cpu",
):
    # 设置设备
    device = torch.device(device)
    print(f"Using device: {device}")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 归一化到[-1, 1]
    ])

    # 加载MNIST数据集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
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
    model = DiT_XS(num_classes=10).to(device)
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

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # 随机采样时间步
            t = torch.randint(0, num_timesteps, (images.shape[0],), device=device)

            # 前向扩散：添加噪声
            noisy_images, noise = q_sample(images, t, schedule)

            # 模型预测噪声
            model_output = model(noisy_images, t, labels)

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
            checkpoint_path = os.path.join(save_dir, f"dit_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'schedule': schedule,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # 保存最终模型
    final_path = os.path.join(save_dir, "dit_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'schedule': schedule,
    }, final_path)
    print(f"Saved final model: {final_path}")

    print("Training completed!")


if __name__ == "__main__":
    train(
        epochs=20,
        batch_size=64,
        learning_rate=1e-4,
        num_timesteps=1000,
        device="cpu"
    )
