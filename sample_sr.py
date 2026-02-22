"""
================================================================================
DiT 超分辨率采样脚本
================================================================================

【超分辨率采样流程】
1. 加载低分辨率(LR)图像
2. 上采样到目标尺寸作为条件cond
3. 从纯噪声开始，逐步去噪
4. 每步去噪时，把cond输入模型

【使用方法】
# 单张图像超分辨率
python sample_sr.py --lr_image input.png --checkpoint checkpoints_sr/dit_sr_4x_final.pt

# 批量测试（从MNIST测试集）
python sample_sr.py --batch --checkpoint checkpoints_sr/dit_sr_4x_final.pt
================================================================================
"""

import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dit import DiT_SR_XS


# =============================================================================
# 第一部分：单步去噪（带条件）
# =============================================================================

def p_sample(model, x: torch.Tensor, t: int, cond: torch.Tensor, schedule: dict):
    """
    单步反向扩散（带条件图像）

    与普通采样的区别：模型额外接收条件图像cond

    Args:
        model: 超分辨率模型
        x: [B, C, H, W] 当前噪声图像
        t: 当前时间步
        cond: [B, C, H, W] 条件图像（上采样后的LR）
        schedule: 噪声调度

    Returns:
        x_prev: [B, C, H, W] 去噪后的图像
    """
    device = x.device
    t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

    # 类别标签对超分辨率不重要，设为0
    y = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    # 模型预测噪声（使用条件）
    predicted_noise = model(x, t_tensor, y, cond=cond)

    # 获取调度参数
    beta = schedule['betas'][t]
    alpha = schedule['alphas'][t]
    alpha_cumprod = schedule['alphas_cumprod'][t]

    # 计算均值
    sqrt_alpha = torch.sqrt(alpha)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
    mean = (1 / sqrt_alpha) * (x - beta / sqrt_one_minus_alpha_cumprod * predicted_noise)

    # 添加噪声
    if t > 0:
        noise = torch.randn_like(x)
        if t > 1:
            alpha_cumprod_prev = schedule['alphas_cumprod'][t - 1]
            posterior_variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
        else:
            posterior_variance = torch.tensor(0.0)
        std = torch.sqrt(posterior_variance)
        x_prev = mean + std * noise
    else:
        x_prev = mean

    return x_prev


def ddpm_sample(model, shape, cond: torch.Tensor, schedule: dict, num_timesteps: int = 1000, device: str = "cpu"):
    """
    完整的DDPM采样过程（带条件）

    Args:
        model: 超分辨率模型
        shape: 输出形状 (B, C, H, W)
        cond: 条件图像 [B, C, H, W]
        schedule: 噪声调度
        num_timesteps: 采样步数
        device: 设备

    Returns:
        sr_image: 超分辨率结果
    """
    model.eval()
    device = torch.device(device)

    # 从纯噪声开始
    x = torch.randn(shape, device=device)

    # 反向扩散
    for t in tqdm(reversed(range(num_timesteps)), desc="超分辨率采样中", total=num_timesteps):
        with torch.no_grad():
            x = p_sample(model, x, t, cond, schedule)

    return x


# =============================================================================
# 第二部分：条件图像准备
# =============================================================================

def prepare_condition(lr_image: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    准备条件图像：将LR图像上采样到目标尺寸

    Args:
        lr_image: [B, C, H, W] 或 [C, H, W] 低分辨率图像
        target_size: 目标尺寸

    Returns:
        cond: [B, C, target_size, target_size] 上采样后的条件图像
    """
    if lr_image.dim() == 3:
        lr_image = lr_image.unsqueeze(0)

    cond = F.interpolate(lr_image, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return cond


# =============================================================================
# 第三部分：单图超分辨率
# =============================================================================

@torch.no_grad()
def super_resolve(
    checkpoint_path: str = "checkpoints_sr/dit_sr_4x_final.pt",
    lr_image_path: str = None,
    output_dir: str = "outputs_sr",
    num_timesteps: int = 1000,
    device: str = "cpu",
):
    """
    对单张图像进行超分辨率

    Args:
        checkpoint_path: 模型路径
        lr_image_path: 低分辨率图像路径（None则生成随机测试图）
        output_dir: 输出目录
        num_timesteps: 采样步数
        device: 设备
    """
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # ============ 加载模型 ============
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    scale_factor = checkpoint.get('scale_factor', 4)
    img_size = checkpoint.get('img_size', 28)

    model = DiT_SR_XS(img_size=img_size, scale_factor=scale_factor).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    schedule = checkpoint['schedule']

    # ============ 加载/生成LR图像 ============
    if lr_image_path is not None:
        print(f"加载图像: {lr_image_path}")
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size // scale_factor, img_size // scale_factor)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        lr_image = transform(Image.open(lr_image_path)).unsqueeze(0).to(device)
    else:
        print("生成随机测试图像...")
        lr_image = torch.randn(1, 1, img_size // scale_factor, img_size // scale_factor, device=device)

    # ============ 准备条件 ============
    cond = prepare_condition(lr_image, img_size)

    print(f"LR图像尺寸: {lr_image.shape}")
    print(f"条件图像尺寸: {cond.shape}")
    print(f"目标HR尺寸: {img_size}x{img_size}")

    # ============ 超分辨率采样 ============
    print("正在超分辨率...")
    shape = (1, 1, img_size, img_size)
    sr_image = ddpm_sample(model, shape, cond, schedule, num_timesteps, device)

    # ============ 后处理 ============
    sr_image = (sr_image + 1) / 2
    sr_image = sr_image.clamp(0, 1)
    cond_vis = (cond + 1) / 2
    cond_vis = cond_vis.clamp(0, 1)

    # ============ 保存结果 ============
    save_image(cond_vis, os.path.join(output_dir, "lr_upsampled.png"))
    save_image(sr_image, os.path.join(output_dir, "sr_result.png"))

    # 拼接对比
    comparison = torch.cat([cond_vis, sr_image], dim=3)
    save_image(comparison, os.path.join(output_dir, "comparison.png"))

    print(f"结果保存到: {output_dir}")

    return sr_image


# =============================================================================
# 第四部分：批量测试
# =============================================================================

@torch.no_grad()
def super_resolve_batch(
    checkpoint_path: str = "checkpoints_sr/dit_sr_4x_final.pt",
    num_samples: int = 16,
    output_dir: str = "outputs_sr",
    num_timesteps: int = 1000,
    device: str = "cpu",
):
    """
    批量超分辨率测试（使用MNIST测试集）

    生成对比图：LR上采样 | SR结果 | 原始HR
    """
    from torchvision import datasets

    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    scale_factor = checkpoint.get('scale_factor', 4)
    img_size = checkpoint.get('img_size', 28)

    model = DiT_SR_XS(img_size=img_size, scale_factor=scale_factor).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    schedule = checkpoint['schedule']

    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    hr_images = []
    lr_images = []
    sr_images = []

    for i in tqdm(range(num_samples), desc="批量超分辨率"):
        hr_image, _ = test_dataset[i]
        hr_image = hr_image.unsqueeze(0).to(device)

        # 创建LR
        lr_size = img_size // scale_factor
        lr = F.interpolate(hr_image, size=(lr_size, lr_size), mode='area')
        cond = F.interpolate(lr, size=(img_size, img_size), mode='bilinear', align_corners=False)

        # SR采样
        shape = (1, 1, img_size, img_size)
        sr = ddpm_sample(model, shape, cond, schedule, num_timesteps, device)

        hr_images.append((hr_image + 1) / 2)
        lr_images.append((cond + 1) / 2)
        sr_images.append((sr + 1) / 2)

    # 合并保存
    hr_all = torch.cat(hr_images, dim=0).clamp(0, 1)
    lr_all = torch.cat(lr_images, dim=0).clamp(0, 1)
    sr_all = torch.cat(sr_images, dim=0).clamp(0, 1)

    save_image(lr_all, os.path.join(output_dir, "batch_lr.png"), nrow=4)
    save_image(sr_all, os.path.join(output_dir, "batch_sr.png"), nrow=4)
    save_image(hr_all, os.path.join(output_dir, "batch_hr.png"), nrow=4)

    # 三者对比（每行一张图的三种版本）
    comparison = torch.cat([lr_all, sr_all, hr_all], dim=0)
    save_image(comparison, os.path.join(output_dir, "batch_comparison.png"), nrow=4)

    print(f"批量结果保存到: {output_dir}")


# =============================================================================
# 第五部分：命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DiT超分辨率")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_sr/dit_sr_4x_final.pt",
                        help="模型checkpoint路径")
    parser.add_argument("--lr_image", type=str, default=None,
                        help="低分辨率输入图像路径")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="批量测试时的样本数")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="采样步数")
    parser.add_argument("--output_dir", type=str, default="outputs_sr",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cpu",
                        help="设备 (cpu/cuda)")
    parser.add_argument("--batch", action="store_true",
                        help="批量测试模式")

    args = parser.parse_args()

    if args.batch:
        super_resolve_batch(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            num_timesteps=args.num_timesteps,
            device=args.device
        )
    else:
        super_resolve(
            checkpoint_path=args.checkpoint,
            lr_image_path=args.lr_image,
            output_dir=args.output_dir,
            num_timesteps=args.num_timesteps,
            device=args.device
        )
