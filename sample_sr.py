"""
DiT超分辨率采样脚本
使用训练好的模型进行图像超分辨率
"""

import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dit import DiT_SR_XS


def p_sample(model, x: torch.Tensor, t: int, cond: torch.Tensor, schedule: dict):
    """
    单步反向扩散采样（带条件）
    """
    device = x.device
    t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
    y = torch.zeros(x.shape[0], dtype=torch.long, device=device)  # 类别标签不重要

    # 模型预测噪声（使用条件）
    predicted_noise = model(x, t_tensor, y, cond=cond)

    # 获取调度参数
    beta = schedule['betas'][t]
    alpha = schedule['alphas'][t]
    alpha_cumprod = schedule['alphas_cumprod'][t]

    # 计算去噪后的图像
    sqrt_alpha = torch.sqrt(alpha)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

    # 计算均值
    mean = (1 / sqrt_alpha) * (x - beta / sqrt_one_minus_alpha_cumprod * predicted_noise)

    # 添加噪声（除了t=0）
    if t > 0:
        noise = torch.randn_like(x)
        # 后验方差
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
    """
    model.eval()
    device = torch.device(device)

    # 从纯噪声开始
    x = torch.randn(shape, device=device)

    # 反向扩散
    for t in tqdm(reversed(range(num_timesteps)), desc="Sampling", total=num_timesteps):
        with torch.no_grad():
            x = p_sample(model, x, t, cond, schedule)

    return x


def prepare_condition(lr_image: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    准备条件图像：将LR图像上采样到目标尺寸
    """
    if lr_image.dim() == 3:
        lr_image = lr_image.unsqueeze(0)

    # 上采样到目标尺寸
    cond = F.interpolate(lr_image, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return cond


@torch.no_grad()
def super_resolve(
    checkpoint_path: str = "checkpoints_sr/dit_sr_4x_final.pt",
    lr_image_path: str = None,
    output_dir: str = "outputs_sr",
    num_timesteps: int = 1000,
    device: str = "cpu",
):
    """
    使用训练好的模型进行超分辨率
    """
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    scale_factor = checkpoint.get('scale_factor', 4)
    img_size = checkpoint.get('img_size', 28)

    model = DiT_SR_XS(img_size=img_size, scale_factor=scale_factor).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    schedule = checkpoint['schedule']

    # 加载LR图像
    if lr_image_path is not None:
        # 从文件加载
        print(f"Loading LR image from {lr_image_path}")
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size // scale_factor, img_size // scale_factor)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        lr_image = transform(Image.open(lr_image_path)).unsqueeze(0).to(device)
    else:
        # 生成随机测试图像
        print("Generating random test LR image...")
        lr_image = torch.randn(1, 1, img_size // scale_factor, img_size // scale_factor, device=device)

    # 准备条件图像
    cond = prepare_condition(lr_image, img_size)

    print(f"LR image size: {lr_image.shape}")
    print(f"Condition size: {cond.shape}")
    print(f"Target HR size: {img_size}x{img_size}")

    # 采样
    print("Super-resolving...")
    shape = (1, 1, img_size, img_size)
    sr_image = ddpm_sample(model, shape, cond, schedule, num_timesteps, device)

    # 后处理：从[-1, 1]转换到[0, 1]
    sr_image = (sr_image + 1) / 2
    sr_image = sr_image.clamp(0, 1)
    cond_vis = (cond + 1) / 2
    cond_vis = cond_vis.clamp(0, 1)

    # 保存结果
    # 上采样的LR图像（用于对比）
    save_image(cond_vis, os.path.join(output_dir, "lr_upsampled.png"))

    # SR结果
    save_image(sr_image, os.path.join(output_dir, "sr_result.png"))

    # 拼接对比
    comparison = torch.cat([cond_vis, sr_image], dim=3)
    save_image(comparison, os.path.join(output_dir, "comparison.png"))

    print(f"Saved results to {output_dir}")

    return sr_image


@torch.no_grad()
def super_resolve_batch(
    checkpoint_path: str = "checkpoints_sr/dit_sr_4x_final.pt",
    num_samples: int = 16,
    output_dir: str = "outputs_sr",
    num_timesteps: int = 1000,
    device: str = "cpu",
):
    """
    批量生成超分辨率结果（用于测试）
    从MNIST测试集加载真实图像
    """
    from torchvision import datasets

    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"Loading model from {checkpoint_path}")
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

    # 采样测试
    hr_images = []
    lr_images = []
    sr_images = []

    for i in tqdm(range(num_samples), desc="Super-resolving"):
        hr_image, _ = test_dataset[i]
        hr_image = hr_image.unsqueeze(0).to(device)

        # 创建LR图像
        lr_size = img_size // scale_factor
        lr = F.interpolate(hr_image, size=(lr_size, lr_size), mode='area')
        cond = F.interpolate(lr, size=(img_size, img_size), mode='bilinear', align_corners=False)

        # SR采样
        shape = (1, 1, img_size, img_size)
        sr = ddpm_sample(model, shape, cond, schedule, num_timesteps, device)

        hr_images.append((hr_image + 1) / 2)
        lr_images.append((cond + 1) / 2)
        sr_images.append((sr + 1) / 2)

    # 合并并保存
    hr_all = torch.cat(hr_images, dim=0).clamp(0, 1)
    lr_all = torch.cat(lr_images, dim=0).clamp(0, 1)
    sr_all = torch.cat(sr_images, dim=0).clamp(0, 1)

    save_image(lr_all, os.path.join(output_dir, "batch_lr.png"), nrow=4)
    save_image(sr_all, os.path.join(output_dir, "batch_sr.png"), nrow=4)
    save_image(hr_all, os.path.join(output_dir, "batch_hr.png"), nrow=4)

    # 三者对比
    comparison = torch.cat([lr_all, sr_all, hr_all], dim=0)
    save_image(comparison, os.path.join(output_dir, "batch_comparison.png"), nrow=4)

    print(f"Saved batch results to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Super-Resolution with DiT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_sr/dit_sr_4x_final.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--lr_image", type=str, default=None,
                        help="Path to low-resolution input image")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of samples for batch testing")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--output_dir", type=str, default="outputs_sr",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--batch", action="store_true",
                        help="Run batch evaluation on test set")

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
