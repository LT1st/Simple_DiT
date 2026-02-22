"""
DiT采样脚本 - DDPM反向扩散采样
"""

import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dit import DiT_XS


def p_sample(model, x: torch.Tensor, t: int, y: torch.Tensor, schedule: dict):
    """
    单步反向扩散采样
    从x_t预测x_{t-1}
    """
    device = x.device
    t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

    # 模型预测噪声
    predicted_noise = model(x, t_tensor, y)

    # 获取调度参数
    beta = schedule['betas'][t]
    alpha = schedule['alphas'][t]
    alpha_cumprod = schedule['alphas_cumprod'][t]

    # 计算去噪后的图像
    # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t / sqrt(1-alpha_cumprod_t) * predicted_noise)
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


def ddpm_sample(model, shape, y: torch.Tensor, schedule: dict, num_timesteps: int = 1000, device: str = "cpu"):
    """
    完整的DDPM采样过程
    """
    model.eval()
    device = torch.device(device)

    # 从纯噪声开始
    x = torch.randn(shape, device=device)

    # 反向扩散
    for t in tqdm(reversed(range(num_timesteps)), desc="Sampling", total=num_timesteps):
        with torch.no_grad():
            x = p_sample(model, x, t, y, schedule)

    return x


@torch.no_grad()
def sample_images(
    checkpoint_path: str = "checkpoints/dit_final.pt",
    num_samples: int = 16,
    class_label: int = None,
    num_timesteps: int = 1000,
    output_dir: str = "samples",
    device: str = "cpu",
):
    """
    从训练好的模型采样图像
    """
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DiT_XS(num_classes=10).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取噪声调度
    if 'schedule' in checkpoint:
        schedule = checkpoint['schedule']
    else:
        # 如果没有保存schedule，重新创建
        from train import setup_ddpm_schedule
        schedule = setup_ddpm_schedule(num_timesteps)

    # 设置类别标签
    if class_label is not None:
        # 生成指定类别的图像
        y = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
        label_str = f"class_{class_label}"
    else:
        # 生成所有类别的图像
        y = torch.arange(10, device=device).repeat(num_samples // 10 + 1)[:num_samples]
        label_str = "mixed"

    # 采样
    print(f"Generating {num_samples} images...")
    shape = (num_samples, 1, 28, 28)
    samples = ddpm_sample(model, shape, y, schedule, num_timesteps, device)

    # 后处理：从[-1, 1]转换到[0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    # 保存图像
    output_path = os.path.join(output_dir, f"samples_{label_str}.png")
    save_image(samples, output_path, nrow=4, normalize=False)
    print(f"Saved samples to {output_path}")

    # 也保存单独的图像
    for i in range(num_samples):
        single_path = os.path.join(output_dir, f"sample_{i}_label_{y[i].item()}.png")
        save_image(samples[i], single_path)

    return samples


def sample_all_classes(
    checkpoint_path: str = "checkpoints/dit_final.pt",
    samples_per_class: int = 4,
    num_timesteps: int = 1000,
    output_dir: str = "samples",
    device: str = "cpu",
):
    """
    为每个类别生成样本
    """
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DiT_XS(num_classes=10).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取噪声调度
    if 'schedule' in checkpoint:
        schedule = checkpoint['schedule']
    else:
        from train import setup_ddpm_schedule
        schedule = setup_ddpm_schedule(num_timesteps)

    all_samples = []

    for class_label in range(10):
        print(f"Generating samples for class {class_label}...")
        y = torch.full((samples_per_class,), class_label, dtype=torch.long, device=device)
        shape = (samples_per_class, 1, 28, 28)
        samples = ddpm_sample(model, shape, y, schedule, num_timesteps, device)
        all_samples.append(samples)

    # 合并所有样本
    all_samples = torch.cat(all_samples, dim=0)
    all_samples = (all_samples + 1) / 2
    all_samples = all_samples.clamp(0, 1)

    # 保存网格图
    output_path = os.path.join(output_dir, "all_classes.png")
    save_image(all_samples, output_path, nrow=samples_per_class, normalize=False)
    print(f"Saved all class samples to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample from trained DiT model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dit_final.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of samples to generate")
    parser.add_argument("--class_label", type=int, default=None,
                        help="Specific class label to generate (0-9), None for mixed")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--output_dir", type=str, default="samples",
                        help="Output directory for samples")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--all_classes", action="store_true",
                        help="Generate samples for all classes")

    args = parser.parse_args()

    if args.all_classes:
        sample_all_classes(
            checkpoint_path=args.checkpoint,
            samples_per_class=4,
            num_timesteps=args.num_timesteps,
            output_dir=args.output_dir,
            device=args.device
        )
    else:
        sample_images(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            class_label=args.class_label,
            num_timesteps=args.num_timesteps,
            output_dir=args.output_dir,
            device=args.device
        )
