"""
================================================================================
DiT 采样脚本 - 图像生成
================================================================================

【扩散模型采样原理】
训练好的模型可以预测噪声。采样时，从纯噪声开始，逐步去噪：

x_T (纯噪声) -> x_{T-1} -> x_{T-2} -> ... -> x_0 (生成图像)

每一步：
1. 模型预测噪声 ε_θ(x_t, t)
2. 用预测的噪声估计 x_{t-1}
3. 加入少量随机噪声（除了最后一步）

【DDPM采样公式】
x_{t-1} = (1/√α_t) × (x_t - (β_t/√(1-ᾱ_t)) × ε_θ(x_t, t)) + σ_t × z

其中 z ~ N(0, I) 是随机噪声

【使用方法】
python sample.py                                    # 默认生成16张图
python sample.py --class_label 5                    # 只生成数字5
python sample.py --num_samples 32                   # 生成32张图
python sample.py --checkpoint my_model.pt           # 使用指定模型
================================================================================
"""

import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dit import DiT_XS


# =============================================================================
# 第一部分：单步去噪
# =============================================================================

def p_sample(model, x: torch.Tensor, t: int, y: torch.Tensor, schedule: dict):
    """
    单步反向扩散：从x_t预测x_{t-1}

    【DDPM公式】
    均值: μ = (1/√α_t) × (x_t - (β_t/√(1-ᾱ_t)) × ε_θ)
    方差: σ² = β_t × (1-ᾱ_{t-1}) / (1-ᾱ_t)

    x_{t-1} = μ + σ × z,  其中 z ~ N(0, I)

    Args:
        model: 训练好的DiT模型
        x: [B, C, H, W] 当前时间步的图像x_t
        t: 当前时间步（整数）
        y: [B] 类别标签
        schedule: 噪声调度参数

    Returns:
        x_prev: [B, C, H, W] 前一时间步的图像x_{t-1}
    """
    device = x.device

    # 创建时间步tensor
    t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

    # 模型预测噪声
    predicted_noise = model(x, t_tensor, y)

    # 获取调度参数
    beta = schedule['betas'][t]
    alpha = schedule['alphas'][t]
    alpha_cumprod = schedule['alphas_cumprod'][t]

    # 计算去噪后的均值
    # μ = (1/√α_t) × (x_t - (β_t/√(1-ᾱ_t)) × ε_θ)
    sqrt_alpha = torch.sqrt(alpha)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

    mean = (1 / sqrt_alpha) * (x - beta / sqrt_one_minus_alpha_cumprod * predicted_noise)

    # 添加噪声（除了t=0）
    if t > 0:
        noise = torch.randn_like(x)

        # 计算后验方差
        # σ² = β_t × (1-ᾱ_{t-1}) / (1-ᾱ_t)
        if t > 1:
            alpha_cumprod_prev = schedule['alphas_cumprod'][t - 1]
            posterior_variance = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
        else:
            posterior_variance = torch.tensor(0.0)

        std = torch.sqrt(posterior_variance)
        x_prev = mean + std * noise
    else:
        # 最后一步不加噪声
        x_prev = mean

    return x_prev


# =============================================================================
# 第二部分：完整采样过程
# =============================================================================

def ddpm_sample(model, shape, y: torch.Tensor, schedule: dict, num_timesteps: int = 1000, device: str = "cpu"):
    """
    完整的DDPM采样过程

    【流程】
    1. 从纯噪声开始: x_T ~ N(0, I)
    2. for t = T-1, T-2, ..., 0:
           x_t = p_sample(model, x_{t+1}, t+1)
    3. 返回 x_0

    Args:
        model: 训练好的模型
        shape: 生成图像的形状 (B, C, H, W)
        y: [B] 类别标签
        schedule: 噪声调度
        num_timesteps: 扩散步数
        device: 设备

    Returns:
        x_0: 生成的图像
    """
    model.eval()
    device = torch.device(device)

    # 从纯噪声开始
    x = torch.randn(shape, device=device)

    # 反向扩散：T-1 -> 0
    for t in tqdm(reversed(range(num_timesteps)), desc="采样中", total=num_timesteps):
        with torch.no_grad():  # 不需要计算梯度
            x = p_sample(model, x, t, y, schedule)

    return x


# =============================================================================
# 第三部分：图像生成函数
# =============================================================================

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
    从训练好的模型生成图像

    Args:
        checkpoint_path: 模型checkpoint路径
        num_samples: 生成图像数量
        class_label: 指定类别（None表示混合类别）
        num_timesteps: 采样步数
        output_dir: 输出目录
        device: 设备
    """
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # ============ 加载模型 ============
    print(f"加载模型: {checkpoint_path}")
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

    # ============ 设置类别标签 ============
    if class_label is not None:
        # 生成指定类别的图像
        y = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
        label_str = f"class_{class_label}"
        print(f"生成类别 {class_label} 的图像...")
    else:
        # 生成混合类别的图像
        y = torch.arange(10, device=device).repeat(num_samples // 10 + 1)[:num_samples]
        label_str = "mixed"
        print(f"生成混合类别图像...")

    # ============ 采样 ============
    print(f"正在生成 {num_samples} 张图像...")
    shape = (num_samples, 1, 28, 28)
    samples = ddpm_sample(model, shape, y, schedule, num_timesteps, device)

    # ============ 后处理 ============
    # 从[-1, 1]转换到[0, 1]（用于保存图像）
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    # ============ 保存图像 ============
    # 保存为网格图
    output_path = os.path.join(output_dir, f"samples_{label_str}.png")
    save_image(samples, output_path, nrow=4, normalize=False)
    print(f"保存到: {output_path}")

    # 保存单独的图像
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

    生成0-9各samples_per_class张图，方便对比效果
    """
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DiT_XS(num_classes=10).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if 'schedule' in checkpoint:
        schedule = checkpoint['schedule']
    else:
        from train import setup_ddpm_schedule
        schedule = setup_ddpm_schedule(num_timesteps)

    all_samples = []

    # 为每个类别生成样本
    for class_label in range(10):
        print(f"生成类别 {class_label}...")
        y = torch.full((samples_per_class,), class_label, dtype=torch.long, device=device)
        shape = (samples_per_class, 1, 28, 28)
        samples = ddpm_sample(model, shape, y, schedule, num_timesteps, device)
        all_samples.append(samples)

    # 合并所有样本
    all_samples = torch.cat(all_samples, dim=0)
    all_samples = (all_samples + 1) / 2
    all_samples = all_samples.clamp(0, 1)

    # 保存网格图（每行是一个类别）
    output_path = os.path.join(output_dir, "all_classes.png")
    save_image(all_samples, output_path, nrow=samples_per_class, normalize=False)
    print(f"保存所有类别图像到: {output_path}")


# =============================================================================
# 第四部分：命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DiT图像生成")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dit_final.pt",
                        help="模型checkpoint路径")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="生成图像数量")
    parser.add_argument("--class_label", type=int, default=None,
                        help="指定类别 (0-9)，None表示混合")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="采样步数")
    parser.add_argument("--output_dir", type=str, default="samples",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cpu",
                        help="设备 (cpu/cuda)")
    parser.add_argument("--all_classes", action="store_true",
                        help="为所有类别生成样本")

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
