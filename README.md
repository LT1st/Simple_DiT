# Simple_DiT

一个简单易懂的 DiT (Diffusion Transformer) PyTorch 实现，带有详细中文注释，适合学习扩散模型和Transformer。

## 🎯 项目特点

- **详细注释**：代码包含大量中文注释，解释每个模块的原理
- **两个任务**：支持图像生成和超分辨率
- **CPU友好**：小模型设计，可在CPU上运行
- **结构清晰**：模块化设计，便于理解和修改

## 📁 文件结构

```
minidit/
├── dit.py        # 模型定义（核心！包含详细注释）
├── train.py      # 图像生成训练脚本
├── sample.py     # 图像生成采样脚本
├── train_sr.py   # 超分辨率训练脚本
└── sample_sr.py  # 超分辨率采样脚本
```

## 🔧 模型架构

### DiT 核心组件

```
┌─────────────────────────────────────────────────────────┐
│                      DiT 模型                            │
├─────────────────────────────────────────────────────────┤
│  输入图像 [B,C,H,W]                                      │
│       ↓                                                  │
│  PatchEmbed (图像切分成patch序列)                         │
│       ↓                                                  │
│  + Positional Encoding (位置编码)                        │
│       ↓                                                  │
│  ┌─────────────────────────────────────┐                │
│  │  DiT Block (×N)                      │                │
│  │  ├── LayerNorm + adaLN调制           │                │
│  │  ├── Self-Attention                  │                │
│  │  ├── LayerNorm + adaLN调制           │                │
│  │  └── MLP                             │                │
│  └─────────────────────────────────────┘                │
│       ↓                                                  │
│  Final Layer (输出patch像素值)                           │
│       ↓                                                  │
│  Unpatchify (恢复图像形状)                               │
│       ↓                                                  │
│  输出 [B,C,H,W] (预测的噪声)                             │
└─────────────────────────────────────────────────────────┘
```

### adaLN-Zero (核心创新)

```
条件信息 c (时间步 + 类别)
       ↓
  MLP预测 (γ, β, gate)
       ↓
x_norm = LayerNorm(x) × (1+γ) + β
       ↓
output = x + gate × Attention(x_norm)
```

### 超分辨率条件注入

```
噪声图像 → PatchEmbed → + pos_embed_1 ─┐
                                       ├→ concat → Linear → Transformer
条件图像 → PatchEmbed → + pos_embed_2 ─┘
```

## 🚀 快速开始

### 1. 图像生成 (MNIST)

**训练：**
```bash
python train.py --epochs 20
```

**生成图像：**
```bash
# 生成16张图
python sample.py

# 生成指定数字
python sample.py --class_label 5

# 生成所有类别
python sample.py --all_classes
```

### 2. 超分辨率

**训练：**
```bash
# 4x超分辨率
python train_sr.py --scale 4 --epochs 20

# 2x超分辨率
python train_sr.py --scale 2 --epochs 20
```

**超分辨率推理：**
```bash
# 单张图像
python sample_sr.py --lr_image input.png --checkpoint checkpoints_sr/dit_sr_4x_final.pt

# 批量测试
python sample_sr.py --batch --checkpoint checkpoints_sr/dit_sr_4x_final.pt
```

## 📊 模型参数

| 模型 | 参数量 | hidden_size | depth | num_heads |
|------|--------|-------------|-------|-----------|
| DiT_XS | ~5M | 256 | 4 | 4 |
| DiT_S | ~12M | 384 | 6 | 6 |
| DiT_SR_XS | ~5M | 256 | 4 | 4 |

## 📚 学习资源

### 核心概念

1. **扩散模型 (Diffusion Model)**
   - 前向扩散：给图像逐步加噪声
   - 反向去噪：训练模型预测噪声
   - 采样：从纯噪声逐步去噪生成图像

2. **Vision Transformer (ViT)**
   - 把图像切成patch
   - 每个patch作为一个token
   - 用Transformer处理patch序列

3. **adaLN (Adaptive Layer Norm)**
   - 用条件信息调制归一化参数
   - 比cross-attention更简单高效

### 推荐阅读

- [DiT论文](https://arxiv.org/abs/2212.09748): Scalable Diffusion Models with Transformers
- [DDPM论文](https://arxiv.org/abs/2006.11239): Denoising Diffusion Probabilistic Models
- [ViT论文](https://arxiv.org/abs/2010.11929): An Image is Worth 16x16 Words

## ⚙️ 环境要求

```bash
pip install torch torchvision tqdm pillow
```

## 📝 代码阅读建议

1. 先读 `dit.py`，理解模型架构
2. 再读 `train.py`，理解训练流程
3. 最后读 `sample.py`，理解采样过程
4. 超分辨率版本是条件生成的扩展

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License
