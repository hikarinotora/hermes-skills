# hermes-skills

ComfyUI image generation skills for Hermes Agent on Apple Silicon Mac (M1–M4 with 24GB unified memory).

## Skills

### comfyui-flux-mlx
Run **FLUX.1-dev** on macOS via ComfyUI + MLX framework. Covers model download, custom node installation, API workflow, and performance benchmarks. MLX is the only reliable path for FLUX.1 on Apple Silicon — GGUF on MPS produces black images.

- **Model:** FLUX.1-dev MLX 4-bit (~9.2GB)
- **Resolution:** 768×768 in ~5–6 min
- **Hardware:** M1–M4, 16GB+ (24GB recommended)

### comfyui-flux2-gguf
Run **FLUX.2 [Klein 4B]** on macOS via ComfyUI + GGUF. Includes model/text-encoder/VAE downloads, native FLUX.2 workflow, ESRGAN 4x upscaling, and Ultimate SD Upscale tile-based enhancement. Tested resolution ceiling: 1664×1664 on M4 24GB.

- **Model:** FLUX.2-klein-4B GGUF Q4_K_M (~2.6GB)
- **Text encoder:** Qwen3-4B (7.5GB, must merge shards)
- **Resolution:** up to 1664² native, 6656² with ESRGAN 4x, 2048² with USDU
- **Upscaling:** ESRGAN 4x UltraSharp (fast) + Ultimate SD Upscale with SD 1.5 (quality)

## Quick Install

```bash
# Add this skill repository to Hermes
hermes skills tap add hikarinotora/hermes-skills

# Install skills
hermes skills install comfyui-flux-mlx
hermes skills install comfyui-flux2-gguf
```

## Prerequisites

- Hermes Agent installed
- ComfyUI already running on `http://127.0.0.1:8188`
- Apple Silicon Mac (M1–M4) with 24GB unified memory recommended
- ComfyUI custom nodes: Mflux-ComfyUI, ComfyUI-GGUF, UltimateSDUpscale (covered in skill instructions)

## Author

Benjamin Sun + Hermes

---

# hermes-skills（中文）

适用于 Apple Silicon Mac（M1–M4，24GB 统一内存）上 Hermes Agent 的 ComfyUI 图像生成技能。

## 技能列表

### comfyui-flux-mlx
在 macOS 上通过 ComfyUI + MLX 框架运行 **FLUX.1-dev**。涵盖模型下载、自定义节点安装、API 工作流和性能基准。MLX 是 Apple Silicon 上 FLUX.1 唯一可靠的路径——MPS 上的 GGUF 会产出全黑图像。

- **模型：** FLUX.1-dev MLX 4-bit（约 9.2GB）
- **分辨率：** 768×768，约 5–6 分钟
- **硬件：** M1–M4，16GB 以上（推荐 24GB）

### comfyui-flux2-gguf
在 macOS 上通过 ComfyUI + GGUF 运行 **FLUX.2 [Klein 4B]**。包含模型/文本编码器/VAE 下载、原生 FLUX.2 工作流、ESRGAN 4x 放大和 Ultimate SD Upscale 分块重绘增强。已在 M4 24GB 上测试分辨率天花板：1664×1664。

- **模型：** FLUX.2-klein-4B GGUF Q4_K_M（约 2.6GB）
- **文本编码器：** Qwen3-4B（7.5GB，需合并分片）
- **分辨率：** 原生最高 1664²，ESRGAN 4x 可达 6656²，USDU 可达 2048²
- **放大：** ESRGAN 4x UltraSharp（快速）+ Ultimate SD Upscale + SD 1.5（高质量）

## 快速安装

```bash
# 添加技能仓库到 Hermes
hermes skills tap add hikarinotora/hermes-skills

# 安装技能
hermes skills install comfyui-flux-mlx
hermes skills install comfyui-flux2-gguf
```

## 前置条件

- 已安装 Hermes Agent
- ComfyUI 已在 `http://127.0.0.1:8188` 运行
- Apple Silicon Mac（M1–M4），推荐 24GB 统一内存
- ComfyUI 自定义节点：Mflux-ComfyUI、ComfyUI-GGUF、UltimateSDUpscale（技能文档中有安装说明）

## 作者

Benjamin Sun + Hermes
