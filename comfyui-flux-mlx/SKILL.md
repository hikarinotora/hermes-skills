---
name: comfyui-flux-mlx
description: Run FLUX.1 on Apple Silicon Mac (M1-M4) with 24GB unified memory via ComfyUI + MLX framework. Works on the only reliable path — MPS/GGUF will produce NaN black images or dimension errors.
author: 大本老师 + Hermes
platforms: [macos]
hardware: Apple Silicon M1-M4, 24GB unified memory recommended
---

# FLUX.1 on macOS via ComfyUI + MLX

> **FLUX.2 is a different path.** FLUX.2 [klein] uses GGUF quantization + Qwen3-4B text encoder, not MLX. See `comfyui-flux2-gguf` skill for FLUX.2 instructions.

## Why MLX is the only path for FLUX.1

**FLUX.1 (dev/schnell)** → MLX framework (this skill's main body below)
**FLUX.2 (Klein 4B/9B)** → GGUF + ComfyUI-GGUF nodes → see `references/flux2-gguf.md`

The paths are completely different — wrong path = black images or load errors.

## FLUX.1: MLX path

On Apple Silicon (M1–M4), standard PyTorch MPS + GGUF-based FLUX nodes produce NaN black images or dimension mismatch errors (e.g., `RuntimeError: mat1 and mat2 shapes cannot be multiplied`). This is because GGUF quantized FLUX models and MPS's `Float8` support are incompatible.

**Solution:** Use Apple's native **MLX** framework via [Mflux-ComfyUI](https://github.com/anthonymwu/Mflux-ComfyUI). MLX bypasses MPS entirely, uses unified memory efficiently (~30% less RAM), and is purpose-built for Apple Silicon.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- ComfyUI already installed
- 16GB+ unified memory recommended (24GB comfortable for dev 4-bit)

## Step 1: Install mflux (exact version lock)

```bash
# In both system Python and ComfyUI's venv
# Version MUST be 0.4.1 — newer versions have API changes that break the node
pip install mflux==0.4.1

# For ComfyUI venv:
/path/to/ComfyUI/venv/bin/pip install mflux==0.4.1
```

## Step 2: Install Mflux-ComfyUI custom node

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/anthonymwu/Mflux-ComfyUI.git
```

Restart ComfyUI after installation. Verify nodes loaded by checking `/object_info` for `QuickMfluxNode`, `MfluxModelsLoader`.

## Step 3: Download MLX-quantized FLUX model

Models go in `ComfyUI/models/Mflux/`. Use HuggingFace `snapshot_download`:

```python
from huggingface_hub import snapshot_download

# For FLUX.1-dev (best quality, ~9GB, 20-step generation)
snapshot_download(
    repo_id="madroid/flux.1-dev-mflux-4bit",
    local_dir="/path/to/ComfyUI/models/Mflux/flux.1-dev-mflux-4bit"
)

# For FLUX.1-schnell (fast, ~6GB, 4-step generation)
snapshot_download(
    repo_id="madroid/flux.1-schnell-mflux-4bit",
    local_dir="/path/to/ComfyUI/models/Mflux/flux.1-schnell-mflux-4bit"
)
```

Model structure after download (dev 4-bit, ~9.2GB total):
```
transformer/  6.2GB (4 shards)
text_encoder_2/ 2.7GB (T5-XXL)
vae/  157MB
text_encoder/  118MB
tokenizer/  ~1.5MB
tokenizer_2/  ~0.8MB
```

## Step 4: Test generation via ComfyUI API

```bash
curl -s -X POST http://127.0.0.1:8188/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "test",
    "prompt": {
      "1": {
        "inputs": {"model_name": "flux.1-dev-mflux-4bit"},
        "class_type": "MfluxModelsLoader"
      },
      "2": {
        "inputs": {
          "prompt": "your prompt here",
          "model": "dev",
          "quantize": "None",
          "seed": 12345,
          "width": 768,
          "height": 768,
          "steps": 20,
          "guidance": 3.5,
          "metadata": true,
          "Local_model": ["1", 0]
        },
        "class_type": "QuickMfluxNode"
      },
      "3": {
        "inputs": {
          "images": ["2", 0],
          "filename_prefix": "flux_test"
        },
        "class_type": "SaveImage"
      }
    }
  }'
```

### Key parameters

| Parameter | dev | schnell |
|-----------|-----|---------|
| `quantize` | `"None"` (already 4-bit) | `"None"` |
| `steps` | 20 (recommended) | 4 |
| `guidance` | 3.5 | 3.5 |
| `model` | `"dev"` | `"schnell"` |

- `quantize`: Set to `"None"` when using pre-quantized models. Options `"4"` / `"8"` are for on-the-fly quantization of full-precision models.
- `seed`: INT, -1 for random. Max `18446744073709551615`.

## Performance benchmarks (M4, 24GB)

| Model | Size | Steps | Resolution | Time |
|-------|------|-------|------------|------|
| FLUX.1-dev MLX 4-bit | 9.2GB | 20 | 768×768 | ~5-6 min |
| FLUX.1-schnell MLX 4-bit | ~6GB | 4 | 512×512 | ~30-60 sec (est.) |
| FLUX.2 Klein 4B GGUF Q4_K_M | 2.6GB | 4 | 1024×1024 | ~30-60 sec (est.) |

---

## FLUX.2 (Klein 4B) — separate skill

FLUX.2 has a completely different path from FLUX.1. See `comfyui-flux2-gguf` skill for full instructions.

Quick summary for cross-reference:
- **4B model**: GGUF Q4_K_M (2.6GB) via ComfyUI-GGUF, NOT MLX/Mflux
- **Text encoder**: Qwen3-4B (7680-dim, 8GB merged), NOT Mistral 3.1 (12288-dim)
- **VAE**: flux2-vae.safetensors (160MB)
- **Performance**: 1024×1024 in ~90s (4 steps), 4096² with ESRGAN in ~100s

## Pitfalls

### FLUX.1 (MLX) pitfalls
1. **`mflux` version lock**: The node requires `mflux==0.4.1`. Newer versions (`>0.5`) changed API signatures and the node will fail silently with import errors. Check with `pip show mflux`.
2. **Never use GGUF FLUX.1 on MPS**: Will produce black images (NaN tensor) or shape mismatch errors. Delete any GGUF FLUX.1 models — they're dead weight on Mac.
3. **`quantize` parameter**: With pre-quantized models (4-bit from HF), always set `"None"`. Setting `"4"` or `"8"` causes double-quantization errors.
4. **Model directory**: Must be `ComfyUI/models/Mflux/` — NOT standard `checkpoints/`. The `MfluxModelsLoader` lists subdirectories of this path.

### Universal pitfalls (all FLUX versions)
5. **Download method**: Use `huggingface_hub.hf_hub_download()` or `snapshot_download()` for HF model downloads. **Never use curl** — HF uses LFS (Git Large File Storage). curl returns 15-byte LFS pointers, not real files. `huggingface_hub` handles LFS transparently.
6. **First run memory**: The text encoder + model + VAE combined can push 18-24GB. Close other apps to avoid OOM.

### FLUX.2 — see `comfyui-flux2-gguf` for full details
7. **Wrong text encoder → dimension mismatch**: FLUX.2-klein 4B uses Qwen3-4B (7680-dim, not 12288-dim Mistral). Wrong encoder gives `linear(): input and weight.T shapes cannot be multiplied`.
8. **Comfy-Org flux2-dev link is dead**: `Comfy-Org/flux2-dev/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors` returns 0 bytes. Use `black-forest-labs/FLUX.2-klein-4B` directly.
