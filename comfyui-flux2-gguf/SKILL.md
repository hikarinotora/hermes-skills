---
name: comfyui-flux2-gguf
description: Run FLUX.2 [klein] on Apple Silicon Mac (M1-M4) with 24GB unified memory via ComfyUI + GGUF. Separate path from FLUX.1 — uses ComfyUI-GGUF nodes, Qwen3-4B text encoder (NOT Mistral).
author: Benjamin Sun + Hermes
platforms: [macos]
hardware: Apple Silicon M1-M4, 24GB unified memory recommended
---

# FLUX.2 on macOS via ComfyUI + GGUF

## Why GGUF, not MLX

FLUX.2 [klein] uses a different architecture from FLUX.1. On Apple Silicon, the path is **GGUF quantization via ComfyUI-GGUF**, NOT MLX. FLUX.2 GGUF is natively supported — unlike FLUX.1 where GGUF produced NaN black images on MPS.

ComfyUI has built-in FLUX.2 support: `Flux2Scheduler`, `EmptyFlux2LatentImage`, `FluxGuidance`.

## Prerequisites

- macOS with Apple Silicon (M1–M4)
- ComfyUI with ComfyUI-GGUF custom node installed
- 24GB+ unified memory recommended (model 2.6GB + VAE 0.16GB + text encoder ~7.5GB + inference overhead)

## Step 1: Download FLUX.2 Klein 4B GGUF

```python
from huggingface_hub import snapshot_download

# Model goes in ComfyUI/models/diffusion_models/
# Q4_K_M is the recommended quantization (best quality/size tradeoff)
snapshot_download(
    repo_id="unsloth/FLUX.2-klein-4B-GGUF",
    local_dir="/path/to/ComfyUI/models/diffusion_models",
    allow_patterns=["flux-2-klein-4b-Q4_K_M.gguf"]
)
```

Model size: ~2.6GB.

## Step 2: Download VAE

```python
from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(
    repo_id="black-forest-labs/FLUX.2-klein-4B",
    filename="vae/diffusion_pytorch_model.safetensors",
)
shutil.copy2(path, os.path.expanduser(
    "~/Documents/comfy/ComfyUI/models/vae/flux2-vae.safetensors"
))
```

VAE size: ~160MB.

## Step 3: Download and merge Qwen3-4B text encoder

**CRITICAL**: FLUX.2-klein uses **Qwen3-4B** as its text encoder (7680-dim embedding). Do NOT use Mistral 3.1 (12288-dim) — it will produce `RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x12288 and 7680x3072)`.

The official BFL repo distributes the text encoder as sharded safetensors. ComfyUI's `CLIPLoader` needs a single `.safetensors` file. Download and merge:

```python
from huggingface_hub import snapshot_download
import json, os, torch
from safetensors.torch import load_file, save_file

# Step A: Download sharded text encoder from BFL official repo
target = os.path.expanduser("~/Documents/comfy/ComfyUI/models/text_encoders/qwen3_4b_flux2")
snapshot_download(
    repo_id="black-forest-labs/FLUX.2-klein-4B",
    local_dir=target,
    allow_patterns=[
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "text_encoder/model.safetensors.index.json",
    ]
)

# Step B: Flatten directory (files are nested under text_encoder/)
import shutil
for f in os.listdir(os.path.join(target, "text_encoder")):
    shutil.move(os.path.join(target, "text_encoder", f), os.path.join(target, f))
os.rmdir(os.path.join(target, "text_encoder"))

# Step C: Merge shards into single file
index = json.load(open(os.path.join(target, "model.safetensors.index.json")))
# Group keys by shard filename first
shard_to_keys = {}
for tensor_name, shard_name in index["weight_map"].items():
    shard_to_keys.setdefault(shard_name, []).append(tensor_name)

merged = {}
for shard_name, keys in shard_to_keys.items():
    shard = load_file(os.path.join(target, shard_name))
    for key in keys:
        merged[key] = shard[key]

out_path = os.path.expanduser("~/Documents/comfy/ComfyUI/models/text_encoders/qwen3_4b_flux2.safetensors")
save_file(merged, out_path)
```

Merged text encoder: ~7.5GB (from 4.6GB + 2.9GB shards). ComfyUI's `CLIPLoader` will now list it.

## Step 4: Workflow

ComfyUI native FLUX.2 workflow (based on official examples):

```
UnetLoaderGGUF (flux-2-klein-4b-Q4_K_M.gguf)
    → MODEL
CLIPLoader (qwen3_4b_flux2.safetensors, type="flux2")
    → CLIP → CLIPTextEncode (your prompt)
        → CONDITIONING → FluxGuidance (4.0)
            → CONDITIONING → BasicGuider (model + conditioning)
                → GUIDER
VAELoader (flux2-vae.safetensors)
    → VAE
Flux2Scheduler (steps, width, height)
    → SIGMAS
EmptyFlux2LatentImage (width, height, batch_size)
    → LATENT
RandomNoise (seed)
    → NOISE
KSamplerSelect (euler)
    → SAMPLER

SamplerCustomAdvanced (noise, guider, sampler, sigmas, latent_image)
    → LATENT → VAEDecode (samples + vae)
        → IMAGE → SaveImage
```

### API submission

```bash
curl -s -X POST http://127.0.0.1:8188/prompt -H "Content-Type: application/json" -d '{
  "client_id": "flux2-test",
  "prompt": {
    "1": {"inputs": {"unet_name": "flux-2-klein-4b-Q4_K_M.gguf"}, "class_type": "UnetLoaderGGUF"},
    "2": {"inputs": {"clip_name": "qwen3_4b_flux2.safetensors", "type": "flux2"}, "class_type": "CLIPLoader"},
    "3": {"inputs": {"vae_name": "flux2-vae.safetensors"}, "class_type": "VAELoader"},
    "4": {"inputs": {"clip": ["2",0], "text": "prompt here"}, "class_type": "CLIPTextEncode"},
    "5": {"inputs": {"conditioning": ["4",0], "guidance": 4}, "class_type": "FluxGuidance"},
    "6": {"inputs": {"model": ["1",0], "conditioning": ["5",0]}, "class_type": "BasicGuider"},
    "7": {"inputs": {"sampler_name": "euler"}, "class_type": "KSamplerSelect"},
    "8": {"inputs": {"steps": 4, "width": 768, "height": 768}, "class_type": "Flux2Scheduler"},
    "9": {"inputs": {"width": 768, "height": 768, "batch_size": 1}, "class_type": "EmptyFlux2LatentImage"},
    "10": {"inputs": {"noise_seed": 12345}, "class_type": "RandomNoise"},
    "11": {"inputs": {"noise":["10",0],"guider":["6",0],"sampler":["7",0],"sigmas":["8",0],"latent_image":["9",0]}, "class_type": "SamplerCustomAdvanced"},
    "12": {"inputs": {"samples":["11",0],"vae":["3",0]}, "class_type": "VAEDecode"},
    "13": {"inputs": {"images":["12",0],"filename_prefix":"flux2_klein"}, "class_type": "SaveImage"}
  }
}'
```

### Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model` steps | 4 | Klein 4B is heavily distilled — only 4 steps needed |
| `guidance` | 4.0 | FluxGuidance node |
| `sampler` | euler | Default for distilled models |
| `scheduler` | Flux2Scheduler | Built-in FLUX.2 native scheduler |

## Performance (M4, 24GB)

| Model | Size | Resolution | Steps | Time | Output |
|-------|------|------------|-------|------|--------|
| FLUX.2-klein 4B Q4_K_M | 2.6GB | 768×768 | 4 | ~80s | ~800KB |
| FLUX.2-klein 4B Q4_K_M | 2.6GB | 1024×1024 | 4 | ~90s | ~1.2MB |
| FLUX.2-klein 4B Q4_K_M | 2.6GB | 1344×1344 | 4 | ~120s | ~2.0MB |
| FLUX.2-klein 4B Q4_K_M | 2.6GB | 1536×1536 | 4 | ~150s | ~2.5MB |
| FLUX.2-klein 4B Q4_K_M | 2.6GB | **1664×1664** | 4 | ~180s | ~3.2MB |
| FLUX.2-klein 4B Q4_K_M | 2.6GB | 1728×1728 | — | ❌ OOM | — |
| FLUX.2-klein 4B Q4_K_M | 2.6GB | 1800×1800 | — | ❌ OOM (twice) | — |
| **1664² + ESRGAN 4x** | 2.6GB + upscaler | 1664→6656² | 4 + upscale | ~200s | ~50MB |

Memory: model 2.6G + VAE 0.16G + text encoder 8.0G + overhead ≈ ~12GB total. 24GB unified memory comfortable up to 1664².

### Resolution ceiling

**M4 24GB hard cap: 1664×1664.** 1728² fails at VAE decode (OOM). The sweet spot for commercial work is **1664² → ESRGAN 4x → 6656²** — print-ready at ~300 DPI for 22-inch prints.

Alternative path for 4096²+: generate at 1024², ESRGAN 4x to 4096² (verified: 27MB, sub-2min), then ESRGAN again if needed. Lower memory pressure, faster, nearly identical quality for most compositions.

### ESRGAN 4x upscale (Photoshop-ready outputs)

1024² → 4096² with `4x-UltraSharp.pth` (placed in `models/upscale_models/`). Add after VAEDecode:

```
UpscaleModelLoader("4x-UltraSharp.pth") → ImageUpscaleWithModel → SaveImage
```

ESRGAN preserves fine detail (dragon scales, textures) better than bicubic/lanczos. 4096² at 27MB is print-ready. For even larger outputs, stack two upscale nodes or generate at higher native resolution.

## Enhanced Nodes & Multi-Tool Workflow

See `references/enhanced-nodes.md` for the full node inventory (1,243 nodes across 8 custom packs). Key additions for commercial product photography:

- **Image Comparer (rgthree)**: Side-by-side A/B comparison of ESRGAN vs USDU results
- **FaceDetailer (Impact Pack)**: Auto-detect and re-render faces at full resolution in portraits
- **WAS Node Suite (220 nodes)**: Image Sharpen, Levels Adjustment, Film Grain, High Pass Filter, Background Removal — all critical for product photography post-processing
- **Multi-tool workflow**: Midjourney/Gemini (creative generation) → ComfyUI (upscaling) → Google Drive (delivery). See `references/enhanced-nodes.md` for details.

## Upscaling Strategy

Three approaches to upscale FLUX.2 outputs, ranked by quality:

### 1. Blind Upscaling: ESRGAN 4x (✅ verified — recommended default)

Node chain: `LoadImage → UpscaleModelLoader(4x-UltraSharp.pth) → ImageUpscaleWithModel → SaveImage`

- **Speed**: ~30s for 1664→6656². Only loads upsacle model (~64MB).
- **Quality**: UltraSharp is the best general-purpose upscaler for photorealistic/commercial work. Preserves material grain, minimal artifacts.
- **For anime/illustration**: Use `4x_NMKD-Siax_200k.pth` instead (better line preservation).
- **Verified result**: 1664² → 6656² = 50MB PNG. Print-ready at 300 DPI for ~22 inch prints.
- **Limitation**: Pure pixel-based math — doesn't understand image content. Complex textures may show mild oil-painting artifacts or edge fragmentation.

API submission (ESRGAN 4x on saved image):

```bash
curl -s -X POST http://127.0.0.1:8188/prompt -H "Content-Type: application/json" -d '{
  "client_id": "esrgan",
  "prompt": {
    "1": {"inputs": {"image": "input_image.png"}, "class_type": "LoadImage"},
    "2": {"inputs": {"model_name": "4x-UltraSharp.pth"}, "class_type": "UpscaleModelLoader"},
    "3": {"inputs": {"upscale_model": ["2",0], "image": ["1",0]}, "class_type": "ImageUpscaleWithModel"},
    "4": {"inputs": {"images": ["3",0], "filename_prefix": "upscaled"}, "class_type": "SaveImage"}
  }
}'
```

### 2. Semantic Upscaling: Ultimate SD Upscale (✅ verified with SD 1.5)

Uses tile-based re-diffusion — splits image into tiles, runs a diffusion model on each tile with the original prompt to generate real detail. Node: `UltimateSDUpscale` from `ComfyUI_UltimateSDUpscale` custom node.

**Node installation:**
```bash
cd ~/Documents/comfy/ComfyUI/custom_nodes
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git
```

**Two-tier compatibility on M4 24GB:**

| Checkpoint | Size | Status | Details |
|-----------|------|--------|---------|
| SDXL Base 1.0 | 6.5GB | ❌ OOM | 15+ min with no output — swap death with 1664² image in memory |
| SD 1.5 (v1-5-pruned-emaonly) | 4.3GB (~2GB active) | ✅ Works | 1024→2048² in ~10 min with optimized parameters |

**Verified parameters (Gemini-optimized, confirmed working):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `checkpoint` | `v1-5-pruned-emaonly.safetensors` | Lightweight SD 1.5 (~2GB active), not SDXL |
| `tile_width/height` | 1024 | Max tile size. Fewer tiles = faster. 4 tiles (2×2) for 2048² |
| `steps` | 10 | Minimum viable — each step costs ~10.7s |
| `denoise` | 0.175 | Sweet spot. Above 0.25 distorts original; below 0.15 adds nothing |
| `sampler` | `dpmpp_2m_sde` | Good quality/speed balance |
| `scheduler` | `karras` | Standard for SD 1.5 |
| `seam_fix_mode` | `Half Tile + Intersections` | Best seam quality at cost of extra passes |
| `seam_fix_denoise` | 0.175 | Match main denoise |

**Verified benchmark (1024² → 2048², 4 tiles on M4 24GB):**
- Per-tile: ~1m46s (10 steps at ~10.7s/step)
- Per-seam fix: ~1m32s (4 seam passes after tiles)
- Total: ~10 minutes
- Output: 3.7MB PNG at 2048²

**Memory strategy**: Run USDU in a SEPARATE ComfyUI session from FLUX.2 generation. FLUX.2 (12GB) + SD 1.5 (~2GB) would fit together on 24GB, but in practice the ComfyUI process accumulates memory. Restart ComfyUI between FLUX.2 generation and USDU upscale to guarantee clean memory.

**API submission (SD 1.5 + USDU, 2x upscale):**
```bash
# First copy the FLUX.2 output to ComfyUI/input/
cp ~/Documents/comfy/ComfyUI/output/flux2_image.png ~/Documents/comfy/ComfyUI/input/to_upscale.png

# Submit USDU job
curl -s -X POST http://127.0.0.1:8188/prompt -H "Content-Type: application/json" -d '{
  "client_id": "usdu",
  "prompt": {
    "1": {"inputs": {"image": "to_upscale.png"}, "class_type": "LoadImage"},
    "2": {"inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}, "class_type": "CheckpointLoaderSimple"},
    "3": {"inputs": {"model_name": "4x-UltraSharp.pth"}, "class_type": "UpscaleModelLoader"},
    "4": {"inputs": {"clip": ["2",1], "text": "ultra photorealistic, 8K, masterpiece"}, "class_type": "CLIPTextEncode"},
    "5": {"inputs": {"clip": ["2",1], "text": "blurry, low quality, distorted, ugly, jpeg artifacts, plastic, cartoon, 3d render"}, "class_type": "CLIPTextEncode"},
    "6": {"inputs": {
      "image": ["1",0], "model": ["2",0], "positive": ["4",0], "negative": ["5",0], "vae": ["2",2],
      "upscale_by": 2.0, "seed": 0, "steps": 10, "cfg": 5, "sampler_name": "dpmpp_2m_sde", "scheduler": "karras",
      "denoise": 0.175, "upscale_model": ["3",0], "mode_type": "Chess",
      "tile_width": 1024, "tile_height": 1024, "mask_blur": 8, "tile_padding": 32,
      "seam_fix_mode": "Half Tile + Intersections", "seam_fix_denoise": 0.175,
      "seam_fix_width": 64, "seam_fix_mask_blur": 4, "seam_fix_padding": 16,
      "force_uniform_tiles": true, "tiled_decode": true, "batch_size": 1
    }, "class_type": "UltimateSDUpscale"},
    "7": {"inputs": {"images": ["6",0], "filename_prefix": "usdu_result"}, "class_type": "SaveImage"}
  }
}'
```

**SD 1.5 checkpoint download:**
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    filename="v1-5-pruned-emaonly.safetensors",
    local_dir="~/Documents/comfy/ComfyUI/models/checkpoints"
)
```

### 3. Latent Upscaling: LatentUpscale

- **Node**: `LatentUpscale` / `LatentUpscaleBy` — upsamples in latent space before VAE decode.
- **Pros**: No extra model, fast. Good for 1.5-2x.
- **Cons**: Quality drops beyond 2x. Not suitable for commercial output.

### Strategy Decision Tree

```
Need upscale?
├─ Quick preview → ESRGAN 4x (seconds)
├─ Print-ready commercial → FLUX.2 @ 1664² → ESRGAN 4x → 6656² (verified ✅)
└─ Need semantic detail restoration → lightweight SD 1.5 + Ultimate SD Upscale (separate session, ✅ verified 1024→2048² in ~10min)
```

---

## Pitfalls

1. **Wrong text encoder → dimension mismatch**: FLUX.2-klein uses Qwen3-4B (7680-dim). Using Mistral 3.1 (12288-dim) produces `RuntimeError: linear(): input and weight.T shapes cannot be multiplied (512x12288 and 7680x3072)`. Always verify the text encoder matches the model architecture.
2. **Comfy-Org text encoder link is broken**: The official ComfyUI examples page links to `Comfy-Org/flux2-dev/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors` which returns 0-byte files on HF. Use `black-forest-labs/FLUX.2-klein-4B` official repo instead, and merge shards.
3. **CLIPLoader needs single .safetensors**: It cannot load sharded models — merge them first. Subdirectories under `text_encoders/` are invisible to the loader.
4. **Mixed format mismatch**: Don't mix GGUF model with safetensors text encoder from a different model family (e.g., 9B encoder with 4B model). Both model and encoder must target the same architecture.
5. **`Flux2ImageNode` is a paid API node**: The built-in `Flux2ImageNode` routes through Comfy.org cloud — use the local node chain above for offline generation.
6. **GGUF FLUX.1 ≠ GGUF FLUX.2**: FLUX.1 GGUF on MPS produces NaN black images (Float8 incompatibility). FLUX.2 GGUF works natively because its architecture supports the quantization path.
