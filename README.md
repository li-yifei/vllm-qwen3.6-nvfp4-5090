# Qwen3.6-27B NVFP4 on RTX 5090 with vLLM

Run [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) on a single **NVIDIA RTX 5090** (32 GB) using [vLLM](https://github.com/vllm-project/vllm) with **NVFP4 quantization**.

Default model: [lyf/Qwen3.6-27B-NVFP4](https://huggingface.co/lyf/Qwen3.6-27B-NVFP4) — quantized locally from the official Qwen3.6-27B release.

## Features

- 256K context length with FP8 KV cache
- NVFP4 quantization via Marlin GEMM backend
- Works out of the box with vLLM v0.17+ — no patches or custom builds needed
- Uses the `vllm/vllm-openai:cu130-nightly` Docker image
- Fully parameterized via `.env` — switch models by changing `MODEL_ID`

## GPU Compatibility

> **⚠️ This setup is tested and verified on NVIDIA RTX 5090 only.**

NVFP4 quantization requires Blackwell architecture FP4 tensor core instructions. The `vllm/vllm-openai:cu130-nightly` Docker image ships with PyTorch kernels compiled for **SM 12.0**, which matches the RTX 5090 but may not work on other Blackwell GPUs with different compute capabilities (e.g. DGX Spark GB10 is SM 12.1).

## Quick Start

```bash
# Clone this repo
git clone https://github.com/li-yifei/vllm-qwen3.6-nvfp4-5090.git
cd vllm-qwen3.6-nvfp4-5090

# Create your .env from the template
cp .env.example .env
# Edit .env with your HF token and cache paths
vim .env

# Start the server
docker compose up -d

# Check logs (model loading takes ~5-10 min on first run)
docker compose logs -f
```

The OpenAI-compatible API will be available at `http://localhost:8000`.

### Serve A Local Quantized Checkpoint

If you quantized the model locally and want to serve it from disk instead of the Hub:

```bash
cp .env.qwen36-local.example .env
docker compose -f docker-compose.yml -f docker-compose.local-model.yml up -d
```

This expects the quantized model to live at `/home/mario/qwen-llm-compressor/Qwen3.6-27B-NVFP4` by default. Adjust `MODEL_ID` in the env file if your path differs.

## Configuration

All user-specific settings live in `.env` (see [`.env.example`](.env.example)):

| Variable | Description |
|---|---|
| `HF_TOKEN` | Your [Hugging Face token](https://huggingface.co/settings/tokens) (required for gated models) |
| `HF_CACHE` | Path to your local HF cache directory (e.g. `/home/user/.cache/huggingface`) |
| `VLLM_CACHE` | Path to vLLM AOT compilation cache (e.g. `/home/user/.cache/vllm`) |
| `MODEL_ID` | HuggingFace model ID (default: `lyf/Qwen3.6-27B-NVFP4`) |

### Key vLLM Parameters

| Parameter | Default | Notes |
|---|---|---|
| `MAX_MODEL_LEN` | `262144` | 256K context window |
| `GPU_MEMORY_UTILIZATION` | `0.8` | ~26 GB of 32 GB VRAM |
| `MAX_NUM_SEQS` | `4` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | Per-batch token budget |
| `KV_CACHE_DTYPE` | `fp8` | KV cache precision |
| `NVFP4_BACKEND` | `marlin` | NVFP4 GEMM backend |

## About the Patch (vLLM v0.16 only)

> **vLLM v0.17+ has fixed this issue natively — the patch is no longer needed.** The default `docker-compose.yml` uses v0.17+ and does not include the patch.

The Qwen3.6 Mamba-hybrid architecture has layers that must remain in BF16 even when the rest of the model is NVFP4-quantized. In vLLM v0.16, the HuggingFace-to-vLLM name mapping didn't correctly translate the checkpoint's quantization ignore list for this architecture. The included `fix_linear_attn_nvfp4_exclusion.py` patched vLLM at container startup to:

1. **Exclude BF16 layers** from NVFP4 quantization: `linear_attn` (Mamba), `shared_expert_gate`, `.mlp.gate` (MoE router), and `mtp.*` layers
2. **Handle weight size mismatches** gracefully during loading, re-materializing affected parameters as unquantized tensors

vLLM v0.17 fixed this upstream via [`apply_vllm_mapper`](https://github.com/vllm-project/vllm/issues/28072), which now properly translates the exclude_modules list from HF names to vLLM names.

If you need to use vLLM v0.16, use the legacy configuration:

```bash
docker compose -f docker-compose.v16.yml up -d
```

## Requirements

- **NVIDIA RTX 5090** (32 GB VRAM) — see [GPU Compatibility](#gpu-compatibility)
- A recent NVIDIA driver (tested with 580.x)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- A [Hugging Face token](https://huggingface.co/settings/tokens) with access to gated models

## License

This configuration is provided as-is. The model itself is subject to the [Qwen License](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE).
