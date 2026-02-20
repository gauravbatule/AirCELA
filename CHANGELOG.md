# 📋 Changelog

All notable changes to AirCELA are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-02-19

### 🎉 Initial Release

The first public release of AirCELA — an optimized AirLLM alternative for running
large language models on consumer GPUs with limited VRAM.

### Added

- **Core Engine** (`CELAEngine`)
  - Layer-by-layer transformer inference with single-layer VRAM usage
  - `from_pretrained()` for HuggingFace models (any size)
  - `from_gguf()` for Ollama/llama.cpp GGUF models
  - Streaming token generation with `generate()`
  - Temperature and top-k sampling
  - KV cache management with `reset()`

- **Double-Buffer Prefetching** (`LayerPrefetcher`)
  - Background thread loads next layer while GPU computes current one
  - Hides ~90% of disk I/O latency
  - Automatic fallback to sequential loading if prefetch fails

- **Fast GGUF Parser** (`GGUFReader`)
  - mmap-based zero-copy parsing (12GB files in <10ms)
  - Skips large tokenizer metadata arrays for speed
  - Supports GGUF v2 and v3 formats

- **Dequantization Kernels** (`Dequantizer`)
  - Vectorized NumPy implementations for: F32, F16, Q4_0, Q4_1, Q8_0, Q6_K
  - 100x faster than naive Python loop implementations

- **Transformer Layer** (`CELATransformerLayer`)
  - Grouped Query Attention (GQA) support
  - Rotary Position Embeddings (RoPE)
  - Flash Attention via PyTorch `scaled_dot_product_attention`
  - RMS Layer Normalization
  - SiLU-gated FFN (LLaMA-style)

- **HuggingFace Loader** (`HFModelLoader`)
  - Automatic model download and caching via `huggingface_hub`
  - safetensors and .bin format support
  - Automatic layer-to-file mapping

- **Hardware Detection** (`DeviceManager`)
  - Automatic GPU/CPU detection
  - VRAM and RAM capacity reporting
  - Compute capability detection

- **CLI Tool**
  - `aircela info` — Hardware information
  - `aircela chat` — Interactive chat mode
  - `aircela run` — Single prompt generation
  - `aircela bench` — Hardware benchmarking (RAM, GPU, SSD)

- **Lazy Module Loading**
  - `import aircela` completes in <0.1s (torch loads only when needed)

### Fixed

- `generate()` now works without a tokenizer (yields raw token IDs)
- RoPE config key compatibility (`rope_freq_base` vs `rope_theta`)
- LM head weight tying fallback when `output.weight` is missing in GGUF
- Null-safe prefetcher calls in `reset()`

---

## [Unreleased]

### Planned

- Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K quantization support
- CUDA-accelerated dequantization kernels
- Phi, Qwen, Gemma architecture support
- Speculative decoding for faster generation
- Web UI for interactive inference
- Batch generation support
- AMD ROCm support

---

_For the full commit history, see [GitHub Commits](https://github.com/gauravbatule/AirCELA/commits/main)._
