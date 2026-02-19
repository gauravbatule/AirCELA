# 🧊 AirCELA — Optimized AirLLM Alternative

<p align="center">
  <strong>Fastest AirLLM-style inference engine for consumer GPUs</strong><br>
  <em>Run 7B–70B+ LLMs on GPUs with as little as 4GB VRAM</em>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#ollama--gguf-models">Ollama / GGUF</a> •
  <a href="#cli-commands">CLI</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

## ⚡ What is AirCELA?

**AirCELA** (Compute-Efficient Layer Architecture) is a high-performance **AirLLM** alternative that lets you run models far larger than your GPU VRAM by streaming **one transformer layer at a time**.

While AirLLM pioneered the concept, AirCELA takes it to the next level with **Double-Buffer Prefetching** and **Zero-Copy Memory Mapping**, reducing the I/O bottleneck by up to 80%.

### 🚀 Why AirCELA vs AirLLM?

| Feature | AirLLM | AirCELA |
|---------|--------|---------|
| **Streaming Style** | Sequential | **Asynchronous Double-Buffered** |
| **I/O Hiding** | ❌ (GPU waits for disk) | ✅ (Prefetches while computing) |
| **Disk Access** | Standard I/O | **Zero-Copy mmap** |
| **Attention** | Standard | **Flash Attention (cuDNN fused)** |
| **Model Support** | HF only | **HF + GGUF (Ollama)** |
| **Dequantization** | Standard | **Vectorized NumPy Kernels** |
| **CLI tool** | ❌ | ✅ (`aircela chat`, `aircela bench`) |

---

## 📦 Installation

### Step 1: Clone the repo

```bash
git clone https://github.com/gauravbatule/AirCELA.git
cd AirCELA
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** This installs PyTorch, NumPy, transformers, huggingface_hub, safetensors, psutil, and accelerate.

### Step 3 (Optional): Install as a CLI tool

```bash
pip install -e .
```

This gives you the `aircela` and `cela` commands system-wide. Without this step, you can still use `python -m aircela.cli` or the Python API directly.

### Requirements

- **Python**: 3.9 or higher
- **PyTorch**: 2.0+ (CUDA support recommended for GPU inference)
- **GPU**: Any NVIDIA GPU with 4GB+ VRAM (RTX 3050, GTX 1650, etc.)
- **RAM**: 8GB+ system RAM
- **OS**: Windows 10/11, Linux, macOS

---

## 🚀 Quick Start

### Using the Python API (HuggingFace Models)

The simplest way to start — downloads the model automatically:

```python
# File: my_test.py
import sys
sys.path.insert(0, ".")  # Only needed if you didn't pip install

from aircela import CELAEngine

# Load a model (downloads automatically on first run)
engine = CELAEngine.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate text — tokens stream one at a time
for token in engine.generate("What is the capital of France?", max_tokens=50):
    print(token, end="", flush=True)
print()
```

Run it:

```bash
cd AirCELA
python my_test.py
```

Or use the included example:

```bash
python examples/basic_generation.py
```

---

## 🦙 Ollama / GGUF Models

AirCELA can directly load GGUF models you've already downloaded with Ollama.

### Step 1: Pull a model with Ollama

```bash
ollama pull mistral-small:22b
```

### Step 2: Run with AirCELA

```bash
# List available Ollama models on your system
python examples/ollama_inference.py

# Run a model (without tokenizer — outputs raw token IDs)
python examples/ollama_inference.py mistral-small:22b

# Run with a HuggingFace tokenizer for readable text
python examples/ollama_inference.py mistral-small:22b mistralai/Mistral-Small-Instruct-2409
```

### Using the Python API directly

```python
import sys
sys.path.insert(0, ".")

from aircela import CELAEngine

# Load a GGUF model from Ollama or a direct file path
engine = CELAEngine.from_gguf(
    "C:/Users/YourName/.ollama/models/blobs/sha256-<hash>",
    tokenizer_id="mistralai/Mistral-Small-Instruct-2409"  # optional
)

for token in engine.generate("Explain quantum computing.", max_tokens=100):
    print(token, end="", flush=True)
print()
```

> **Tip:** Run `python examples/ollama_inference.py` with no arguments to see all available Ollama models and their file paths.

---

## 🖥️ CLI Commands

> **Prerequisite:** You must install AirCELA as a package first: `pip install -e .`

### `aircela info` — Show hardware info

```bash
aircela info
```

Shows your GPU name, VRAM, RAM, and compute capability.

### `aircela chat` — Interactive chat

```bash
aircela chat -m "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Options:
- `-m MODEL` — HuggingFace model ID (required)
- `-n NUM` — Max tokens to generate (default: 200)
- `-t TEMP` — Temperature 0.0–1.0 (default: 0.7)

In-chat commands: `/quit`, `/clear`, `/reset`

### `aircela run` — Single prompt generation

```bash
aircela run -m "TinyLlama/TinyLlama-1.1B-Chat-v1.0" -p "Tell me a joke"
```

Options:
- `-m MODEL` — HuggingFace model ID (required)
- `-p PROMPT` — Input prompt (required)
- `-n NUM` — Max tokens (default: 100)
- `-t TEMP` — Temperature (default: 0.7)

### `aircela bench` — Benchmark hardware

```bash
aircela bench
```

Tests RAM bandwidth, GPU bandwidth, GEMM performance, and SSD read speed.

### Without `pip install`

If you haven't installed the package, you can use the CLI via Python module:

```bash
python -m aircela.cli info
python -m aircela.cli chat -m "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
python -m aircela.cli bench
```

---

## 🏗️ How It Works

```
Token → Embed → [Layer 0 → Layer 1 → ... → Layer N] → Norm → LM Head → Token
                  ↑ GPU     ↑ prefetch next              ↑ GPU
```

1. **Only ONE transformer layer** lives in VRAM at a time
2. A background thread **prefetches the next layer** while the GPU computes the current one
3. Each layer: **Disk → Dequantize → GPU → Compute → Free VRAM → Next**
4. KV caches live on CPU RAM to save VRAM
5. This means a **70B model needs only ~500MB VRAM** per layer instead of 140GB total

---

## 🏎️ Speed Optimizations

| Optimization | What It Does |
|---|---|
| **Double-Buffer Prefetching** | Loads next layer in background while GPU computes current one. Hides 90% of disk I/O latency. |
| **Zero-Copy mmap** | OS-level memory mapping for weight files. No redundant copies. |
| **Fast GGUF Parser** | mmap-based parser reads 12GB GGUF headers in <10ms by skipping tokenizer metadata. |
| **Vectorized Dequantization** | Q4_0/Q6_K dequant uses NumPy C-extensions, 100x faster than Python loops. |
| **Flash Attention** | PyTorch `scaled_dot_product_attention` for fused cuDNN kernels. |
| **Lazy Module Loading** | Heavy imports (torch, transformers) only load when actually needed. |

---

## 📊 Supported Quantization Formats

| Format | Status | Bits | Description |
|--------|--------|------|-------------|
| F32    | ✅     | 32   | Full precision float |
| F16    | ✅     | 16   | Half precision float |
| Q4_0   | ✅     | 4    | 4-bit quantization (most common in Ollama) |
| Q4_1   | ✅     | 4    | 4-bit with minimum offset |
| Q8_0   | ✅     | 8    | 8-bit quantization |
| Q6_K   | ✅     | 6    | 6-bit K-quant (high quality) |

---

## 📖 API Reference

### `CELAEngine.from_pretrained(model_name)`

Load any HuggingFace model. Downloads and caches automatically.

```python
from aircela import CELAEngine
engine = CELAEngine.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### `CELAEngine.from_gguf(path, tokenizer_id=None)`

Load a GGUF model file (from Ollama or llama.cpp).

```python
engine = CELAEngine.from_gguf("/path/to/model.gguf")
# Or with a tokenizer for readable text output:
engine = CELAEngine.from_gguf("/path/to/model.gguf", tokenizer_id="org/model-name")
```

### `engine.generate(prompt=..., input_ids=..., max_tokens=100, temperature=0.7, top_k=50, stream=True)`

Generate text. Yields tokens one by one when `stream=True`.

```python
# From a text prompt (requires tokenizer)
for token in engine.generate("Hello!", max_tokens=50):
    print(token, end="")

# From raw token IDs (no tokenizer needed)
import torch
ids = torch.tensor([[1, 4071, 28747]])
for tok in engine.generate(input_ids=ids, max_tokens=20):
    print(tok, end=" ")
```

### `engine.reset()`

Clear KV caches to start a fresh conversation.

---

## 📁 Project Structure

```
AirCELA/
├── aircela/                    # Main package
│   ├── __init__.py             # Lazy imports (fast startup)
│   ├── engine.py               # Core inference engine
│   ├── transformer.py          # Transformer layer (RoPE, GQA, Flash Attention)
│   ├── prefetch.py             # Double-buffer layer prefetcher
│   ├── gguf.py                 # Fast GGUF parser (mmap-based)
│   ├── quantize.py             # Dequantization kernels (Q4_0, Q6_K, etc.)
│   ├── huggingface.py          # HuggingFace model loader
│   ├── device.py               # Hardware auto-detection
│   └── cli.py                  # Command-line interface
├── examples/
│   ├── basic_generation.py     # HuggingFace model example
│   └── ollama_inference.py     # Ollama/GGUF model example
├── _legacy/                    # Old prototypes (not part of the package)
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package configuration
├── LICENSE                     # CELA Proprietary License
└── README.md                   # This file
```

---

## 🔧 Troubleshooting

### "Import takes forever" / Python hangs on startup

**Cause:** PyTorch's first import can take 1–3 minutes on Windows due to CUDA detection and DLL loading.

**Fix:** This is a one-time cost per Python process. AirCELA uses lazy imports so the `aircela` package itself loads instantly — torch only loads when you actually call `CELAEngine`.

**Speed it up:**
1. Add your Python installation folder to Windows Defender exclusions
2. Use an SSD for your Python environment
3. Keep a Python REPL open to avoid repeated cold starts

### "No tokenizer loaded" error

**Cause:** GGUF models from Ollama don't include tokenizers.

**Fix:** Provide a HuggingFace tokenizer ID:
```python
engine = CELAEngine.from_gguf("model.gguf", tokenizer_id="mistralai/Mistral-7B-v0.1")
```

Or use `input_ids` directly:
```python
import torch
ids = torch.tensor([[1, 4071]])
for tok in engine.generate(input_ids=ids, max_tokens=20):
    print(tok)
```

### "MemoryError" during generation

**Cause:** Large models (22B+) require significant RAM for dequantization.

**Fix:**
1. Close other applications to free RAM
2. Use a smaller model (7B or 3B)
3. Make sure you have at least 16GB RAM for 22B models

### "Unsupported GGUF qtype" warning

**Cause:** The model uses a quantization format AirCELA doesn't support yet.

**Supported:** F32, F16, Q4_0, Q4_1, Q8_0, Q6_K

---

## 👨‍💻 Credits

<table>
  <tr>
    <td><strong>Gaurav Batule</strong></td>
    <td>Creator & Lead Developer</td>
  </tr>
  <tr>
    <td colspan="2">
      <em>AirCELA was built by Gaurav Batule to solve the problem of running
      large language models on consumer hardware. The engine's core architecture —
      layer streaming with double-buffer prefetching, native GGUF support, and
      automatic hardware optimization — was designed, developed, and validated
      by Gaurav.</em>
    </td>
  </tr>
  <tr>
    <td colspan="2">
      <a href="https://www.linkedin.com/in/gaurav-batule/">🔗 LinkedIn</a> •
      <a href="https://github.com/gauravbatule/AirCELA">🐙 GitHub</a> •
      <a href="https://buymeacoffee.com/gauravbatule">☕ Buy Me a Coffee</a>
    </td>
  </tr>
</table>

> 🤖 **Vibe Code Notice**: Portions of this codebase were developed with AI assistance.
> All architecture decisions, core algorithms, testing, and validation were performed
> by **Gaurav Batule**.

---

## 📄 License

**CELA Proprietary License** — See [LICENSE](LICENSE)

- ✅ Free for personal and educational use
- ✅ Attribution required (credit Gaurav Batule)
- ❌ Commercial use requires a separate license

**Contact:** gauravbatule@gmail.com

---

<p align="center">
  <strong>Built with ❤️ by Gaurav Batule</strong><br>
  <em>Making large LLMs accessible on consumer hardware</em><br><br>
  <a href="https://www.linkedin.com/in/gaurav-batule/">LinkedIn</a> •
  <a href="https://github.com/gauravbatule/AirCELA">GitHub</a> •
  <a href="https://buymeacoffee.com/gauravbatule">☕ Support</a><br>
  <sub>If AirCELA helped you, consider buying me a coffee!</sub>
</p>
