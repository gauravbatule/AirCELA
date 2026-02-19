# 🧊 AirCELA — Optimized AirLLM Alternative

<p align="center">
  <strong>Fastest AirLLM-style inference engine for consumer GPUs</strong><br>
  <em>Run 7B–70B+ LLMs on GPUs with as little as 4GB VRAM</em>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#speed-optimizations">Speed Optimizations</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#credits">Credits</a>
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
| **CLI tool** | ❌ | ✅ (`aircela chat`) |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/gauravbatule/AirCELA.git
cd AirCELA

# Install dependencies
pip install -r requirements.txt

# Or install as a package (editable mode)
pip install -e .
```

### Requirements
- Python 3.9+
- PyTorch 2.0+ (with CUDA support recommended)
- 4GB+ GPU VRAM (NVIDIA, any generation)
- 8GB+ System RAM

---

## 🚀 Quick Start

### Option 1: HuggingFace Models (Recommended for first-time)

```python
from aircela import CELAEngine

# Load ANY HuggingFace model — no VRAM limit!
engine = CELAEngine.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Stream tokens as they generate
for token in engine.generate("What is the capital of France?", max_tokens=50):
    print(token, end="", flush=True)
```

```bash
# Run the example script
python examples/basic_generation.py
```

### Option 2: Ollama / GGUF Models

AirCELA can load GGUF models you've already downloaded with Ollama:

```bash
# First, pull a model with Ollama
ollama pull mistral-small:22b

# Then run with AirCELA (much more control over GPU usage)
python examples/ollama_inference.py mistral-small:22b

# With a HuggingFace tokenizer for proper text output
python examples/ollama_inference.py mistral-small:22b mistralai/Mistral-Small-Instruct-2409
```

```python
# Or use the Python API directly
from aircela import CELAEngine

engine = CELAEngine.from_gguf(
    "/path/to/model.gguf",
    tokenizer_id="mistralai/Mistral-Small-Instruct-2409"  # optional
)

for token in engine.generate("Explain quantum computing.", max_tokens=100):
    print(token, end="", flush=True)
```

### Option 3: Command Line

```bash
# Interactive chat
aircela chat -m "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Single prompt
aircela run -m "mistralai/Mistral-7B-v0.1" -p "Tell me a joke"

# System info & benchmarks
aircela info
aircela bench
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
Load a HuggingFace model for layer-by-layer inference.
```python
engine = CELAEngine.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### `CELAEngine.from_gguf(path, tokenizer_id=None)`
Load a GGUF model (from Ollama or llama.cpp).
```python
engine = CELAEngine.from_gguf("/path/to/model.gguf", tokenizer_id="org/model-name")
```

### `engine.generate(prompt, max_tokens=100, temperature=0.7, top_k=50, stream=True)`
Generate text. Yields tokens one at a time when `stream=True`.
```python
for token in engine.generate("Hello!", max_tokens=50):
    print(token, end="")
```

### `engine.generate(input_ids=tensor, ...)`
Generate from raw token IDs (useful when no tokenizer is available).
```python
import torch
ids = torch.tensor([[1, 4071, 28747]])
for tok in engine.generate(input_ids=ids, max_tokens=20):
    print(tok, end=" ")
```

### `engine.reset()`
Clear KV caches for a new conversation.

---

## 📁 Project Structure

```
AirCELA/
├── aircela/
│   ├── __init__.py       # Package init (lazy imports)
│   ├── engine.py         # Core inference engine
│   ├── transformer.py    # Transformer layer (RoPE, GQA, Flash Attention)
│   ├── prefetch.py       # Double-buffer layer prefetcher
│   ├── gguf.py           # Fast GGUF parser (mmap-based)
│   ├── quantize.py       # Dequantization kernels (Q4_0, Q6_K, etc.)
│   ├── huggingface.py    # HuggingFace model loader
│   ├── device.py         # Hardware auto-detection
│   └── cli.py            # Command-line interface
├── examples/
│   ├── basic_generation.py    # HuggingFace model example
│   └── ollama_inference.py    # Ollama/GGUF model example
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 👨‍💻 Credits

<table>
  <tr>
    <td><strong>Gaurav Batule</strong></td>
    <td>Creator & Lead Developer</td>
  </tr>
  <tr>
    <td colspan="2">
      <em>AirCELA was conceived and developed by Gaurav Batule to solve the problem
      of running large language models on consumer hardware. The engine's core
      architecture — layer streaming with double-buffer prefetching, native GGUF
      support, and automatic hardware optimization — was designed and validated
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

---

<p align="center">
  <strong>Built with ❤️ by Gaurav Batule</strong><br>
  <em>Making large LLMs accessible on consumer hardware</em><br><br>
  <a href="https://www.linkedin.com/in/gaurav-batule/">LinkedIn</a> •
  <a href="https://github.com/gauravbatule/AirCELA">GitHub</a> •
  <a href="https://buymeacoffee.com/gauravbatule">☕ Support</a><br>
  <sub>If AirCELA helped you, consider buying me a coffee!</sub>
</p>
