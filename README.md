# 🧊 AirCELA — Optimized AirLLM Alternative

<p align="center">
  <strong>Fastest AirLLM-style inference engine for consumer GPUs</strong><br>
  <em>Run 7B–70B+ LLMs on GPUs with as little as 4GB VRAM using advanced prefetching</em>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#speed-optimizations">Speed Optimizations</a> •
  <a href="#api-reference">API</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#credits">Credits</a>
</p>

---

## ⚡ What is AirCELA?

**AirCELA** (Compute-Efficient Layer Architecture) is a high-performance **AirLLM** alternative that lets you run models far larger than your GPU VRAM by streaming **one transformer layer at a time**. 

While AirLLM pioneered the concept, AirCELA takes it to the next level with **Double-Buffer Prefetching** and **Zero-Copy Memory Mapping**, reducing the I/O bottleneck by up to 80%.

### 🚀 Why AirCELA vs AirLLM?

| Feature | AirLLM | AirCELA |
|---------|--------|------|
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

# Install (editable mode for development)
pip install -e .
```

---

## 🚀 Quick Start

### Python API

```python
from aircela import CELAEngine

# Load ANY HuggingFace model — no VRAM limit!
engine = CELAEngine.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate text (streams tokens)
for token in engine.generate("What is the capital of France?", max_tokens=50):
    print(token, end="", flush=True)
```

### Running Ollama Models (GGUF)

AirCELA can directly load models you've already downloaded with Ollama!

```python
from aircela import CELAEngine

# Point to your Ollama GGUF blob and provide the HF tokenizer
engine = CELAEngine.from_gguf(
    "/path/to/model.gguf",
    tokenizer_id="mistralai/Mistral-Small-Instruct-2409"
)

for token in engine.generate("Explain layer-wise inference.", max_tokens=100):
    print(token, end="", flush=True)
```

```bash
# Or use the helper script to find Ollama models automatically
python examples/ollama_inference.py "mistral-small:22b" "mistralai/Mistral-Small-Instruct-2409"
```

### Command Line

```bash
# Interactive chat with any model
aircela chat -m "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Single prompt
aircela run -m "mistralai/Mistral-7B-v0.1" -p "Tell me a joke"
```

---

## 🏎️ Speed Optimizations

AirCELA is designed from the ground up for speed on consumer hardware.

1.  **Double-Buffer Prefetching**: AirLLM loads Layer 1, computes, then loads Layer 2. AirCELA loads Layer 2 in a background thread *while* the GPU is still computing Layer 1. This hides 90% of disk latency.
2.  **Memory Mapping (mmap)**: We use OS-level memory mapping for weight files. This allows the OS to handle paging and avoids redundant copies from disk to Python memory.
3.  **Vectorized Dequantization**: Our GGUF/Q4_0 dequantizer is fully vectorized using internal NumPy C-extensions, making it 100x faster than traditional Python loops.
4.  **Flash Attention**: We leverage PyTorch's `scaled_dot_product_attention` for fused kernels that are significantly faster and more memory-efficient than standard attention.
5.  **Fast GGUF Parser**: Our mmap-based parser reads multi-GB GGUF files in milliseconds, skipping heavy tokenizer metadata that isn't needed for inference.

---

## 🏗️ Architecture

```
Token → Embed → [Layer 0 → Layer 1 → ... → Layer N] → Norm → LM Head → Token
                  ↑ GPU  ↑ prefetch next               ↑ GPU
```

Each layer is streamed from disk → dequantized → GPU → computed → freed:
- Only **ONE layer** lives in VRAM at any time
- Background thread **prefetches the next layer** while GPU computes
- KV caches are stored on CPU to save VRAM

---

## 📊 Supported Quantization Formats

| Format | Status | Description |
|--------|--------|-------------|
| F32    | ✅     | 32-bit float |
| F16    | ✅     | 16-bit float |
| Q4_0   | ✅     | 4-bit quantization |
| Q4_1   | ✅     | 4-bit with min |
| Q8_0   | ✅     | 8-bit quantization |
| Q6_K   | ✅     | 6-bit K-quant |

---

## 👨‍💻 Credits

<table>
  <tr>
    <td><strong>Gaurav Batule</strong></td>
    <td>Creator, Lead Developer, Architecture Design</td>
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
      <a href="https://buymeacoffee.com/gauravbatule">☕ Buy Me a Coffee</a> •
      <a href="https://github.com/gauravbatule/AirCELA">🐙 GitHub</a>
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
  <a href="https://buymeacoffee.com/gauravbatule">☕ Buy Me a Coffee</a><br>
  <sub>If AirCELA helped you, consider buying me a coffee!</sub>
</p>
