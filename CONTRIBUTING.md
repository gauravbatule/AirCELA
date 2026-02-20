# 🤝 Contributing to AirCELA

Thank you for your interest in contributing to AirCELA! Whether it's fixing a bug, adding a feature, improving documentation, or sharing feedback — every contribution matters.

---

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Coding Guidelines](#coding-guidelines)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Code of Conduct](#code-of-conduct)

---

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/AirCELA.git
   cd AirCELA
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/gauravbatule/AirCELA.git
   ```
4. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.9+
- PyTorch 2.0+ (with CUDA recommended)
- Git

### Install dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install in editable mode (for development)
pip install -e .
```

### Verify your setup

```bash
# Quick import test (should complete in <1 second)
python -c "from aircela.gguf import GGUFReader; print('OK')"

# Run the basic example
python examples/basic_generation.py
```

---

## 📁 Project Structure

```
AirCELA/
├── aircela/                    # Main package
│   ├── __init__.py             # Lazy imports
│   ├── engine.py               # Core inference engine (CELAEngine)
│   ├── transformer.py          # Transformer layer (RoPE, GQA, Flash Attention)
│   ├── prefetch.py             # Double-buffer layer prefetcher
│   ├── gguf.py                 # GGUF file parser (mmap-based)
│   ├── quantize.py             # Dequantization kernels
│   ├── huggingface.py          # HuggingFace model loader
│   ├── device.py               # Hardware detection
│   └── cli.py                  # CLI interface
├── examples/                   # Example scripts
├── _legacy/                    # Old prototypes (do not modify)
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Key modules

| Module | Purpose | Modify when... |
|--------|---------|----------------|
| `engine.py` | Core engine orchestration | Adding new model formats, generation strategies |
| `transformer.py` | Layer computation | Optimizing attention, adding new architectures |
| `prefetch.py` | I/O pipeline | Improving layer streaming performance |
| `gguf.py` | GGUF parsing | Supporting new GGUF versions or metadata |
| `quantize.py` | Dequantization | Adding new quantization formats (Q5_K, IQ, etc.) |
| `huggingface.py` | HF model loading | Supporting new HF model architectures |
| `device.py` | Hardware detection | Adding new GPU/accelerator support |
| `cli.py` | Command-line tool | Adding new CLI commands |

---

## 💡 How to Contribute

### Good First Issues

If you're new to the project, look for issues labeled **`good first issue`** or **`help wanted`**.

Some areas where contributions are welcome:

- 🐛 **Bug fixes** — Fix any issues you encounter
- 📝 **Documentation** — Improve README, docstrings, or add tutorials
- ⚡ **Performance** — Optimize dequantization, attention, or I/O
- 🧩 **New quant formats** — Add support for Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, IQ formats
- 🏗️ **New architectures** — Add support for Phi, Qwen, Gemma, etc.
- 🧪 **Tests** — Add unit tests for any module
- 🖥️ **Platform support** — Improve Windows/macOS/Linux compatibility

---

## 📏 Coding Guidelines

### Style

- **Python 3.9+** compatible code
- Follow **PEP 8** conventions
- Use **type hints** where practical
- Use **docstrings** for all public methods
- Keep functions focused and under 50 lines when possible

### Naming

- Classes: `PascalCase` (e.g., `CELAEngine`, `GGUFReader`)
- Functions/methods: `snake_case` (e.g., `load_layer`, `get_model_config`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `GGUF_MAGIC`, `QTYPE_F16`)
- Private methods: prefix with `_` (e.g., `_parse`, `_read_value_fast`)

### Performance

- **Always profile** before and after optimizations
- Prefer **NumPy vectorized operations** over Python loops for dequantization
- Use **mmap** for file I/O when possible
- Keep **lazy imports** — don't import `torch` at module level unless the module requires it
- Memory matters — free large tensors with `del` when no longer needed

### Testing

- Test your changes with at least one small model (e.g., TinyLlama 1.1B)
- If adding a new quantization format, verify output against `llama.cpp` reference
- If modifying the engine, test both HuggingFace and GGUF loading paths

---

## 📝 Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `perf` | Performance improvement |
| `refactor` | Code refactoring (no behavior change) |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks, dependency updates |

### Examples

```
feat(quantize): add Q5_K dequantization support
fix(engine): handle missing lm_head weight in GGUF models
docs(readme): add troubleshooting section for Windows users
perf(gguf): reduce parse time with mmap-based metadata skip
```

---

## 🔀 Pull Request Process

1. **Update your branch** with the latest `main`:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Test your changes** thoroughly

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub with:
   - A clear title following the commit message format
   - Description of what changed and why
   - Any relevant issue numbers (e.g., "Fixes #12")
   - Screenshots or benchmarks if applicable

5. **Address review feedback** — the maintainer may request changes

### PR Checklist

- [ ] My code follows the project's coding guidelines
- [ ] I have tested my changes with a real model
- [ ] I have added/updated docstrings for new public methods
- [ ] My commit messages follow the conventional format
- [ ] I have not introduced unnecessary dependencies

---

## 🐛 Reporting Bugs

[Open an issue](https://github.com/gauravbatule/AirCELA/issues/new?template=bug_report.md) with:

1. **Description** — What happened?
2. **Expected behavior** — What should have happened?
3. **Steps to reproduce** — Exact commands/code to trigger the bug
4. **Environment** — OS, Python version, GPU, PyTorch version
5. **Error output** — Full traceback if available

---

## 💡 Suggesting Features

[Open an issue](https://github.com/gauravbatule/AirCELA/issues/new?template=feature_request.md) with:

1. **Problem** — What limitation are you facing?
2. **Proposed solution** — How would you like it solved?
3. **Alternatives** — Any workarounds you've considered?
4. **Use case** — Why is this important?

---

## 📜 Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold a welcoming and respectful environment.

---

## 🙏 Thank You!

Every contribution, no matter how small, helps make AirCELA better. Whether it's a typo fix in docs or a new quantization kernel — we appreciate your time and effort.

**Questions?** Open an issue or reach out:
- GitHub: [@gauravbatule](https://github.com/gauravbatule)
- Email: gauravbatule@gmail.com
- LinkedIn: [Gaurav Batule](https://www.linkedin.com/in/gaurav-batule/)
