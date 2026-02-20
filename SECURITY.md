# 🔒 Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Active |

## Reporting a Vulnerability

If you discover a security vulnerability in AirCELA, **please report it responsibly**.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. **Email** the maintainer directly at: **gauravbatule@gmail.com**
3. Include the following in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment** within 48 hours
- **Assessment** within 1 week
- **Fix and disclosure** coordinated with the reporter

### Scope

Security concerns for AirCELA primarily relate to:

- **Arbitrary code execution** through malicious GGUF or model files
- **Path traversal** vulnerabilities in file loading
- **Memory safety** issues during dequantization or mmap operations
- **Dependency vulnerabilities** in PyTorch, NumPy, transformers, etc.

### Out of Scope

- Model output quality or accuracy
- Performance on specific hardware
- Features not yet implemented

## Best Practices for Users

1. **Only load models from trusted sources** (HuggingFace Hub, Ollama official registry)
2. **Keep dependencies updated** — run `pip install -r requirements.txt --upgrade` regularly
3. **Do not expose AirCELA as a public-facing service** without proper input validation

## Thank You

We appreciate responsible disclosure. Security researchers who report valid vulnerabilities
will be credited in the project's release notes (unless they prefer to remain anonymous).
