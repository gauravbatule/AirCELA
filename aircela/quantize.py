"""
CELA Dequantizer — Convert quantized GGUF weights to float tensors.

Supports Q4_0, Q4_1, Q8_0, F16, F32 quantization formats.
Vectorized with NumPy for speed; future versions will use CUDA kernels.

Developed by Gaurav Batule | 🤖 AI-assisted vibe code
"""

import struct
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class Dequantizer:
    """
    Dequantize GGUF tensor data to PyTorch float16 tensors.

    Supported formats: F32, F16, Q4_0, Q4_1, Q8_0.

    Usage::

        raw = reader.load_tensor_data("blk.0.attn_q.weight")
        tensor = Dequantizer.dequantize(raw, shape=(4096, 4096), qtype=2)
    """

    @staticmethod
    def dequantize(raw_data: bytes, shape: tuple, qtype: int) -> "torch.Tensor":
        """
        Dequantize raw GGUF data to a float16 PyTorch tensor.

        Args:
            raw_data: Raw bytes from GGUFReader.load_tensor_data()
            shape:    Original tensor shape from GGUF metadata
            qtype:    GGUF quantization type ID

        Returns:
            torch.Tensor in float16
        """
        if torch is None:
            raise ImportError("PyTorch is required for dequantization")

        n_elements = 1
        for s in shape:
            n_elements *= s

        if qtype == 0:    # F32
            arr = np.frombuffer(raw_data, dtype=np.float32).copy()
            return torch.from_numpy(arr).to(torch.float16).reshape(shape)

        elif qtype == 1:  # F16
            arr = np.frombuffer(raw_data, dtype=np.float16).copy()
            return torch.from_numpy(arr).reshape(shape)

        elif qtype == 2:  # Q4_0 — 32 values per block, 18 bytes/block
            return Dequantizer._dequant_q4_0(raw_data, n_elements, shape)

        elif qtype == 3:  # Q4_1 — 32 values per block, 20 bytes/block
            return Dequantizer._dequant_q4_1(raw_data, n_elements, shape)

        elif qtype == 8:  # Q8_0 — 32 values per block, 34 bytes/block
            return Dequantizer._dequant_q8_0(raw_data, n_elements, shape)

        else:
            # Unsupported format — return zeros with a warning
            import warnings
            warnings.warn(f"Unsupported GGUF qtype {qtype}, returning zeros")
            return torch.zeros(shape, dtype=torch.float16)

    # ------------------------------------------------------------------
    #  Q4_0: 32 values per block, 2 bytes scale + 16 bytes data = 18 bytes
    # ------------------------------------------------------------------
    @staticmethod
    def _dequant_q4_0(raw: bytes, n_elements: int, shape: tuple) -> "torch.Tensor":
        block_size = 18
        n_blocks = len(raw) // block_size
        data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, block_size)

        # Scale is stored as float16 in first 2 bytes
        scales = data[:, :2].copy().view(np.float16).astype(np.float32).flatten()

        # Quantized nibbles in bytes 2..17 (16 bytes = 32 nibbles)
        qs = data[:, 2:]  # (n_blocks, 16)
        low = (qs & 0x0F).astype(np.float32) - 8.0
        high = ((qs >> 4) & 0x0F).astype(np.float32) - 8.0

        # Interleave low/high nibbles
        values = np.empty((n_blocks, 32), dtype=np.float32)
        values[:, :16] = low
        values[:, 16:] = high

        # Apply scale
        values *= scales[:, np.newaxis]

        result = values.flatten()[:n_elements]
        return torch.from_numpy(result).to(torch.float16).reshape(shape)

    # ------------------------------------------------------------------
    #  Q4_1: 32 values per block, 2+2 bytes (scale+min) + 16 bytes = 20 bytes
    # ------------------------------------------------------------------
    @staticmethod
    def _dequant_q4_1(raw: bytes, n_elements: int, shape: tuple) -> "torch.Tensor":
        block_size = 20
        n_blocks = len(raw) // block_size
        data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, block_size)

        scales = data[:, :2].copy().view(np.float16).astype(np.float32).flatten()
        mins = data[:, 2:4].copy().view(np.float16).astype(np.float32).flatten()
        qs = data[:, 4:]

        low = (qs & 0x0F).astype(np.float32)
        high = ((qs >> 4) & 0x0F).astype(np.float32)

        values = np.empty((n_blocks, 32), dtype=np.float32)
        values[:, :16] = low
        values[:, 16:] = high
        values = values * scales[:, np.newaxis] + mins[:, np.newaxis]

        result = values.flatten()[:n_elements]
        return torch.from_numpy(result).to(torch.float16).reshape(shape)

    # ------------------------------------------------------------------
    #  Q8_0: 32 values per block, 2 bytes scale + 32 bytes data = 34 bytes
    # ------------------------------------------------------------------
    @staticmethod
    def _dequant_q8_0(raw: bytes, n_elements: int, shape: tuple) -> "torch.Tensor":
        block_size = 34
        n_blocks = len(raw) // block_size
        data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, block_size)

        scales = data[:, :2].copy().view(np.float16).astype(np.float32).flatten()
        qs = data[:, 2:].view(np.int8).astype(np.float32)

        values = qs * scales[:, np.newaxis]
        result = values.flatten()[:n_elements]
        return torch.from_numpy(result).to(torch.float16).reshape(shape)
