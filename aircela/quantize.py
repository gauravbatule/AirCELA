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

        """
        if torch is None:
            raise ImportError("PyTorch is required for dequantization")
            
        # print(f"  Dequantizing: shape={shape} qtype={qtype}")

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

        elif qtype == 14: # Q6_K — 256 values per block, 210 bytes/block
            return Dequantizer._dequant_q6_k(raw_data, n_elements, shape)

        else:
            # Unsupported format — return zeros with a warning
            import warnings
            warnings.warn(f"Unsupported GGUF qtype {qtype}, returning zeros")
            return torch.zeros(shape, dtype=torch.float16)

    # ------------------------------------------------------------------
    #  Q6_K: 256 values per block, 210 bytes/block
    # ------------------------------------------------------------------
    @staticmethod
    def _dequant_q6_k(raw: bytes, n_elements: int, shape: tuple) -> "torch.Tensor":
        block_size = 210
        n_blocks = len(raw) // block_size
        data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, block_size)

        # ql: 4-bit part (128 bytes -> 256 nibbles)
        ql = data[:, :128]
        # qh: 2-bit part (64 bytes -> 256 bits)
        qh = data[:, 128:192]
        # scales: 8-bit part (16 bytes)
        sc = data[:, 192:208].copy().view(np.int8).astype(np.float32)
        # d: global scale (2 bytes float16)
        d = data[:, 208:210].copy().view(np.float16).astype(np.float32)

        # Reconstruct 6-bit values
        # This is a simplified vectorized version
        values = np.empty((n_blocks, 256), dtype=np.float32)
        
        for i in range(64):
            # Each byte of qh provides 2 bits for 4 different elements
            # Elements: i, i+64, i+128, i+192
            h = qh[:, i]
            
            # Element i
            l0 = ql[:, i] & 0xF
            h0 = (h & 0x03) << 4
            values[:, i] = (l0 | h0).astype(np.float32) - 32
            
            # Element i+64
            l1 = ql[:, i+64] & 0xF
            h1 = (h & 0x0C) << 2
            values[:, i+64] = (l1 | h1).astype(np.float32) - 32
            
            # Element i+128
            l2 = ql[:, i] >> 4
            h2 = (h & 0x30)
            values[:, i+128] = (l2 | h2).astype(np.float32) - 32
            
            # Element i+192
            l3 = ql[:, i+64] >> 4
            h3 = (h & 0xC0) >> 2
            values[:, i+192] = (l3 | h3).astype(np.float32) - 32

        # Apply sub-scales and global scale
        # values is (n_blocks, 256), sc is (n_blocks, 16)
        # Reshape to use broadcasting: (n_blocks, 16, 16) * (n_blocks, 16, 1)
        values = values.reshape(n_blocks, 16, 16)
        values *= sc[:, :, np.newaxis]
        values *= d[:, np.newaxis]
        
        result = values.ravel()[:n_elements]
        return torch.from_numpy(result).to(torch.float16).reshape(shape)

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
        
        # Allocate output buffer directly
        values = np.empty((n_blocks, 32), dtype=np.float32)
        
        # Write directly into buffer to save memory
        # low nibbles
        values[:, :16] = (qs & 0x0F).astype(np.float32)
        values[:, :16] -= 8.0
        
        # high nibbles
        values[:, 16:] = ((qs >> 4) & 0x0F).astype(np.float32)
        values[:, 16:] -= 8.0

        # Apply scale in-place
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
