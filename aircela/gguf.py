"""
CELA GGUF Reader — Parse GGUF model files (Ollama / llama.cpp).

Reads GGUF v2/v3 headers, extracts tensor metadata and raw data
for on-the-fly dequantization. Supports models from Ollama's
blob storage.

Developed by Gaurav Batule | 🤖 AI-assisted vibe code
"""

import os
import sys
import struct
import mmap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# GGUF constants
GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian
GGUF_TYPES = {
    0: ("F32",  4), 1: ("F16",  2), 2: ("Q4_0", 0), 3: ("Q4_1", 0),
    6: ("Q5_0", 0), 7: ("Q5_1", 0), 8: ("Q8_0", 0), 9: ("Q8_1", 0),
    10: ("Q2_K", 0), 11: ("Q3_K", 0), 12: ("Q4_K", 0),
    13: ("Q5_K", 0), 14: ("Q6_K", 0), 15: ("IQ2_XXS", 0),
}


class GGUFReader:
    """
    Parse a GGUF file and provide access to tensor data.

    Usage::

        reader = GGUFReader("model.gguf")
        print(reader.summary())
        layer_data = reader.get_layer_tensors(0)
        raw = reader.load_tensor_data("blk.0.attn_q.weight")
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.metadata: Dict[str, object] = {}
        self.tensors: Dict[str, dict] = {}
        self._data_offset: int = 0
        self._parse()

    # ------------------------------------------------------------------
    #  Parsing
    # ------------------------------------------------------------------
    def _parse(self):
        """Read GGUF header and build tensor index."""
        with open(self.path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: bad magic 0x{magic:08X}")

            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_metadata = struct.unpack("<Q", f.read(8))[0]

            # Read metadata key-value pairs
            for _ in range(n_metadata):
                key = self._read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                value = self._read_value(f, vtype)
                self.metadata[key] = value

            # Read tensor info
            for _ in range(n_tensors):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
                qtype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]

                n_elements = 1
                for s in shape:
                    n_elements *= s

                # Compute size in bytes
                type_name, bpe = GGUF_TYPES.get(qtype, (f"unknown_{qtype}", 2))
                if qtype == 2:       # Q4_0
                    size_bytes = (n_elements // 32) * 18
                elif qtype == 3:     # Q4_1
                    size_bytes = (n_elements // 32) * 20
                elif qtype == 8:     # Q8_0
                    size_bytes = (n_elements // 32) * 34
                elif qtype == 12:    # Q4_K
                    size_bytes = (n_elements // 256) * 144
                elif qtype == 14:    # Q6_K
                    size_bytes = (n_elements // 256) * 210
                elif bpe > 0:
                    size_bytes = n_elements * bpe
                else:
                    size_bytes = n_elements * 2  # fallback

                self.tensors[name] = {
                    "shape": shape,
                    "qtype": qtype,
                    "type_name": type_name,
                    "offset": offset,
                    "n_elements": n_elements,
                    "size_bytes": size_bytes,
                }

            # Data starts at next 32-byte aligned boundary
            pos = f.tell()
            self._data_offset = pos + (32 - pos % 32) % 32

    # ------------------------------------------------------------------
    #  Low-level readers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_string(f) -> str:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")

    @staticmethod
    def _read_value(f, vtype):
        if vtype == 0:   return struct.unpack("<B", f.read(1))[0]      # uint8
        elif vtype == 1: return struct.unpack("<b", f.read(1))[0]      # int8
        elif vtype == 2: return struct.unpack("<H", f.read(2))[0]      # uint16
        elif vtype == 3: return struct.unpack("<h", f.read(2))[0]      # int16
        elif vtype == 4: return struct.unpack("<I", f.read(4))[0]      # uint32
        elif vtype == 5: return struct.unpack("<i", f.read(4))[0]      # int32
        elif vtype == 6: return struct.unpack("<f", f.read(4))[0]      # float32
        elif vtype == 7: return struct.unpack("<?", f.read(1))[0]      # bool
        elif vtype == 8:                                                # string
            return GGUFReader._read_string(f)
        elif vtype == 9:                                                # array
            atype = struct.unpack("<I", f.read(4))[0]
            alen = struct.unpack("<Q", f.read(8))[0]
            return [GGUFReader._read_value(f, atype) for _ in range(alen)]
        elif vtype == 10: return struct.unpack("<Q", f.read(8))[0]     # uint64
        elif vtype == 11: return struct.unpack("<q", f.read(8))[0]     # int64
        elif vtype == 12: return struct.unpack("<d", f.read(8))[0]     # float64
        else:
            return None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def load_tensor_data(self, name: str) -> bytes:
        """Load raw bytes for a tensor from the GGUF file using mmap for speed."""
        info = self.tensors[name]
        offset = self._data_offset + info["offset"]
        
        with open(self.path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return mm[offset : offset + info["size_bytes"]]

    def get_n_layers(self) -> int:
        """Return the number of transformer layers."""
        layer_indices = set()
        for name in self.tensors:
            if name.startswith("blk."):
                idx_str = name.split(".")[1]
                if idx_str.isdigit():
                    layer_indices.add(int(idx_str))
        return len(layer_indices) if layer_indices else 0

    def get_layer_tensors(self, layer_idx: int) -> Dict[str, dict]:
        """Return tensor info for a specific layer."""
        prefix = f"blk.{layer_idx}."
        return {k: v for k, v in self.tensors.items() if k.startswith(prefix)}

    def get_layer_size_bytes(self) -> int:
        """Estimate the size of one layer in bytes."""
        layer0 = self.get_layer_tensors(0)
        return sum(t["size_bytes"] for t in layer0.values()) if layer0 else 0

    def get_model_config(self) -> dict:
        """Extract model architecture config from metadata."""
        def _m(key, default=None):
            for k, v in self.metadata.items():
                if key in k:
                    return v
            return default

        arch = _m("general.architecture", "unknown")
        return {
            "architecture": arch,
            "d_model": _m("embedding_length", 4096),
            "n_heads": _m("attention.head_count", 32),
            "n_kv_heads": _m("attention.head_count_kv", 8),
            "d_ff": _m("feed_forward_length", 14336),
            "n_layers": _m("block_count", self.get_n_layers()),
            "vocab_size": _m("vocab_size", 32000),
            "context_length": _m("context_length", 4096),
            "rope_freq_base": _m("rope.freq_base", 10000.0),
        }

    def summary(self) -> str:
        config = self.get_model_config()
        n_layers = self.get_n_layers()
        layer_size = self.get_layer_size_bytes()
        total_size = self.path.stat().st_size
        return "\n".join([
            f"  GGUF Model: {self.path.name}",
            f"  Architecture: {config['architecture']}",
            f"  Params: d={config['d_model']} h={config['n_heads']} kv={config['n_kv_heads']}",
            f"  Layers: {n_layers} x {layer_size/1024**2:.0f} MB = {total_size/1024**3:.1f} GB",
            f"  Vocab: {config['vocab_size']}",
            f"  Context: {config['context_length']}",
            f"  Tensors: {len(self.tensors)}",
        ])
