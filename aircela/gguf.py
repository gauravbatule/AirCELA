"""
CELA GGUF Reader — Parse GGUF model files (Ollama / llama.cpp).

Reads GGUF v2/v3 headers, extracts tensor metadata and raw data
for on-the-fly dequantization. Supports models from Ollama's
blob storage.

FAST: Uses mmap + bulk reads to parse multi-GB files in seconds.

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

    Uses mmap for fast parsing of multi-GB files.

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
    #  Fast Parsing using mmap
    # ------------------------------------------------------------------
    def _parse(self):
        """Read GGUF header and build tensor index using mmap for speed."""
        file_size = self.path.stat().st_size
        
        with open(self.path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            pos = 0
            
            # Header
            magic = struct.unpack_from("<I", mm, pos)[0]; pos += 4
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: bad magic 0x{magic:08X}")

            version = struct.unpack_from("<I", mm, pos)[0]; pos += 4
            n_tensors = struct.unpack_from("<Q", mm, pos)[0]; pos += 8
            n_metadata = struct.unpack_from("<Q", mm, pos)[0]; pos += 8

            # Read metadata (FAST: skip large arrays we don't need)
            for _ in range(n_metadata):
                key, pos = self._read_string_fast(mm, pos)
                vtype = struct.unpack_from("<I", mm, pos)[0]; pos += 4
                value, pos = self._read_value_fast(mm, pos, vtype, key)
                if value is not None:  # Only store non-skipped values
                    self.metadata[key] = value

            # Read tensor info
            for _ in range(n_tensors):
                name, pos = self._read_string_fast(mm, pos)
                n_dims = struct.unpack_from("<I", mm, pos)[0]; pos += 4
                shape = tuple(struct.unpack_from("<Q", mm, pos + i*8)[0] for i in range(n_dims))
                pos += n_dims * 8
                qtype = struct.unpack_from("<I", mm, pos)[0]; pos += 4
                offset = struct.unpack_from("<Q", mm, pos)[0]; pos += 8

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
            self._data_offset = pos + (32 - pos % 32) % 32
            mm.close()

    # ------------------------------------------------------------------
    #  Fast low-level readers (mmap-based)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_string_fast(mm, pos):
        length = struct.unpack_from("<Q", mm, pos)[0]; pos += 8
        s = mm[pos:pos+length].decode("utf-8", errors="replace"); pos += length
        return s, pos

    @staticmethod
    def _read_value_fast(mm, pos, vtype, key=""):
        """Read a value, but SKIP large arrays (tokenizer vocabs) for speed."""
        if vtype == 0:    # uint8
            v = struct.unpack_from("<B", mm, pos)[0]; pos += 1; return v, pos
        elif vtype == 1:  # int8
            v = struct.unpack_from("<b", mm, pos)[0]; pos += 1; return v, pos
        elif vtype == 2:  # uint16
            v = struct.unpack_from("<H", mm, pos)[0]; pos += 2; return v, pos
        elif vtype == 3:  # int16
            v = struct.unpack_from("<h", mm, pos)[0]; pos += 2; return v, pos
        elif vtype == 4:  # uint32
            v = struct.unpack_from("<I", mm, pos)[0]; pos += 4; return v, pos
        elif vtype == 5:  # int32
            v = struct.unpack_from("<i", mm, pos)[0]; pos += 4; return v, pos
        elif vtype == 6:  # float32
            v = struct.unpack_from("<f", mm, pos)[0]; pos += 4; return v, pos
        elif vtype == 7:  # bool
            v = struct.unpack_from("<?", mm, pos)[0]; pos += 1; return v, pos
        elif vtype == 8:  # string
            return GGUFReader._read_string_fast(mm, pos)
        elif vtype == 9:  # array
            atype = struct.unpack_from("<I", mm, pos)[0]; pos += 4
            alen = struct.unpack_from("<Q", mm, pos)[0]; pos += 8
            
            # SKIP large arrays (tokenizer vocabs can have 32k+ entries)
            # These are not needed for inference — we use HF tokenizers
            if alen > 1000:
                # Fast-skip: calculate byte size and jump
                pos = GGUFReader._skip_array_fast(mm, pos, atype, alen)
                return None, pos
            
            result = []
            for _ in range(alen):
                v, pos = GGUFReader._read_value_fast(mm, pos, atype)
                result.append(v)
            return result, pos
        elif vtype == 10:  # uint64
            v = struct.unpack_from("<Q", mm, pos)[0]; pos += 8; return v, pos
        elif vtype == 11:  # int64
            v = struct.unpack_from("<q", mm, pos)[0]; pos += 8; return v, pos
        elif vtype == 12:  # float64
            v = struct.unpack_from("<d", mm, pos)[0]; pos += 8; return v, pos
        else:
            return None, pos

    @staticmethod
    def _skip_array_fast(mm, pos, atype, alen):
        """Skip over an array without reading its contents."""
        # Fixed-size types can be skipped immediately
        fixed_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if atype in fixed_sizes:
            return pos + alen * fixed_sizes[atype]
        
        # String arrays: must walk through (each has 8-byte len + data)
        if atype == 8:
            for _ in range(alen):
                slen = struct.unpack_from("<Q", mm, pos)[0]; pos += 8
                pos += slen
            return pos
        
        # Nested arrays or unknown: walk element by element
        for _ in range(alen):
            _, pos = GGUFReader._read_value_fast(mm, pos, atype)
        return pos

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def load_tensor_data(self, name: str) -> bytes:
        """Load raw bytes for a tensor from the GGUF file using mmap for speed."""
        info = self.tensors[name]
        offset = self._data_offset + info["offset"]
        
        with open(self.path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return bytes(mm[offset : offset + info["size_bytes"]])

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
            # Try specific architecture-prefixed keys first
            arch = self.metadata.get("general.architecture", "llama")
            specific_key = f"{arch}.{key}"
            if specific_key in self.metadata:
                return self.metadata[specific_key]
            
            # fallback to loose search but be careful about head_count vs head_count_kv
            for k, v in self.metadata.items():
                if key in k:
                    # If looking for head_count, don't match head_count_kv
                    if key == "attention.head_count" and "head_count_kv" in k:
                        continue
                    return v
            return default

        arch = self.metadata.get("general.architecture", "unknown")
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
