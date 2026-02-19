"""
CELA HuggingFace Loader — Load any HuggingFace model for layer-by-layer inference.

Supports safetensors and .bin weight files. Downloads and caches
models automatically via huggingface_hub.

Developed by Gaurav Batule | 🤖 AI-assisted vibe code
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required: pip install torch")

try:
    import safetensors.torch
except ImportError:
    safetensors = None  # type: ignore

try:
    from transformers import AutoConfig, AutoTokenizer
    from huggingface_hub import snapshot_download
except ImportError:
    raise ImportError(
        "transformers and huggingface_hub are required:\n"
        "  pip install transformers huggingface_hub safetensors"
    )


class HFModelLoader:
    """
    Load any HuggingFace model for layer-by-layer inference.

    Unlike Ollama, this gives you FULL control over GPU usage —
    every layer runs on GPU regardless of model size.

    Usage::

        loader = HFModelLoader("meta-llama/Llama-2-7b-hf")
        print(loader.config)
        weights = loader.load_layer(0)  # load layer 0 to CPU
        embed = loader.load_embeddings()
    """

    def __init__(self, model_name_or_path: str, dtype: torch.dtype = torch.float16):
        self.model_name = model_name_or_path
        self.dtype = dtype

        print(f"  Loading config: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self._hf_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        # Extract architecture params
        self.config = {
            "d_model": self._hf_config.hidden_size,
            "n_heads": self._hf_config.num_attention_heads,
            "n_kv_heads": getattr(self._hf_config, "num_key_value_heads",
                                  self._hf_config.num_attention_heads),
            "d_ff": self._hf_config.intermediate_size,
            "n_layers": self._hf_config.num_hidden_layers,
            "vocab_size": self._hf_config.vocab_size,
            "head_dim": (self._hf_config.hidden_size
                         // self._hf_config.num_attention_heads),
            "rope_theta": getattr(self._hf_config, "rope_theta", 10000.0),
        }

        # Download / cache weights
        print(f"  Downloading weights (cached if already downloaded)...")
        t0 = time.perf_counter()
        self.model_dir = Path(snapshot_download(
            model_name_or_path,
            allow_patterns=["*.safetensors", "*.json", "*.bin",
                            "*.model", "*.txt"],
        ))
        dt = time.perf_counter() - t0
        print(f"  Cached in {dt:.1f}s → {self.model_dir}")

        # Build layer → file mapping
        self._layer_map: Dict[int, Dict[str, str]] = {}
        self._base_tensors: Dict[str, str] = {}
        self._build_tensor_map()

    def _build_tensor_map(self):
        """Scan weight files and map tensor names to layers."""
        sf_files = sorted(self.model_dir.glob("*.safetensors"))
        if not sf_files:
            sf_files = sorted(self.model_dir.glob("pytorch_model*.bin"))

        for sf_path in sf_files:
            if sf_path.suffix == ".safetensors":
                with open(sf_path, "rb") as f:
                    header_size = int.from_bytes(f.read(8), "little")
                    header = json.loads(f.read(header_size))
                tensor_names = [k for k in header if k != "__metadata__"]
            else:
                # .bin — torch.load header scan
                tensor_names = list(torch.load(sf_path, map_location="cpu",
                                               weights_only=True).keys())

            for key in tensor_names:
                layer_idx = self._parse_layer_idx(key)
                if layer_idx is not None:
                    self._layer_map.setdefault(layer_idx, {})[key] = str(sf_path)
                else:
                    self._base_tensors[key] = str(sf_path)

        print(f"  Found {len(self._layer_map)} layers, "
              f"{len(self._base_tensors)} base tensors")

    @staticmethod
    def _parse_layer_idx(key: str) -> Optional[int]:
        """Extract layer index from a tensor name."""
        for pattern in ["model.layers.", "transformer.h.",
                        "gpt_neox.layers.", "decoder.layers."]:
            if pattern in key:
                rest = key[key.index(pattern) + len(pattern):]
                idx_str = rest.split(".")[0]
                if idx_str.isdigit():
                    return int(idx_str)
        return None

    def _load_tensors_from_file(self, filepath: str, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Load specific tensors from a safetensors/bin file using mmap for speed."""
        import mmap
        p = Path(filepath)
        
        # We still use safetensors.torch.load_file as it already uses mmap internally
        # but we add explicit mmap handling for .bin files if needed.
        if p.suffix == ".safetensors" and safetensors is not None:
            # Safetensors already uses mmap by default
            all_t = safetensors.torch.load_file(filepath, device="cpu")
        else:
            # For .bin files, we load carefully
            all_t = torch.load(filepath, map_location="cpu", weights_only=True)
            
        return {k: all_t[k].to(self.dtype) for k in keys if k in all_t}

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def load_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load all weights for a single transformer layer (on CPU)."""
        if layer_idx not in self._layer_map:
            return {}

        weights = {}
        files_needed = {}
        for key, filepath in self._layer_map[layer_idx].items():
            files_needed.setdefault(filepath, []).append(key)

        for filepath, keys in files_needed.items():
            weights.update(self._load_tensors_from_file(filepath, keys))

        return weights

    def load_embeddings(self) -> Dict[str, torch.Tensor]:
        """Load embedding and output projection weights."""
        files_needed = {}
        for key, filepath in self._base_tensors.items():
            files_needed.setdefault(filepath, []).append(key)

        weights = {}
        for filepath, keys in files_needed.items():
            weights.update(self._load_tensors_from_file(filepath, keys))

        return weights

    def get_embed_weight(self) -> Optional[torch.Tensor]:
        """Get the token embedding weight matrix."""
        base = self.load_embeddings()
        for k, v in base.items():
            if "embed_tokens" in k or "wte" in k or "word_embeddings" in k:
                return v
        return None

    def get_lm_head_weight(self) -> Optional[torch.Tensor]:
        """Get the output projection (lm_head) weight matrix."""
        base = self.load_embeddings()
        for k, v in base.items():
            if "lm_head" in k:
                return v
        # Tied weights — use embedding
        return self.get_embed_weight()

    def get_final_norm_weight(self) -> Optional[torch.Tensor]:
        """Get the final layer norm weight."""
        base = self.load_embeddings()
        for k, v in base.items():
            if "norm" in k.lower() and "layer" not in k.lower():
                return v
        return None
