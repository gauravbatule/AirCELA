"""
CELA Engine — Main inference engine for layer-by-layer LLM generation.

This is the core of CELA: load ANY model (HuggingFace or GGUF),
stream layers through GPU one at a time, generate text at speed.

Architecture:
    Token → Embed → [Layer 0 → Layer 1 → ... → Layer N] → Norm → LM Head → Token
                      ↑ GPU  ↑ prefetch                    ↑ GPU

Each layer: Load from disk/cache → GPU compute → Free VRAM → Next layer

Developed by Gaurav Batule | 🤖 AI-assisted vibe code
"""

import sys
import time
import gc
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from aircela.device import DeviceManager
from aircela.transformer import CELATransformerLayer, RMSNorm, RotaryEmbedding
from aircela.prefetch import LayerPrefetcher


class CELAEngine:
    """
    Main CELA inference engine.

    Run any LLM on consumer hardware by streaming layers through GPU one at a time.

    Why First Token Is Slow (TTFT):
        The first token requires the ENTIRE prompt to pass through ALL layers.
        This is called "prefill" — it builds the KV cache that all future tokens reuse.
        After prefill, each new token only processes ONE token through the layers
        (the "decode" phase), which is much faster.

        Additionally, on first use, the model weights must be loaded from disk
        into the HuggingFace cache (download) and then mmap'd for reading.

    Supports:
        - HuggingFace models (safetensors): ``CELAEngine.from_pretrained("model/name")``
        - GGUF models (Ollama): ``CELAEngine.from_gguf("model.gguf")``

    Usage::

        engine = CELAEngine.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        for token in engine.generate("What is the capital of France?", max_tokens=50):
            print(token, end="", flush=True)
    """

    def __init__(self):
        self.dm = DeviceManager()
        self.device = self.dm.device
        self.dtype = torch.float16 if self.dm.has_cuda else torch.float32

        # Model components (set by from_pretrained or from_gguf)
        self.config: dict = {}
        self.tokenizer = None
        self.embed_weight: Optional[torch.Tensor] = None
        self.lm_head_weight: Optional[torch.Tensor] = None
        self.final_norm_weight: Optional[torch.Tensor] = None
        self._load_layer_fn: Optional[Callable[[int], Dict[str, torch.Tensor]]] = None

        # Runtime state
        self._layer: Optional[CELATransformerLayer] = None
        self._rope: Optional[RotaryEmbedding] = None
        self._prefetcher: Optional[LayerPrefetcher] = None
        self._kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []

    # ==================================================================
    #  Factory Methods
    # ==================================================================
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "CELAEngine":
        """
        Load a HuggingFace model for layer-by-layer inference.

        This gives you FULL GPU control — unlike Ollama, every layer
        runs on GPU regardless of model size.

        Args:
            model_name: HuggingFace model ID (e.g. "meta-llama/Llama-2-7b-hf")
        """
        from aircela.huggingface import HFModelLoader

        engine = cls()
        loader = HFModelLoader(model_name, dtype=engine.dtype)

        engine.config = loader.config
        engine.tokenizer = loader.tokenizer
        engine.embed_weight = loader.get_embed_weight()
        engine.lm_head_weight = loader.get_lm_head_weight()
        engine.final_norm_weight = loader.get_final_norm_weight()
        engine._load_layer_fn = loader.load_layer

        engine._init_runtime()
        return engine

    @classmethod
    def from_gguf(cls, path: str, tokenizer_id: Optional[str] = None) -> "CELAEngine":
        """
        Load a GGUF model (from Ollama or llama.cpp) for inference.

        Args:
            path:         Path to .gguf file
            tokenizer_id: Optional HF model ID for the tokenizer
                          (e.g. "mistralai/Mistral-7B-v0.1")
        """
        from aircela.gguf import GGUFReader
        from aircela.quantize import Dequantizer
        from transformers import AutoTokenizer

        engine = cls()
        reader = GGUFReader(path)
        config = reader.get_model_config()

        engine.config = config

        if tokenizer_id:
            print(f"  Loading tokenizer: {tokenizer_id}")
            engine.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        else:
            print("  Warning: No tokenizer_id provided for GGUF model.")
            print("  Generation will return raw token IDs.")

        # Load base weights (embed, head, final_norm)
        def find_weight(patterns):
            if isinstance(patterns, str): patterns = [patterns]
            for name in reader.tensors:
                for pattern in patterns:
                    if pattern in name and not name.startswith("blk."):
                        print(f"  Found base weight: {name} for {patterns}")
                        raw = reader.load_tensor_data(name)
                        info = reader.tensors[name]
                        w = Dequantizer.dequantize(raw, info["shape"], info["qtype"])
                        
                        # Fix orientation (Common in GGUF/llama.cpp)
                        # e.g. token_embd: (hidden, vocab) -> (vocab, hidden)
                        if "embd" in name and w.shape[0] == config["d_model"]:
                            w = w.T
                        # head: (hidden, vocab) -> (vocab, hidden)
                        if "output.weight" in name and w.shape[1] == config["d_model"]:
                            pass # PyTorch head expects (vocab, hidden)
                        elif "output.weight" in name and w.shape[0] == config["d_model"]:
                            w = w.T
                        return w
            return None

        engine.embed_weight = find_weight(["token_embd", "wte"])
        engine.lm_head_weight = find_weight(["output.weight", "lm_head"])
        engine.final_norm_weight = find_weight(["output_norm", "final_norm"])

        if engine.embed_weight is None:
            print("  [ERROR] Could not find token embeddings in GGUF!")
        if engine.lm_head_weight is None:
            print("  [ERROR] Could not find LM head in GGUF!")
        if engine.final_norm_weight is None:
            print("  [ERROR] Could not find final norm in GGUF!")

        # Set up layer loader that dequantizes on-the-fly
        def load_layer(idx: int) -> Dict[str, torch.Tensor]:
            tensors = reader.get_layer_tensors(idx)
            weights = {}
            for name, info in tensors.items():
                raw = reader.load_tensor_data(name)
                w = Dequantizer.dequantize(
                    raw, info["shape"], info["qtype"]
                )
                
                # GGUF weights are often (in, out), we want (out, in)
                if len(w.shape) == 2:
                    if any(x in name for x in ["attn_q", "attn_k", "attn_v", "attn_output"]):
                        if w.shape[0] == config["d_model"]:
                            w = w.T
                    elif any(x in name for x in ["ffn_up", "ffn_gate"]):
                        if w.shape[0] == config["d_model"]:
                            w = w.T
                    elif "ffn_down" in name:
                        if w.shape[1] == config["d_model"]:
                            w = w.T
                            
                weights[name] = w
            return weights

        engine._load_layer_fn = load_layer
        engine._init_runtime()
        return engine

    # ==================================================================
    #  Runtime Init
    # ==================================================================
    def _init_runtime(self):
        """Initialize transformer layer, RoPE, prefetcher, KV caches."""
        c = self.config
        n_layers = c.get("n_layers", c.get("n_layers", 32))
        head_dim = c.get("head_dim", c["d_model"] // c["n_heads"])

        self._layer = CELATransformerLayer(
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            n_kv_heads=c["n_kv_heads"],
            d_ff=c["d_ff"],
            head_dim=head_dim,
        )

        self._rope = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=8192,
            base=c.get("rope_theta", 10000.0),
            dtype=self.dtype,
        )

        self._prefetcher = LayerPrefetcher(max_workers=2)
        self._kv_caches = [None] * n_layers

        print(f"  CELA Engine ready: {c['d_model']}d, {c['n_heads']}h, "
              f"{n_layers} layers, {c.get('vocab_size', '?')} vocab")
        print(f"  {self.dm.summary()}")

    # ==================================================================
    #  Generation
    # ==================================================================
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        stream: bool = True,
    ):
        """
        Generate text from a prompt.

        Yields tokens one at a time if stream=True.

        Args:
            prompt:      Input text
            max_tokens:  Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k:       Top-K sampling parameter
            stream:      If True, yield tokens as they are generated

        Yields:
            str: Generated tokens (one at a time)
        """
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer loaded. Use from_pretrained().")
        if self._load_layer_fn is None:
            raise RuntimeError("No model loaded.")

        n_layers = self.config.get("n_layers", 32)
        device = self.device

        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        all_ids = input_ids.clone()
        generated = []

        # Reset KV caches
        self._kv_caches = [None] * n_layers

        t_start = time.perf_counter()

        for step in range(max_tokens):
            t_step = time.perf_counter()

            # Embed
            if step == 0:
                x = F.embedding(all_ids, self.embed_weight).to(device, self.dtype)
                start_pos = 0
            else:
                tok = torch.tensor([[generated[-1]]], dtype=torch.long)
                x = F.embedding(tok, self.embed_weight).to(device, self.dtype)
                start_pos = all_ids.shape[1] - 1

            # Process each layer
            for layer_idx in range(n_layers):
                # Prefetch next layer
                if layer_idx + 1 < n_layers:
                    self._prefetcher.prefetch(layer_idx + 1, self._load_layer_fn)

                # Get current layer weights
                weights = self._prefetcher.get(
                    layer_idx, self._load_layer_fn, device=device
                )

                # Forward through layer
                x, kv = self._layer.forward(
                    x, weights, self._rope, self._kv_caches[layer_idx], start_pos
                )
                self._kv_caches[layer_idx] = kv

                # Free GPU memory
                del weights
                if device == "cuda":
                    torch.cuda.empty_cache()

            # Final norm + LM head
            if self.final_norm_weight is not None:
                x = RMSNorm.forward(x, self.final_norm_weight.to(device))

            logits = x[:, -1:, :] @ self.lm_head_weight.to(device).T
            logits = logits[:, -1, :]

            # Sample
            if temperature <= 0:
                next_id = logits.argmax(-1).item()
            else:
                logits = logits / temperature
                if top_k > 0:
                    top_vals, top_idx = logits.topk(min(top_k, logits.shape[-1]))
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, top_idx, top_vals)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            generated.append(next_id)
            all_ids = torch.cat([all_ids, torch.tensor([[next_id]])], dim=1)

            # Decode and yield
            token_text = self.tokenizer.decode([next_id])

            if step == 0:
                ttft = time.perf_counter() - t_start
                if stream:
                    yield token_text

            elif stream:
                yield token_text

            # EOS check
            if (self.tokenizer.eos_token_id is not None
                    and next_id == self.tokenizer.eos_token_id):
                break

        # Stats
        elapsed = time.perf_counter() - t_start
        n_tok = len(generated)
        tps = n_tok / elapsed if elapsed > 0 else 0

        if not stream:
            yield self.tokenizer.decode(generated, skip_special_tokens=True)

        # Print stats
        print(f"\n  [{tps:.1f} tok/s | {n_tok} tokens | TTFT {ttft:.1f}s | "
              f"Total {elapsed:.1f}s]")
        self._prefetcher.print_stats()

    def reset(self):
        """Reset KV caches for a new conversation."""
        n_layers = self.config.get("n_layers", 32)
        self._kv_caches = [None] * n_layers
        self._prefetcher.clear()
        if self.dm.has_cuda:
            torch.cuda.empty_cache()
        gc.collect()
