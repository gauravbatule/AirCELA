"""
CELA Transformer — Layer-by-layer transformer with RoPE, GQA, Flash Attention.

Implements the core transformer computation that runs on each layer:
    RMSNorm → Attention (w/ RoPE, GQA, KV cache) → RMSNorm → SwiGLU FFN

Each layer is loaded to GPU, computed, and freed — only ONE layer in VRAM
at a time. This allows running models far larger than available VRAM.

Developed by Gaurav Batule | 🤖 AI-assisted vibe code
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    @staticmethod
    def forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


class RotaryEmbedding:
    """Precomputed Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0,
                 dtype: torch.dtype = torch.float16):
        self.dim = dim
        self.max_seq_len = max_seq_len

        pos = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
        self.cos = torch.cos(angles).to(dtype)
        self.sin = torch.sin(angles).to(dtype)

    def apply(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply rotary embeddings to Q or K tensor."""
        T = x.shape[2]
        cos = self.cos[start_pos:start_pos + T].to(x.device)
        sin = self.sin[start_pos:start_pos + T].to(x.device)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)


class CELATransformerLayer:
    """
    Single transformer layer with layer-by-layer GPU execution.

    Loads weights to GPU, computes attention + FFN, then frees GPU memory.
    Supports:
      - Grouped-Query Attention (GQA)
      - Rotary Position Embeddings (RoPE)
      - Flash Attention (via PyTorch scaled_dot_product_attention)
      - SwiGLU FFN (Llama/Mistral style)
      - KV cache for autoregressive generation

    Usage::

        layer = CELATransformerLayer(
            d_model=4096, n_heads=32, n_kv_heads=8, d_ff=14336, head_dim=128
        )
        x, kv = layer.forward(x, weights, rope, kv_cache, start_pos)
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, head_dim: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.head_dim = head_dim
        self.gqa_groups = n_heads // n_kv_heads if n_kv_heads > 0 else 1

    def forward(
        self,
        x: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        rope: RotaryEmbedding,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for one transformer layer.

        Args:
            x:        Input tensor (B, T, d_model)
            weights:  Dict of weight tensors on GPU
            rope:     Rotary embedding instance
            kv_cache: Previous (K, V) cache or None
            start_pos: Position offset for RoPE

        Returns:
            (output, (K_cache, V_cache))
        """
        B, T, D = x.shape
        device = x.device

        # Helper to find weight by partial key match
        def w(pattern: str) -> Optional[torch.Tensor]:
            for k, v in weights.items():
                if pattern in k:
                    return v
            return None

        # ---- Attention ----
        attn_norm = w("input_layernorm")
        if attn_norm is None: attn_norm = w("attn_norm")
        
        q_w = w("q_proj")
        if q_w is None: q_w = w("attn_q")
        
        k_w = w("k_proj")
        if k_w is None: k_w = w("attn_k")
        
        v_w = w("v_proj")
        if v_w is None: v_w = w("attn_v")
        
        o_w = w("o_proj")
        if o_w is None: o_w = w("attn_output")

        if attn_norm is None or q_w is None:
            return x, kv_cache or (torch.empty(0), torch.empty(0))

        residual = x
        x_n = RMSNorm.forward(x, attn_norm)

        q = (x_n @ q_w.T).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = (x_n @ k_w.T).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = (x_n @ v_w.T).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = rope.apply(q, start_pos)
        k = rope.apply(k, start_pos)

        # Append to KV cache
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k.to(device), k], dim=2)
            v = torch.cat([prev_v.to(device), v], dim=2)
        new_kv = (k.cpu(), v.cpu())  # Store on CPU to save VRAM

        # GQA expansion
        if self.gqa_groups > 1:
            k = k.repeat_interleave(self.gqa_groups, dim=1)
            v = v.repeat_interleave(self.gqa_groups, dim=1)

        # Flash Attention (cuDNN fused, much faster)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + (attn @ o_w.T)

        # ---- FFN ----
        ffn_norm = w("post_attention_layernorm")
        if ffn_norm is None: ffn_norm = w("ffn_norm")
        
        gate_w = w("gate_proj")
        if gate_w is None: gate_w = w("ffn_gate")
        
        up_w = w("up_proj")
        if up_w is None: up_w = w("ffn_up")
        
        down_w = w("down_proj")
        if down_w is None: down_w = w("ffn_down")

        if ffn_norm is not None and gate_w is not None and down_w is not None:
            residual = x
            x_n = RMSNorm.forward(x, ffn_norm)

            if up_w is not None:
                # SwiGLU (Llama, Mistral, Qwen)
                x = residual + (F.silu(x_n @ gate_w.T) * (x_n @ up_w.T)) @ down_w.T
            else:
                # GELU FFN (GPT-2 style)
                x = residual + F.gelu(x_n @ gate_w.T) @ down_w.T

        return x, new_kv
