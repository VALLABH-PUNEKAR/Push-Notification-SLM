"""
Full SLM Model — wires your ready modules together
====================================================

Pipeline per your architecture:

    token_ids
      -> Embedding
      -> [ RMSNorm -> QKVProjection -> RoPE -> GQAAttention -> residual add
          -> RMSNorm -> SwiGLU FFN            -> residual add ]  x num_layers
      -> final RMSNorm
      -> lm_head (optionally tied to embedding weight)
      -> logits

*** ASSUMPTIONS — ADJUST THESE IMPORT LINES / CALL SIGNATURES TO MATCH
    YOUR ACTUAL FILES. I only have your qkv_projections.py; the other
    three are assumed to have the interfaces below. If your real
    modules differ, paste them and I'll rewire this exactly. ***

Assumed interfaces:
    RMSNorm(dim, eps=1e-6).forward(x) -> x_normed                (same shape)
    QKVProjection(hidden_size, num_heads, num_kv_groups, bias).forward(x)
        -> (Q, K, V)   each (batch, heads, seq_len, head_dim)
    GQAAttention(hidden_size, num_heads, num_kv_groups).forward(Q, K, V, causal=True)
        -> attn_out  (batch, seq_len, hidden_size)   <- already merged
                                                          heads + output-projected
        (assumed to apply RoPE internally — flag this if it doesn't;
         if RoPE is a separate module you already have, tell me and
         I'll insert it explicitly between QKVProjection and attention)
    SwiGLUFFN(hidden_size, ffn_dim, bias=False).forward(x) -> x    (same shape)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .qvk_projections import QKVProjection
from .norms import RMSNorm
from .attention import GQAAttention
from .swiglu import SwiGLUFeedForward
from .config import SLMConfig
from ..Embedders.token_embedder import TokenEmbedding, LMHead
from ..Embedders.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, cfg: SLMConfig, rope: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.qkv = QKVProjection(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_kv_groups=cfg.num_kv_groups,
            bias=cfg.bias,
            init_std=cfg.init_std,
        )
        self.rope = rope

        self.attn = GQAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_kv_groups=cfg.num_kv_groups,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            init_std=cfg.init_std,
        )

        self.ffn_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.ffn = SwiGLUFeedForward(
            hidden_size=cfg.hidden_size,
            intermediate_size=None,
            init_std=cfg.init_std,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        h = self.attn_norm(x)
        Q, K, V = self.qkv(h)
        
        # RoPE applies based on current token sequence length and position
        past_k, past_v = past_kv if past_kv is not None else (None, None)
        past_len = past_k.shape[2] if past_k is not None else 0
        Q, K = self.rope(Q, K,position_offset=past_len)          

        # Call updated attention layer that returns output and updated (K, V)
        
        attn_res = self.attn(Q, K, V, past_key=past_k, past_value=past_v, use_cache=use_cache)
        
        if use_cache:
            attn_out, present_k, present_v = attn_res
            present_kv = (present_k, present_v)
        else:
            attn_out = attn_res[0] if isinstance(attn_res, (tuple, list)) else attn_res
            present_kv = None

        x = residual + self.dropout(attn_out)

        # FFN sub-layer
        residual = x
        h = self.ffn_norm(x)
        ffn_out = self.ffn(h)
        x = residual + self.dropout(ffn_out)

        return x, present_kv


class SLM(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = TokenEmbedding(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            init_std=cfg.init_std,
        )
        self.lm_head = LMHead(self.token_embedding, bias=False)
        self.rope = RotaryEmbedding(head_dim=cfg.head_dim, max_seq_len=cfg.max_seq_len)

        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, self.rope) for _ in range(cfg.num_layers)]
        )
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)

        if not cfg.tie_embeddings:
            raise NotImplementedError("cfg.tie_embeddings=False is not supported.")

    def forward(
        self, 
        token_ids: torch.Tensor, 
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ):
        """
        If use_cache=True:
            Returns: logits, present_key_values
        If use_cache=False:
            Returns: logits
        """
        x = self.token_embedding(token_ids)
        presents = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                presents.append(present_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, presents
        return logits

    def num_params(self, exclude_embeddings: bool = False) -> int:
        if exclude_embeddings:
            return sum(
                p.numel() for n, p in self.named_parameters()
                if "token_embedding" not in n and "lm_head" not in n
            )
        return sum(p.numel() for p in self.parameters())