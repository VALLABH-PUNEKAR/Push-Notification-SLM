import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAAttention(nn.Module):
    """
    Back-half of the attention sub-layer: takes already-projected,
    already-RoPE-rotated Q/K (V unrotated) and produces the block's
    attention output, ready for the residual add.

    Does NOT do: QKV linear projections (QKVProjection), RoPE (RotaryEmbedding),
    or normalization (RMSNorm, applied before this module at the block level).
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_groups,
        head_dim,
        max_seq_len,
        init_std=0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.q_per_kv = num_heads // num_kv_groups

        self.W_o = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Precomputed causal mask, cached the same way RoPE caches cos/sin:
        # a non-persistent buffer that travels with the model to GPU but is
        # never trained and never saved in the state_dict.
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self._init_weights(init_std)

    def _init_weights(self, std):
        nn.init.normal_(self.W_o.weight, mean=0.0, std=std)

    def _expand_kv(self, x):
        # x: (batch, num_kv_groups, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim), each kv head repeated
        # q_per_kv times contiguously (kv head 0 -> q heads 0..q_per_kv-1, etc.)
        batch_size, num_kv_groups, seq_len, head_dim = x.shape
        x = x.unsqueeze(2)
        x = x.expand(batch_size, num_kv_groups, self.q_per_kv, seq_len, head_dim)
        x = x.reshape(batch_size, num_kv_groups * self.q_per_kv, seq_len, head_dim)
        return x

    def forward(self, q, k, v, past_key=None, past_value=None,use_cache: bool = False,):
        # q: (batch, num_heads, seq_len_new, head_dim) — RoPE-rotated,
        #    already at the correct absolute position (RoPE offset is the
        #    caller's responsibility, not this module's)
        # k, v: (batch, num_kv_groups, seq_len_new, head_dim) — freshly
        #    computed K/V for only the new token(s) this call; k RoPE-rotated, v not
        # past_key, past_value: optional (batch, num_kv_groups, seq_len_past, head_dim)
        #    — unexpanded cached K/V from previous generation steps. When None,
        #    behavior is identical to the original no-cache design (training path).
        batch_size, num_heads, seq_len_new, head_dim = q.shape

        if past_key is not None:
            # Concatenate on the SMALL, unexpanded K/V — this is what keeps
            # the cache itself cheap; expansion happens fresh every call.
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        # These are what the caller stores and passes back in next step.
        new_past_key, new_past_value = k, v

        k_expanded = self._expand_kv(k)  # (batch, num_heads, total_len, head_dim)
        v_expanded = self._expand_kv(v)  # (batch, num_heads, total_len, head_dim)

        if past_key is None:
            # Original, unmodified training/full-sequence path: query and
            # key are the same length and both start at position 0, so
            # SDPA's fused causal path is valid and correct as before.
            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            # Cached path: query is short (new tokens only) and key is long
            # (full history). is_causal=True is NOT valid here since it
            # assumes query/key are equal length and aligned at position 0.
            # Build an explicit, offset-aware causal mask instead.
            total_len = k_expanded.shape[2]
            offset = total_len - seq_len_new
            query_pos = torch.arange(offset, offset + seq_len_new, device=q.device).unsqueeze(1)
            key_pos = torch.arange(total_len, device=q.device).unsqueeze(0)
            allowed_mask = key_pos <= query_pos  # (seq_len_new, total_len), True = attend

            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=allowed_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_new, num_heads * head_dim)
        )

        output = self.W_o(attn_output)  # (batch, seq_len_new, hidden_size)
        return output, new_past_key, new_past_value