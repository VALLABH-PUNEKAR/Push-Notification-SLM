"""
Rotary Positional Embeddings (RoPE).

No learned parameters. Encodes absolute position by rotating Q/K vectors
in 2D sub-planes of the head dimension, such that dot products between
rotated Q and K naturally depend on relative position.
"""

import torch
import torch.nn as nn


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Build the angle table: rotation angle for every (position, freq-pair) combo.

    head_dim is split into head_dim // 2 pairs of dimensions. Each pair gets
    its own frequency — pair 0 rotates fastest, the last pair rotates slowest.
    freq_i = 1 / theta^(2i / head_dim)  for i in [0, head_dim/2)

    angle(pos, i) = pos * freq_i

    Returns:
        angles: (max_seq_len, head_dim // 2) tensor of angles in radians.
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # exponents: 0, 2, 4, ..., head_dim-2  -> shape (head_dim // 2,)
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    freqs = 1.0 / (theta ** exponents)  # (head_dim // 2,) — decreasing values

    positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)

    # outer product: each position times each frequency
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim // 2)
    return angles


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Split the last dim in half and swap with a sign flip:
    [x1, x2] -> [-x2, x1]

    This is the standard trick that lets a real-valued multiply/add
    simulate a complex-plane rotation.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embedding to a Q or K tensor.

    Args:
        x:   (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim) — cosine of angles, already duplicated to full head_dim
        sin: (seq_len, head_dim) — sine of angles, already duplicated to full head_dim

    Returns:
        Rotated tensor, same shape as x.
    """
    # cos/sin broadcast over (batch, num_heads, seq_len, head_dim)
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    Precomputes RoPE angle tables once and applies rotation to Q/K at call time.

    Usage:
        rope = RotaryEmbedding(head_dim=64, max_seq_len=2048)
        q_rot, k_rot = rope(q, k)   # q, k: (batch, num_heads, seq_len, head_dim)
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        angles = precompute_rope_freqs(head_dim, max_seq_len, theta)  # (max_seq_len, head_dim // 2)

        # Duplicate each angle across the pair of dims it controls so cos/sin
        # have shape (max_seq_len, head_dim) and line up directly with x and
        # _rotate_half(x) for the elementwise multiply in apply_rope.
        angles = torch.cat([angles, angles], dim=-1)  # (max_seq_len, head_dim)

        cos = angles.cos()
        sin = angles.sin()

        # Buffers: travel with .to(device)/state_dict, but never trained
        # and never appear in model.parameters().
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k: (batch, num_heads, seq_len, head_dim) — works for GQA too,
                  since K's num_heads (kv groups) is just a different size
                  in dim 1; rotation is applied per-position, per-head,
                  identically regardless of how many heads there are.

        Returns:
            (q_rotated, k_rotated), same shapes as inputs.
        """
        seq_len = q.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len}) "
                f"the RotaryEmbedding was built for."
            )

        cos = self.cos_cached[:seq_len]  # (seq_len, head_dim)
        sin = self.sin_cached[:seq_len]  # (seq_len, head_dim)

        q_rot = apply_rope(q, cos, sin)
        k_rot = apply_rope(k, cos, sin)
        return q_rot, k_rot


if __name__ == "__main__":
    torch.manual_seed(0)

    batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
    max_seq_len = 64

    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)

    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=max_seq_len)
    q_rot, k_rot = rope(q, k)

    # 1. Shape must be unchanged
    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} vs {k.shape}"
    print(f"Shape check passed: {q_rot.shape}")

    # 2. Norm must be unchanged (rotation preserves magnitude, only changes direction)
    q_norm_before = q.norm(dim=-1)
    q_norm_after = q_rot.norm(dim=-1)
    k_norm_before = k.norm(dim=-1)
    k_norm_after = k_rot.norm(dim=-1)

    assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5), "Q norm changed after rotation!"
    assert torch.allclose(k_norm_before, k_norm_after, atol=1e-5), "K norm changed after rotation!"
    print(f"Norm check passed: max diff Q={ (q_norm_before - q_norm_after).abs().max().item():.2e}, "
          f"K={ (k_norm_before - k_norm_after).abs().max().item():.2e}")

    # 3. Relative-position property: take ONE fixed (q, k) vector pair, and
    # rotate that *same pair* as if placed at two different pairs of positions
    # that share the same offset. RoPE's core guarantee is that the rotated
    # dot product depends only on (pos_q - pos_k), not on absolute position.
    fixed_q = torch.randn(1, 1, 1, head_dim)
    fixed_k = torch.randn(1, 1, 1, head_dim)

    def rotated_dot(pos_q: int, pos_k: int) -> torch.Tensor:
        cos_q, sin_q = rope.cos_cached[pos_q], rope.sin_cached[pos_q]
        cos_k, sin_k = rope.cos_cached[pos_k], rope.sin_cached[pos_k]
        q_r = apply_rope(fixed_q, cos_q, sin_q)
        k_r = apply_rope(fixed_k, cos_k, sin_k)
        return (q_r * k_r).sum()

    offset = 4
    dot_low = rotated_dot(5, 5 - offset)    # positions 5, 1
    dot_high = rotated_dot(50, 50 - offset)  # positions 50, 46 — same offset, far away

    assert torch.allclose(dot_low, dot_high, atol=1e-4), (
        f"Relative-position property failed: {dot_low.item()} vs {dot_high.item()}"
    )
    print(f"Relative-position check passed: dot@offset{offset} consistent "
          f"({dot_low.item():.4f} ≈ {dot_high.item():.4f}) regardless of absolute position")

    print("All self-tests passed.")