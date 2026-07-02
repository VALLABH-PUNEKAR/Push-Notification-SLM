**`rope.py` — Module Overview**

Takes: `head_dim` and `max_seq_len` as config. Operates on Q and K tensors of shape `(batch, num_heads, seq_len, head_dim)`. Returns those same tensors rotated, same shape, no learned weights involved.

---

**`precompute_rope_freqs(head_dim, max_seq_len, theta=10000.0)`**
Builds a table of rotation angles.
- Takes: head dimension size, max sequence length, theta constant
- Returns: `(max_seq_len, head_dim//2)` float tensor of angles in radians
- One row per position, one column per dimension-pair. Early columns = large angles (fast rotation), later columns = tiny angles (slow rotation).

---

**`_rotate_half(x)`**
Mechanical helper for the rotation math.
- Takes: any tensor, operates on last dimension
- Returns: same shape tensor with last dim split in half, second half negated and swapped to front: `[x1, x2] → [-x2, x1]`

---

**`apply_rope(x, cos, sin)`**
Applies the actual rotation to one Q or K tensor.
- Takes: `x (batch, num_heads, seq_len, head_dim)`, precomputed `cos` and `sin` each of shape `(seq_len, head_dim)`
- Returns: rotated tensor, identical shape to `x`
- Formula: `(x * cos) + (rotate_half(x) * sin)`

---

**`RotaryEmbedding` (nn.Module)**
The class you actually instantiate and use.
- Init takes: `head_dim`, `max_seq_len`, optional `theta`
- Internally calls `precompute_rope_freqs()`, duplicates angles to full `head_dim`, stores `cos` and `sin` as non-persistent buffers (travel with model to GPU, never trained)
- `forward(q, k)` takes Q and K both `(batch, num_heads, seq_len, head_dim)`, slices the cached tables down to current `seq_len`, calls `apply_rope` on both
- Returns: `(q_rotated, k_rotated)`, same shapes as input