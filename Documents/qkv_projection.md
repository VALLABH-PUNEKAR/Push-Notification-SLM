**File-level: what goes in, what comes out**
RMSNorm-normalized hidden states `(batch_size, seq_len, hidden_size)` go in, three tensors Q `(batch_size, num_heads, seq_len, head_dim)`, K `(batch_size, num_kv_groups, seq_len, head_dim)`, V `(batch_size, num_kv_groups, seq_len, head_dim)` come out. Nothing is written to disk.

---

**`QKVProjection` class variables**
- `hidden_size` — width of the incoming hidden state vector (512 placeholder)
- `num_heads` — number of Q attention heads (8 placeholder)
- `num_kv_groups` — number of K and V heads, smaller than `num_heads` because of GQA (2 placeholder)
- `head_dim` — size of each individual head vector, `hidden_size // num_heads = 64`, derived from `num_heads` not `num_kv_groups`
- `q_per_kv` — how many Q heads share one K/V head, `num_heads // num_kv_groups = 4`
- `W_q` — `nn.Linear(hidden_size, num_heads * head_dim)`, projects input to Q
- `W_k` — `nn.Linear(hidden_size, num_kv_groups * head_dim)`, projects input to K
- `W_v` — `nn.Linear(hidden_size, num_kv_groups * head_dim)`, projects input to V

---

**`QKVProjection` methods**
- `_init_weights(std)` — applies normal init (mean 0, std 0.02) to all three weight matrices and zeros any biases; nothing in, nothing out
- `forward(x)` — takes `(batch_size, seq_len, hidden_size)`, runs three independent linear projections, reshapes and transposes each into per-head layout, returns the `(Q, K, V)` tuple

---

**Inside `forward()` — the two-step shape transformation**
- `q_flat, k_flat, v_flat` — raw linear projection outputs, still flat on the last dimension before head splitting
- `.view()` — splits the last dimension into `(num_heads, head_dim)` or `(num_kv_groups, head_dim)`
- `.transpose(1, 2)` — swaps `seq_len` and `heads` axes to produce the `(batch, heads, seq_len, head_dim)` layout that RoPE and attention score computation expect