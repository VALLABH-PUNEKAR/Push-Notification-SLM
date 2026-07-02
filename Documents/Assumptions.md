# Token Embedder

Here are all assumptions made in this chat session specifically:

**From `token_embedding.py`**
- `vocab_size = 8000` as default (acknowledged as approximate — exact final number from your tokenizer may differ slightly)
- `hidden_size = 512` as placeholder (real value pending full architecture lock)
- `init_std = 0.02` matching GPT-2/Llama convention
- `lm_head_bias = False` (no bias on output projection)

**From `use_token_embedding.py`**
- `BATCH_SIZE = 8`
- `SEQ_LEN = 128` (matching your locked `max_seq_len`)
- `.bin` files are flat `uint16` memmapped token ID arrays with no header bytes — raw token IDs start at byte 0
- `dtype=np.uint16, mode="r"` is the correct memmap config for your shards

**From `qkv_projection.py`**
- `num_heads = 8` (Q heads, placeholder)
- `num_kv_groups = 2` (K/V heads, placeholder)
- `head_dim = hidden_size // num_heads = 64` (derived from `num_heads`, not `num_kv_groups` — this one is locked per your architecture decision)
- `bias = False` on all three linear layers
- Same `init_std = 0.02` applied to `W_Q`, `W_K`, `W_V`
- Input to `QKVProjection` is already RMSNorm-normalized — raw embedding output does NOT go in directly

**Cross-cutting assumptions**
- RMSNorm sits between `TokenEmbedding` output and `QKVProjection` input — no code written for it yet but its placement is assumed throughout
- K and V head expansion from `num_kv_groups` → `num_heads` (repeating each group `q_per_kv` times) is deferred to the attention block, not handled in `QKVProjection`
- RoPE is applied to Q and K after this module, never to V

The ones most likely to break something when you lock real dimensions are `num_heads`, `num_kv_groups`, and `hidden_size` — those three ripple through everything built so far.

# RoPE

Here are all the assumptions made so far, in order:

---

**Tokenizer & Data**
- Token shards (`train.bin`, `validation.bin`) are written as flat `uint16` binary files
- Custom BBPE tokenizer with vocab size ~8,000–12,000 (exact value you supply at runtime)
- Tokens were finalized from the custom BBPE pipeline, not the interim GPT-2/tiktoken one

---

**Sequence Length**
- `max_seq_len = 128` — chosen based on your 10–30 token notifications, 4GB VRAM constraint, and lower-end phone inference target

---

**Model Architecture**
- Decoder-only Transformer (GQA-based)
- `head_dim = hidden_size // num_heads` — NOT `num_kv_groups`
- `head_dim` is always even (required for RoPE's half-split)
- RoPE `theta = 10000.0` (standard default, no custom value)
- Pre-norm RMSNorm, SwiGLU FFN, tied embeddings — locked earlier, not yet implemented

---

**RoPE Specifics**
- cos/sin cached at full `(max_seq_len, head_dim)` shape (duplicated from `head_dim//2`)
- Buffers are non-persistent (travel with model to device, not saved in checkpoint)
- Q and K are both shaped `(batch, num_heads, seq_len, head_dim)` before RoPE is applied
- V is never rotated — passes through untouched

---

**Hardware & Environment**
- Training GPU: GTX 1650, 4GB VRAM
- Windows, Python venv at `D:\codescope\push_notification_SLM\venv\`
- Project root: `D:\codescope\push_notification_SLM\`
- Inference target: lower-end mobile phones

---

**Dataset Loader**
- `get_batch()` re-opens the memmap on every call (intentional — avoids memory leak on Windows over long runs)
- Batches are random windows, not sequential — no epoch tracking in the loader itself
- `pin_memory + non_blocking=True` used for GPU transfers
- No DataLoader workers — single-process memmap only

---

Anything here that doesn't match your actual setup, flag it now before we go deeper into the GQA block — wrong assumptions at this stage compound into hard-to-trace bugs later.