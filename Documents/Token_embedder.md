**File-level: what goes in, what comes out**
Token IDs (integers, `uint16` from `.bin` → cast to `int64`) go in, dense float vectors of shape `(batch_size, seq_len, hidden_size)` come out from `TokenEmbedding`, and those same float vectors go into `LMHead` which returns `(batch_size, seq_len, vocab_size)` logits. One shared weight matrix `(vocab_size, hidden_size)` is the only parameter underlying both.

---

**`TokenEmbedding` class**
- `vocab_size` — how many unique tokens exist (8000 for your BBPE tokenizer)
- `hidden_size` — length of each token's dense vector (512 placeholder)
- `init_std` — standard deviation for random weight init (0.02, GPT-2 style)
- `self.embedding` — the actual `nn.Embedding` holding the `(vocab_size, hidden_size)` weight table
- `_init_weights(std)` — fills the weight table with small random values at construction; nothing in, nothing out
- `weight` (property) — exposes `self.embedding.weight` directly so `LMHead` can grab the same tensor object
- `forward(token_ids)` — takes `(batch_size, seq_len)` LongTensor, returns `(batch_size, seq_len, hidden_size)` FloatTensor

---

**`LMHead` class**
- `token_embedding` — receives the `TokenEmbedding` instance; reads its `.weight` live, never copies it
- `bias` — optional `(vocab_size,)` parameter, off by default
- `forward(hidden_states)` — takes `(batch_size, seq_len, hidden_size)` FloatTensor (final transformer output), returns `(batch_size, seq_len, vocab_size)` logits via `F.linear(hidden_states, embedding.weight)` which is `hidden_states @ weight.T`

---

**`build_tied_embedding_and_head()`**
- Inputs: `vocab_size`, `hidden_size`, `init_std`, `lm_head_bias`
- Output: a `(TokenEmbedding, LMHead)` tuple that share one weight tensor — call this once in your model's `__init__`, never build them separately