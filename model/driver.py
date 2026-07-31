"""
Driver Script
=============
Run: python driver.py

What this does:
    1. Builds the SLM from config.py
    2. Prints parameter count so you can tune hidden_size / num_layers
       toward your 15M target
    3. Runs a dummy forward pass to sanity-check tensor shapes
    4. Runs ONE dummy training step (loss.backward + optimizer.step)
       to confirm gradients flow through every module correctly
"""

import torch
import torch.nn as nn

from .config import SLMConfig
from .model import SLM


def main():
    torch.manual_seed(0)

    cfg = SLMConfig()
    model = SLM(cfg)

    total_params = model.num_params()
    non_embed_params = model.num_params(exclude_embeddings=True)

    print("=" * 60)
    print("CONFIG")
    print("=" * 60)
    for k, v in cfg.__dict__.items():
        print(f"  {k:16s}: {v}")

    print("\n" + "=" * 60)
    print("PARAMETER COUNT")
    print("=" * 60)
    print(f"  Total params        : {total_params:,}  ({total_params / 1e6:.2f}M)")
    print(f"  Non-embedding params: {non_embed_params:,}  ({non_embed_params / 1e6:.2f}M)")
    print(f"  Target              : ~15.00M")
    if abs(total_params - 15_000_000) / 15_000_000 > 0.15:
        print("  -> Off target by >15%. Adjust hidden_size / num_layers in config.py and re-run.")
    else:
        print("  -> Within range of 15M target.")

    # ---------------- Dummy KV-Cache pass check ----------------
    print("\n" + "=" * 60)
    print("KV-CACHE FORWARD PASS CHECK")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        # Step 1: Prompt step
        logits_prompt, past_kv = model(dummy_ids, use_cache=True)
        print(f"  prompt logits shape: {tuple(logits_prompt.shape)}")
        print(f"  cached layers      : {len(past_kv)}")
        print(f"  layer 0 key shape  : {tuple(past_kv[0][0].shape)}")

        # Step 2: Next token step (seq_len = 1)
        next_token_id = torch.randint(0, cfg.vocab_size, (batch_size, 1))
        logits_next, past_kv = model(next_token_id, past_key_values=past_kv, use_cache=True)
        print(f"  next-step logits   : {tuple(logits_next.shape)} (expect: ({batch_size}, 1, {cfg.vocab_size}))")
        assert logits_next.shape == (batch_size, 1, cfg.vocab_size), "KV Cache shape mismatch!"
        print("  -> KV Cache check OK.")

    # ---------------- Dummy forward pass ----------------
    print("\n" + "=" * 60)
    print("FORWARD PASS SHAPE CHECK")
    print("=" * 60)
    batch_size, seq_len = 4, 32
    dummy_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        logits = model(dummy_ids)
    print(f"  input  shape: {tuple(dummy_ids.shape)}")
    print(f"  output shape: {tuple(logits.shape)}  (expect: ({batch_size}, {seq_len}, {cfg.vocab_size}))")
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size), "Shape mismatch!"
    print("  -> Shapes OK.")

    # ---------------- Dummy training step ----------------
    print("\n" + "=" * 60)
    print("DUMMY TRAINING STEP (gradient flow check)")
    print("=" * 60)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    dummy_targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    optimizer.zero_grad()
    logits = model(dummy_ids)
    loss = loss_fn(logits.view(-1, cfg.vocab_size), dummy_targets.view(-1))
    loss.backward()

    # check that gradients actually reached every parameter
    missing_grad = [n for n, p in model.named_parameters() if p.grad is None]
    optimizer.step()

    print(f"  loss (random init, expect ~ln({cfg.vocab_size})={torch.log(torch.tensor(float(cfg.vocab_size))):.3f}): {loss.item():.4f}")
    if missing_grad:
        print(f"  WARNING: no gradient reached: {missing_grad}")
    else:
        print("  -> Gradients reached every parameter. Optimizer step OK.")

    print("\nDone. Model builds, forward pass runs, and training step works end-to-end.")


if __name__ == "__main__":
    main()
