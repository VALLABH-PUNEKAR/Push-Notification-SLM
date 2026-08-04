"""
generate.py
===========
Load a trained checkpoint and generate text from a prompt.

Run (same relative-import style as train.py, so from the parent directory
of your package):

    python -m V3.model.generate \\
        --checkpoint_dir V3\\checkpoints \\
        --vocab_file push_notif_tokenizer-vocab.json \\
        --merges_file push_notif_tokenizer-merges.txt \\
        --prompt "Your order is"

If you don't pass --checkpoint (a specific file), this script scans
checkpoint_dir for all checkpoint_step*.pt files and automatically picks
the one with the lowest saved val_loss — matching the "protect the best
checkpoint" logic already in checkpoint.py's prune_old_checkpoints().

NOTE on context length: this model has no KV-cache — every generation
step re-runs the full forward pass over everything generated so far
(simplest correct approach for a first version). That means generation
gets slower as it gets longer, and is hard-capped at cfg.max_seq_len
(512 tokens total, prompt + generated, per your config.py). Fine for
short outputs like push notifications; if you need long-form generation
later, a KV-cache is the next upgrade.
"""

import argparse
import glob
import os

import torch
from tokenizers import ByteLevelBPETokenizer

from .config import SLMConfig
from .model import SLM


# --------------------------------------------------------------------------
# Checkpoint discovery / loading
# --------------------------------------------------------------------------

def find_best_checkpoint(checkpoint_dir):
    """
    Scan checkpoint_step*.pt files in checkpoint_dir and return the path
    to the one with the lowest saved val_loss. Mirrors the "best" logic
    in checkpoint.py's prune_old_checkpoints(), so this always agrees
    with what training protected from deletion.
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint_step*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No checkpoint_step*.pt files found in {checkpoint_dir}. "
            "Pass --checkpoint to point at a specific file instead."
        )

    best_path, best_val_loss = None, float("inf")
    for f in files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        val_loss = data.get("val_loss")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f

    if best_path is None:
        raise RuntimeError(
            f"Found checkpoint files in {checkpoint_dir} but none had a "
            "usable val_loss. Pass --checkpoint to pick one explicitly."
        )

    print(f"Auto-selected best checkpoint: {best_path} (val_loss={best_val_loss:.4f})")
    return best_path


def load_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild the exact architecture the checkpoint was trained with,
    # rather than assuming the current config.py hasn't changed since.
    saved_config = ckpt["config"]
    cfg = SLMConfig(**saved_config)

    model = SLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"Loaded checkpoint from step {step} (val_loss={val_loss})")
    return model, cfg


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

@torch.no_grad()
def generate(model, cfg, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    eos_id = tokenizer.token_to_id("<|eos|>")

    ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

    past_key_values = None
    
    for i in range(max_new_tokens):
        # On token 0, feed entire prompt. On token >0, feed ONLY the last predicted token.
        if past_key_values is None:
            model_input = tokens[:, -cfg.max_seq_len:]
        else:
            model_input = tokens[:, -1:]

        logits, past_key_values = model(
            model_input, 
            past_key_values=past_key_values, 
            use_cache=True
        )

        logits = logits[:, -1, :]  # last position -> (1, vocab_size)
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_keep = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_keep, torch.full_like(logits, float("-inf")), logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        tokens = torch.cat([tokens, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    return tokens[0].tolist()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a trained SLM checkpoint")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Directory to auto-search for the best checkpoint (checkpoint_step*.pt)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a specific checkpoint file. Overrides --checkpoint_dir auto-search.")

    p.add_argument("--vocab_file", type=str, required=True,
                   help="e.g. push_notif_tokenizer-vocab.json")
    p.add_argument("--merges_file", type=str, required=True,
                   help="e.g. push_notif_tokenizer-merges.txt")

    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Lower = more predictable/repetitive, higher = more random")
    p.add_argument("--top_k", type=int, default=50,
                   help="Only sample from the top_k most likely tokens each step")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.checkpoint is None and args.checkpoint_dir is None:
        raise ValueError("Pass either --checkpoint or --checkpoint_dir.")

    checkpoint_path = args.checkpoint or find_best_checkpoint(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
    print(f"Using device: {device}")

    tokenizer = ByteLevelBPETokenizer(args.vocab_file, args.merges_file)
    model, cfg = load_model_from_checkpoint(checkpoint_path, device)

    output_ids = generate(
        model, cfg, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    text = tokenizer.decode(output_ids)
    print("\n--- Generated ---")
    print(text)


if __name__ == "__main__":
    main()
