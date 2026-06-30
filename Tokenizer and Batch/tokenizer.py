import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import torch
from tqdm.auto import tqdm

# ─────────────────────────────────────────────
# CONSTANTS — safe to define at top level
# ─────────────────────────────────────────────
VOCAB_FILE   = "push_notif_tokenizer-vocab.json"
MERGES_FILE  = "push_notif_tokenizer-merges.txt"
BLOCK_SIZE   = 128   # FIX 4: was 8, push notifications need at least 128
BATCH_SIZE   = 32


# ─────────────────────────────────────────────
# FIX 1+3: process() now uses your custom
# tokenizer correctly — no more tiktoken refs
# ─────────────────────────────────────────────
def process(example):
    # Load tokenizer inside the function so each
    # worker process can access it safely
    tok = ByteLevelBPETokenizer(VOCAB_FILE, MERGES_FILE)

    text = example["text"]
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"ids": [], "len": 0}

    encoding = tok.encode(text.strip())
    ids = encoding.ids

    # Add end-of-sequence token at end of every notification
    eos_id = tok.token_to_id("<|eos|>")
    ids.append(eos_id)

    return {"ids": ids, "len": len(ids)}


# ─────────────────────────────────────────────
# FIX 5: get_batch stays outside main but
# does NOT depend on global mutable state
# ─────────────────────────────────────────────
def get_batch(split, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    filename = "train.bin" if split == "train" else "validation.bin"
    data = np.memmap(filename, dtype=np.uint16, mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i     : i +  block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
        for i in ix
    ])

    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


# ─────────────────────────────────────────────
# MAIN BLOCK — everything that "runs" goes here
# FIX 2: tokenizer training moved inside here
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # STEP 1 — Train tokenizer only if not already saved
    if not os.path.exists(VOCAB_FILE) or not os.path.exists(MERGES_FILE):
        print("Training custom BPE tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=["your_notifications.txt"],
            vocab_size=12000,
            min_frequency=2,
            special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
        )
        tokenizer.save_model(".", "push_notif_tokenizer")
        print(f"Tokenizer saved! Vocab size: {tokenizer.get_vocab_size()}")
    else:
        print("Tokenizer already trained — loading from disk")

    # STEP 2 — Load dataset
    csv_files = {
        "train":      ["ic1.csv", "ic2.csv", "ic3.csv", "ic4.csv"],
        "validation": ["icv.csv"]
    }
    ds = load_dataset("csv", data_files=csv_files)

    # STEP 3 — Tokenize and write .bin files
    if not os.path.exists("train.bin") or not os.path.exists("validation.bin"):
        tokenized = ds.map(
            process,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            num_proc=4   # lowered from 8 — safer on most machines
        )

        for split, dset in tokenized.items():
            # Filter out empty examples
            dset = dset.filter(lambda x: x["len"] > 0)

            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = f"{split}.bin"
            arr = np.memmap(filename, mode="w+",
                            shape=(arr_len,), dtype=np.uint16)

            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches),
                                  desc=f"Writing {split}.bin"):
                batch = dset.shard(num_shards=total_batches,
                                   index=batch_idx, contiguous=True)
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
            print(f"{split}.bin written — {arr_len:,} tokens total")
    else:
        print("Bin files already exist — skipping tokenization")

    # STEP 4 — Quick sanity check
    print("\n--- Sanity Check ---")
    tok_check = ByteLevelBPETokenizer(VOCAB_FILE, MERGES_FILE)
    print(f"Vocab size : {tok_check.get_vocab_size()}")

    sample = tok_check.encode("🔔 Your order is out for delivery!")
    print(f"Sample     : 🔔 Your order is out for delivery!")
    print(f"Token IDs  : {sample.ids}")
    print(f"Tokens     : {sample.tokens}")

    xb, yb = get_batch("train")
    print(f"\nBatch x shape : {xb.shape}")  # should be [32, 128]
    print(f"Batch y shape : {yb.shape}")  # should be [32, 128]
    print("All good! Ready to build the model.")