import os
import sys
import glob
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import torch
from tqdm.auto import tqdm

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
VOCAB_FILE   = "push_notif_tokenizer-vocab.json"
MERGES_FILE  = "push_notif_tokenizer-merges.txt"
BLOCK_SIZE   = 128
BATCH_SIZE   = 32
VAL_SPLIT    = 0.1   # fraction held out for validation when only one CSV is given

CONTEXT_KEYS = [
    "scenario", "product", "category", "customer_type",
    "user_activity", "time_of_day", "day_type", "season",
    "weather", "discount", "urgency", "tone", "emoji"
]


# ─────────────────────────────────────────────
# HELPER: ROBUST CSV READER & CLEANER
# ─────────────────────────────────────────────
def load_and_clean_csvs(file_paths):
    """
    Reads CSV files safely handling corrupt rows, unescaped commas, and
    extra fields via Python's CSV engine and on_bad_lines='skip'.
    """
    dfs = []
    for fp in file_paths:
        try:
            # First attempt: Python engine to handle tricky quotes and commas
            df = pd.read_csv(fp, engine="python", on_bad_lines="skip")
        except Exception:
            # Fallback attempt: Standard C engine skipping bad lines
            df = pd.read_csv(fp, on_bad_lines="skip")

        # Strip trailing/phantom un-named columns
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        # Keep only valid expected columns
        expected_cols = CONTEXT_KEYS + ["notification"]
        existing_cols = [c for c in expected_cols if c in df.columns]
        df = df[existing_cols]

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.fillna("")
    return combined_df


# ─────────────────────────────────────────────
# HELPER: FORMAT ROW INTO PROMPT STRING
# ─────────────────────────────────────────────
def format_example_prompt(example):
    ctx_str = ", ".join(f"{k}={example.get(k, '')}" for k in CONTEXT_KEYS)
    full_text = f"Context: {ctx_str} -> Notification: {example.get('notification', '')}"
    return full_text


# ─────────────────────────────────────────────
# TOKENIZATION PROCESS FOR DATASET MAP
# ─────────────────────────────────────────────
def process(example):
    tok = ByteLevelBPETokenizer(VOCAB_FILE, MERGES_FILE)
    full_text = format_example_prompt(example)

    if not isinstance(full_text, str) or len(full_text.strip()) == 0:
        return {"ids": [], "len": 0}

    encoding = tok.encode(full_text.strip())
    ids = encoding.ids

    eos_id = tok.token_to_id("<|eos|>")
    ids.append(eos_id)

    return {"ids": ids, "len": len(ids)}


# ─────────────────────────────────────────────
# BATCH GENERATOR FOR MODEL TRAINING
# ─────────────────────────────────────────────
def get_batch(split, data_dir=".", block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    filename = os.path.join(data_dir, "train.bin" if split == "train" else "validation.bin")
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
# MAIN EXECUTION PIPELINE
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tokenize push-notification CSV data")
    parser.add_argument("csv_path", nargs="?", default=None,
                         help="Path to a specific CSV file. Defaults to glob('all.csv') in cwd.")
    parser.add_argument("--data_dir", type=str, default=".",
                         help="Where to write train.bin / validation.bin. Must match the "
                              "--data_dir you pass to train.py.")
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT,
                         help="Fraction of rows held out for validation when only one CSV is given.")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # 1. Discover CSV file(s) to use
    if args.csv_path:
        all_csvs = [args.csv_path]
    else:
        all_csvs = sorted(glob.glob("all.csv"))
        all_csvs = [f for f in all_csvs if not f.startswith("train_corpus")]

    print(f"Found {len(all_csvs)} CSV dataset file(s).")

    # Split into train/validation.
    #
    # FIX: with a single CSV file, the old logic set train_files == val_files,
    # meaning "validation" was literally the training data — val_loss was
    # meaningless and checkpoint.py's "keep best val_loss" protection had
    # nothing real to select on. Now we always produce a genuine held-out
    # split, either by file (multi-file case) or by row (single-file case).
    if len(all_csvs) > 1:
        train_files = all_csvs[:-1]
        val_files = [all_csvs[-1]]
        ds = DatasetDict({
            "train": Dataset.from_pandas(load_and_clean_csvs(train_files)),
            "validation": Dataset.from_pandas(load_and_clean_csvs(val_files)),
        })
    else:
        full_df = load_and_clean_csvs(all_csvs)
        train_df, val_df = train_test_split(
            full_df, test_size=args.val_split, random_state=42
        )
        ds = DatasetDict({
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        })

    print(f"  train rows      : {len(ds['train']):,}")
    print(f"  validation rows : {len(ds['validation']):,}")

    # 2. Train BPE Tokenizer on dataset corpus
    if not os.path.exists(VOCAB_FILE) or not os.path.exists(MERGES_FILE):
        print("Building formatted prompt corpus for tokenizer training...")
        corpus_filename = "train_corpus.txt"

        with open(corpus_filename, "w", encoding="utf-8") as f:
            for example in ds["train"]:
                prompt_text = format_example_prompt(example)
                f.write(prompt_text + "\n")

        print("Training custom Byte-Level BPE tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[corpus_filename],
            vocab_size=1000,
            min_frequency=2,
            special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
        )
        tokenizer.save_model(".", "push_notif_tokenizer")
        tokenizer.save("tokenizer.json")  # unified format for ExecuTorch / mobile runtime
        print(f"Tokenizer trained & saved! Vocab size: {tokenizer.get_vocab_size()}")

        if os.path.exists(corpus_filename):
            os.remove(corpus_filename)
    else:
        print("Tokenizer already exists — loading from disk.")

    # 3. Tokenize and build train.bin / validation.bin (written into --data_dir)
    train_bin_path = os.path.join(args.data_dir, "train.bin")
    val_bin_path = os.path.join(args.data_dir, "validation.bin")

    if not os.path.exists(train_bin_path) or not os.path.exists(val_bin_path):
        remove_cols = ds["train"].column_names

        tokenized = ds.map(
            process,
            remove_columns=remove_cols,
            desc="Tokenizing dataset",
            num_proc=4
        )

        for split, dset in tokenized.items():
            dset = dset.filter(lambda x: x["len"] > 0)
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(args.data_dir, f"{split}.bin")
            arr = np.memmap(filename, mode="w+", shape=(arr_len,), dtype=np.uint16)

            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {split}.bin"):
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
                if len(batch["ids"]) > 0:
                    arr_batch = np.concatenate(batch["ids"])
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
            arr.flush()
            print(f"{filename} written — {arr_len:,} total tokens")
    else:
        print(f"Bin files already exist in {args.data_dir} — skipping tokenization step.")

    # 4. Sanity Check
    print("\n--- Pipeline Verification ---")
    tok_check = ByteLevelBPETokenizer(VOCAB_FILE, MERGES_FILE)
    print(f"Vocab size : {tok_check.get_vocab_size()}")

    sample_str = "Context: scenario=Discounts & Promotions, product=Triple Chocolate Brownie Sundae -> Notification: 🍨 Discount unlocked!"
    encoded = tok_check.encode(sample_str)
    print(f"Sample Input : {sample_str}")
    print(f"Token IDs    : {encoded.ids[:12]}...")
    print(f"Tokens       : {encoded.tokens[:12]}...")

    xb, yb = get_batch("train", data_dir=args.data_dir)
    print(f"\nBatch x shape : {xb.shape}")
    print(f"Batch y shape : {yb.shape}")
    print("Ready to begin 15M SLM model training!")


if __name__ == "__main__":
    main()
