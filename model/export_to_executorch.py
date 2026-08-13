"""
export_to_executorch.py

Exports the custom SLM (config.py / model.py) to an ExecuTorch .pte file
for on-device deployment (e.g. via react-native-executorch).

Usage:
    python -m V3.model.export_to_executorch \
        --checkpoint V3/checkpoints/checkpoint_latest.pt \
        --out model.pte \
        --seq_len 128 \
        --quantize

Notes:
- seq_len is only the example length used to trace the graph. It's marked
  as a dynamic dimension (see export_pte), so actual prompts of any length
  up to cfg.max_seq_len will work at inference time.
- This model has no KV cache (see generate.py docstring), so every
  generation step recomputes the full forward pass over the sequence so
  far. Fine for short outputs; if you add a KV cache later, this script's
  export call will need updating to also trace the cache tensors as
  inputs/outputs.
"""

import argparse
import torch

from .config import SLMConfig
from .model import SLM


def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = SLMConfig(**ckpt["config"])
    model = SLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def quantize_model(model):
    """
    Quantizes model weights for ExecuTorch using PyTorch 2.x compatible torchao APIs.
    """
    try:
        from torchao.quantization import quantize_, int8_dynamic_activation_int4_weight
        print("Applying Int8 Dynamic Activation + Int4 Weight quantization (torchao)...")
        quantize_(model, int8_dynamic_activation_int4_weight())
        return model
    except (ImportError, AttributeError):
        pass

    try:
        from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
        print("Applying Int8 Dynamic Activation + Int8 Weight quantization (torchao)...")
        quantize_(model, int8_dynamic_activation_int8_weight())
        return model
    except (ImportError, AttributeError) as e:
        print(f"torchao quantization unavailable: {e}")
        print("Proceeding with unquantized model export for ExecuTorch.")
        return model


def export_pte(model, cfg, seq_len, out_path):
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    example_tokens = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.long)

    # seq_len varies at inference (prompt + generated tokens so far, up to
    # cfg.max_seq_len) so it must be dynamic, not fixed at the traced value.
    seq_len_dim = torch.export.Dim("seq_len", min=1, max=cfg.max_seq_len)
    dynamic_shapes = {"token_ids": {1: seq_len_dim}}

    exported_program = torch.export.export(
        model,
        (example_tokens,),
        dynamic_shapes=dynamic_shapes,
    )

    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    executorch_program = edge_program.to_executorch()

    with open(out_path, "wb") as f:
        f.write(executorch_program.buffer)

    print(f"Wrote {out_path} ({len(executorch_program.buffer) / 1e6:.2f} MB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to a checkpoint_step*.pt or checkpoint_latest.pt file")
    p.add_argument("--out", default="model.pte")
    p.add_argument("--seq_len", type=int, default=128,
                   help="Example prompt length used to trace the graph")
    p.add_argument("--quantize", action="store_true",
                   help="Apply int8 activation / int4 weight quantization before export")
    return p.parse_args()


def main():
    args = parse_args()
    model, cfg = load_model(args.checkpoint)

    if args.quantize:
        print("Quantizing model (int8 dynamic activation, int4 weight)...")
        model = quantize_model(model)

    export_pte(model, cfg, args.seq_len, args.out)


if __name__ == "__main__":
    main()
