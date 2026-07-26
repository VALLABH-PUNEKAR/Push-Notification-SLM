import json

config = {
    "add_bos_token": False,
    "add_eos_token": True,
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
    "model_max_length": 128,
    "tokenizer_class": "PreTrainedTokenizerFast"
}

with open("tokenizer_config.json", "w") as f:
    json.dump(config, f, indent=2)