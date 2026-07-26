from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing

VOCAB_FILE  = "push_notif_tokenizer-vocab.json"
MERGES_FILE = "push_notif_tokenizer-merges.txt"

tok = ByteLevelBPETokenizer(VOCAB_FILE, MERGES_FILE)

# bake EOS-appending into the tokenizer itself,
# instead of doing it manually in process()
eos_id = tok.token_to_id("<|eos|>")
tok._tokenizer.post_processor = TemplateProcessing(
    single="$A <|eos|>",
    special_tokens=[("<|eos|>", eos_id)],
)

tok.save("tokenizer.json")
print("Saved tokenizer.json")