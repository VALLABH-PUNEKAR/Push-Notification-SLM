from tokenizers import Tokenizer
t = Tokenizer.from_file("tokenizer.json")
out = t.encode("🔔 Your order is out for delivery!")
print(out.ids, out.tokens)