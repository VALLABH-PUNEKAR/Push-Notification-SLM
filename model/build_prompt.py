"""
build_prompt.py

Builds a --prompt string for generate.py that matches the exact format
tokenizer.py used during training:

    Context: scenario=X, product=Y, ... -> Notification: <text>

generate.py should be handed everything up to and including "Notification:"
so the model continues naturally from where training examples end.
"""

CONTEXT_KEYS = [
    "scenario", "product", "category", "customer_type",
    "user_activity", "time_of_day", "day_type", "season",
    "weather", "discount", "urgency", "tone", "emoji"
]


def build_prompt(context: dict) -> str:
    """
    context: dict with any subset of CONTEXT_KEYS. Missing keys are left
    blank, matching format_example_prompt()'s example.get(k, '') behavior
    in tokenizer.py.
    """
    ctx_str = ", ".join(f"{k}={context.get(k, '')}" for k in CONTEXT_KEYS)
    return f"Context: {ctx_str} -> Notification:"


if __name__ == "__main__":
    example_context = {
        "scenario": "New Flavor",
        "product": "Blue Velvet Cloud",
        "category": "Ice Cream",
        "customer_type": "New",
        "user_activity": "Browsing",
        "time_of_day": "Afternoon",
        "day_type": "Weekday",
        "season": "Spring",
        "weather": "Clear",
        "discount": "15%",
        "urgency": "Medium",
        "tone": "Exciting",
        "emoji": "☁️",
    }
    prompt = build_prompt(example_context)
    print(prompt)
    print(
        "\nPass this to generate.py, e.g.:\n"
        f'  python -m V3.model.generate --checkpoint_dir V3/checkpoints '
        f'--vocab_file push_notif_tokenizer-vocab.json '
        f'--merges_file push_notif_tokenizer-merges.txt '
        f'--prompt "{prompt}"'
    )
