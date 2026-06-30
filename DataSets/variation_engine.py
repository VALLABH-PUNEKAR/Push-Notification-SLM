"""
Ice Cream App Notification Dataset - Variation Engine (Final)
================================================================
Usage:
    python variation_engine.py --target 1000000 --output final_dataset.csv
"""

import csv
import json
import random
import re
import argparse


def load_menu(menu_path):
    with open(menu_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_seeds(seed_csv_path):
    with open(seed_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    return [r[0] for r in rows[1:]]


SPICY_CONTEXT_RE = re.compile(r'\b(spicy|chili|chilli|kick|fiery|heat)\b', re.IGNORECASE)
NO_SWAP_PREFIXES = ('Flavor spotlight:',)


def build_templates(seeds, menu):
    flavor_objs = menu['ice_cream_flavors'] + menu['bakery_sweets']
    flavor_names = [f['name'] for f in flavor_objs]
    flavors_sorted = sorted(flavor_names, key=len, reverse=True)

    templates = []
    for text in seeds:
        if text.startswith(NO_SWAP_PREFIXES):
            templates.append({
                'template': text,
                'flavor_slots': 0,
                'wants_spicy': False,
            })
            continue

        new_text = text
        slot_count = 0
        for fl in flavors_sorted:
            if fl in new_text:
                new_text = new_text.replace(fl, '{FLAVOR}')
                slot_count += 1
        templates.append({
            'template': new_text,
            'flavor_slots': slot_count,
            'wants_spicy': bool(SPICY_CONTEXT_RE.search(text)),
        })
    return templates


def get_flavor_pools(menu):
    flavor_objs = menu['ice_cream_flavors'] + menu['bakery_sweets']
    all_names = [f['name'] for f in flavor_objs]
    spicy_names = [f['name'] for f in flavor_objs if f.get('spicy')]
    non_spicy_names = [f['name'] for f in flavor_objs if not f.get('spicy')]
    return all_names, spicy_names, non_spicy_names


PERCENT_POOL = [10, 15, 20, 25, 30, 35, 40, 45, 50]
RUPEE_POOL = [49, 99, 149, 199, 249, 299, 349, 399, 499]
MINUTES_POOL = [10, 15, 20, 25, 30, 45]
HOURS_POOL = [1, 2, 3, 4, 5, 6]
CLOCK_POOL = ['12 AM', '1 AM', '2 AM', '11 PM', '10 PM', '9 PM']

PERCENT_RE = re.compile(r'\d{1,2}%')
RUPEE_RE = re.compile(r'₹\d{2,4}')
MINUTES_RE = re.compile(r'\b\d{1,2} minutes?\b')
HOURS_RE = re.compile(r'\b\d{1,2} hours?\b')
CLOCK_RE = re.compile(r'\b\d{1,2} ?(AM|PM)\b')


def vary_numbers(text, rng, probability=0.5):
    if len(PERCENT_RE.findall(text)) >= 2:
        return text
    if rng.random() > probability:
        return text

    def sub_percent(m):
        return f"{rng.choice(PERCENT_POOL)}%"

    def sub_rupee(m):
        return f"₹{rng.choice(RUPEE_POOL)}"

    def sub_minutes(m):
        n = rng.choice(MINUTES_POOL)
        return f"{n} minute" + ("s" if n != 1 else "")

    def sub_hours(m):
        n = rng.choice(HOURS_POOL)
        return f"{n} hour" + ("s" if n != 1 else "")

    def sub_clock(m):
        return rng.choice(CLOCK_POOL)

    text = PERCENT_RE.sub(sub_percent, text)
    text = RUPEE_RE.sub(sub_rupee, text)
    text = MINUTES_RE.sub(sub_minutes, text)
    text = HOURS_RE.sub(sub_hours, text)
    text = CLOCK_RE.sub(sub_clock, text)
    return text


SYNONYM_SWAPS = {
    'right now': ['immediately', 'this instant', 'right away', 'at once'],
    'today only': ['for today only', 'just for today', 'today alone'],
    'order now': ['order today', 'grab yours now', 'get it now', 'order it now'],
    'on us': ['free', 'complimentary', 'our treat'],
    "don't miss it": ["don't miss out", "don't sleep on it", 'act fast'],
    'limited stock': ['limited quantity', 'while stocks last', 'limited batch'],
    'free delivery': ['free shipping', 'no delivery fee', 'zero delivery cost'],
}


def vary_synonyms(text, rng, probability=0.35):
    for phrase, alternatives in SYNONYM_SWAPS.items():
        if phrase in text.lower() and rng.random() < probability:
            idx = text.lower().find(phrase)
            if idx != -1:
                replacement = rng.choice(alternatives)
                text = text[:idx] + replacement + text[idx + len(phrase):]
    return text


EMOJI_GENERAL = ["🍦", "🍨", "🍧", "🍫", "🧁", "🎂", "🍪", "✨", "💛", "🤎", "💙", "❤️"]
EMOJI_WEATHER = ["☀️", "🌞", "🌧️", "☔", "❄️", "🥶", "🔥", "🌡️", "⛈️", "☁️"]
EMOJI_FRUIT = ["🥭", "🍓", "🥥", "🍉", "🫐", "🍋"]
EMOJI_MOOD = ["😄", "😅", "😊", "😍", "🤗", "😌", "🥳", "👀"]

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]"
)


def vary_emoji(text, rng, probability=0.25):
    if rng.random() > probability:
        return text
    matches = list(EMOJI_RE.finditer(text))
    if not matches:
        return text

    safe_matches = matches[1:] if len(matches) > 1 else matches

    m = rng.choice(safe_matches)
    original_emoji = m.group(0)
    if original_emoji in EMOJI_WEATHER:
        pool = EMOJI_WEATHER
    elif original_emoji in EMOJI_FRUIT:
        pool = EMOJI_FRUIT
    elif original_emoji in EMOJI_MOOD:
        pool = EMOJI_MOOD
    elif original_emoji in EMOJI_GENERAL:
        pool = EMOJI_GENERAL
    else:
        return text
    replacement = rng.choice([e for e in pool if e != original_emoji] or pool)
    return text[:m.start()] + replacement + text[m.end():]


def fill_flavor_slots(template_str, slot_count, wants_spicy, all_names, spicy_names, non_spicy_names, rng):
    if slot_count == 0:
        return template_str
    pool = spicy_names if (wants_spicy and spicy_names) else non_spicy_names
    if len(pool) < slot_count:
        pool = all_names
    chosen = rng.sample(pool, k=min(slot_count, len(pool)))
    if slot_count > len(chosen):
        chosen += [rng.choice(pool) for _ in range(slot_count - len(chosen))]
    result = template_str
    for fl in chosen:
        result = result.replace('{FLAVOR}', fl, 1)
    return result


def generate_row(template_entry, all_names, spicy_names, non_spicy_names, rng):
    text = fill_flavor_slots(
        template_entry['template'],
        template_entry['flavor_slots'],
        template_entry['wants_spicy'],
        all_names, spicy_names, non_spicy_names,
        rng,
    )
    text = vary_numbers(text, rng)
    text = vary_synonyms(text, rng)
    text = vary_emoji(text, rng)
    return text


def generate_dataset(templates, menu, target_rows, output_path, max_attempts_per_row=8, chunk_size=10000, seed=None):
    rng = random.Random(seed)
    all_names, spicy_names, non_spicy_names = get_flavor_pools(menu)

    seen = set()
    written = 0

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['text'])

        buffer = []
        while written < target_rows:
            template_entry = rng.choice(templates)
            row_text = None
            for _ in range(max_attempts_per_row):
                candidate = generate_row(template_entry, all_names, spicy_names, non_spicy_names, rng)
                if candidate not in seen:
                    row_text = candidate
                    break
            if row_text is None:
                continue

            seen.add(row_text)
            buffer.append(row_text)
            written += 1

            if len(buffer) >= chunk_size:
                writer.writerows([[t] for t in buffer])
                buffer = []
                print(f"  ...{written:,} / {target_rows:,} rows written")

        if buffer:
            writer.writerows([[t] for t in buffer])

    print(f"Done. {written:,} rows written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Scale up ice cream app notification dataset.")
    parser.add_argument('--seeds', default='icecream_notifications_seed_FINAL.csv')
    parser.add_argument('--menu', default='menu_v2.json')
    parser.add_argument('--target', type=int, default=10000)
    parser.add_argument('--output', default='icecream_notifications_generated.csv')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    print(f"Loading seeds from {args.seeds} ...")
    seeds = load_seeds(args.seeds)
    print(f"  {len(seeds):,} seed rows loaded.")

    print(f"Loading menu from {args.menu} ...")
    menu = load_menu(args.menu)

    print("Building templates ...")
    templates = build_templates(seeds, menu)
    flavor_templates = sum(1 for t in templates if t['flavor_slots'] > 0)
    generic_templates = len(templates) - flavor_templates
    print(f"  {flavor_templates:,} flavor-slotted templates, {generic_templates:,} generic templates.")

    print(f"Generating {args.target:,} rows -> {args.output}")
    generate_dataset(templates, menu, args.target, args.output, seed=args.seed)


if __name__ == '__main__':
    main()
