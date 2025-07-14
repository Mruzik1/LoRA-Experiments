import json
import re
import argparse
from pathlib import Path


def is_bad_text(text, min_len=5, max_len=1000):
    """
    Heuristic to detect bad messages:
    - Empty or whitespace-only
    - Too short (< min_len)
    - Too long (> max_len)
    - Contains only emojis or punctuation
    - Contains disallowed content (e.g., spammy URLs)
    Supports both Latin and Cyrillic alphabets.
    """
    if not text or not text.strip():
        return True
    length = len(text)
    if length < min_len or length > max_len:
        return True
    cleaned = re.sub(r"[^\w\s\u0400-\u04FF]", "", text)  # keep cyrillic
    if not re.search(r"[A-Za-z\u0400-\u04FF0-9]", cleaned):
        return True
    if re.search(r"https?://\S+", text):
        return True
    return False


def clean_json(input_path, output_path, min_len=5, max_len=1000):
    input_path = Path(input_path)
    output_path = Path(output_path)

    removed = 0
    kept = 0

    with input_path.open('r', encoding='utf-8') as fin, output_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            rec = json.loads(line)
            orig = rec.get('original_content', '')
            reply = rec.get('reply_content', '')
            if is_bad_text(orig, min_len, max_len) or is_bad_text(reply, min_len, max_len):
                removed += 1
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            kept += 1

    print(f"Removed {removed} bad pairs, kept {kept} good pairs.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean JSON training data by removing bad message pairs.')
    parser.add_argument('input', help='Path to the raw JSON file')
    parser.add_argument('output', help='Path to save the cleaned JSON file')
    parser.add_argument('--min_len', type=int, default=5, help='Minimum allowed message length')
    parser.add_argument('--max_len', type=int, default=1000, help='Maximum allowed message length')
    args = parser.parse_args()
    clean_json(args.input, args.output, args.min_len, args.max_len)