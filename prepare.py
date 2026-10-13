"""Download Alice in Wonderland and make a contiguous 90/10 split."""
import argparse
import re
import urllib.request
from pathlib import Path

from tokenizer import CharTokenizer

URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
DATA = Path(__file__).resolve().parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="Use a local UTF-8 text instead of downloading.")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    if args.source:
        text = args.source.read_text(encoding="utf-8")
        source = str(args.source)
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "tiny-gpt-educational/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8-sig")
        (DATA / "alice-original.txt").write_text(raw, encoding="utf-8")
        start = re.search(r"\*\*\* START OF .*?\*\*\*", raw)
        end = re.search(r"\*\*\* END OF .*?\*\*\*", raw)
        if start is None or end is None or start.end() >= end.start():
            raise ValueError("Gutenberg boundary markers missing; inspect the source before using it.")
        text = raw[start.end():end.start()].strip() + "\n"
        source = URL
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) < 2000:
        raise ValueError("Provide at least 2,000 characters for the training/validation split.")
    split = int(len(text) * 0.9)
    train, val = text[:split], text[split:]
    # Only the alphabet is shared: validation text is never a training target.
    # Using the full alphabet avoids unknown validation characters.
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(DATA / "tokenizer.json")
    (DATA / "train.txt").write_text(train, encoding="utf-8")
    (DATA / "val.txt").write_text(val, encoding="utf-8")
    (DATA / "source.txt").write_text(f"Source: {source}\nSplit: contiguous 90% train / 10% validation\n", encoding="utf-8")
    print(f"Train: {len(train):,} characters; validation: {len(val):,}")
    print(f"Vocabulary: {tokenizer.vocab_size} characters")
    sample = train[:60]
    print(f"Text: {sample!r}\nIDs: {tokenizer.encode(sample)}")
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print("Full-text encode/decode round trip: PASS")


if __name__ == "__main__":
    main()
