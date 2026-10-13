"""Run stage-one checks without needing PyTorch installed."""
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizer import CharTokenizer


def main():
    text = "Hello, GPT!\nCafé — hello."
    tokenizer = CharTokenizer.from_text(text)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert len(ids) == len(text)
    assert all(0 <= i < tokenizer.vocab_size for i in ids)
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        tokenizer.save(path)
        restored = CharTokenizer.load(path)
        assert restored.encode(text) == ids
    for operation in (lambda: tokenizer.encode("🦊"), lambda: tokenizer.decode([-1])):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input must fail clearly")
    print("PASS: Unicode round trip, token count, ID bounds, empty input, saved vocabulary, invalid input")


if __name__ == "__main__":
    main()
