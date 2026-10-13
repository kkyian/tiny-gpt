"""A character maps to one integer; no pretrained vocabulary is used."""
import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, characters: list[str]):
        if not characters or any(len(c) != 1 for c in characters):
            raise ValueError("Vocabulary must contain single characters.")
        if len(set(characters)) != len(characters):
            raise ValueError("Vocabulary contains duplicates.")
        self.characters = list(characters)
        self.stoi = {char: index for index, char in enumerate(characters)}

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        unknown = set(text) - self.stoi.keys()
        if unknown:
            raise ValueError(f"Characters outside the vocabulary: {sorted(unknown)!r}")
        return [self.stoi[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        if any(type(i) is not int or not 0 <= i < self.vocab_size for i in ids):
            raise ValueError("Token IDs must be integers inside the vocabulary.")
        return "".join(self.characters[i] for i in ids)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.characters, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
