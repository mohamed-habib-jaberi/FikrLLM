"""
tokenizer.py -- loads the bilingual BPE tokenizer trained in ch03
"""

from pathlib import Path

from tokenizers import Tokenizer as BackingTokenizer

DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "assets" / "tokenizer.json"

class Tokenizer:

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    SEP = 4
    AR = 5
    EN = 6
    SYS = 7
    USER = 8
    ASST = 9

    _SPECIAL_TOKENS = {  # noqa: RUF012
        "[PAD]": PAD, "[UNK]": UNK, "[BOS]": BOS, "[EOS]": EOS, "[SEP]": SEP,
        "[AR]": AR, "[EN]": EN, "[SYS]": SYS, "[USER]": USER, "[ASST]": ASST,
    }

    def __init__(self, backing_tokenizer: BackingTokenizer):
        self._tokenizer = backing_tokenizer

    @classmethod
    def from_file(cls, path=DEFAULT_TOKENIZER_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No tokenizer at {path}."
            )

        backing_tokenizer = BackingTokenizer.from_file(str(path))
        vocab = backing_tokenizer.get_vocab()
        mismatches = {
            token: (expected_id, vocab.get(token))
            for token, expected_id in cls._SPECIAL_TOKENS.items()
            if vocab.get(token) != expected_id
        }
        if mismatches:
            details = ", ".join(
                f"{token}: expected {expected}, got {actual}"
                for token, (expected, actual) in mismatches.items()
            )
            raise ValueError(f"Tokenizer special-token IDs do not match: {details}")

        return cls(backing_tokenizer)


    def normalize(self, text: str) -> str:
        return self._tokenizer.normalizer.normalize_str(text)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[list[int]]:
        encodings = self._tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )

        return [
            encoding.ids
            for encoding in encodings
        ]

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
