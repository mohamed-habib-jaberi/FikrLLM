"""Shared interface between text and the IDs understood by GPT.

The BPE vocabulary and normalization rules are not learned here: they are
loaded from ``assets/tokenizer.json``, produced by chapter 3. This class wraps
the Hugging Face tokenizer so the whole project uses the same API and the same
IDs for special tokens.

Main flow::

    raw text   -> normalize/tokenize/encode -> IDs -> GPT model
    final text <- decode                   <- IDs <- GPT model

The notebook ``assets/notebooks/tokenizer_embeddings.ipynb`` can be used to
test these operations interactively without launching model training.
"""

from pathlib import Path

from tokenizers import Tokenizer as BackingTokenizer

DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "assets" / "tokenizer.json"


class Tokenizer:
    """Load and use the bilingual Arabic-English tokenizer for FikrLLM.

    The class delegates BPE operations to ``tokenizers.Tokenizer`` and exposes
    plain Python lists of IDs or token pieces. Constants such as ``BOS``,
    ``EOS``, and ``USER`` avoid scattering magic numbers across packing,
    fine-tuning, and generation.
    """

    # These IDs must stay synchronized with the SPECIAL_TOKENS order in
    # scripts/ch03-build-tokenizer/02_build_tokenizer.py.
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
        "[PAD]": PAD,
        "[UNK]": UNK,
        "[BOS]": BOS,
        "[EOS]": EOS,
        "[SEP]": SEP,
        "[AR]": AR,
        "[EN]": EN,
        "[SYS]": SYS,
        "[USER]": USER,
        "[ASST]": ASST,
    }

    # Initialize this object with the state it needs later.
    def __init__(self, backing_tokenizer: BackingTokenizer) -> None:
        """Keep the Hugging Face tokenizer that does the actual work."""
        self._tokenizer = backing_tokenizer

    # Purpose: from file.
    @classmethod
    def from_file(
        cls, path: str | Path = DEFAULT_TOKENIZER_PATH
    ) -> "Tokenizer":
        """Build a ``Tokenizer`` from a saved JSON file.

        In addition to loading the vocabulary, this method verifies special
        token IDs. That validation prevents silently using a vocabulary that is
        incompatible with the data or checkpoints.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No tokenizer at {path}.")

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

    # Normalize text with the tokenizer backend rules.
    def normalize(self, text: str) -> str:
        """Apply only the tokenizer normalization rules."""
        return self._tokenizer.normalizer.normalize_str(text)

    # Convert one text string into token IDs.
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Normalize and convert one text string into integer IDs."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    # Convert many text strings into token ID sequences.
    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = False
    ) -> list[list[int]]:
        """Encode several texts and return one ID list per text."""
        encodings = self._tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )

        return [encoding.ids for encoding in encodings]

    # Convert text into readable token pieces.
    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[str]:
        """Return readable BPE pieces instead of their IDs."""
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).tokens

    # Convert token IDs back into text.
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Rebuild text from IDs, hiding control tokens by default."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    # Return the number of tokens known by the tokenizer.
    @property
    def vocab_size(self) -> int:
        """Total number of tokens the model can predict."""
        return self._tokenizer.get_vocab_size()
