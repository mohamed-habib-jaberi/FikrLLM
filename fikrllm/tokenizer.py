"""Interface commune entre le texte et les identifiants compris par GPT.

Le vocabulaire BPE et les règles de normalisation ne sont pas appris ici :
ils sont chargés depuis ``assets/tokenizer.json``, produit par le chapitre 3.
Cette classe enveloppe le tokenizer Hugging Face afin que tout le projet utilise
la même API et les mêmes identifiants pour les tokens spéciaux.

Flux principal::

    texte brut -> normalize/tokenize/encode -> IDs -> modèle GPT
    texte final <- decode                 <- IDs <- modèle GPT

Le notebook ``assets/notebooks/tokenizer_embeddings.ipynb`` permet de tester
ces opérations interactivement sans lancer l'entraînement du modèle.
"""

from pathlib import Path

from tokenizers import Tokenizer as BackingTokenizer

DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "assets" / "tokenizer.json"


class Tokenizer:
    """Charge et utilise le tokenizer bilingue arabe–anglais de FikrLLM.

    La classe délègue les opérations BPE à ``tokenizers.Tokenizer`` et expose
    directement des listes Python d'IDs ou de morceaux. Les constantes
    ``BOS``, ``EOS``, ``USER`` et autres évitent de disperser des nombres
    magiques dans le packing, le fine-tuning et la génération.
    """

    # Ces IDs doivent rester synchronisés avec l'ordre de SPECIAL_TOKENS dans
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

    def __init__(self, backing_tokenizer: BackingTokenizer):
        """Conserver le tokenizer Hugging Face qui réalise le vrai travail."""
        self._tokenizer = backing_tokenizer

    @classmethod
    def from_file(cls, path=DEFAULT_TOKENIZER_PATH):
        """Construire un ``Tokenizer`` à partir d'un fichier JSON sauvegardé.

        En plus de charger le vocabulaire, cette méthode vérifie les IDs des
        tokens spéciaux. Cette validation empêche d'utiliser silencieusement
        un vocabulaire incompatible avec les données ou les checkpoints.
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

    def normalize(self, text: str) -> str:
        """Appliquer uniquement les règles de normalisation du tokenizer."""
        return self._tokenizer.normalizer.normalize_str(text)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Normaliser et convertir un texte en liste d'identifiants entiers."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = False
    ) -> list[list[int]]:
        """Encoder plusieurs textes et retourner une liste d'IDs par texte."""
        encodings = self._tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )

        return [encoding.ids for encoding in encodings]

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[str]:
        """Retourner les morceaux BPE lisibles plutôt que leurs IDs."""
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Reconstruire du texte depuis des IDs, en masquant les contrôles par défaut."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        """Nombre total de tokens que le modèle peut prédire."""
        return self._tokenizer.get_vocab_size()
