"""Source de vérité des paramètres de FikrLLM.

Ce fichier sépare trois questions différentes :

``ModelConfig``
    Quelle est la forme du réseau ? Ces valeurs doivent correspondre exactement
    au checkpoint : vocabulaire, largeur, nombre de blocs, têtes et contexte.
``GenerationConfig``
    Comment choisir les prochains tokens à partir des logits du modèle ?
``TrainingConfig``
    Comment optimiser, évaluer, suivre et sauvegarder les poids ?

Cette séparation permet de changer le learning rate ou la stratégie de
sampling sans changer l'architecture stockée dans un checkpoint.
"""

import math
import random
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelConfig:
    """Décrit toutes les dimensions qui déterminent la forme du GPT.

    Modifier l'un de ces champs change généralement la taille ou la structure
    des tenseurs du modèle. Un checkpoint ne peut donc être chargé que dans un
    ``GPT`` construit avec la même configuration.
    """

    vocab_size: int = 32_000  # Nombre de sorties possibles du tokenizer.
    d_model: int = 768  # Largeur d'un vecteur de token dans le réseau.
    d_ff: int | None = None  # MLP interne ; par défaut 4 * d_model, comme GPT-2.
    max_seq_len: int = 1024  # Nombre maximal de tokens dans le contexte.
    dropout: float = 0.1  # Régularisation appliquée pendant l'entraînement.
    n_layers: int = 12  # Nombre de blocs Transformer empilés.

    n_heads: int = 12  # Nombre de vues d'attention en parallèle.
    qkv_bias: bool = False  # Ajoute ou non un biais aux projections Q/K/V.
    pad_id: int = 0  # Cible ignorée par la loss.

    tie_weights: bool = True  # Partage embedding d'entrée et projection de sortie.

    def __post_init__(self):
        """Compléter les valeurs dérivées et refuser les formes impossibles."""
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )

    @property
    def d_k(self):
        """Largeur des vecteurs Query, Key et Value d'une seule tête."""
        return self.d_model // self.n_heads

    @classmethod
    def fikrllm(cls):
        """Configuration principale du projet : 8 blocs de largeur 512."""
        return cls(d_model=512, n_heads=8, n_layers=8, dropout=0.05)

    @classmethod
    def tiny(cls):
        """Petit modèle destiné aux tests rapides et aux machines sans GPU."""
        return cls(d_model=128, n_heads=4, n_layers=2, max_seq_len=256)


@dataclass
class GenerationConfig:
    """Contrôle la transformation des logits en texte.

    Le sampling est l'étape où le modèle peut devenir non déterministe. Les
    paramètres ci-dessous règlent le compromis entre cohérence et diversité.
    """

    max_new_tokens: int = 100  # Limite la longueur de la continuation.
    temperature: float = 0.8  # 0 = greedy ; < 1 prudent ; > 1 plus varié.

    top_k: int | None = 50  # Ne conserve que les k meilleurs candidats.
    top_p: float | None = 0.95  # Nucleus sampling : masse cumulée maximale.

    use_cache: bool = True  # Réutilise les Key/Value des tokens précédents.
    seed: int | None = None  # Rend l'échantillonnage reproductible.

    @classmethod
    def greedy(cls):
        """Toujours sélectionner le token le plus probable."""
        return cls(temperature=0.0, top_k=None, top_p=None)


@dataclass
class TrainingConfig:
    """Contrôle l'optimisation sans modifier la forme du modèle.

    Contrairement à ``ModelConfig``, ces valeurs peuvent changer lorsque les
    mêmes poids sont repris sur un nouveau corpus ou pour un fine-tuning.
    """

    # Optimiseur AdamW et stabilité des gradients.
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Durée et schedule du learning rate.
    max_steps: int = 10_000
    warmup_steps: int = 500

    min_lr_ratio: float = 0.1

    # Batch effectif = batch_size * accumulation_steps.
    batch_size: int = 24
    accumulation_steps: int = 1

    # Fréquence du suivi, de l'évaluation et des sauvegardes.
    log_every: int = 10
    print_every: int = 0
    eval_every: int = 500
    eval_batches: int = 50

    save_every: int = 1_000

    project: str = "fikrllm"
    run_name: str = "train-phase"

    checkpoint_dir: str = "checkpoints"
    seed: int = 0

    sample_every: int = 500
    sample_chat: bool = False  # Utiliser le template chat pendant le fine-tuning.
    sample_max_new_tokens: int = 60
    sample_prompts: tuple[str, ...] = (
        "وُلد في مدينة بغداد عام",  # biography
        "The early life of",  # biography
        "تقع هذه المدينة في شمال",  # geography
        "This city is located on the eastern coast of",  # geography
        "شهدت الحرب العالمية الثانية",  # history
        "During the nineteenth century,",  # history
        "يُعرف هذا الفنان بأعماله في مجال",  # arts
        "The film was directed by",  # arts
        "فاز الفريق بالبطولة بعد",  # sports
        "The scientific study found that",  # science
    )


def unique_run_name(base: str) -> str:
    """Créer un nom de run lisible et suffisamment unique pour TrackIO."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{random.randint(0, 9999):04d}-{base}-{stamp}"


def steps_for_epochs(num_examples, batch_size, accumulation_steps, epochs):
    """Convertir un nombre d'epochs en mises à jour réelles de l'optimiseur."""
    effective_batch = batch_size * accumulation_steps
    steps_per_epoch = math.ceil(num_examples / effective_batch)
    return max(1, math.ceil(epochs * steps_per_epoch))
