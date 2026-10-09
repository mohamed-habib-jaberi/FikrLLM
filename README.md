# FikrLLM — guide pédagogique complet

FikrLLM est un GPT bilingue arabe–anglais construit avec PyTorch. Ce dépôt est pensé pour comprendre les mécanismes : préparation de données, tokenizer BPE, Transformer, entraînement, génération et adaptation chat.

## Vue générale

Un LLM ne lit pas directement du texte : il manipule des identifiants numériques.

```text
corpus brut
  → nettoyer et filtrer
  → entraîner le tokenizer BPE
  → texte → IDs de tokens
  → écrire train.bin / eval.bin
  → entraîner GPT à prédire le token suivant
  → sauvegarder un checkpoint
  → générer du texte ou adapter le modèle au chat
```

| Étape | Fichiers principaux | Produit |
|---|---|---|
| Données | `scripts/ch03-build-tokenizer/01_clean_corpus.py`, `cleaning.py` | corpus tokenizer et paires Q/R |
| Tokenizer | `02_build_tokenizer.py`, `fikrllm/tokenizer.py` | `tokenizer.json`, texte ↔ IDs |
| Packing | `03_pack_corpus.py`, `fikrllm/data/dataset.py` | `train.bin`, `eval.bin`, `meta.json` |
| Modèle | `fikrllm/model/` | GPT qui donne un score à chaque token |
| Entraînement | `run_pretrain.py`, `fikrllm/training/` | checkpoints et métriques |
| Génération | `fikrllm/generation/` | continuation / réponse |
| Fine-tuning | `run_finetune_lora.py`, `data/chat.py`, `model/lora.py` | assistant Q/R |

## Installation

Python 3.10+ est nécessaire.

```bash
conda create --name fikrllm python=3.11 pip -y
conda activate fikrllm
cd FikrLLM
python -m pip install -e .
```

`-e` signifie *editable* : modifier `fikrllm/` rend le changement immédiatement disponible sans réinstaller. `pyproject.toml` définit les dépendances : PyTorch (réseau), Tokenizers (BPE), Pandas/PyArrow (Parquet), Hugging Face (données) et TrackIO (métriques).

## Arborescence expliquée

```text
FikrLLM/
├── fikrllm/
│   ├── config.py          dimensions et hyperparamètres
│   ├── tokenizer.py       charge tokenizer.json, encode et décode
│   ├── model/             embeddings, attention, GPT, LoRA
│   ├── data/              fichiers binaires et batches
│   ├── generation/        sampling et boucle auto-régressive
│   ├── training/          optimisation, checkpoints, tracking
│   └── assets/            tokenizer et notebook d'exploration
├── scripts/
│   ├── ch03-build-tokenizer/       pipeline minimal
│   ├── wiki_data_collecting/       collecte bilingue étendue
│   └── clean_and_build_dataset.py  fusion de données étendues
├── run_pretrain.py        pré-entraînement
├── run_finetune_lora.py   adaptation chat LoRA
└── pyproject.toml         package et dépendances
```

Les `__init__.py` ré-exportent les objets publics afin que `from fikrllm import GPT` fonctionne. Ils ne lancent aucune tâche.

---

## 1. Corpus : nettoyage et normalisation

Wikipedia contient du contenu utile mais aussi des catégories, références, liens, gabarits et répétitions. `cleaning.py` distingue deux opérations, qu'il ne faut pas confondre.

| Type | Fonctions | Quand ? | But |
|---|---|---|---|
| A — nettoyage structurel | `prepare_document()`, `clean_dataframe()` | une fois, hors entraînement | enlever le bruit des dumps |
| B — normalisation légère | `build_normalizer()`, `normalize_text()` | tokenizer **et** inférence | garantir les mêmes règles de texte |

Le nettoyage Type A retire les restes de citations (`cite web`, `استشهاد ويب`), les lignes de catégories, les sections « References / See also », les sections trop courtes et les doublons consécutifs. C'est destructif : on ne doit jamais l'appliquer à une question libre de l'utilisateur.

Le Type B applique NFC, supprime les diacritiques/tatweel arabes, unifie `أ إ آ` vers `ا`, `ى` vers `ي`, puis réduit les espaces. Ces règles sont embarquées dans `tokenizer.json`; `Tokenizer.encode()` les réapplique donc automatiquement aux prompts.

Test local, sans téléchargement ni entraînement :

```bash
cd scripts/ch03-build-tokenizer
python -c '
from cleaning import normalize_text
for x in ["جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّة", "أحمد إبراهيم آدم علىّ", "The   United    States"]:
    print(repr(x), "->", repr(normalize_text(x)))
'
```

### `01_clean_corpus.py`

Ce script télécharge des shards Parquet, prépare les fichiers de chat et fabrique l'échantillon qui entraîne le tokenizer :

```bash
python scripts/ch03-build-tokenizer/01_clean_corpus.py \
  --max-shards 1 --tokenizer-docs 1000
```

| Sortie dans `scripts/ch03-build-tokenizer/output/` | Usage |
|---|---|
| `tokenizer_corpus.parquet` | documents Type A nettoyés pour `02_build_tokenizer.py` |
| `ft_train_filtered.parquet` | paires question/réponse pour le fine-tuning |
| `ft_eval_filtered.parquet` | paires question/réponse de validation |

`--max-shards 1` et `--tokenizer-docs 1000` sont adaptés aux tests; les valeurs par défaut téléchargent plusieurs Go et ciblent 400 000 documents.

`03_pack_corpus.py` applique le même nettoyage Type A à chaque shard avant la tokenisation. La normalisation Type B reste ensuite appliquée automatiquement par le tokenizer. Le fichier `meta.json` conserve aussi le hash SHA-256 du tokenizer, ce qui permet de vérifier précisément avec quel vocabulaire les binaires ont été construits.

---

## 2. Tokenizer : du texte aux IDs

BPE (*Byte Pair Encoding*) apprend des sous-morceaux fréquents. Il ne découpe ni strictement par mots ni strictement par lettres : un mot courant peut devenir un seul token, tandis qu'un mot rare peut devenir plusieurs morceaux. Chaque morceau est représenté par un entier.

`02_build_tokenizer.py` lit `tokenizer_corpus.parquet`, entraîne un vocabulaire de 32 000 tokens et sauvegarde `tokenizer.json`.

```bash
python scripts/ch03-build-tokenizer/02_build_tokenizer.py
```

| Token | ID | Signification |
|---|---:|---|
| `[PAD]` | 0 | remplissage ignoré par la loss |
| `[UNK]` | 1 | élément inconnu |
| `[BOS]` / `[EOS]` | 2 / 3 | début / fin de document ou réponse |
| `[SEP]` | 4 | séparateur disponible |
| `[AR]` / `[EN]` | 5 / 6 | marqueurs de langue disponibles |
| `[SYS]`, `[USER]`, `[ASST]` | 7 / 8 / 9 | rôles conversationnels |

Le script sauvegarde directement le tokenizer officiel dans `fikrllm/assets/tokenizer.json`. Le packing, les notebooks et l'entraînement chargent tous ce même fichier afin d'éviter toute divergence d'IDs.

### `fikrllm/tokenizer.py`

Ce module n'entraîne rien : il charge le tokenizer sauvegardé et présente une interface stable au reste du projet.

```python
from fikrllm.tokenizer import Tokenizer

tok = Tokenizer.from_file()
ids = tok.encode("أحمد lives in Cairo")
morceaux = tok.tokenize("أحمد lives in Cairo")
texte = tok.decode(ids)
```

`from_file()` vérifie le fichier, laisse la bibliothèque Hugging Face reconstruire le tokenizer BPE, puis renvoie l'objet FikrLLM. Cet objet expose aussi `tok.BOS`, `tok.EOS`, `tok.USER` et `tok.ASST`; ainsi les autres modules ne recopient jamais les IDs spéciaux à la main.

Le notebook `fikrllm/assets/notebooks/tokenizer_embeddings.ipynb` sert de banc de test interactif pour `fikrllm/tokenizer.py`. Il permet de vérifier le chargement de `tokenizer.json`, la normalisation arabe–anglaise, la tokenisation, l'encodage, le décodage et la transformation des IDs en embeddings, sans lancer un entraînement complet.

---

## 3. Packing : construire un dataset performant

Encoder des chaînes à chaque epoch serait lent. `03_pack_corpus.py` encode les documents une fois puis écrit un flux compact de `uint16`. Ce format suffit, car 32 000 IDs tiennent dans la plage 0–65 535.

```bash
python scripts/ch03-build-tokenizer/03_pack_corpus.py --max-shards 1
```

Le script ajoute les frontières de documents **avant** de concaténer :

```text
document A  [BOS] a b c [EOS]
document B  [BOS] d e   [EOS]
train.bin   [BOS] a b c [EOS] [BOS] d e [EOS] ...
```

| Fichier | Usage |
|---|---|
| `train.bin` | tokens qui mettent à jour les poids |
| `eval.bin` | corpus séparé pour mesurer la généralisation |
| `meta.json` | provenance, nombre de tokens et vocabulaire |

`data/dataset.py` utilise `numpy.memmap`: le gros fichier reste sur disque et seules les fenêtres demandées sont lues. Une fenêtre a `max_seq_len + 1` tokens, afin que `data/collate.py` crée la tâche de prédiction suivante :

```text
fenêtre : [BOS, t1, t2, t3, EOS]
entrée  : [BOS, t1, t2, t3]
cible   : [t1,   t2, t3, EOS]
```

Le modèle apprend « étant donné ce qui précède, prédire ce qui suit ». Le padding ajouté à la fin des batches est ignoré dans la loss.

---

## 4. Architecture GPT : module par module

`config.py` est la source de vérité des dimensions. `ModelConfig.fikrllm()` crée le modèle normal (8 blocs, largeur 512, 8 têtes, contexte 1024). `ModelConfig.tiny()` crée un modèle CPU rapide (2 blocs, largeur 128, contexte 256). `d_model` doit être divisible par `n_heads`, car une tête reçoit `d_model / n_heads` dimensions.

| Fichier | Rôle |
|---|---|
| `model/embeddings.py` | transforme IDs et positions en vecteurs, puis applique dropout |
| `model/attention.py` | Query, Key, Value, attention multi-têtes et masque causal |
| `model/feedforward.py` | MLP par position : `d_model → d_ff → d_model` avec GELU |
| `model/block.py` | LayerNorm, attention résiduelle, puis MLP résiduel |
| `model/gpt.py` | assemble les couches et produit un score par token du vocabulaire |
| `model/lora.py` | ajoute puis fusionne les adaptateurs de fine-tuning |

`GPT.forward(input_ids)` retourne des logits de forme `(batch, longueur, vocab_size)`. Un logit est un score, pas une probabilité. Le masque causal interdit à une position de regarder les tokens futurs : sans lui, le modèle verrait la réponse pendant l'entraînement et tricherait.

`gpt.py` utilise le *weight tying* : les poids de l'embedding et de la projection finale sont partagés. Cela réduit les paramètres et emploie un même espace pour lire et scorer un token.

---

## 5. Pré-entraînement

`run_pretrain.py` relie tokenizer, binaires, DataLoader, modèle et `Trainer`.

Commence par un test complet sur CPU :

```bash
python run_pretrain.py --tiny --steps 60 --warmup 10 \
  --limit 2000 --eval-limit 500 --batch-size 8 \
  --print-every 10 --sample-every 30 --device cpu
```

`--limit` restreint les fenêtres et `--steps` les mises à jour. C'est un smoke test : il valide le code mais ne crée pas un modèle de langue compétent.

### `training/trainer.py`

Une mise à jour suit ce chemin :

1. déplacer le batch vers CPU/GPU ;
2. faire passer les IDs dans GPT ;
3. comparer logits et cibles avec la cross-entropy ;
4. appeler `backward()` ;
5. limiter éventuellement la norme des gradients ;
6. appliquer AdamW ;
7. avancer le learning rate ;
8. selon la configuration, faire validation, sauvegarde et génération d'exemples.

`document_cross_entropy()` moyenne d'abord les tokens d'un exemple, puis les exemples. Cela évite qu'un document long domine une mise à jour uniquement grâce à son nombre de tokens.

L'accumulation permet de simuler un batch plus grand sans le garder en mémoire. `--batch-size 8 --accumulation-steps 4` correspond à quatre micro-batches de 8 et à un batch effectif de 32.

`training/schedule.py` réalise un warmup linéaire suivi d'une décroissance cosinus. `training/checkpoint.py` enregistre les poids, le `ModelConfig`, le step et, pour le pré-entraînement, l'optimiseur et le scheduler afin que `--resume` reprenne réellement l'entraînement.

```bash
python run_pretrain.py --epochs 15 \
  --batch-size 56 --accumulation-steps 4 \
  --lr 6e-4 --warmup 500 --weight-decay 0.1 \
  --eval-every 100 --print-every 100 --sample-every 100 \
  --shuffle-seed 42 \
  --checkpoint-dir ../checkpoints/fikrllm-512x8-ep15 \
  --save-every 100
```

---

## 6. Génération

`generation/generate.py` répète :

```text
prompt → GPT → logits du dernier token → choisir un ID → ajouter l'ID → recommencer
```

| Paramètre dans `GenerationConfig` | Effet |
|---|---|
| `temperature=0` | greedy : toujours le token le plus probable |
| température < 1 | texte moins varié |
| température > 1 | texte plus varié mais plus risqué |
| `top_k=50` | garde les 50 meilleurs candidats |
| `top_p=0.95` | garde les candidats qui totalisent 95 % de probabilité |

`sampling.py` implémente ces règles. La génération emploie en général un cache KV : les clés/valeurs d'attention des anciens tokens sont conservées, afin de ne pas recalculer tout le prompt à chaque nouveau token.

```bash
python - <<'PY'
from fikrllm import GPT, Tokenizer, GenerationConfig, generate
from fikrllm.training import load_checkpoint, model_config_from_checkpoint

path = "../checkpoints/fikrllm-512x8-ep15/pretrain_final.pt"
model = GPT(model_config_from_checkpoint(path))
load_checkpoint(path, model)
print(generate(model, Tokenizer.from_file(), "The Nile flows", GenerationConfig.greedy()))
PY
```

---

## 7. Fine-tuning chat avec LoRA

Un modèle pré-entraîné complète du texte; il ne sait pas automatiquement respecter le format question-réponse. `data/chat.py` transforme une paire Q/R en :

```text
[BOS] [SYS] instruction [USER] question [ASST] réponse [EOS]
```

La loss ignore toute la partie prompt et ne note que la réponse. Le modèle n'est donc pas entraîné à reproduire les questions utilisateur.

`model/lora.py` gèle les poids de base et apprend une petite correction de rang `r` :

```text
sortie = xW + (alpha / r) × x Aᵀ Bᵀ
```

Au lieu de modifier chaque grande matrice `W`, LoRA apprend `A` et `B` sur les projections d'attention `W_q`, `W_k`, `W_v`, `W_o`. Après l'entraînement, `merge_lora()` replie ces corrections dans les matrices ordinaires.

```bash
python run_finetune_lora.py \
  --resume ../checkpoints/fikrllm-512x8-ep15/pretrain_final.pt \
  --steps 60 --warmup 10 --limit 2000 --batch-size 8 \
  --lr 2e-5 --lora-r 8 --lora-alpha 16 \
  --print-every 10 --sample-every 30
```

`--resume` est essentiel en pratique : LoRA appliqué à un réseau aléatoire ne produit pas un assistant utile.

---

## Pipeline bilingue étendu

Ces scripts sont facultatifs pour le parcours minimal et font des appels réseau. Ils permettent de reconstruire un corpus Égypte/MENA arabe–anglais plus riche.

| Fichier | Rôle |
|---|---|
| `wiki_data_collecting/step05-collect-english-base-model-dataset.py` | récupère les équivalents anglais des pages et produit les JSONL des phases 1/2 |
| `wiki_data_collecting/step06-translate-finetune-dataset-to-english.py` | traduit ensemble titre, question et réponse avec vLLM/TranslateGemma pour préserver les noms propres |
| `wiki_data_collecting/step07-augment-english-dataset.py` | récupère résumés de figures et catégories Wikipedia EN/AR, avec débit limité et déduplication |
| `clean_and_build_dataset.py` | normalise le schéma, filtre bruit/stubs, tronque les textes et peut publier sur Hugging Face |

Ordre logique : étape 05, étape 06, étape 07 si nécessaire, puis `clean_and_build_dataset.py --no-upload` pour inspecter les sorties avant publication.

## Suivi et dépannage

Les métriques sont enregistrées avec TrackIO :

```bash
trackio show --project fikrllm --host 0.0.0.0
```

Checklist :

1. `Tokenizer.from_file()` échoue : vérifie `fikrllm/assets/tokenizer.json`.
2. `PackedDataset` échoue : crée `train.bin`/`eval.bin` avec `03_pack_corpus.py` ou utilise le bon `--data-dir`.
3. Un checkpoint refuse de charger : construis le GPT avec `model_config_from_checkpoint(path)`, pas une configuration différente.
4. Mémoire insuffisante : utilise `--tiny`, diminue `--batch-size`, augmente l'accumulation et commence avec `--limit`.
5. Texte incohérent après quelques steps : c'est normal pour un smoke test.
6. Loss `nan` ou instable : réduis le learning rate et examine les données.

## Parcours minimal reproductible

```bash
python -m pip install -e .

python scripts/ch03-build-tokenizer/01_clean_corpus.py --max-shards 1 --tokenizer-docs 1000
python scripts/ch03-build-tokenizer/02_build_tokenizer.py
python scripts/ch03-build-tokenizer/03_pack_corpus.py --max-shards 1

python run_pretrain.py --tiny --steps 10 --warmup 2 --limit 100 --eval-limit 20 \
  --batch-size 2 --device cpu --print-every 1 --sample-every 0
```

Cette dernière commande vérifie que tokenizer, données, modèle, loss et optimisation coopèrent. Elle ne suffit pas à apprendre une langue : un modèle utile demande davantage de données, de steps et généralement un GPU.
