# Notes techniques FikrLLM

Ce fichier résume les notions techniques principales à retenir du projet FikrLLM. Les mots techniques importants sont gardés en anglais.

## 1. Vue globale

FikrLLM est un GPT bilingue Arabic-English construit from scratch avec PyTorch.

Le pipeline complet est :

```text
raw dataset
  -> cleaning
  -> tokenizer training
  -> tokenization
  -> binary packing
  -> Dataset + DataLoader
  -> GPT pretraining
  -> checkpoint
  -> generation
  -> LoRA chat fine-tuning
```

L'idée principale : un LLM ne s'entraîne pas directement sur du raw text. Le texte est converti en token IDs, puis le modèle apprend à prédire le next token.

## 2. Dataset et cleaning

Le raw dataset vient de Hugging Face sous forme de fichiers Parquet.

Fichiers importants :

| File | Rôle |
|---|---|
| `01_clean_corpus.py` | Télécharge les dataset shards et prépare les fichiers locaux. |
| `cleaning.py` | Nettoie le texte bruité et normalise Arabic/English. |
| `03_pack_corpus.py` | Convertit le texte nettoyé en binary token files. |

Il y a deux niveaux de cleaning :

| Type | But |
|---|---|
| Type A cleaning | Offline destructive cleaning : retirer bruit, références, catégories, doublons. |
| Type B normalization | Lightweight normalization utilisée par le tokenizer et l'inference. |

À retenir :

```text
Type A = seulement pour dataset preparation
Type B = safe pour tokenizer et inference
```

## 3. Tokenizer

Le tokenizer transforme le texte en token IDs.

Fichiers importants :

| File | Rôle |
|---|---|
| `02_build_tokenizer.py` | Entraîne le BPE tokenizer. |
| `fikrllm/tokenizer.py` | Charge `tokenizer.json` et expose `encode`, `decode`, `tokenize`. |
| `fikrllm/assets/tokenizer.json` | Tokenizer officiel utilisé par training et inference. |

Special tokens :

| Token | Signification |
|---|---|
| `[PAD]` | Padding token ignoré par la loss. |
| `[UNK]` | Unknown token. |
| `[BOS]` | Beginning of sequence. |
| `[EOS]` | End of sequence. |
| `[SYS]` | System message role. |
| `[USER]` | User message role. |
| `[ASST]` | Assistant message role. |

Si `tokenizer.json` change, il faut reconstruire `train.bin` et `eval.bin`, car les token IDs peuvent changer.

## 4. Binary packing

Le packing consiste à encoder le corpus une seule fois et sauvegarder les token IDs dans des fichiers binaires compacts.

Fichiers produits :

```text
train.bin
eval.bin
meta.json
```

Pourquoi le packing est important :

- éviter de tokenizer le texte à chaque epoch ;
- rendre le training plus rapide ;
- stocker les tokens de façon compacte avec `uint16` ;
- garder les métadonnées dans `meta.json`.

Exemple :

```text
document A: [BOS] a b c [EOS]
document B: [BOS] d e   [EOS]
train.bin : [BOS] a b c [EOS] [BOS] d e [EOS] ...
```

## 5. Dataset et collate

`PackedDataset` lit `train.bin` / `eval.bin` avec `numpy.memmap`.

Cela garde le gros fichier binaire sur le disque au lieu de tout charger en RAM.

`collate_fn` crée la tâche de next-token prediction :

```text
window : [BOS, t1, t2, t3, EOS]
input  : [BOS, t1, t2, t3]
target : [t1,  t2, t3, EOS]
```

Le modèle reçoit `input_ids` et apprend à prédire `targets`.

Le padding est ignoré par la loss grâce à `ignore_index=pad_id`.

## 6. GPT model

Fichiers principaux :

| File | Rôle |
|---|---|
| `embeddings.py` | Convertit token IDs et positions en vectors. |
| `attention.py` | Implémente causal multi-head self-attention. |
| `feedforward.py` | Applique le MLP dans chaque Transformer block. |
| `block.py` | Combine LayerNorm, attention, residuals et MLP. |
| `gpt.py` | Assemble le GPT model complet. |
| `lora.py` | Ajoute les LoRA adapters pour fine-tuning. |

Architecture principale :

```text
input_ids
  -> token + position embeddings
  -> Transformer blocks
  -> final LayerNorm
  -> lm_head
  -> logits
```

Le modèle produit des `logits`, pas directement des probabilities. Softmax est appliqué ensuite pour calculer des probabilities ou faire du sampling.

## 7. Causal self-attention

La causal attention empêche un token de voir les tokens futurs.

Exemple :

```text
position 0 peut voir : 0
position 1 peut voir : 0, 1
position 2 peut voir : 0, 1, 2
```

C'est essentiel pour la next-token prediction. Si le modèle voyait les futurs tokens, il tricherait pendant le training.

Multi-head attention signifie que plusieurs attention patterns sont appris en parallèle.

## 8. Training

Fichiers importants :

| File | Rôle |
|---|---|
| `run_pretrain.py` | Entry point pour le pretraining. |
| `trainer.py` | Training loop. |
| `schedule.py` | Learning-rate schedule. |
| `checkpoint.py` | Save/load model checkpoints. |
| `tracking.py` | Log metrics et generated samples. |

Une training step :

```text
batch
  -> model forward
  -> logits
  -> cross-entropy loss
  -> backward
  -> gradient clipping
  -> AdamW step
  -> scheduler step
```

Gradient accumulation simule un batch plus grand :

```text
batch_size 8 * accumulation_steps 4 = effective batch size 32
```

## 9. Checkpoint

Un checkpoint est une sauvegarde de l'état du training run.

Il peut contenir :

- model weights ;
- model config ;
- current step ;
- optimizer state ;
- scheduler state ;
- metrics.

Il sert à :

- reprendre le training ;
- charger un modèle entraîné pour generation ;
- démarrer un LoRA fine-tuning depuis un pretrained model.

## 10. Generation

La generation est autoregressive :

```text
prompt
  -> model predicts next token
  -> append token
  -> repeat
```

Sampling strategies :

| Strategy | Signification |
|---|---|
| `greedy` | Toujours choisir le token avec le score le plus haut. |
| `temperature` | Contrôle le niveau de randomness. |
| `top_k` | Garde seulement les `k` meilleurs candidats. |
| `top_p` | Garde les candidats couvrant une probability mass. |

Le KV cache rend la generation plus rapide en réutilisant les previous attention keys and values.

## 11. LoRA fine-tuning

LoRA est une technique de parameter-efficient fine-tuning.

Au lieu de mettre à jour tous les weights, on entraîne de petites adapter matrices.

Concept :

```text
frozen base model + trainable LoRA adapters
```

Avantages :

- moins de trainable parameters ;
- moins de memory usage ;
- fine-tuning plus rapide ;
- adaptation plus simple au chat format.

Dans FikrLLM, LoRA est appliqué aux attention projections comme `W_q`, `W_k`, `W_v`, et `W_o`.

## 12. À retenir

- GPT apprend la next-token prediction.
- Le tokenizer fait partie du model contract.
- `tokenizer.json`, `train.bin` et les checkpoints doivent rester compatibles.
- `collate_fn` crée `input_ids` et `targets`.
- Le causal mask empêche le modèle de tricher.
- Les checkpoints permettent resume, generation et fine-tuning.
- LoRA adapte le modèle sans réentraîner tous les weights.
- Clean data et reproducible pipelines sont aussi importants que le model code.
