# Questions d'interview FikrLLM

Ce fichier contient des questions/réponses pour préparer un interview autour du projet FikrLLM. Les mots techniques importants sont gardés en anglais.

## Project overview

### 1. What is FikrLLM?

FikrLLM est un GPT bilingue Arabic-English construit from scratch avec PyTorch. Il couvre tout le pipeline LLM : dataset preparation, tokenizer training, binary packing, GPT architecture, pretraining, generation et LoRA chat fine-tuning.

### 2. Quel est l'objectif principal du projet ?

L'objectif est de comprendre comment fonctionne un LLM end to end, au lieu d'utiliser seulement un pretrained model via une API.

### 3. Quel est le pipeline complet ?

```text
raw data -> cleaning -> tokenizer -> packing -> dataset -> GPT -> training -> checkpoint -> generation -> LoRA fine-tuning
```

## Data et tokenizer

### 4. Pourquoi a-t-on besoin d'un tokenizer ?

Un GPT model ne traite pas directement le raw text. Le tokenizer transforme le texte en integer token IDs que le modèle peut convertir en embeddings.

### 5. C'est quoi BPE ?

BPE, ou Byte Pair Encoding, est un algorithme de subword tokenization. Il apprend les combinaisons fréquentes depuis le corpus, ce qui permet de représenter les mots fréquents efficacement tout en gérant les mots rares.

### 6. Pourquoi `tokenizer.json` est important ?

Il définit le vocabulary et les normalization rules. Si le tokenizer change, les token IDs peuvent changer, donc les packed datasets et checkpoints peuvent devenir incompatibles.

### 7. Quelle est la différence entre Type A cleaning et Type B normalization ?

Type A cleaning est destructif et utilisé offline pour retirer le bruit du corpus. Type B normalization est léger et utilisé de manière cohérente par le tokenizer pendant training et inference.

### 8. Pourquoi ne pas appliquer Type A cleaning aux user prompts ?

Parce qu'il peut supprimer du contenu de façon agressive, comme des sections ou des lignes. Les user prompts doivent seulement passer par la tokenizer normalization.

## Packing et dataset

### 9. C'est quoi binary packing ?

Binary packing signifie encoder le texte une seule fois en token IDs et l'écrire dans des fichiers binaires compacts comme `train.bin` et `eval.bin`.

### 10. Pourquoi utiliser `numpy.memmap` ?

`numpy.memmap` permet de lire des morceaux d'un gros fichier binaire sans charger tout le fichier en RAM.

### 11. Pourquoi `PackedDataset` utilise `max_seq_len + 1` tokens ?

Parce qu'il faut un token supplémentaire pour créer la target sequence. L'input retire le dernier token, et la target retire le premier token.

Exemple :

```text
window : [BOS, t1, t2, t3, EOS]
input  : [BOS, t1, t2, t3]
target : [t1,  t2, t3, EOS]
```

### 12. Que fait `collate_fn` ?

Elle regroupe les exemples dans un batch, ajoute du padding si nécessaire, et crée `input_ids` et `targets` pour la next-token prediction.

## GPT architecture

### 13. C'est quoi les embeddings ?

Les embeddings convertissent les token IDs en dense vectors. Le modèle ajoute aussi position embeddings pour connaître l'ordre des tokens.

### 14. C'est quoi causal self-attention ?

Causal self-attention permet à chaque token de regarder seulement lui-même et les tokens précédents. Elle empêche le modèle de voir les futurs tokens.

### 15. Pourquoi le causal mask est important ?

Sans causal mask, le modèle pourrait voir le answer token pendant le training, ce qui rendrait la next-token prediction invalide.

### 16. C'est quoi multi-head attention ?

Multi-head attention exécute plusieurs attention heads en parallèle. Chaque head peut apprendre une relation différente entre les tokens.

### 17. C'est quoi un Transformer block ?

Un Transformer block combine généralement LayerNorm, causal self-attention, residual connections, une autre LayerNorm et un FeedForward network.

### 18. C'est quoi les logits ?

Les logits sont les raw scores produits par le modèle pour chaque token du vocabulary. Ce ne sont pas des probabilities avant softmax.

### 19. C'est quoi weight tying ?

Weight tying signifie partager la token embedding matrix avec la output projection matrix. Cela réduit le nombre de parameters et aligne les espaces input/output.

## Training

### 20. Quel est l'objectif du training ?

L'objectif est la next-token prediction : à partir des previous tokens, prédire le next token.

### 21. Quelle loss est utilisée ?

On utilise cross-entropy loss. Les padding positions sont ignorées avec `ignore_index=pad_id`.

### 22. C'est quoi gradient accumulation ?

Gradient accumulation additionne les gradients de plusieurs micro-batches avant de faire un optimizer step. Cela simule un batch plus grand sans demander plus de mémoire.

### 23. Pourquoi utiliser gradient clipping ?

Gradient clipping empêche les gradients trop grands de déstabiliser le training.

### 24. Pourquoi utiliser un learning-rate scheduler ?

Le scheduler contrôle le learning rate dans le temps. FikrLLM utilise warmup puis cosine decay pour améliorer la stabilité.

### 25. C'est quoi AdamW ?

AdamW est un optimizer qui sépare weight decay et gradient updates. Il est très utilisé pour entraîner des Transformer models.

## Checkpoints

### 26. C'est quoi un checkpoint ?

Un checkpoint est une sauvegarde du model state et du training state à un step précis.

### 27. Que contient un checkpoint ?

Il peut contenir model weights, model configuration, current step, optimizer state, scheduler state et metrics.

### 28. Pourquoi sauvegarder des checkpoints ?

Les checkpoints permettent de reprendre le training, d'utiliser le modèle pour generation, et de démarrer un fine-tuning.

## Generation

### 29. C'est quoi autoregressive generation ?

Autoregressive generation signifie générer un token à la fois, l'ajouter au context, puis répéter.

### 30. C'est quoi greedy decoding ?

Greedy decoding choisit toujours le token avec le score le plus haut. C'est deterministic mais parfois répétitif.

### 31. Que fait temperature ?

Temperature contrôle la randomness. Une valeur basse rend les sorties plus conservatrices ; une valeur haute rend les sorties plus diverses.

### 32. C'est quoi top-k sampling ?

Top-k sampling garde seulement les `k` tokens avec les meilleurs scores avant le sampling.

### 33. C'est quoi top-p sampling ?

Top-p sampling garde le plus petit ensemble de tokens dont la cumulative probability atteint `p`.

### 34. C'est quoi le KV cache ?

Le KV cache stocke les attention keys and values des tokens précédents pour éviter de recalculer tout le prompt à chaque step.

## LoRA et chat fine-tuning

### 35. C'est quoi LoRA ?

LoRA, ou Low-Rank Adaptation, est une méthode de parameter-efficient fine-tuning qui entraîne de petites adapter matrices tout en gardant le base model principalement frozen.

### 36. Pourquoi utiliser LoRA ?

LoRA réduit la memory usage et le training cost tout en permettant au modèle de s'adapter à une nouvelle tâche ou à un nouveau format.

### 37. Que change le chat fine-tuning ?

Il apprend au modèle à suivre un conversation format structuré avec `[SYS]`, `[USER]` et `[ASST]`.

### 38. Pourquoi masquer le prompt pendant le chat fine-tuning ?

Parce qu'on veut que le modèle apprenne la réponse, pas qu'il apprenne à reproduire le system prompt ou la user question.

## Project quality

### 39. Pourquoi la reproducibility est importante dans ce projet ?

Parce que tokenizer, dataset, model config et checkpoint doivent rester compatibles. La reproducibility rend les experiments plus faciles à inspecter, comparer et reprendre.

### 40. Quelle serait une bonne amélioration suivante ?

De bonnes améliorations seraient d'ajouter DVC pour les dataset/model artifacts, Airflow pour l'orchestration, plus d'evaluation metrics et un meilleur experiment tracking.

## Questions probables de recruteur et hiring manager

### 41. Peux-tu expliquer ce projet en deux minutes ?

FikrLLM est un projet éducatif LLM où j'ai reconstruit les principales parties d'un pipeline GPT from scratch. J'ai préparé un dataset bilingue Arabic-English, entraîné un BPE tokenizer, converti le corpus en binary token files, implémenté une architecture GPT avec PyTorch, entraîné le modèle en next-token prediction, sauvegardé des checkpoints, ajouté la generation, et préparé le LoRA fine-tuning pour le chat behavior.

### 42. Pourquoi as-tu construit ce projet ?

Je l'ai construit pour aller au-delà de l'utilisation simple des LLM APIs et comprendre l'engineering réel derrière les language models : data preparation, tokenization, model architecture, training stability, checkpoints et fine-tuning.

### 43. Quelle a été la partie la plus difficile ?

La partie la plus difficile était de garder tout le pipeline cohérent. Le tokenizer, les packed data, la model configuration et les checkpoints doivent rester compatibles. Si une pièce change, par exemple `tokenizer.json`, les binary files et checkpoints peuvent devenir incompatibles.

### 44. Quelle décision technique te rend le plus fier ?

J'ai séparé le destructive corpus cleaning de la lightweight tokenizer normalization. Cela rend le data pipeline plus propre et plus safe, car les user prompts ne doivent pas passer par un cleaning agressif.

### 45. Comment expliquerais-tu ce projet à un recruteur non technique ?

J'ai construit un petit pipeline de language model pour comprendre comment les systèmes comme ChatGPT fonctionnent en interne. Le projet part du raw text, le transforme en nombres, entraîne un neural network à prédire le prochain mot/token, puis utilise le modèle entraîné pour générer du texte.

### 46. Est-ce que c'est un LLM prêt pour la production ?

Non. C'est surtout un projet éducatif et portfolio. Mais il utilise des idées proches de la production : modular code, data preparation claire, checkpoints, concepts d'experiment tracking, et une trajectoire vers DVC, Airflow et MLOps.

### 47. Qu'est-ce qu'il faudrait ajouter pour le rendre plus production-ready ?

J'ajouterais des automated tests sur le data pipeline, un vrai experiment tracking, DVC pour les dataset/model artifacts, des CI/CD checks, model evaluation, monitoring, containerized training/inference, et une API de deployment claire.

### 48. Quel est le lien avec ton background iOS ?

Mon expérience iOS m'a donné de solides réflexes de software engineering : architecture, clean code, testing, performance et product thinking. FikrLLM ajoute la profondeur AI/ML, donc je peux travailler sur des produits qui combinent mobile apps avec LLMs, RAG, agents ou on-device AI.

### 49. Pour quels rôles ce projet te prépare ?

Il me prépare pour des rôles comme AI Engineer, GenAI Engineer, ML Engineer, Mobile AI Engineer, ou iOS Engineer travaillant sur des AI-powered features.

### 50. Comment intégrerais-tu FikrLLM dans une application ?

J'exposerais l'inference via un service FastAPI, avec request validation, chargement d'un trained checkpoint, generation avec safe limits, puis connexion à une frontend app ou mobile app via REST API.

### 51. Comment évaluerais-tu le modèle ?

Je suivrais validation loss et perplexity pendant le pretraining, puis j'ajouterais une evaluation spécifique pour chat quality, factuality, latency, memory usage et failure cases.

### 52. Quels sont les principaux risques dans un projet LLM ?

Les principaux risques sont poor data quality, overfitting, unstable training, hallucinations, unsafe outputs, high compute cost, weak evaluation et manque de reproducibility.

### 53. Comment debugger un modèle qui génère du mauvais texte ?

Je vérifierais le tokenizer, training/eval loss, dataset quality, overfitting sur un tiny sample, generation parameters, et je comparerais les outputs entre plusieurs checkpoints.

### 54. Comment debugger un training qui plante avec out-of-memory ?

Je réduirais le batch size, je baisserais `max_seq_len`, j'utiliserais gradient accumulation, je réduirais la taille du modèle, je vérifierais les tensor shapes, j'activerais mixed precision si disponible, et je surveillerais GPU/CPU memory.

### 55. Pourquoi le modular code est important dans ce projet ?

Le modular code rend le projet plus simple à tester, expliquer, maintenir et étendre. Par exemple, tokenizer, dataset, model, training, generation et LoRA sont séparés au lieu d'être cachés dans un seul grand script.

### 56. Quelle est la différence entre AI Engineer et MLOps Engineer ?

Un AI Engineer se concentre sur la construction d'AI features et d'applications basées sur des modèles. Un MLOps Engineer se concentre davantage sur pipelines, deployment, monitoring, model registry, reproducibility et production reliability.

### 57. Où placerais-tu DVC dans ce projet ?

DVC servirait à tracker les gros artifacts comme raw Parquet files, cleaned datasets, `train.bin`, `eval.bin`, tokenizer outputs, et éventuellement checkpoints sans les committer directement dans Git.

### 58. Où placerais-tu Airflow dans ce projet ?

Airflow servirait à orchestrer le pipeline : download data, clean corpus, train tokenizer, pack binary files, run training, evaluate checkpoints et publish artifacts.

### 59. Que répondre si le recruteur demande : “Tu es plutôt iOS ou AI ?”

Je répondrais que mon expérience production la plus forte est en iOS engineering, et que je développe maintenant une vraie profondeur en Applied AI. Mon positionnement est mobile/product engineering avec de solides compétences AI/LLM implementation.

### 60. Quel est ton principal selling point ?

Mon principal selling point est la combinaison entre production software engineering et hands-on AI. Je peux construire des applications propres côté utilisateur, tout en comprenant le ML pipeline derrière les intelligent features.

## IA, model training, PyTorch et hyperparameters

### 61. Quelle est la différence entre parameter et hyperparameter ?

Un parameter est appris par le modèle pendant le training, comme les weights et biases. Un hyperparameter est choisi avant le training, comme learning rate, batch size, number of layers, embedding dimension, dropout ou `max_seq_len`.

### 62. Donne des exemples de parameters dans un GPT model.

Exemples : token embeddings, position embeddings, attention projection matrices `W_q`, `W_k`, `W_v`, `W_o`, FeedForward weights, LayerNorm weights et output projection weights.

### 63. Donne des exemples de hyperparameters dans ce projet.

Exemples : `vocab_size`, `max_seq_len`, `n_layers`, `n_heads`, `d_model`, `dropout`, `batch_size`, `learning_rate`, `weight_decay`, `warmup_steps` et `max_steps`.

### 64. Que se passe-t-il si le learning rate est trop élevé ?

Le training peut devenir instable. La loss peut exploser, osciller ou ne jamais converger, car l'optimizer fait des steps trop grands.

### 65. Que se passe-t-il si le learning rate est trop faible ?

Le training devient très lent et peut rester bloqué avant d'atteindre une bonne solution.

### 66. Pourquoi utilise-t-on un validation set ?

Le validation set mesure la capacité du modèle à généraliser sur des données non vues. Il aide à détecter l'overfitting et à comparer les checkpoints.

### 67. C'est quoi overfitting ?

Overfitting signifie que le modèle apprend trop spécifiquement le training data et performe mal sur des données non vues.

### 68. Comment réduire l'overfitting ?

On peut utiliser plus de data, dropout, weight decay, early stopping, un modèle plus petit, un meilleur data cleaning, et choisir les checkpoints selon la validation loss.

### 69. C'est quoi underfitting ?

Underfitting signifie que le modèle est trop faible, pas assez entraîné, ou mal configuré. Il performe mal sur training data et validation data.

### 70. Que fait `model.train()` en PyTorch ?

`model.train()` met le modèle en training mode. Des couches comme dropout deviennent actives, et BatchNorm mettrait à jour ses running statistics si utilisé.

### 71. Que fait `model.eval()` en PyTorch ?

`model.eval()` met le modèle en evaluation mode. Dropout est désactivé, ce qui rend l'inference plus déterministe pour les couches affectées par le training mode.

### 72. Pourquoi utiliser `torch.no_grad()` pendant evaluation ou generation ?

`torch.no_grad()` désactive le gradient tracking. Cela économise la mémoire et rend evaluation/generation plus rapide, car PyTorch ne construit pas de backward graph.

### 73. C'est quoi backpropagation ?

Backpropagation calcule les gradients de la loss par rapport aux model parameters. Ensuite, l'optimizer utilise ces gradients pour mettre à jour les parameters.

### 74. C'est quoi un optimizer ?

Un optimizer met à jour les model parameters à partir des gradients. Dans le training des Transformers, AdamW est souvent utilisé car il marche bien avec des gradients sparse et noisy.

### 75. Quelle est la différence entre Adam et AdamW ?

AdamW sépare weight decay du gradient update. Cela donne souvent une meilleure regularization pour les Transformer models.

### 76. C'est quoi un gradient ?

Un gradient indique combien chaque parameter contribue à la loss et dans quelle direction il faut le modifier pour réduire cette loss.

### 77. Pourquoi appeler `optimizer.zero_grad()` ?

PyTorch accumule les gradients par défaut. `optimizer.zero_grad()` efface les anciens gradients avant le prochain backward pass.

### 78. Quel est le rôle de `loss.backward()` ?

`loss.backward()` calcule les gradients pour tous les parameters impliqués dans le calcul de la loss.

### 79. Quel est le rôle de `optimizer.step()` ?

`optimizer.step()` met à jour les model parameters avec les gradients calculés pendant la backpropagation.

### 80. C'est quoi batch size ?

Batch size est le nombre d'exemples traités avant un gradient update. Un batch plus grand peut être plus stable, mais demande plus de mémoire.

### 81. C'est quoi gradient accumulation ?

Gradient accumulation simule un batch plus grand en accumulant les gradients sur plusieurs micro-batches avant d'appeler `optimizer.step()`.

### 82. C'est quoi dropout ?

Dropout est une technique de regularization qui désactive aléatoirement certaines activations pendant le training. Cela aide à réduire l'overfitting.

### 83. C'est quoi weight decay ?

Weight decay est une technique de regularization qui pénalise les weights trop grands. Cela aide le modèle à ne pas trop se spécialiser sur le training data.

### 84. C'est quoi perplexity ?

Perplexity est une métrique dérivée de la cross-entropy loss. Une perplexity plus basse signifie généralement que le modèle prédit mieux le next token.

### 85. Quelle est la différence entre logits, probabilities et tokens ?

Les logits sont les raw scores du modèle. Les probabilities sont obtenues après softmax. Les tokens sont les discrete IDs sélectionnés depuis le vocabulary.

### 86. C'est quoi softmax ?

Softmax transforme les logits en probabilities dont la somme vaut 1. Il sert à interpréter les scores du modèle sur le vocabulary.

### 87. Pourquoi utiliser cross-entropy pour language modeling ?

Cross-entropy mesure l'écart entre la probability distribution prédite par le modèle et le vrai next token. Elle est adaptée à une classification sur les vocabulary tokens.

### 88. C'est quoi mixed precision training ?

Mixed precision training utilise des formats de précision plus basse comme FP16 ou BF16 pour certaines opérations afin de réduire la memory usage et accélérer le training, tout en gardant de la stabilité avec certaines opérations en précision plus élevée.

### 89. C'est quoi un tensor ?

Un tensor est un tableau multidimensionnel utilisé par PyTorch pour stocker inputs, parameters, activations, gradients et outputs.

### 90. Quelle est la shape typique d'un GPT input batch ?

Un GPT input batch a généralement la shape `[batch_size, sequence_length]`, où chaque valeur est un token ID.

### 91. Quelle est la shape des logits dans un GPT model ?

Les logits ont généralement la shape `[batch_size, sequence_length, vocab_size]`. Pour chaque position, le modèle donne un score à chaque token du vocabulary.

### 92. Pourquoi déplacer les tensors vers un device ?

Les tensors et model parameters doivent être sur le même device, comme CPU, CUDA GPU ou Apple Silicon MPS. Sinon PyTorch donne une device mismatch error.

### 93. Quel est le rôle de `DataLoader` ?

`DataLoader` lit les exemples depuis un Dataset, les regroupe en batches, peut les mélanger, et utilise une `collate_fn` pour préparer les batch tensors.

### 94. Quelle est la différence entre `Dataset` et `DataLoader` ?

`Dataset` définit comment accéder à un seul exemple. `DataLoader` définit comment parcourir plusieurs exemples en batches.

### 95. À quoi sert un model checkpoint pendant le training ?

Un checkpoint permet de reprendre le training après interruption, comparer différents training steps, et réutiliser les trained weights pour generation ou fine-tuning.

### 96. C'est quoi fine-tuning ?

Fine-tuning continue le training depuis un pretrained model sur un dataset ou une tâche plus spécifique, par exemple chat-style question answering.

### 97. Quelle est la différence entre pretraining et fine-tuning ?

Pretraining apprend les patterns généraux du langage à partir de grands text corpora. Fine-tuning adapte le modèle à une tâche, un format ou un comportement spécifique.

### 98. Pourquoi la GPU memory est importante dans LLM training ?

LLM training stocke parameters, gradients, optimizer states, activations et batch tensors. Des modèles plus grands, des séquences plus longues et des batches plus grands demandent plus de mémoire.

### 99. Que vérifier en premier si la training loss devient NaN ?

Je vérifierais le learning rate, gradient explosion, invalid data, mixed precision stability, tensor values, et si gradient clipping est correctement appliqué.

### 100. Comment expliquer honnêtement ton niveau AI en interview ?

Je dirais que j'ai une forte expérience en software engineering et une pratique hands-on en Applied AI. J'ai compris et implémenté le pipeline principal d'un LLM : data, tokenizer, PyTorch model, training loop, checkpointing, generation et LoRA fine-tuning, et je continue à approfondir l'expérience ML à l'échelle production.

## Serving, inference, deployment et monitoring

### 101. C'est quoi inference ?

Inference est la phase où un trained model est utilisé pour produire des prédictions ou générer du texte à partir de nouveaux inputs. Pour un LLM, inference signifie prendre un prompt et générer les next tokens.

### 102. C'est quoi model serving ?

Model serving signifie exposer un trained model via une interface, souvent une API, pour que des applications puissent envoyer des requests et recevoir des predictions.

### 103. Quelle est la différence entre training et inference ?

Training met à jour les model parameters avec data et loss. Inference utilise des trained parameters fixes pour générer des outputs sans modifier le modèle.

### 104. Quels fichiers sont nécessaires pour l'inference dans FikrLLM ?

Inference nécessite un checkpoint compatible, une model configuration, `tokenizer.json`, et le generation code. Le tokenizer et le checkpoint doivent correspondre au même vocabulary et à la même model architecture.

### 105. Comment servir FikrLLM avec FastAPI ?

Je créerais un endpoint FastAPI comme `/generate`, je chargerais tokenizer et checkpoint au startup, je validerais le prompt et les generation parameters, je lancerais la generation avec `torch.no_grad()`, puis je retournerais le generated text en JSON.

### 106. Pourquoi charger le modèle une seule fois au startup ?

Charger le modèle à chaque request serait très lent. Le charger une seule fois au startup garde le modèle en mémoire et rend chaque inference request plus rapide.

### 107. C'est quoi latency ?

Latency est le temps entre la réception d'une request et le retour de la response. Pour les LLMs, elle dépend de prompt length, generated tokens, model size, hardware et decoding strategy.

### 108. C'est quoi throughput ?

Throughput est le nombre de requests ou tokens que le système peut traiter par seconde. Un serving system doit équilibrer low latency et high throughput.

### 109. C'est quoi batching en inference ?

Batching regroupe plusieurs inference requests pour que le modèle les traite plus efficacement. Cela peut améliorer throughput, mais parfois augmenter latency pour certains users.

### 110. C'est quoi streaming generation ?

Streaming generation envoie les tokens au client au fur et à mesure de leur génération au lieu d'attendre la réponse complète. Cela améliore la responsiveness perçue.

### 111. Pourquoi `torch.no_grad()` est important pendant inference ?

Il désactive le gradient tracking, réduit memory usage et améliore la speed parce qu'aucun backward graph n'est créé.

### 112. Pourquoi appeler `model.eval()` avant inference ?

`model.eval()` désactive les comportements spécifiques au training comme dropout, ce qui rend l'inference plus stable et déterministe.

### 113. C'est quoi cold start ?

Cold start est le délai au démarrage d'un service quand il doit charger model, tokenizer, configuration et éventuellement déplacer les weights vers GPU avant de servir les requests.

### 114. C'est quoi warmup en model serving ?

Warmup signifie lancer quelques dummy inference calls après le startup pour initialiser les kernels, allouer la mémoire, et réduire la latency des premières vraies requests.

### 115. C'est quoi autoscaling ?

Autoscaling augmente ou diminue automatiquement le nombre d'instances de serving selon traffic, CPU/GPU usage, queue length ou latency.

### 116. Quelles métriques monitorer en LLM serving ?

Je monitorerais request count, error rate, latency, tokens per second, GPU/CPU memory, GPU utilization, timeout rate, prompt length, output length et user feedback.

### 117. Quels logs sont utiles pendant inference ?

Logs utiles : request ID, timestamp, model version, prompt length, generation parameters, latency, output length, status code et error messages. Les données sensibles doivent être masquées ou évitées.

### 118. Pourquoi le model versioning est important en serving ?

Model versioning permet de savoir exactement quel checkpoint a produit quel output, comparer les versions, faire un rollback safe et debugger les regressions.

### 119. Comment déployer une model API ?

Je packagerais le service avec Docker, j'exposerais un endpoint FastAPI, je configurerais les environment variables, j'ajouterais health checks, je déploierais sur VM ou Kubernetes, puis je monitorerais logs et metrics.

### 120. C'est quoi un health check endpoint ?

Un health check endpoint indique si le service est vivant et prêt. Par exemple, `/health` peut confirmer que l'API fonctionne et que le modèle est chargé.

### 121. Quelle est la différence entre CPU, GPU et MPS inference ?

CPU inference est plus simple mais plus lente. GPU inference est plus rapide pour les grandes matrix operations. MPS est le backend d'accélération Apple pour les machines Apple Silicon.

### 122. C'est quoi quantization ?

Quantization réduit la précision des model weights, par exemple de FP32 vers INT8 ou 4-bit. Cela peut réduire memory usage et accélérer inference, parfois avec un trade-off sur la qualité.

### 123. C'est quoi le KV cache en serving ?

Le KV cache stocke les attention keys and values des tokens précédents pendant la generation. Il évite de recalculer tout le context à chaque nouveau token et améliore inference speed.

### 124. Pourquoi les longs prompts ralentissent inference ?

Les longs prompts augmentent la sequence length traitée par attention. Comme le coût d'attention augmente avec la longueur de séquence, les longs contexts utilisent plus de memory et compute.

### 125. Comment protéger une inference API contre les abus ?

J'ajouterais authentication, rate limiting, request size limits, timeout limits, max generated tokens, logging, monitoring et content safety checks si nécessaire.

### 126. C'est quoi timeout handling ?

Timeout handling arrête les requests qui prennent trop de temps. Cela protège le service contre les requests bloquées et garde les resources disponibles.

### 127. C'est quoi une fallback strategy ?

Une fallback strategy définit quoi faire si le main model échoue : retourner une safe error, utiliser un modèle plus petit, retry une fois, ou envoyer la request vers un autre service.

### 128. Comment debugger une inference lente ?

Je mesurerais prompt length, generated token count, model size, device placement, batch size, KV cache usage, CPU/GPU utilization, et si le modèle est accidentellement rechargé à chaque request.

### 129. Comment debugger des outputs différents pour le même prompt ?

Je vérifierais decoding strategy, temperature, top-k/top-p settings, random seed, dropout mode, et si `model.eval()` est utilisé.

### 130. Quelle est la différence entre offline evaluation et online monitoring ?

Offline evaluation teste le modèle avant deployment sur des datasets préparés. Online monitoring observe le comportement réel en production : latency, errors, usage patterns et user feedback.

## MLOps, LLMOps, AIOps et production operations

### 131. C'est quoi MLOps ?

MLOps est l'ensemble des pratiques pour construire, déployer, monitorer, versionner et maintenir des machine learning systems en production. Cela combine machine learning, DevOps, data engineering et software engineering.

### 132. C'est quoi LLMOps ?

LLMOps est une spécialisation de MLOps pour les large language models. Cela couvre prompt management, model serving, fine-tuning, evaluation, RAG pipelines, vector databases, safety, cost tracking, latency et monitoring du comportement LLM en production.

### 133. C'est quoi AIOps ?

AIOps utilise l'AI pour améliorer IT operations. Cela peut aider à détecter des incidents, analyser des logs, prédire des pannes, réduire le bruit des alertes, et aider au root-cause analysis.

### 134. Quelle est la différence entre MLOps, LLMOps et AIOps ?

MLOps sert à opérer des ML systems. LLMOps sert à opérer des LLM-based systems. AIOps utilise l'AI pour opérer infrastructure et IT operations. En résumé : MLOps/LLMOps font tourner des modèles AI ; AIOps utilise l'AI pour faire tourner les opérations.

### 135. Où placerais-tu MLflow dans ce projet ?

MLflow peut tracker experiments, parameters, metrics, artifacts, model versions et checkpoints. Il aide à comparer les training runs et garder l'historique des experiments organisé.

### 136. C'est quoi un model registry ?

Un model registry stocke des modèles versionnés avec metadata comme stage, metrics, owner, training data version et deployment status. Il aide à gérer la promotion de development vers production.

### 137. C'est quoi data versioning ?

Data versioning permet de tracker exactement quel dataset a été utilisé pour chaque experiment. C'est important car un modèle n'est pas reproductible si la data version est inconnue.

### 138. Pourquoi la reproducibility est importante en MLOps ?

Reproducibility permet de reconstruire le même modèle depuis le même code, data, configuration et environment. Cela rend debugging, audit et comparison fiables.

### 139. C'est quoi model drift ?

Model drift arrive quand la performance du modèle se dégrade dans le temps parce que les production data changent par rapport aux training data.

### 140. C'est quoi data drift ?

Data drift signifie que la distribution des input data change avec le temps. Par exemple, les user prompts peuvent devenir plus longs, utiliser d'autres langues ou couvrir de nouveaux sujets.

### 141. C'est quoi concept drift ?

Concept drift signifie que la relation entre inputs et expected outputs change. Le même type d'input peut nécessiter une réponse différente parce que user behavior, business rules ou environment ont changé.

### 142. Comment monitorer un LLM en production ?

Je monitorerais latency, error rate, token usage, cost, prompt length, output length, user feedback, safety violations, hallucination reports, retrieval quality pour RAG, et comportement par model version.

### 143. C'est quoi observability ?

Observability est la capacité à comprendre l'état interne d'un système à partir de signaux externes comme logs, metrics, traces et events.

### 144. Quelle est la différence entre logs, metrics et traces ?

Logs décrivent des événements. Metrics sont des mesures numériques dans le temps. Traces montrent le chemin d'une request entre services et aident à identifier les bottlenecks.

### 145. C'est quoi CI/CD pour ML ?

CI/CD pour ML automatise testing, packaging, validation et deployment du ML code, data pipelines et model services. Il doit vérifier la software quality et la model quality.

### 146. Que faut-il tester dans un LLMOps pipeline ?

Il faut tester data validation, tokenizer compatibility, model loading, inference endpoint behavior, generation limits, safety rules, latency budget, et regression tests sur des evaluation prompts.

### 147. C'est quoi rollback ?

Rollback signifie revenir à une version précédente stable du modèle ou service quand la nouvelle version cause errors, poor quality, high latency ou unexpected behavior.

### 148. C'est quoi canary deployment ?

Canary deployment déploie une nouvelle version de modèle ou service sur un petit pourcentage du traffic d'abord. Si les metrics sont bonnes, on augmente progressivement le traffic.

### 149. C'est quoi un A/B test pour LLMs ?

Un A/B test compare deux versions de model ou prompt sur du traffic réel ou contrôlé pour mesurer quality, latency, cost, user satisfaction ou business impact.

### 150. C'est quoi prompt versioning ?

Prompt versioning permet de tracker les changements des system prompts, templates, examples et instructions. C'est important car les prompt changes peuvent fortement changer la output quality.

### 151. C'est quoi RAG observability ?

RAG observability monitore les parties retrieval et generation d'un RAG system. Elle vérifie retrieved documents, relevance, context length, answer quality, latency et failure cases.

### 152. À quoi sert une vector database en LLMOps ?

Une vector database stocke des embeddings et permet semantic search. Dans RAG, elle récupère les documents pertinents ajoutés au prompt avant generation.

### 153. C'est quoi un embedding model ?

Un embedding model transforme du texte en dense vectors qui capturent le sens sémantique. Des textes similaires doivent avoir des vector representations similaires.

### 154. C'est quoi cost monitoring en LLMOps ?

Cost monitoring suit GPU usage, API calls, token usage, storage et infrastructure cost. Il aide à éviter les dépenses incontrôlées en production.

### 155. C'est quoi rate limiting ?

Rate limiting limite le nombre de requests qu'un user ou service peut envoyer dans une fenêtre de temps. Cela protège le système contre abuse et overload.

### 156. C'est quoi incident response dans les AI systems ?

Incident response définit comment l'équipe détecte, investigue, mitige et communique les problèmes comme model outage, unsafe outputs, latency spikes ou quality regressions.

### 157. C'est quoi governance dans les AI/LLM systems ?

Governance définit les règles autour de model usage, data privacy, access control, auditability, safety, compliance et responsible AI practices.

### 158. C'est quoi human-in-the-loop system ?

Un human-in-the-loop system inclut une validation ou review humaine dans le workflow AI, surtout pour décisions sensibles, low-confidence outputs ou safety-critical use cases.

### 159. Quelle est la différence entre batch inference et real-time inference ?

Batch inference traite beaucoup d'inputs à des moments planifiés. Real-time inference répond immédiatement aux user requests avec des latency requirements stricts.

### 160. Comment expliquer ton Ops mindset en interview ?

Je dirais que je ne pense pas seulement à construire un modèle. Je pense aussi à comment le versionner, le déployer, le monitorer, faire un rollback safe, contrôler le cost, gérer les incidents, et garder le système reliable dans le temps.
