# FikrLLM Interview Questions and Answers

This file contains interview-style questions based on the FikrLLM project.

## Project overview

### 1. What is FikrLLM?

FikrLLM is a bilingual Arabic-English GPT built from scratch with PyTorch. It covers the full LLM pipeline: dataset preparation, tokenizer training, binary packing, GPT architecture, pretraining, generation, and LoRA chat fine-tuning.

### 2. What is the main learning objective of the project?

The goal is to understand how an LLM works end to end, instead of only using a pretrained model through an API.

### 3. What is the full pipeline?

```text
raw data -> cleaning -> tokenizer -> packing -> dataset -> GPT -> training -> checkpoint -> generation -> LoRA fine-tuning
```

## Data and tokenizer

### 4. Why do we need a tokenizer?

A GPT model does not process raw text directly. The tokenizer converts text into integer token IDs that the model can embed and process.

### 5. What is BPE?

BPE, or Byte Pair Encoding, is a subword tokenization algorithm. It learns frequent character or subword merges from the corpus, allowing common words to become short token sequences while still handling rare words.

### 6. Why is `tokenizer.json` important?

It defines the vocabulary and normalization rules. If the tokenizer changes, the token IDs can change, so packed datasets and checkpoints may become incompatible.

### 7. What is the difference between Type A cleaning and Type B normalization?

Type A cleaning is destructive and used offline to remove noise from raw corpus data. Type B normalization is lightweight and used consistently by the tokenizer during training and inference.

### 8. Why should Type A cleaning not be applied to user prompts?

Because it can remove content aggressively, such as sections or lines. User prompts should only go through safe tokenizer normalization.

## Packing and dataset

### 9. What is binary packing?

Binary packing means encoding text once into token IDs and writing them into compact binary files such as `train.bin` and `eval.bin`.

### 10. Why use `numpy.memmap`?

`numpy.memmap` allows the dataset to read slices from a large binary file without loading the whole file into RAM.

### 11. Why does `PackedDataset` use `max_seq_len + 1` tokens?

Because one extra token is needed to create the target sequence. The input drops the last token, and the target drops the first token.

Example:

```text
window : [BOS, t1, t2, t3, EOS]
input  : [BOS, t1, t2, t3]
target : [t1,  t2, t3, EOS]
```

### 12. What does `collate_fn` do?

It groups individual examples into a batch, adds padding when needed, and creates `input_ids` and `targets` for next-token prediction.

## GPT architecture

### 13. What are embeddings?

Embeddings convert token IDs into dense vectors. The model also adds position embeddings so it knows the order of tokens.

### 14. What is causal self-attention?

Causal self-attention lets each token attend only to itself and previous tokens. It prevents the model from seeing future tokens during training.

### 15. Why is the causal mask important?

Without the causal mask, the model could see the answer token while training, which would make next-token prediction invalid.

### 16. What is multi-head attention?

Multi-head attention runs several attention heads in parallel. Each head can learn a different relationship between tokens.

### 17. What is a Transformer block?

A Transformer block usually combines LayerNorm, causal self-attention, residual connections, another LayerNorm, and a FeedForward network.

### 18. What are logits?

Logits are raw scores produced by the model for every token in the vocabulary. They are not probabilities until softmax is applied.

### 19. What is weight tying?

Weight tying means sharing the token embedding matrix with the output projection matrix. This reduces parameters and keeps input/output token spaces aligned.

## Training

### 20. What is the training objective?

The objective is next-token prediction: given previous tokens, predict the next token.

### 21. What loss is used?

Cross-entropy loss is used. Padding positions are ignored with `ignore_index=pad_id`.

### 22. What is gradient accumulation?

Gradient accumulation adds gradients across several micro-batches before one optimizer step. It simulates a larger batch without needing more memory.

### 23. Why use gradient clipping?

Gradient clipping prevents very large gradients from destabilizing training.

### 24. Why use a learning-rate scheduler?

The scheduler controls the learning rate over time. FikrLLM uses warmup followed by cosine decay to improve stability.

### 25. What is AdamW?

AdamW is an optimizer that separates weight decay from gradient-based updates. It is widely used for Transformer training.

## Checkpoints

### 26. What is a checkpoint?

A checkpoint is a saved snapshot of the model and training state at a specific step.

### 27. What does a checkpoint contain?

It can contain model weights, model configuration, current step, optimizer state, scheduler state, and metrics.

### 28. Why save checkpoints?

Checkpoints allow training to resume, enable later generation, and provide a starting point for fine-tuning.

## Generation

### 29. What is autoregressive generation?

Autoregressive generation means generating one token at a time, appending it to the context, and repeating the process.

### 30. What is greedy decoding?

Greedy decoding always chooses the token with the highest score. It is deterministic but can be repetitive.

### 31. What does temperature do?

Temperature controls randomness. Lower values make outputs more conservative; higher values make outputs more diverse.

### 32. What is top-k sampling?

Top-k sampling keeps only the `k` highest-scoring tokens before sampling.

### 33. What is top-p sampling?

Top-p sampling keeps the smallest set of tokens whose cumulative probability reaches `p`.

### 34. What is the KV cache?

The KV cache stores attention keys and values from previous tokens so generation does not recompute the full prompt at every step.

## LoRA and chat fine-tuning

### 35. What is LoRA?

LoRA, or Low-Rank Adaptation, is a parameter-efficient fine-tuning method that trains small adapter matrices while keeping the base model mostly frozen.

### 36. Why use LoRA?

It reduces memory usage and training cost while still allowing the model to adapt to a new task or format.

### 37. What does chat fine-tuning change?

It teaches the model to follow a structured conversation format with `[SYS]`, `[USER]`, and `[ASST]` tokens.

### 38. Why mask the prompt during chat fine-tuning?

Because we want the model to learn the answer, not to learn to reproduce the system prompt or the user question.

## Project quality

### 39. Why is reproducibility important in this project?

Because tokenizer, dataset, model config, and checkpoint must stay compatible. Reproducibility makes experiments easier to inspect, compare, and resume.

### 40. What would be a good next improvement?

Good next improvements include adding DVC for dataset/model artifacts, adding Airflow for orchestration, adding more evaluation metrics, and improving experiment tracking.

## Recruiter and hiring-manager questions

### 41. Can you explain this project in two minutes?

FikrLLM is an educational LLM project where I rebuilt the main parts of a GPT pipeline from scratch. I prepared a bilingual Arabic-English dataset, trained a BPE tokenizer, packed the corpus into binary token files, implemented a GPT-style Transformer in PyTorch, trained it with next-token prediction, saved checkpoints, added text generation, and prepared LoRA fine-tuning for chat behavior.

### 42. Why did you build this project?

I built it to move beyond using LLM APIs and understand the real engineering behind language models: data preparation, tokenization, model architecture, training stability, checkpoints, and fine-tuning.

### 43. What was the hardest part of the project?

The hardest part was keeping the full pipeline consistent. The tokenizer, packed data, model configuration, and checkpoints must all match. If one piece changes, for example `tokenizer.json`, the downstream binary files and trained checkpoints may become incompatible.

### 44. What technical decision are you most proud of?

I separated destructive corpus cleaning from lightweight tokenizer normalization. That makes the data pipeline cleaner and safer because user prompts should not go through aggressive dataset cleaning.

### 45. How would you explain this project to a non-technical recruiter?

I built a small language model pipeline to understand how systems like ChatGPT work internally. The project starts from raw text, transforms it into numbers, trains a neural network to predict the next word or token, and then uses the trained model to generate text.

### 46. Is this a production-ready LLM?

No. It is mainly an educational and portfolio project. However, it uses production-inspired ideas such as modular code, clear data preparation, checkpoints, experiment tracking concepts, and a path toward DVC, Airflow, and MLOps.

### 47. What would you need to make it more production-ready?

I would add automated tests around the data pipeline, proper experiment tracking, DVC for dataset/model artifacts, CI/CD checks, model evaluation, monitoring, containerized training/inference, and a clear deployment API.

### 48. How does this project connect to your iOS background?

My iOS experience gives me strong software engineering habits: architecture, clean code, testing, performance, and product thinking. FikrLLM adds AI/ML depth, so I can work on products that combine mobile apps with LLMs, RAG, agents, or on-device AI.

### 49. What type of role does this project prepare you for?

It prepares me for roles such as AI Engineer, GenAI Engineer, ML Engineer, Mobile AI Engineer, or iOS Engineer working on AI-powered features.

### 50. How would you integrate FikrLLM into an application?

I would expose inference through a FastAPI service, add request validation, load a trained checkpoint, run generation with safe limits, and connect it to a frontend or mobile app through a REST API.

### 51. How would you evaluate the model?

I would track validation loss and perplexity during pretraining, then add task-specific evaluation for chat quality, factuality, latency, memory usage, and failure cases.

### 52. What are the main risks in an LLM project?

The main risks are poor data quality, overfitting, unstable training, hallucinations, unsafe outputs, high compute cost, poor evaluation, and weak reproducibility.

### 53. How would you debug a model that generates poor text?

I would check the tokenizer, inspect training/eval loss, verify the dataset quality, test overfitting on a tiny sample, review generation parameters, and compare outputs across checkpoints.

### 54. How would you debug a training run that crashes with out-of-memory errors?

I would reduce batch size, lower `max_seq_len`, use gradient accumulation, reduce model size, check tensor shapes, enable mixed precision if available, and monitor GPU/CPU memory usage.

### 55. Why is modular code important in this project?

Modular code makes the project easier to test, explain, maintain, and extend. For example, tokenizer, dataset, model, training, generation, and LoRA are separated instead of being hidden in one large script.

### 56. What is the difference between an AI Engineer and an MLOps Engineer?

An AI Engineer focuses on building AI features and model-based applications. An MLOps Engineer focuses more on pipelines, deployment, monitoring, model registry, reproducibility, and production reliability.

### 57. Where would DVC fit in this project?

DVC would track large artifacts such as raw Parquet files, cleaned datasets, `train.bin`, `eval.bin`, tokenizer outputs, and possibly checkpoints without committing them directly to Git.

### 58. Where would Airflow fit in this project?

Airflow would orchestrate the pipeline: download data, clean corpus, train tokenizer, pack binary files, run training, evaluate checkpoints, and publish artifacts.

### 59. What would you say if the recruiter asks: “Are you more iOS or AI?”

I would say that my strongest production experience is iOS engineering, and I am now expanding deeply into Applied AI. My positioning is mobile/product engineering with strong AI/LLM implementation skills.

### 60. What is your strongest selling point?

My strongest selling point is the combination of production software engineering and hands-on AI. I can build clean, user-facing applications and also understand the ML pipeline behind intelligent features.

## AI, model training, PyTorch, and hyperparameters

### 61. What is the difference between a parameter and a hyperparameter?

A parameter is learned by the model during training, such as weights and biases. A hyperparameter is chosen before training, such as learning rate, batch size, number of layers, embedding dimension, dropout, or `max_seq_len`.

### 62. Give examples of parameters in a GPT model.

Examples include token embeddings, position embeddings, attention projection matrices `W_q`, `W_k`, `W_v`, `W_o`, FeedForward weights, LayerNorm weights, and the output projection weights.

### 63. Give examples of hyperparameters in this project.

Examples include `vocab_size`, `max_seq_len`, `n_layers`, `n_heads`, `d_model`, `dropout`, `batch_size`, `learning_rate`, `weight_decay`, `warmup_steps`, and `max_steps`.

### 64. What happens if the learning rate is too high?

Training can become unstable. The loss may explode, oscillate, or fail to converge because the optimizer takes steps that are too large.

### 65. What happens if the learning rate is too low?

Training becomes very slow and may get stuck before reaching a good solution.

### 66. Why do we use a validation set?

The validation set measures how well the model generalizes to unseen data. It helps detect overfitting and compare checkpoints.

### 67. What is overfitting?

Overfitting happens when the model learns the training data too specifically and performs poorly on unseen data.

### 68. How can you reduce overfitting?

You can use more data, dropout, weight decay, early stopping, smaller models, better data cleaning, and validation-based checkpoint selection.

### 69. What is underfitting?

Underfitting happens when the model is too weak, trained for too few steps, or poorly configured, so it performs badly on both training and validation data.

### 70. What does `model.train()` do in PyTorch?

`model.train()` switches the model to training mode. Layers like dropout become active, and BatchNorm would update running statistics if used.

### 71. What does `model.eval()` do in PyTorch?

`model.eval()` switches the model to evaluation mode. Dropout is disabled, and inference becomes deterministic for layers affected by training mode.

### 72. Why do we use `torch.no_grad()` during evaluation or generation?

`torch.no_grad()` disables gradient tracking. This saves memory and makes evaluation or generation faster because no backward graph is built.

### 73. What is backpropagation?

Backpropagation computes gradients of the loss with respect to model parameters. The optimizer then uses those gradients to update the parameters.

### 74. What is an optimizer?

An optimizer updates model parameters using gradients. In Transformer training, AdamW is commonly used because it works well with sparse and noisy gradients.

### 75. What is the difference between Adam and AdamW?

AdamW decouples weight decay from the gradient update. This usually gives better regularization behavior for Transformer models.

### 76. What is a gradient?

A gradient tells us how much each parameter contributes to the loss and in which direction it should be changed to reduce the loss.

### 77. Why do we call `optimizer.zero_grad()`?

PyTorch accumulates gradients by default. `optimizer.zero_grad()` clears old gradients before computing the next backward pass.

### 78. What is the role of `loss.backward()`?

`loss.backward()` computes gradients for all parameters involved in producing the loss.

### 79. What is the role of `optimizer.step()`?

`optimizer.step()` updates model parameters using the gradients computed during backpropagation.

### 80. What is batch size?

Batch size is the number of training examples processed before one gradient update. Larger batches can be more stable but require more memory.

### 81. What is gradient accumulation?

Gradient accumulation simulates a larger batch by accumulating gradients over several smaller micro-batches before calling `optimizer.step()`.

### 82. What is dropout?

Dropout is a regularization technique that randomly disables some activations during training. It helps reduce overfitting.

### 83. What is weight decay?

Weight decay is a regularization technique that penalizes large weights. It helps keep the model from becoming too specialized to the training data.

### 84. What is perplexity?

Perplexity is an evaluation metric derived from cross-entropy loss. Lower perplexity usually means the model is better at predicting the next token.

### 85. What is the difference between logits, probabilities, and tokens?

Logits are raw model scores. Probabilities are obtained after applying softmax to logits. Tokens are discrete IDs selected from the vocabulary.

### 86. What is softmax?

Softmax converts logits into probabilities that sum to 1. It is used to interpret model scores over the vocabulary.

### 87. Why is cross-entropy used for language modeling?

Cross-entropy measures how far the model's predicted probability distribution is from the true next token. It is well suited for classification over vocabulary tokens.

### 88. What is mixed precision training?

Mixed precision training uses lower-precision formats such as FP16 or BF16 for some operations to reduce memory usage and speed up training, while keeping stability with selected higher-precision operations.

### 89. What is a tensor?

A tensor is a multidimensional array used by PyTorch to store inputs, parameters, activations, gradients, and outputs.

### 90. What is the shape of a typical GPT input batch?

A typical GPT input batch has shape `[batch_size, sequence_length]`, where each value is a token ID.

### 91. What is the shape of logits in a GPT model?

The logits usually have shape `[batch_size, sequence_length, vocab_size]`. For each position, the model scores every token in the vocabulary.

### 92. Why do we move tensors to a device?

Tensors and model parameters must be on the same device, such as CPU, CUDA GPU, or Apple Silicon MPS. Otherwise PyTorch raises a device mismatch error.

### 93. What is the role of `DataLoader`?

`DataLoader` reads examples from a Dataset, groups them into batches, optionally shuffles them, and uses a `collate_fn` to prepare batch tensors.

### 94. What is the difference between `Dataset` and `DataLoader`?

`Dataset` defines how to access one example. `DataLoader` defines how to iterate over many examples in batches.

### 95. What is a model checkpoint used for during training?

A checkpoint lets you resume training after interruption, compare different training steps, and reuse trained weights for generation or fine-tuning.

### 96. What is fine-tuning?

Fine-tuning continues training from a pretrained model on a more specific dataset or task, such as chat-style question answering.

### 97. What is the difference between pretraining and fine-tuning?

Pretraining teaches broad language patterns from large text corpora. Fine-tuning adapts the model to a specific task, format, or behavior.

### 98. Why is GPU memory important in LLM training?

LLM training stores parameters, gradients, optimizer states, activations, and batch tensors. Larger models, longer sequences, and larger batches require more memory.

### 99. What would you check first if the training loss becomes NaN?

I would check the learning rate, gradient explosion, invalid data, mixed precision stability, tensor values, and whether gradient clipping is correctly applied.

### 100. How would you explain your AI level honestly in an interview?

I would say I have strong software engineering experience and hands-on Applied AI practice. I understand and implemented the main LLM pipeline from data to tokenizer, PyTorch model, training loop, checkpointing, generation, and LoRA fine-tuning, and I am continuing to deepen production-scale ML experience.

## Serving, inference, deployment, and monitoring

### 101. What is inference?

Inference is the phase where a trained model is used to produce predictions or generated text from new inputs. In an LLM, inference means taking a prompt and generating the next tokens.

### 102. What is model serving?

Model serving means exposing a trained model through an interface, usually an API, so applications can send requests and receive predictions.

### 103. What is the difference between training and inference?

Training updates model parameters using data and loss. Inference uses fixed trained parameters to generate outputs without updating the model.

### 104. What files are needed for inference in FikrLLM?

Inference needs a compatible checkpoint, model configuration, `tokenizer.json`, and generation code. The tokenizer and checkpoint must match the same vocabulary and model architecture.

### 105. How would you serve FikrLLM with FastAPI?

I would create a FastAPI endpoint such as `/generate`, load the tokenizer and checkpoint at startup, validate the prompt and generation parameters, run generation under `torch.no_grad()`, and return the generated text as JSON.

### 106. Why should the model be loaded once at startup?

Loading the model for every request would be very slow. Loading it once at startup keeps the model in memory and makes each inference request faster.

### 107. What is latency?

Latency is the time between receiving a request and returning a response. For LLMs, latency depends on prompt length, generated tokens, model size, hardware, and decoding strategy.

### 108. What is throughput?

Throughput is the number of requests or tokens the system can process per second. A serving system should balance low latency and high throughput.

### 109. What is batching in inference?

Batching groups multiple inference requests together so the model can process them more efficiently. It can improve throughput but may increase latency for some users.

### 110. What is streaming generation?

Streaming generation sends tokens to the client as they are generated instead of waiting for the full answer. This improves perceived responsiveness.

### 111. Why is `torch.no_grad()` important during inference?

It disables gradient tracking, reducing memory usage and improving speed because no backward graph is created.

### 112. Why should we call `model.eval()` before inference?

`model.eval()` disables training-specific behavior such as dropout, making inference more stable and deterministic.

### 113. What is cold start?

Cold start is the delay when a service starts and must load the model, tokenizer, configuration, and possibly move weights to GPU before serving requests.

### 114. What is warmup in model serving?

Warmup means running a few dummy inference calls after startup to initialize kernels, allocate memory, and reduce latency for the first real requests.

### 115. What is autoscaling?

Autoscaling automatically increases or decreases the number of serving instances based on traffic, CPU/GPU usage, queue length, or latency.

### 116. What metrics would you monitor in LLM serving?

I would monitor request count, error rate, latency, tokens per second, GPU/CPU memory, GPU utilization, timeout rate, prompt length, output length, and user feedback.

### 117. What logs are useful during inference?

Useful logs include request ID, timestamp, model version, prompt length, generation parameters, latency, output length, status code, and error messages. Sensitive user data should be masked or avoided.

### 118. Why is model versioning important in serving?

Model versioning makes it possible to know exactly which checkpoint produced which output, compare versions, rollback safely, and debug regressions.

### 119. How would you deploy a model API?

I would package the service with Docker, expose a FastAPI endpoint, configure environment variables, add health checks, deploy to a VM or Kubernetes, and monitor logs and metrics.

### 120. What is a health check endpoint?

A health check endpoint returns whether the service is alive and ready. For example, `/health` can confirm that the API is running and the model is loaded.

### 121. What is the difference between CPU, GPU, and MPS inference?

CPU inference is easier but slower. GPU inference is faster for large matrix operations. MPS is Apple's acceleration backend for Apple Silicon machines.

### 122. What is quantization?

Quantization reduces model weight precision, for example from FP32 to INT8 or 4-bit. It can reduce memory usage and speed up inference, sometimes with a quality trade-off.

### 123. What is the KV cache in serving?

The KV cache stores attention keys and values from previous tokens during generation. It avoids recomputing the full context at every new token and improves inference speed.

### 124. Why can long prompts make inference slower?

Long prompts increase the sequence length processed by attention. Since attention cost grows with sequence length, longer contexts use more memory and compute.

### 125. How would you protect an inference API from abuse?

I would add authentication, rate limiting, request size limits, timeout limits, max generated tokens, logging, monitoring, and content safety checks if needed.

### 126. What is timeout handling?

Timeout handling stops requests that take too long. It protects the service from hanging requests and helps keep resources available.

### 127. What is a fallback strategy?

A fallback strategy defines what to do if the main model fails, such as returning a safe error, using a smaller model, retrying once, or sending the request to another service.

### 128. How would you debug slow inference?

I would measure prompt length, generated token count, model size, device placement, batch size, KV cache usage, CPU/GPU utilization, and whether the model is accidentally reloaded per request.

### 129. How would you debug different outputs for the same prompt?

I would check the decoding strategy, temperature, top-k/top-p settings, random seed, dropout mode, and whether `model.eval()` is used.

### 130. What is the difference between offline evaluation and online monitoring?

Offline evaluation tests the model before deployment on prepared datasets. Online monitoring observes real production behavior such as latency, errors, usage patterns, and user feedback.

## MLOps, LLMOps, AIOps, and production operations

### 131. What is MLOps?

MLOps is the set of practices used to build, deploy, monitor, version, and maintain machine learning systems in production. It combines machine learning, DevOps, data engineering, and software engineering.

### 132. What is LLMOps?

LLMOps is MLOps specialized for large language models. It focuses on prompt management, model serving, fine-tuning, evaluation, RAG pipelines, vector databases, safety, cost tracking, latency, and monitoring LLM behavior in production.

### 133. What is AIOps?

AIOps uses AI to improve IT operations. It can help detect incidents, analyze logs, predict failures, reduce alert noise, and support root-cause analysis.

### 134. What is the difference between MLOps, LLMOps, and AIOps?

MLOps operates ML systems. LLMOps operates LLM-based systems. AIOps uses AI to operate IT systems. In short: MLOps and LLMOps are about running AI models; AIOps is about using AI to run infrastructure and operations.

### 135. Where would MLflow fit in this project?

MLflow can track experiments, parameters, metrics, artifacts, model versions, and checkpoints. It helps compare training runs and keep experiment history organized.

### 136. What is a model registry?

A model registry stores versioned models with metadata such as stage, metrics, owner, training data version, and deployment status. It helps manage promotion from development to production.

### 137. What is data versioning?

Data versioning tracks the exact dataset used for each experiment. It is important because a model cannot be reproduced if the data version is unknown.

### 138. Why is reproducibility important in MLOps?

Reproducibility allows the team to rebuild the same model from the same code, data, configuration, and environment. It makes debugging, auditing, and comparison reliable.

### 139. What is model drift?

Model drift happens when model performance degrades over time because production data changes compared with the training data.

### 140. What is data drift?

Data drift means the distribution of input data changes over time. For example, user prompts may become longer, use different languages, or cover new topics.

### 141. What is concept drift?

Concept drift means the relationship between inputs and expected outputs changes. The same input pattern may require a different answer because user behavior, business rules, or the environment changed.

### 142. How would you monitor an LLM in production?

I would monitor latency, error rate, token usage, cost, prompt length, output length, user feedback, safety violations, hallucination reports, retrieval quality for RAG, and model version behavior.

### 143. What is observability?

Observability is the ability to understand the internal state of a system from external signals such as logs, metrics, traces, and events.

### 144. What is the difference between logs, metrics, and traces?

Logs describe events. Metrics are numerical measurements over time. Traces show the path of a request across services and help identify bottlenecks.

### 145. What is CI/CD for ML?

CI/CD for ML automates testing, packaging, validation, and deployment of ML code, data pipelines, and model services. It should check both software quality and model quality.

### 146. What should be tested in an LLMOps pipeline?

Tests should cover data validation, tokenizer compatibility, model loading, inference endpoint behavior, generation limits, safety rules, latency budget, and regression tests on evaluation prompts.

### 147. What is a rollback?

A rollback means returning to a previous stable model or service version when the new version causes errors, poor quality, high latency, or unexpected behavior.

### 148. What is a canary deployment?

A canary deployment releases a new model or service version to a small percentage of traffic first. If metrics look good, traffic is gradually increased.

### 149. What is an A/B test for LLMs?

An A/B test compares two model or prompt versions on real or controlled traffic to measure quality, latency, cost, user satisfaction, or business impact.

### 150. What is prompt versioning?

Prompt versioning tracks changes to system prompts, templates, examples, and instructions. It is important because prompt changes can strongly affect output quality.

### 151. What is RAG observability?

RAG observability monitors the retrieval and generation parts of a RAG system. It checks retrieved documents, relevance, context length, answer quality, latency, and failure cases.

### 152. What is a vector database used for in LLMOps?

A vector database stores embeddings and enables semantic search. In RAG, it retrieves relevant documents that are added to the prompt before generation.

### 153. What is an embedding model?

An embedding model converts text into dense vectors that capture semantic meaning. Similar texts should have similar vector representations.

### 154. What is cost monitoring in LLMOps?

Cost monitoring tracks GPU usage, API calls, token usage, storage, and infrastructure cost. It helps prevent uncontrolled spending in production.

### 155. What is rate limiting?

Rate limiting restricts how many requests a user or service can send in a time window. It protects the system from abuse and overload.

### 156. What is incident response in AI systems?

Incident response defines how the team detects, investigates, mitigates, and communicates issues such as model outage, unsafe outputs, latency spikes, or quality regressions.

### 157. What is governance in AI/LLM systems?

Governance defines rules around model usage, data privacy, access control, auditability, safety, compliance, and responsible AI practices.

### 158. What is a human-in-the-loop system?

A human-in-the-loop system includes human review or validation in the AI workflow, especially for sensitive decisions, low-confidence outputs, or safety-critical use cases.

### 159. What is the difference between batch inference and real-time inference?

Batch inference processes many inputs at scheduled times. Real-time inference responds immediately to user requests with strict latency requirements.

### 160. How would you explain your Ops mindset in an interview?

I would say that I do not only think about building a model. I also think about how to version it, deploy it, monitor it, rollback safely, control cost, handle incidents, and keep the system reliable over time.
