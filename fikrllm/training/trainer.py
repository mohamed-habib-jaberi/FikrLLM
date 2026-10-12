"""
trainer.py -- the training loop
"""

import itertools
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from fikrllm.config import GenerationConfig, TrainingConfig
from fikrllm.generation import generate
from fikrllm.model.gpt import GPT
from fikrllm.tokenizer import Tokenizer
from fikrllm.training.checkpoint import save_checkpoint
from fikrllm.training.schedule import cosine_with_warmup
from fikrllm.training.tracking import Tracker


# Compute loss while ignoring padding tokens.
def document_cross_entropy(logits: Tensor, targets: Tensor, pad_id: int) -> Tensor:
    """Next-token loss, averaged once per document rather than once per token.

    Standard language-model training averages over every token, which lets a
    long chunk outvote a short one just by having more positions in it. ch02's
    Check-8 flagged exactly that imbalance (an 83-chunk article dominating a
    single-chunk one); the fix this course applies is capping chunks per
    article before training ever sees them (ch02's hard cap), so no per-row
    weight needs to travel through the batch. This function carries the other
    half of that lesson: within a batch, a chunk's own token count should not
    decide its influence either, so the average happens in two stages:

      1. mean over the real tokens WITHIN a document, so length stops mattering
      2. plain mean ACROSS documents in the batch

    Padded positions are excluded from both stages.
    """

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    ).view(targets.shape)  # (B, T)
    
    real = (targets != pad_id).float()                  # (B, T)
    tokens_per_example = real.sum(dim=1).clamp(min=1.0)  # never divide by zero
    per_example = (per_token * real).sum(dim=1) / tokens_per_example   # (B,)

    return per_example.mean()
    

class Trainer:
    # Initialize this object with the state it needs later.
    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        config: TrainingConfig,
        eval_loader: DataLoader | None = None,
        tracker: Tracker | None = None,
        tokenizer: Tokenizer | None = None,
        device: str | torch.device | None = None,
    ) -> None:

        if not device:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.tracker = tracker
        self.config = config
        self.tokenizer = tokenizer

        self.step = 0

        torch.manual_seed(config.seed)

        self.optimizer = self._build_optimizer()
        self.scheduler = cosine_with_warmup(self.optimizer, config)

        self.sample_config = GenerationConfig(
            max_new_tokens=config.sample_max_new_tokens, seed=config.seed
        )

    # Create the AdamW optimizer for trainable model parameters.
    def _build_optimizer(self) -> torch.optim.AdamW:
        """AdamW, with weight decay on matrices only.
        """

        decay, no_decay = [], []
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue

            if parameter.dim() >= 2:
                # Linear layer weight [768, 768], embedding [vocab, 768]
                decay.append(parameter)
            else:
                # bias [768], LayerNorm gain [768]
                no_decay.append(parameter)

        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

        # betas control Adam's memory:
        #   0.90 is how much it remembers the direction of past steps.
        #   0.95 is how much it remembers the size of past steps.
        #  This project uses 0.95, not PyTorch's default 0.999, 
        #  because GPT-2 used 0.95. 
       
    def _to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    # Yield batches forever by repeatedly cycling through the DataLoader.
    def _batches(self):
        return itertools.chain.from_iterable(
            itertools.repeat(self.train_loader) # DataLoader
        )

    # Run one optimizer step, including gradient accumulation.
    def train_step(self, batches):
        """One optimizer step, over `accumulation_steps` micro-batches.

        micro 1 (24) → gradients added
        micro 2 (24) → gradients added
        ...
        micro 6 (24) → gradients added
                 ↓
                1 weight update  = like a batch of 144

        """

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(self.config.accumulation_steps):
            batch = self._to_device(next(batches))
            logits, _, _ = self.model(batch["input_ids"])
            loss = document_cross_entropy(
                logits, batch["targets"], self.model.config.pad_id,
            ) / self.config.accumulation_steps
            loss.backward()
            total_loss += loss.item()

        grad_norm = None
        if self.config.grad_clip > 0:
            # Returns the norm BEFORE clipping, which is the number worth
            # logging: a sudden spike is the earliest sign of trouble.

            # norm = 0.6: it's under 1.0, so nothing changes.
            # norm = 5.0: every gradient is multiplied by 1/5, so the norm becomes 1.0.
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        self.optimizer.step()
        self.scheduler.step()

        return total_loss, grad_norm

    # Measure average validation loss without updating model weights.
    @torch.no_grad()
    def evaluate(self) -> float | None:
        """Plain, flat mean loss on held-out data.

        Deliberately not train_step's per-document average: an eval number
        exists to be compared -- against the last eval, against phase 1
        (ch10), against another run -- so it should depend only on what the
        model has learned, not on how documents happened to be grouped into
        this batch.
        """

        if self.eval_loader is None:
            return None

        self.model.eval()
        total, batches = 0.0, 0
        for batch in itertools.islice(self.eval_loader, self.config.eval_batches):
            batch = self._to_device(batch)
            _, loss, _ = self.model(batch["input_ids"], targets=batch["targets"])
            if loss is None:
                raise RuntimeError("GPT must return a loss when targets are provided")
            total += loss.item()
            batches += 1

        self.model.train()
        return total / max(1, batches)

    # Send scalar metrics to the configured tracker.
    def _log(self, metrics: dict[str, float]) -> None:
        if self.tracker is not None:
            self.tracker.log(metrics, step=self.step)

    # Write a checkpoint for the current trainer state.
    def save(self, filename: str) -> Path:
        return save_checkpoint(
            f"{self.config.checkpoint_dir}/{filename}",
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
        )

    # Run the full training loop.
    def train(self):
        config = self.config

        started = time.time()
        history = []

        print(f"training on {self.device} for {config.max_steps} steps")

        batches = self._batches()

        progress = tqdm(
            total=config.max_steps, initial=self.step, unit="step",
            desc=config.run_name, dynamic_ncols=True,
        )

        while self.step < config.max_steps:
            loss, grad_norm = self.train_step(batches)
            self.step += 1
            learning_rate = self.scheduler.get_last_lr()[0]

            progress.update(1)
            progress.set_postfix(loss=f"{loss:.4f}", lr=f"{learning_rate:.2e}")

            if self.step % config.log_every == 0:
                metrics = {"train_loss": loss, "learning_rate": learning_rate}
                if grad_norm is not None:
                    metrics["grad_norm"] = float(grad_norm)

                self._log(metrics)
                history.append({
                    "step": self.step,
                    **metrics
                })

            print_every = config.print_every or config.log_every
            if self.step % print_every == 0:
                tqdm.write(
                    f"  step {self.step:>6}  loss {loss:.4f}"
                    f"  lr {learning_rate:.2e}"
                )

            if config.eval_every and self.step % config.eval_every == 0:
                eval_loss = self.evaluate()
                if eval_loss is not None:
                    self._log(
                        {"eval_loss": eval_loss}
                    )
                    tqdm.write(f"  step {self.step:>6}  eval_loss {eval_loss:.4f}")

            if config.save_every and self.step % config.save_every == 0:
                self.save(
                    f"{config.run_name}_step{self.step}.pt"
                )

            if config.sample_every and self.step % config.sample_every == 0:
                self._sample()

        progress.close()
        elapsed = time.time() - started
        print(f"done: {self.step} steps in {elapsed:.1f}s")
        return history

    # Generate sample text during training for qualitative inspection.
    @torch.no_grad()
    def _sample(self):
        """Continue each sample prompt, print it, and log the rows as a table.
        """
        prompts = self.config.sample_prompts

        if not prompts or self.tokenizer is None:
            return

        rows = []
        columns = ["step", "prompt", "completion"]

        for i, prompt in enumerate(prompts):
            response = generate(
                self.model, self.tokenizer, prompt, self.sample_config,
                device=self.device,
                chat=self.config.sample_chat
            )

            row = [self.step, prompt, response]
            log_line = f"  step {self.step:>6}  sample {prompt!r} -> {response!r}"

            rows.append(row)
            tqdm.write(log_line)

        if self.tracker is not None:
            self.tracker.log_table(
                "samples",
                columns=columns, rows=rows, step=self.step
            )

        self.model.train()



