from importlib.resources import files

import pytest
import torch

from fikrllm import GPT, ModelConfig, Tokenizer, apply_lora, merge_lora
from fikrllm.data.collate import collate_fn


def test_model_config_derives_head_and_feedforward_dimensions():
    config = ModelConfig(d_model=32, n_heads=4, n_layers=1, vocab_size=64)

    assert config.d_k == 8
    assert config.d_ff == 128


def test_model_config_rejects_incompatible_head_count():
    with pytest.raises(ValueError, match="not divisible"):
        ModelConfig(d_model=30, n_heads=8)


def test_packaged_tokenizer_has_expected_special_tokens():
    tokenizer_path = files("fikrllm").joinpath("assets/tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    assert tokenizer.vocab_size == 32_000
    assert tokenizer.encode("[BOS]", add_special_tokens=False) == [tokenizer.BOS]
    assert tokenizer.normalize("أَحْمَد   على") == "احمد علي"


def test_collate_shifts_inputs_and_targets():
    examples = [
        {"input_ids": torch.tensor([2, 10, 11, 3])},
        {"input_ids": torch.tensor([2, 12, 3])},
    ]

    batch = collate_fn(examples, pad_id=0)

    assert batch["input_ids"].tolist() == [[2, 10, 11], [2, 12, 3]]
    assert batch["targets"].tolist() == [[10, 11, 3], [12, 3, 0]]


def test_gpt_forward_shapes_and_loss():
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_seq_len=16,
        dropout=0.0,
    )
    model = GPT(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss, cache = model(inputs, targets=targets)

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss.ndim == 0
    assert cache is None


def test_lora_merge_preserves_output():
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        max_seq_len=16,
        dropout=0.0,
    )
    model = GPT(config).eval()
    apply_lora(model, r=4, alpha=8)

    with torch.no_grad():
        for block in model.blocks:
            block.attn.W_q.B.normal_()

    inputs = torch.randint(0, config.vocab_size, (1, 6))
    before, _, _ = model(inputs)
    merged = merge_lora(model)
    after, _, _ = model(inputs)

    assert merged == 4
    torch.testing.assert_close(before, after)
