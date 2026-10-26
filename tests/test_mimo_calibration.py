"""Exercise MiMo's layer-dependent input contract without downloading weights."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)

from flagos_compressor.calibration.inputs import prepare_calibration_inputs
from flagos_compressor.calibration.modeling import decoder_layers, forward_layer_samples
from flagos_compressor.calibration.runner import quantize_model_sequential
from flagos_compressor.core.policy import AWQPolicy, AutoRoundPolicy, QuantizationPolicy


class _HybridModel(nn.Module):
    """Real attention blocks with MiMo's global/SWA mask and RoPE selection."""

    # MiMo remote code does not supply the usual decoder discovery metadata.
    _no_split_modules = []

    def __init__(self, pattern):
        super().__init__()
        self.config = SimpleNamespace(model_type="mimo_v2")
        config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=len(pattern),
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
            max_position_embeddings=32,
            use_cache=False,
            sliding_window=3,
            layer_types=["sliding_attention" if p else "full_attention" for p in pattern],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
        config._attn_implementation = "eager"
        self.model = nn.Module()
        self.model.config = config
        self.model.embed_tokens = nn.Embedding(32, 16)
        self.model.layers = nn.ModuleList(
            LlamaDecoderLayer(config, index) for index in range(len(pattern))
        )
        for layer, is_swa in zip(self.model.layers, pattern):
            layer.attention_type = (
                "sliding_window_attention" if is_swa else "full_attention"
            )
        self.model.rotary_emb = LlamaRotaryEmbedding(config)
        swa_config = copy.deepcopy(config)
        swa_config.rope_parameters = {"rope_type": "default", "rope_theta": 100.0}
        self.model.swa_rotary_emb = LlamaRotaryEmbedding(swa_config)
        self.eval()

    def reference_kwargs(self, batch, layer):
        hidden = self.model.embed_tokens(batch["input_ids"])
        length = hidden.shape[1]
        positions = batch.get("position_ids", torch.arange(length).unsqueeze(0))
        indices = torch.arange(length)
        visible = indices[None, :] <= indices[:, None]
        rotary = self.model.rotary_emb
        if layer.attention_type == "sliding_window_attention":
            visible &= indices[None, :] > indices[:, None] - 3
            rotary = self.model.swa_rotary_emb
        visible = visible[None, None] & batch["attention_mask"][:, None, None, :].bool()
        mask = torch.zeros(visible.shape, dtype=hidden.dtype)
        mask.masked_fill_(~visible, torch.finfo(hidden.dtype).min)
        return {
            "attention_mask": mask,
            "position_embeddings": rotary(hidden, positions),
            "position_ids": positions,
            "cache_position": indices,
            "past_key_values": None,
            "use_cache": False,
        }


def _batches():
    return [
        {"input_ids": torch.randint(0, 32, (1, 8)),
         "attention_mask": torch.ones(1, 8, dtype=torch.long)},
        {"input_ids": torch.randint(0, 32, (1, 8)),
         "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]]),
         "position_ids": torch.arange(4, 12).unsqueeze(0)},
    ]


@pytest.mark.parametrize("pattern", [(0, 1, 0), (1, 0, 1)])
def test_hybrid_layer_replay_matches_full_forward(pattern):
    torch.manual_seed(42)
    model = _HybridModel(pattern)
    layers = decoder_layers(model)
    batches = _batches()
    samples, layer_kwargs = prepare_calibration_inputs(
        model, layers, batches, device=torch.device("cpu")
    )
    with torch.no_grad():
        for sample_index, batch in enumerate(batches):
            expected = model.model.embed_tokens(batch["input_ids"])
            actual = samples[sample_index][0][0]
            wrong = actual
            for name, layer in layers:
                reference = model.reference_kwargs(batch, layer)
                options = layer_kwargs[name][sample_index]
                torch.testing.assert_close(options["attention_mask"], reference["attention_mask"])
                torch.testing.assert_close(
                    options["position_embeddings"], reference["position_embeddings"]
                )
                expected = layer(expected, **reference)
                actual = layer(actual, **options)
                wrong = layer(wrong, **samples[sample_index][1])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert not torch.allclose(wrong, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
def test_all_calibrators_use_each_layers_mask_and_rope(method):
    torch.manual_seed(7)
    model = _HybridModel((0, 1, 0))
    layers = decoder_layers(model)
    # Two equal-length samples exercise both AutoRound input streams.
    batches = _batches()[:1] * 2
    samples, layer_kwargs = prepare_calibration_inputs(
        model, layers, batches, device=torch.device("cpu")
    )
    calls = {name: 0 for name, _ in layers}
    handles = []
    for name, layer in layers:
        reference = model.reference_kwargs(batches[0], layer)

        def check(_module, _args, kwargs, name=name, reference=reference):
            calls[name] += 1
            torch.testing.assert_close(kwargs["attention_mask"], reference["attention_mask"])
            torch.testing.assert_close(
                kwargs["position_embeddings"], reference["position_embeddings"]
            )

        handles.append(layer.register_forward_pre_hook(check, with_kwargs=True))
    try:
        results = quantize_model_sequential(
            layers, samples,
            QuantizationPolicy(
                selections=("linear",), method=method, num_bits=4, group_size=8,
                awq=AWQPolicy(n_grid=1),
                autoround=AutoRoundPolicy(iters=2, batch_size=1),
            ),
            device=torch.device("cpu"), layer_kwargs=layer_kwargs,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert len(results) == 21
    assert all(count >= 2 for count in calls.values())


def test_non_mimo_uses_existing_first_layer_capture():
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
    )).eval()
    layers = decoder_layers(model)
    samples, layer_kwargs = prepare_calibration_inputs(
        model, layers, _batches()[:1], device=torch.device("cpu")
    )
    assert layer_kwargs is None
    assert len(forward_layer_samples(layers[0][1], samples, device=torch.device("cpu"))) == 1
