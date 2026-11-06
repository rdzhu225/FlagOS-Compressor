"""AWQ must replay each sample's actual attention and MoE arguments."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flagos_compressor.calibration.awq_inputs import AWQCall, AWQReplay
from flagos_compressor.calibration.modeling import capture_first_layer_inputs, decoder_layers
from flagos_compressor.calibration.moe import linearize_fused_experts
from flagos_compressor.calibration.runner import quantize_layer_awq
from flagos_compressor.core.policy import AWQPolicy, QuantizationPolicy


@pytest.mark.parametrize("keyword_input", [False, True])
def test_replay_keeps_positional_ids_masks_and_mutable_state_per_call(keyword_input):
    class Module(nn.Module):
        def forward(self, hidden_states, input_ids, *, mask, state):
            assert state.count == 0
            state.count += 1
            return hidden_states + input_ids.unsqueeze(-1) * mask

    module = Module()
    calls, features, expected = [], [], []
    for rows in [1, 2]:
        hidden = torch.randn(rows, 3, 4)
        ids = torch.arange(rows * 3).reshape(rows, 3)
        state = SimpleNamespace(count=0)
        kwargs = {"mask": torch.full((rows, 3, 1), rows), "state": state}
        args = (hidden, ids)
        if keyword_input:
            kwargs.update(hidden_states=hidden, input_ids=ids)
            args = ()
        calls.append(AWQCall.capture(args, kwargs))
        expected.append(module(*args, **kwargs))
        features.append(hidden)
        assert state.count == 1  # The captured snapshot predates this mutation.
    replay = AWQReplay(module, calls)
    for _ in range(2):
        torch.testing.assert_close(replay(torch.cat(features)), torch.cat(expected))
    with pytest.raises(ValueError, match="feature rows"):
        replay(torch.cat(features)[:1])


@pytest.mark.parametrize("mlp_type", ["moe", "hash_moe"])
@pytest.mark.parametrize("attention_type", [
    "sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention",
])
def test_v4_awq_multiple_samples_preserve_attention_and_router_arguments(mlp_type, attention_type):
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    torch.manual_seed(31)
    config = DeepseekV4Config(
        vocab_size=32, hidden_size=32, moe_intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, qk_rope_head_dim=4, q_lora_rank=16,
        num_experts_per_tok=1, n_routed_experts=2, n_shared_experts=1,
        max_position_embeddings=512, layer_types=[attention_type],
        mlp_layer_types=[mlp_type], hc_mult=2, hc_sinkhorn_iters=2,
        o_groups=2, o_lora_rank=16, index_n_heads=2, index_head_dim=16,
        index_topk=2, num_nextn_predict_layers=0, use_cache=False,
    )
    model = DeepseekV4ForCausalLM(config).eval()
    layer_name, layer = decoder_layers(model)[0]
    if mlp_type == "hash_moe":
        layer.mlp.gate.tid2eid.copy_((torch.arange(config.vocab_size) % 2).unsqueeze(-1))
    linearize_fused_experts(model)
    batches = [
        {"input_ids": (torch.arange(256) + offset).remainder(32).unsqueeze(0),
         "attention_mask": torch.ones(1, 256, dtype=torch.long)}
        for offset in [0, 7]
    ]
    samples = capture_first_layer_inputs(model, layer, batches, device=torch.device("cpu"))
    observed = []
    handle = layer.mlp.register_forward_pre_hook(
        lambda _m, args, kwargs: observed.append(
            (args[1] if len(args) > 1 else kwargs["input_ids"]).clone()
        ), with_kwargs=True,
    )
    try:
        results = quantize_layer_awq(
            layer_name, layer, samples,
            QuantizationPolicy(selections=("linear",), method="awq", num_bits=4,
                               group_size=8, awq=AWQPolicy(n_grid=2)),
            device=torch.device("cpu"),
        )
    finally:
        handle.remove()
    assert results
    assert len(observed) > len(samples)  # Includes scale-search replays.
    assert all(any(torch.equal(ids, batch["input_ids"]) for batch in batches) for ids in observed)
    for batch in batches:
        assert sum(torch.equal(ids, batch["input_ids"]) for ids in observed) > 1
