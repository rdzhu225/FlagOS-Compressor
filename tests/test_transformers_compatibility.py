"""Model behavior required from the Transformers calibration baseline.

Use real, randomly initialized V4 definitions, including compression boundaries
and FP32 norm parameters. No model-specific norm hook or cache copying is used.
"""

import pytest
import torch

from flagos_compressor.calibration.modeling import (
    capture_first_layer_inputs,
    decoder_layers,
    forward_layer_samples,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA"
    )),
])
@pytest.mark.parametrize("attention_type", [
    "sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention",
])
@torch.no_grad()
def test_v4_cache_free_replay_matches_full_model(dtype, device, attention_type):
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    torch.manual_seed(17)
    config = DeepseekV4Config(
        vocab_size=32, hidden_size=16, moe_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, qk_rope_head_dim=4, q_lora_rank=8,
        num_experts_per_tok=1, n_routed_experts=2, n_shared_experts=1,
        max_position_embeddings=512, layer_types=[attention_type],
        mlp_layer_types=["hash_moe"], hc_mult=2, hc_sinkhorn_iters=2,
        o_groups=2, o_lora_rank=8, index_n_heads=2, index_head_dim=8,
        index_topk=2, num_nextn_predict_layers=0, use_cache=False,
    )
    model = DeepseekV4ForCausalLM(config).eval().to(dtype)
    # Match the FP32 exclusions used by from_pretrained with BF16 linears.
    for name, parameter in model.named_parameters():
        if any(part in model._keep_in_fp32_modules_strict for part in name.split(".")):
            parameter.data = parameter.data.float()
    layer = decoder_layers(model)[0][1]
    layer.mlp.gate.tid2eid.copy_((torch.arange(config.vocab_size) % 2).unsqueeze(-1))
    batch = {
        "input_ids": torch.arange(256).remainder(config.vocab_size).unsqueeze(0),
        "attention_mask": torch.ones(1, 256, dtype=torch.long),
    }
    reference = []
    handle = layer.register_forward_hook(
        lambda _module, _args, output: reference.append(
            (output[0] if isinstance(output, (tuple, list)) else output).clone()
        )
    )
    try:
        model.to(device)
        result = model(**{key: value.to(device) for key, value in batch.items()}, use_cache=False)
    finally:
        handle.remove()
    assert result.past_key_values is None
    assert torch.isfinite(result.logits).all()
    model.cpu()
    samples = capture_first_layer_inputs(
        model, layer, [batch], device=torch.device("cpu")
    )
    assert samples[0][1].get("past_key_values") is None
    previous = None
    for _ in range(2):
        actual = forward_layer_samples(layer, samples, device=torch.device(device))[0][0][0]
        assert actual.dtype == reference[0].dtype
        # Calibration prepares positions on CPU; full CUDA forward prepares
        # them on CUDA. FP32 rotary arithmetic can differ by a few ULPs.
        tolerance = {"rtol": 1e-5, "atol": 1e-7} if dtype == torch.float32 else {"rtol": 0, "atol": 0}
        torch.testing.assert_close(actual, reference[0].cpu(), **tolerance)
        if previous is not None:
            torch.testing.assert_close(actual, previous, rtol=0, atol=0)
        previous = actual


def test_v4_fp32_norm_preserves_activation_dtype_and_gradients():
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RMSNorm

    norm = DeepseekV4RMSNorm(8).float()
    hidden = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    actual = norm(hidden)
    assert norm.weight.dtype == torch.float32
    assert actual.dtype == hidden.dtype
    actual.float().square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
