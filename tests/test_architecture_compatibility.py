import pytest
import torch

from flagos_compressor.calibration.mappings import (
    gptq_sequential_groups,
    infer_awq_mappings,
)
from flagos_compressor.calibration.modeling import (
    capture_first_layer_inputs,
    decoder_layers,
    sanitize_kwargs,
)
from flagos_compressor.calibration.moe import linearize_fused_experts
from flagos_compressor.calibration.runner import (
    quantize_model_sequential,
    selected_linears,
)
from flagos_compressor.core.policy import (
    AWQPolicy,
    AutoRoundPolicy,
    GPTQPolicy,
    QuantizationPolicy,
)


def _policy(method: str) -> QuantizationPolicy:
    return QuantizationPolicy(
        selections=("linear",),
        method=method,
        num_bits=4,
        group_size=8,
        gptq=GPTQPolicy(desc_act=False),
        awq=AWQPolicy(n_grid=1),
        autoround=AutoRoundPolicy(iters=1, batch_size=1),
    )


def _run_tiny_model(model, method: str, sequence_length: int = 8) -> dict:
    linearize_fused_experts(model)
    layers = decoder_layers(model)
    batches = [
        {
            "input_ids": torch.randint(0, model.config.vocab_size, (1, sequence_length)),
            "attention_mask": torch.ones(1, sequence_length, dtype=torch.long),
        }
        for _ in range(2)
    ]
    samples = capture_first_layer_inputs(
        model,
        layers[0][1],
        batches,
        device=torch.device("cpu"),
    )
    return quantize_model_sequential(
        layers,
        samples,
        _policy(method),
        device=torch.device("cpu"),
    )


def test_sanitize_kwargs_preserves_inputs_for_var_keyword_layer():
    class Layer(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states, kwargs

    kwargs = {"position_ids": torch.ones(1), "attention_mask": torch.ones(1)}

    assert sanitize_kwargs(Layer(), kwargs) == kwargs


def test_deepseek_mla_has_explicit_gptq_and_awq_projection_order():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_a_proj = torch.nn.Linear(16, 8, bias=False)
            self.q_a_layernorm = torch.nn.LayerNorm(8)
            self.q_b_proj = torch.nn.Linear(8, 16, bias=False)
            self.kv_a_proj_with_mqa = torch.nn.Linear(16, 16, bias=False)
            self.kv_a_layernorm = torch.nn.LayerNorm(8)
            self.kv_b_proj = torch.nn.Linear(8, 24, bias=False)
            self.o_proj = torch.nn.Linear(8, 16, bias=False)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(16)
            self.self_attn = Attention()

    block = Block()
    selected = {
        name
        for name, module in block.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    groups = gptq_sequential_groups(sorted(selected))
    mappings = infer_awq_mappings(block, selected)

    assert set(groups[0]) == {
        "self_attn.q_a_proj",
        "self_attn.kv_a_proj_with_mqa",
    }
    assert set(groups[1]) == {
        "self_attn.q_b_proj",
        "self_attn.kv_b_proj",
    }
    assert groups[2] == ["self_attn.o_proj"]
    assert any(
        mapping.previous_name == "input_layernorm"
        and set(mapping.quantized_names)
        == {"self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"}
        for mapping in mappings
    )
    assert any(
        mapping.previous_name == "self_attn.q_a_layernorm"
        and mapping.quantized_names == ("self_attn.q_b_proj",)
        for mapping in mappings
    )
    assert any(
        mapping.previous_name == "self_attn.kv_a_layernorm"
        and mapping.quantized_names == ("self_attn.kv_b_proj",)
        for mapping in mappings
    )


def test_tiny_glm4_and_deepseek_v2_v3_run_all_calibrators():
    from transformers import (
        DeepseekV2Config,
        DeepseekV2ForCausalLM,
        DeepseekV3Config,
        DeepseekV3ForCausalLM,
        Glm4MoeConfig,
        Glm4MoeForCausalLM,
    )

    common = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "use_cache": False,
    }
    moe = {
        "moe_intermediate_size": 8,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "first_k_dense_replace": 0,
        "n_group": 1,
        "topk_group": 1,
    }
    factories = (
        lambda: Glm4MoeForCausalLM(Glm4MoeConfig(**common, **moe)),
        lambda: DeepseekV2ForCausalLM(
            DeepseekV2Config(
                **common,
                **moe,
                head_dim=16,
                q_lora_rank=8,
                kv_lora_rank=8,
                qk_nope_head_dim=8,
                qk_rope_head_dim=8,
                v_head_dim=4,
            )
        ),
        lambda: DeepseekV3ForCausalLM(
            DeepseekV3Config(
                **common,
                **moe,
                q_lora_rank=8,
                kv_lora_rank=8,
                qk_nope_head_dim=8,
                qk_rope_head_dim=8,
                v_head_dim=4,
                routed_scaling_factor=1.0,
            )
        ),
    )

    for factory in factories:
        for method in ("gptq", "awq", "autoround"):
            torch.manual_seed(7)
            quantized = _run_tiny_model(factory().eval(), method)
            assert quantized


@pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
@pytest.mark.parametrize("layer_type", [
    "sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention",
])
def test_tiny_deepseek_v4_preserves_forward_kwargs_and_runs_calibrators(method, layer_type):
    import transformers

    if not hasattr(transformers, "DeepseekV4Config"):
        pytest.skip("installed Transformers release does not include DeepSeek-V4")
    DeepseekV4Config = transformers.DeepseekV4Config
    DeepseekV4ForCausalLM = transformers.DeepseekV4ForCausalLM

    def factory():
        config = DeepseekV4Config(
            vocab_size=32,
            hidden_size=16,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            qk_rope_head_dim=4,
            q_lora_rank=8,
            num_experts_per_tok=1,
            n_routed_experts=2,
            n_shared_experts=1,
            max_position_embeddings=512,
            layer_types=[layer_type],
            mlp_layer_types=["moe"],
            hc_mult=2,
            hc_sinkhorn_iters=2,
            o_groups=2,
            o_lora_rank=8,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            num_nextn_predict_layers=0,
            use_cache=False,
        )
        return DeepseekV4ForCausalLM(config).eval()

    torch.manual_seed(11)
    model = factory()
    linearize_fused_experts(model)
    layer_name, layer = decoder_layers(model)[0]
    selected = selected_linears(layer_name, layer, _policy(method))

    assert "self_attn.kv_proj" in selected
    assert "self_attn.o_b_proj" in selected
    assert "self_attn.o_a_proj" not in selected
    # Cross the 128-token compression/window boundary. Eight-token tests did
    # not reveal cache accumulation between repeated calibration forwards.
    assert _run_tiny_model(factory(), method, sequence_length=256)
