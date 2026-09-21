import torch
import pytest
from torch import nn

from flagos_compressor.packing.autoawq import pack_autoawq_gemm
from flagos_compressor.quantizers.awq import (
    apply_awq_scale,
    pseudo_quantize_awq,
    search_awq_clip,
    search_awq_scale,
)


def _unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    inverse = [order.index(index) for index in range(8)]
    words = packed.to(torch.int64) & 0xFFFFFFFF
    ordered = torch.stack([(words >> (4 * i)) & 0xF for i in range(8)], dim=2)
    return ordered[..., inverse].reshape(packed.shape[0], packed.shape[1] * 8)


def test_awq_pseudo_quantize_matches_asymmetric_minmax():
    weight = torch.tensor(
        [[-2.0, -1.0, 0.0, 3.0, -4.0, 1.0, 2.0, 4.0]],
        dtype=torch.float32,
    )
    result = pseudo_quantize_awq(weight, group_size=4, zero_point=True)
    assert result.scales.shape == (1, 2)
    assert result.zeros is not None and result.zeros.shape == (1, 2)
    expected_scale = torch.tensor([5 / 15, 8 / 15])
    torch.testing.assert_close(result.scales[0], expected_scale)


def test_autoawq_gemm_pack_round_trip():
    torch.manual_seed(11)
    weight = torch.randn(8, 16)
    quantized = pseudo_quantize_awq(weight, group_size=8, zero_point=True)
    assert quantized.zeros is not None
    packed = pack_autoawq_gemm(
        quantized.weight,
        quantized.scales,
        quantized.zeros,
        group_size=8,
    )
    assert packed.qweight.shape == (16, 1)
    assert packed.qzeros.shape == (2, 1)
    assert packed.scales.shape == (2, 8)

    codes = _unpack_awq(packed.qweight)
    zeros = _unpack_awq(packed.qzeros)
    groups = torch.arange(16) // 8
    reconstructed = (
        packed.scales.float()[groups] * (codes - zeros[groups])
    ).t()
    torch.testing.assert_close(
        reconstructed, quantized.weight, atol=2e-3, rtol=2e-3
    )


def test_awq_scale_and_clip_search_return_channelwise_values():
    torch.manual_seed(5)
    linear = nn.Linear(8, 8, bias=False)
    inputs = torch.randn(2, 4, 8)
    scales = search_awq_scale(
        linear,
        [linear],
        inputs,
        group_size=4,
        n_grid=4,
        max_chunk_memory=1024 * 1024,
    )
    assert scales.shape == (8,)
    assert torch.isfinite(scales).all()

    maxima = search_awq_clip(
        linear.weight,
        inputs,
        group_size=4,
        n_grid=4,
        sample_tokens=8,
        output_chunk_size=4,
    )
    assert maxima.shape == (8, 2, 1)
    assert torch.isfinite(maxima).all()


def test_awq_scale_preserves_zero_centered_qwen_rmsnorm_linear_pair():
    class Qwen3_5RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.1, -0.2, 0.3, -0.4]))

        def forward(self, inputs):
            return inputs * (1 + self.weight)

    norm = Qwen3_5RMSNorm()
    linear = nn.Linear(4, 3, bias=False)
    inputs = torch.randn(2, 4)
    expected = linear(norm(inputs))

    apply_awq_scale(norm, [linear], torch.tensor([0.5, 2.0, 0.25, 4.0]))

    torch.testing.assert_close(linear(norm(inputs)), expected)


def test_awq_chunked_forward_keeps_global_search_and_batched_kwargs():
    class BatchedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 16, bias=False)
            self.batch_limit = 5
            self.observed = []

        def forward(self, x, *, position_embeddings, attention_mask, input_ids,
                    cache_position, past_key_values):
            batch, seq, _ = x.shape
            assert batch <= self.batch_limit
            assert attention_mask.shape == (batch, 1, seq, seq)
            assert input_ids.shape == (batch, seq)
            assert all(value.shape == (batch, seq, 16) for value in position_embeddings)
            assert cache_position.shape == (seq,)
            assert not past_key_values['used']
            past_key_values['used'] = True
            self.observed.append(batch)
            return self.proj(x).tanh() + position_embeddings[0] * input_ids[..., None]

    torch.manual_seed(26)
    block = BatchedBlock()
    inputs = torch.randn(5, 3, 8)
    inputs[-1].mul_(20)  # Uneven final batch must receive its true sample weight.
    kwargs = {'position_embeddings': (torch.randn(5, 3, 16), torch.randn(5, 3, 16)),
              'attention_mask': torch.ones(5, 1, 3, 3), 'input_ids': torch.arange(15).reshape(5, 3),
              'cache_position': torch.arange(3), 'past_key_values': {'used': False}}
    original = block.proj.weight.detach().clone()
    whole = search_awq_scale(block, [block.proj], inputs, kwargs=kwargs, group_size=4, n_grid=20)
    block.batch_limit = 2
    block.observed.clear()
    chunked = search_awq_scale(block, [block.proj], inputs, kwargs=kwargs,
                               group_size=4, n_grid=20, forward_batch_size=2)
    torch.testing.assert_close(chunked, whole)
    torch.testing.assert_close(block.proj.weight, original, rtol=0, atol=0)
    assert block.observed == [2, 2, 1] * 21
    assert kwargs['past_key_values'] == {'used': False}


def test_awq_forward_limit_does_not_split_flat_expert_rows():
    torch.manual_seed(2)
    linear = nn.Linear(8, 8, bias=False)
    sizes = []
    handle = linear.register_forward_pre_hook(lambda _m, args: sizes.append(args[0].shape[0]))
    try:
        search_awq_scale(linear, [linear], torch.randn(17, 8), group_size=4,
                         n_grid=4, forward_batch_size=2)
    finally:
        handle.remove()
    assert sizes == [17] * 5


def test_awq_forward_limit_cli_and_policy_validation():
    from flagos_compressor.cli.main import build_parser
    from flagos_compressor.cli.helpers import build_quantization_policy
    from flagos_compressor.core.policy import AWQPolicy
    args = build_parser().parse_args(['quantize', '--input', 'unused', '--output', 'unused-out',
                                     '--method', 'awq', '--select', 'attention', '--awq-forward-batch-size', '4'])
    assert build_quantization_policy(args).awq.forward_batch_size == 4
    with pytest.raises(ValueError, match='forward_batch_size'):
        AWQPolicy(forward_batch_size=0)
