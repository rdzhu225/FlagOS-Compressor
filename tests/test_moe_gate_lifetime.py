"""Bound expert gates must not retain a hidden copy of the BF16 weight banks."""
import gc
import weakref

import pytest
import torch
from torch import nn

from flagos_compressor.calibration.moe import LinearExperts2D


class ClampedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 8
        self.gate_up_proj = nn.Parameter(torch.randn(2, 12, 8))
        self.down_proj = nn.Parameter(torch.randn(2, 8, 6))
        self.limit = 0.7
        self.act_fn = nn.SiLU()

    def _apply_gate(self, value):
        gate, up = value.chunk(2, dim=-1)
        return self.act_fn(gate.clamp(max=self.limit)) * up.clamp(-self.limit, self.limit)


@pytest.mark.parametrize('real_deepseek', [False, True])
def test_gate_preserves_values_gradients_and_releases_original(real_deepseek):
    if real_deepseek:
        from transformers import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
        original = DeepseekV4Experts(DeepseekV4Config(
            hidden_size=8, intermediate_size=6, num_local_experts=2, swiglu_limit=0.7))
        for parameter in original.parameters():
            nn.init.normal_(parameter)
        del parameter
    else:
        original = ClampedExperts()
    x = torch.randn(4, 8, requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    weights = torch.tensor([[0.3, 0.7]]).expand(4, 2)
    expected = torch.zeros_like(x)
    for position in range(2):
        for expert in range(2):
            selected = indices[:, position] == expert
            value = torch.nn.functional.linear(original._apply_gate(
                torch.nn.functional.linear(x[selected], original.gate_up_proj[expert])),
                original.down_proj[expert])
            expected[selected] += value * weights[selected, position, None]
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    linearized = LinearExperts2D(original)
    actual = linearized(x, indices, weights)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0], expected_gradient)
    reference = weakref.ref(original)
    # A dtype transfer replaces each 2D storage, just as a CPU/CUDA transfer.
    old_banks = [weakref.ref(p) for p in original.parameters()]
    del actual, expected, expected_gradient, value
    linearized.double()
    del original
    gc.collect()
    assert reference() is None
    assert all(bank() is None for bank in old_banks)
    assert not linearized[0].apply_gate.__self__._parameters
