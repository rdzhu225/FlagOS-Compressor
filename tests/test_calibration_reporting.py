import json

import pytest
import torch
from torch import nn

from flagos_compressor.calibration.moe import LinearExperts2D
from flagos_compressor.calibration.runner import quantize_model_sequential
from flagos_compressor.core.policy import AWQPolicy, AutoRoundPolicy, GPTQPolicy, QuantizationPolicy


class RoutedBlock(nn.Module):
    def __init__(self, only_first=False):
        super().__init__()
        source = nn.Module()
        source.hidden_dim = 8
        source.gate_up_proj = nn.Parameter(torch.randn(2, 16, 8) * 0.1)
        source.down_proj = nn.Parameter(torch.randn(2, 8, 8) * 0.1)
        self.mlp = nn.Module()
        self.mlp.experts = LinearExperts2D(source)
        self.mlp.shared_experts = nn.Sequential()
        self.mlp.shared_experts.add_module('gate_proj', nn.Linear(8, 8, bias=False))
        self.only_first = only_first

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, 8)
        indices = torch.zeros((flat.shape[0], 1), dtype=torch.long, device=flat.device)
        if not self.only_first:
            indices[2:] = 1
        weights = torch.ones_like(indices, dtype=flat.dtype)
        result = self.mlp.experts(flat, indices, weights) + self.mlp.shared_experts(flat)
        return result.reshape_as(hidden_states)


def run_case(tmp_path, method, only_first=False):
    torch.manual_seed(42)
    layer = RoutedBlock(only_first)
    policy = QuantizationPolicy(selections=('linear',), method=method, num_bits=4, group_size=8,
        gptq=GPTQPolicy(desc_act=False), awq=AWQPolicy(n_grid=2),
        autoround=AutoRoundPolicy(iters=3, batch_size=1))
    path = tmp_path/'coverage.json'
    result = quantize_model_sequential([('model.layers.0', layer)],
        [((torch.randn(1, 6, 8),), {})], policy, device=torch.device('cpu'), report_path=path)
    return result, json.loads(path.read_text())


@pytest.mark.parametrize('method', ['gptq', 'awq', 'autoround'])
def test_report_counts_routed_tokens_without_optimizer_repetitions(tmp_path, method):
    result, report = run_case(tmp_path, method)
    assert report['state'] == 'completed'
    assert report['quantized_modules'] == len(result) == 7
    rows = report['layers'][0]['input_rows']
    for projection in ('gate_proj', 'up_proj', 'down_proj'):
        assert rows[f'mlp.experts.0.{projection}'] == 2
        assert rows[f'mlp.experts.1.{projection}'] == 4
    assert rows['mlp.shared_experts.gate_proj'] == 6
    assert report['layers'][0]['unobserved_modules'] == []


@pytest.mark.parametrize('method', ['gptq', 'awq', 'autoround'])
def test_failed_calibration_preserves_missing_expert_report(tmp_path, method):
    with pytest.raises(RuntimeError, match='did not route any tokens'):
        run_case(tmp_path, method, only_first=True)
    report = json.loads((tmp_path/'coverage.json').read_text())
    assert report['state'] == 'failed'
    layer = report['layers'][0]
    assert layer['state'] == 'failed'
    assert 'mlp.experts.1.gate_proj' in layer['unobserved_modules']
    assert layer['input_rows']['mlp.experts.0.gate_proj'] == 6
    assert layer['elapsed_seconds'] >= 0
