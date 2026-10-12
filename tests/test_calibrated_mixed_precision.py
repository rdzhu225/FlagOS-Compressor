"""Calibration, serialized values and fallback provenance across algorithms."""
from dataclasses import replace
import json

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from flagos_compressor.calibration.plan import build_calibration_plan
from flagos_compressor.calibration.runner import quantize_model_sequential
from flagos_compressor.cli.helpers import build_quantization_policy
from flagos_compressor.cli.main import build_parser
from flagos_compressor.core.policy import AWQPolicy, AutoRoundPolicy, TargetSchemeRule
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.formats.native_quantized import save_native_quantized_model
from flagos_compressor.quantizers import awq
from test_gptq_mixed_precision import MixedBlock, policy
from test_gptq_quantization import _unpack_rows, _unpack_columns


@pytest.mark.parametrize('method,awq_zero_point', [('gptq', True), ('awq', True), ('awq', False), ('autoround', True)])
@pytest.mark.parametrize('shared8', [False, True])
def test_mixed_calibration_export_matches_actual_weights(tmp_path, method, awq_zero_point, shared8):
    torch.manual_seed(43)
    block = MixedBlock(only_first=True)
    model = nn.Module(); model.model = nn.Module(); model.model.layers = nn.ModuleList([block])
    settings = replace(policy(shared8), method=method, format=None,
        awq=AWQPolicy(n_grid=4, zero_point=awq_zero_point), autoround=AutoRoundPolicy(iters=4, batch_size=1))
    layers = [('model.layers.0', block)]
    plan = build_calibration_plan(layers, settings)
    results = quantize_model_sequential(layers, [((torch.randn(1, 16, 8),), {})], settings,
        device=torch.device('cpu'), report_path=tmp_path/'calibration.json')
    for name, result in results.items():
        bits = plan['selected_modules'][name]['bits']
        assert result.num_bits == bits and result.group_size == 8 and result.packing == 'gptq'
        assert result.algorithm == ('rtn' if '.experts.1.' in name else method)
        packed = result.packed
        codes = _unpack_rows(packed.qweight, bits)
        zeros = (_unpack_columns(packed.qzeros, bits) + 1) & ((1 << bits) - 1)
        restored = (packed.scales[packed.g_idx.long()] * (codes - zeros[packed.g_idx.long()])).t()
        torch.testing.assert_close(restored, model.get_submodule(name).weight, rtol=1e-5, atol=1e-6)
        if method == 'awq':
            assert result.symmetric == (bits == 8 or not awq_zero_point)
            if result.symmetric:
                assert (zeros == 1 << (bits-1)).all()
    report = json.loads((tmp_path/'calibration.json').read_text())
    assert report['fallback_modules'] == 3 and report['calibrated_modules'] == 7
    source = tmp_path/'source'; source.mkdir()
    save_file({'dummy': torch.ones(1)}, source/'model.safetensors')
    (source/'config.json').write_text(json.dumps({'model_type': 'toy'}))
    output = tmp_path/'output'
    save_native_quantized_model(source, output, model, results, method=method, bits=4,
        group_size=8, awq_zero_point=awq_zero_point)
    config = json.loads((output/'config.json').read_text())['quantization_config']
    assert config['quant_method'] == 'gptq'
    assert config['algorithm'] == method+'+rtn'
    assert config['fallback_quantization']['module_count'] == 3
    validation = validate_artifact(output)
    assert validation['valid'], validation['errors']
    assert validation['int8_tensors'] == (4 if shared8 else 3)
    # Symmetry overrides are as significant as packed bit-width overrides.
    if method == 'awq':
        for name, scheme in config['flagos_module_quantization'].items():
            assert scheme['sym'] == (scheme['bits'] == 8 or not awq_zero_point)
        for value in config['dynamic'].values():
            if value.get('bits') == 8:
                value['sym'] = False
        cfg = json.loads((output/'config.json').read_text()); cfg['quantization_config'] = config
        (output/'config.json').write_text(json.dumps(cfg))
        assert not validate_artifact(output)['valid']


@pytest.mark.parametrize('method', ['gptq', 'awq', 'autoround'])
def test_cli_and_invalid_fusion_rules_are_consistent(method):
    args = build_parser().parse_args(['quantize', '--input', 'in', '--output', 'out', '--method', method,
        '--select', 'attention=int8', 'activation-bits=16', 'strategy=group', 'group-size=8',
        '--select', 'moe=int4', 'activation-bits=16', 'strategy=group', 'group-size=8'])
    settings = build_quantization_policy(args)
    assert settings.format == 'gptq'
    broken = replace(settings, target_scheme_rules=settings.target_scheme_rules + (
        TargetSchemeRule(name_pattern=r'\.kv_proj\.weight$', weight_format='int4',
            activation_num_bits=16, strategy='group', group_size=8),))
    with pytest.raises(ValueError, match='fused'):
        build_calibration_plan([('model.layers.0', MixedBlock())], broken)


def test_awq_joint_scale_search_uses_each_projection_scheme(monkeypatch):
    class Pair(nn.Module):
        def __init__(self):
            super().__init__(); self.a = nn.Linear(16, 16, bias=False); self.b = nn.Linear(16, 16, bias=False)
        def forward(self, x):
            return self.a(x) + self.b(x)
    pair = Pair(); seen = []; original = awq.pseudo_quantize_awq
    def track(weight, **kwargs):
        seen.append((kwargs['bits'], kwargs['group_size'], kwargs['zero_point']))
        return original(weight, **kwargs)
    monkeypatch.setattr(awq, 'pseudo_quantize_awq', track)
    awq.search_awq_scale(pair, [pair.a, pair.b], torch.randn(4, 16), n_grid=4,
        linear_schemes=[(4, 8, True), (8, 16, False)])
    assert seen == [(4, 8, True), (8, 16, False)]*4
    seen.clear()
    awq.search_awq_clip(pair.b.weight, torch.randn(8, 16), bits=8, group_size=16, zero_point=False, n_grid=4)
    assert seen and all(item == (8, 16, False) for item in seen)


def test_awq_w8_base_config_matches_symmetric_packed_weights(tmp_path):
    block = MixedBlock()
    model = nn.Module(); model.model = nn.Module(); model.model.layers = nn.ModuleList([block])
    settings = replace(policy(), method='awq', format='gptq', num_bits=8,
        target_scheme_rules=(), selections=('attention',), group_size=8, awq=AWQPolicy(n_grid=4))
    results = quantize_model_sequential([('model.layers.0', block)],
        [((torch.randn(1, 16, 8),), {})], settings, device=torch.device('cpu'))
    assert len(results) == 3 and all(result.symmetric for result in results.values())
    source = tmp_path/'source'; source.mkdir()
    save_file({'dummy': torch.ones(1)}, source/'model.safetensors')
    (source/'config.json').write_text(json.dumps({'model_type': 'toy'}))
    output = tmp_path/'output'
    save_native_quantized_model(source, output, model, results, method='awq', bits=8, group_size=8)
    quant = json.loads((output/'config.json').read_text())['quantization_config']
    # The loader constructs its base config before applying dynamic overrides.
    assert quant['bits'] == 8 and quant['sym'] is True
    assert all(scheme['sym'] for scheme in quant['flagos_module_quantization'].values())
    assert validate_artifact(output)['valid']
