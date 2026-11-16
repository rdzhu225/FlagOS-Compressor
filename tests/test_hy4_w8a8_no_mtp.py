"""Verify the HY4 entry point on synthetic shards, including eight CUDA workers."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_parallel_execution import assert_same_artifact


@pytest.mark.parametrize("backend,devices", [("cpu", "0,1"), ("cuda", "0,1,2,3,4,5,6,7")])
def test_hy4_parallel_matches_serial_and_preserves_prediction_branch(tmp_path, backend, devices):
    if backend == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < 8):
        pytest.skip("requires eight CUDA devices")
    source, serial, parallel = (tmp_path / name for name in ("source", "serial", "parallel"))
    source.mkdir()
    generator = torch.Generator().manual_seed(19)

    def weight(*shape):
        return torch.randn(*shape, generator=generator).bfloat16()

    tensors = {
        "model.layers.0.self_attn.q_a_proj.weight": weight(64, 64),
        "model.layers.1.mlp.experts.gate_up_proj": weight(2, 64, 64),
        "model.layers.1.mlp.experts.down_proj": weight(2, 64, 32),
        "model.layers.1.mlp.shared_experts.gate_proj.weight": weight(32, 64),
        "model.layers.1.mlp.shared_experts.up_proj.weight": weight(32, 64),
        "model.layers.1.mlp.shared_experts.down_proj.weight": weight(64, 32),
        "model.mtp_layers.0.self_attn.q_a_proj.weight": weight(64, 64),
        "model.mtp_layers.0.mlp.experts.gate_up_proj": weight(2, 64, 64),
        "model.mtp_layers.0.mlp.experts.down_proj": weight(2, 64, 32),
        "model.dfly_layers.0.self_attn.q_a_proj.weight": weight(64, 64),
        "model.dfly_layers.hidden_correction.up_proj.weight": weight(64, 64),
        "model.dfly_layers.mtp_fusion.eh_proj.weight": weight(64, 128),
        "vit.layers.0.self_attn.q_proj.weight": weight(64, 64),
        "model.layers.0.self_attn.indexer.wk.weight": weight(64, 64),
        "model.layers.1.mlp.gate.weight": weight(2, 64),
        "lm_head.weight": weight(64, 64),
    }
    mapping = {}
    for index, (name, tensor) in enumerate(tensors.items()):
        shard = f"shard-{index:02d}.safetensors"
        save_file({name: tensor}, source / shard)
        mapping[name] = shard
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (source / "config.json").write_text(json.dumps({"model_type": "hy_v4_vl"}))
    script = Path(__file__).resolve().parents[1] / "examples/hy4_w8a8_no_mtp.py"
    command = [sys.executable, str(script), "--input", str(source), "--backend", backend]
    for output, extra in ((serial, []), (parallel, ["--devices", devices])):
        result = subprocess.run(command + ["--output", str(output)] + extra,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    assert_same_artifact(serial, parallel)
    generated = {}
    for shard in parallel.glob("*.safetensors"):
        generated.update(load_file(shard))
    assert generated["model.layers.0.self_attn.q_a_proj.weight"].dtype == torch.int8
    for name, tensor in tensors.items():
        if name.startswith(("model.mtp_layers.", "model.dfly_layers.", "vit.")):
            assert generated[name].dtype == tensor.dtype, name
            assert torch.equal(generated[name], tensor), name
    assert {name for name in generated if ".mtp_layers." in name or ".dfly_layers." in name} == {
        name for name in tensors if ".mtp_layers." in name or ".dfly_layers." in name
    }
    report = json.loads((parallel / "quantization_report.json").read_text())
    assert report["extras"]["parallel"]["workers"] == len(devices.split(","))
    assert all(not event["fallback"] and event["actual_backend"] == backend
               for event in report["op_events"])
