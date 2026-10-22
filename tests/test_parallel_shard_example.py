"""Check shard orchestration against the serial executor, including split banks."""
import json
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file


def test_parallel_matches_serial(tmp_path):
    import flagos_compressor.formats.register  # noqa: F401
    import flagos_compressor.quantizers.register  # noqa: F401
    from flagos_compressor.backends.registry import build_backend
    from flagos_compressor.core.executor import execute_plan
    from flagos_compressor.cli.quantize import _load_moe_layout
    from flagos_compressor.core.planner import build_quantize_plan
    from flagos_compressor.core.policy import QuantizationPolicy
    from flagos_compressor.core.validation import validate_artifact
    from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        import pytest
        pytest.skip("requires two CUDA GPUs")
    script = Path(__file__).resolve().parents[1] / "examples/parallel_w8a8.py"
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "test_fused"}))
    torch.manual_seed(7)
    tensors = {
        "a.safetensors": {"model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 8, 6).bfloat16()},
        "b.safetensors": {"model.layers.0.mlp.experts.down_proj": torch.randn(2, 6, 4).bfloat16()},
        "c.safetensors": {"model.layers.0.self_attn.q_proj.weight": torch.randn(6, 6).bfloat16()},
        "d.safetensors": {"vit.layers.0.self_attn.q_proj.weight": torch.randn(6, 6).bfloat16(),
                          "model.norm.weight": torch.ones(6).bfloat16()},
    }
    for name, state in tensors.items():
        save_file(state, source / name)
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {}, "weight_map": {key: name for name, state in tensors.items() for key in state},
    }))
    parallel, serial = tmp_path / "parallel", tmp_path / "serial"
    subprocess.run([sys.executable, str(script), "--input", str(source), "--output", str(parallel),
                    "--work-dir", str(tmp_path / "work"), "--gpus", "0,1",
                    "--exclude-name", r"^vit\."], check=True, timeout=120)
    profile = scan_hf_safetensors(source)
    policy = QuantizationPolicy(selections=("linear",), exclude_names=(r"^vit\.",),
                                num_bits=8, activation_num_bits=8, strategy="channel")
    plan = build_quantize_plan(profile, policy, _load_moe_layout(str(source), profile, policy))
    execute_plan(source, serial, plan, build_backend("cuda", "cuda:0"))
    assert validate_artifact(parallel)["valid"]
    for name in tensors:
        left, right = load_file(parallel / name), load_file(serial / name)
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key]), key
    for filename in ("config.json", "model.safetensors.index.json", "quantization_manifest.json"):
        assert json.loads((parallel / filename).read_text()) == json.loads((serial / filename).read_text())
    assert not (parallel / "_INCOMPLETE").exists()
