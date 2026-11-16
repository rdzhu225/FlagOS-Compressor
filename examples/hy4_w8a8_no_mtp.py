#!/usr/bin/env python3
"""HY4 INT8 W8A8: weight-only MSE, dynamic per-token activations, no MTP."""

import argparse
import re
from pathlib import Path

import flagos_compressor.formats.register
import flagos_compressor.quantizers.register
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.cli.helpers import ensure_no_unmatched, print_plan
from flagos_compressor.cli.quantize import _load_moe_layout
from flagos_compressor.core.executor import execute_plan
from flagos_compressor.core.parallel import execute_plan_parallel, normalize_devices
from flagos_compressor.core.planner import build_quantize_plan
from flagos_compressor.core.policy import QuantizationPolicy
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


DEFAULT_INPUT = "/public-nvme/models/Hy4-testing-3"
DEFAULT_OUTPUT = "/public-nvme/models/Hy4-testing-3-W8A8-no-MTP"
# Cover model.mtp_layers.*, mtp_fusion, nextn variants, and the complete
# model.dfly_layers.* auxiliary branch in Hy4-testing-3.
MTP_PATTERN = r"(^|\.)(mtp(?:_[^.]+)?|nextn(?:_[^.]+)?|dfly_layers)(\.|$)"


def make_policy():
    return QuantizationPolicy(
        selections=("linear",),
        exclude_names=(MTP_PATTERN, r"^vit\.", r"(^|\.)indexer(\.|$)"),
        method="mse",                  # Uses weights only; no calibration data.
        format="compressed-tensors",
        num_bits=8,
        activation_num_bits=8,          # Dynamic per-token INT8 at inference.
        strategy="channel",            # Per-output-channel INT8 weights.
        group_size=None,
        scale_dtype="fp32",
        n_candidates=200,
        chunk_size=1024,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", default="cuda")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--device", default=None)
    execution.add_argument("--devices", nargs="+", help="Parallel devices, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("Input and output must be separate, non-nested directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output must be absent or empty")

    policy = make_policy()
    profile = scan_hf_safetensors(source)
    layout = _load_moe_layout(str(source), profile, policy)
    plan = build_quantize_plan(profile, policy, layout)
    ensure_no_unmatched(plan)

    quantized = [
        action for action in plan.actions
        if action.output_format.name.startswith("compressed_tensors_")
    ]
    if not quantized:
        raise RuntimeError("No supported Linear / MoE weights selected")
    for action in quantized:
        if re.search(MTP_PATTERN, action.tensor.effective_name):
            raise RuntimeError(f"MTP unexpectedly selected: {action.tensor.name}")

    devices = normalize_devices(args.backend, args.devices) if args.devices else None
    print_plan(plan)
    if devices:
        print("Parallel devices:", ", ".join(devices), flush=True)
    if args.dry_run:
        return

    if devices:
        # The parallel executor validates once before clearing _INCOMPLETE.
        execute_plan_parallel(source, output, plan, backend=args.backend, devices=devices)
        print(f"Parallel quantization complete: {output}")
        return

    backend = build_backend(args.backend, args.device)
    if not backend.is_available():
        raise RuntimeError(f"Backend is unavailable: {args.backend}")
    execute_plan(source, output, plan, backend)
    result = validate_artifact(output)
    if not result["valid"]:
        raise RuntimeError(f"Output validation failed: {result['errors']}")
    print(f"Quantization complete: {output}")


if __name__ == "__main__":
    main()
