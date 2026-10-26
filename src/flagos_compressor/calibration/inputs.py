"""Prepare model-specific inputs for sequential, cache-free calibration."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn

from flagos_compressor.calibration.modeling import (
    capture_first_layer_inputs,
    move_to_device,
)

LayerSamples = list[tuple[tuple[Any, ...], dict[str, Any]]]
LayerKwargs = dict[str, list[dict[str, Any]]]


@torch.no_grad()
def prepare_calibration_inputs(
    model: nn.Module,
    layers: list[tuple[str, nn.Module]],
    batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[LayerSamples, LayerKwargs | None]:
    """Return initial hidden states and optional kwargs for each decoder layer.

    Most models reuse the first block's kwargs. MiMo instead selects distinct
    masks and rotary embeddings for global and sliding-window attention.
    """
    if getattr(getattr(model, "config", None), "model_type", None) == "mimo_v2":
        return _mimo_inputs(model, layers, batches)
    return capture_first_layer_inputs(
        model, layers[0][1], batches, device=device
    ), None


def _mimo_inputs(
    model: nn.Module,
    layers: list[tuple[str, nn.Module]],
    batches: list[dict[str, torch.Tensor]],
) -> tuple[LayerSamples, LayerKwargs]:
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    backbone = model.model
    attention_types = {layer.attention_type for _, layer in layers}
    factories = {
        "full_attention": (create_causal_mask, backbone.rotary_emb),
        "sliding_window_attention": (
            create_sliding_window_causal_mask, backbone.swa_rotary_emb
        ),
    }
    if attention_types - factories.keys():
        raise ValueError(f"Unsupported MiMo attention types: {attention_types}")
    if (
        "sliding_window_attention" in attention_types
        and getattr(backbone.config, "sliding_window", None) is None
    ):
        raise ValueError("MiMo sliding attention requires config.sliding_window")

    samples: LayerSamples = []
    layer_kwargs: LayerKwargs = {name: [] for name, _ in layers}
    for batch in batches:
        batch = move_to_device(batch, torch.device("cpu"))
        # Like the generic catcher, run only embeddings and input preparation
        # on CPU. No decoder forward or full-model GPU allocation is needed.
        hidden = backbone.embed_tokens(batch["input_ids"])
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = batch.get("position_ids", cache_position.unsqueeze(0))
        by_type: dict[str, dict[str, Any]] = {}
        for attention_type in attention_types:
            mask_factory, rotary = factories[attention_type]
            mask_kwargs = {
                "config": backbone.config,
                "attention_mask": batch.get("attention_mask"),
                "past_key_values": None,
                "position_ids": position_ids,
            }
            # Transformers 5.x renamed input_embeds and removed cache_position.
            # Use its installed API instead of MiMo remote code's older call.
            parameters = inspect.signature(mask_factory).parameters
            embed_key = "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
            mask_kwargs[embed_key] = hidden
            if "cache_position" in parameters:
                mask_kwargs["cache_position"] = cache_position
            by_type[attention_type] = {
                "attention_mask": mask_factory(**mask_kwargs),
                "position_embeddings": rotary(hidden, position_ids),
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": None,
                "use_cache": False,
            }
        for name, layer in layers:
            layer_kwargs[name].append(by_type[layer.attention_type])
        samples.append(((hidden.detach(),), layer_kwargs[layers[0][0]][-1]))
    return samples, layer_kwargs


def samples_with_layer_kwargs(
    samples: LayerSamples, kwargs: list[dict[str, Any]]
) -> LayerSamples:
    """Keep propagated hidden states, replacing only layer-specific inputs."""
    if len(samples) != len(kwargs):
        raise ValueError("Layer kwargs must match the number of calibration samples")
    return [(args, dict(options)) for (args, _), options in zip(samples, kwargs)]
