"""Transformers-v5-style MoE expert linearization for calibration.

Fused 3D expert parameters are exposed as ordinary 2D ``nn.Linear`` modules so
GPTQ/AWQ hooks and native per-expert checkpoint loaders can use them. The outer
router and decoder implementations remain the original Transformers modeling.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
import types

import torch
from torch import nn


def _default_apply_gate(value: torch.Tensor) -> torch.Tensor:
    gate, up = value.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def _gate_without_expert_weights(original: nn.Module):
    """Retain a gate's configuration without retaining its fused weight banks.

    A bound _apply_gate otherwise owns the entire old experts module. Moving
    the new 2D linears to CUDA then leaves a second, unused BF16 copy on CPU.
    Rebind the original method to a shallow context with only those redundant
    parameters removed; activation modules and scalar settings stay identical.
    """
    gate = getattr(original, "_apply_gate", None)
    if isinstance(gate, types.MethodType) and gate.__self__ is original:
        context = copy.copy(original)
        context._parameters = {
            name: parameter for name, parameter in original._parameters.items()
            if name not in {"gate_up_proj", "up_proj", "down_proj",
                            "gate_up_proj_bias", "up_proj_bias", "down_proj_bias"}
        }
        gate = types.MethodType(gate.__func__, context)
    return gate


def _linear(weight: torch.Tensor, bias: torch.Tensor | None = None) -> nn.Linear:
    out_features, in_features = weight.shape
    module = nn.Linear(
        in_features,
        out_features,
        bias=bias is not None,
        device="meta",
        dtype=weight.dtype,
    )
    module.weight = nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        module.bias = nn.Parameter(bias, requires_grad=False)
    return module


class _GatedExpert(nn.Module):
    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        apply_gate: Callable[[torch.Tensor], torch.Tensor],
        gate_bias: torch.Tensor | None = None,
        up_bias: torch.Tensor | None = None,
        down_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = _linear(gate_weight, gate_bias)
        self.up_proj = _linear(up_weight, up_bias)
        self.down_proj = _linear(down_weight, down_bias)
        self.apply_gate = apply_gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        combined = torch.cat(
            [self.gate_proj(hidden_states), self.up_proj(hidden_states)], dim=-1
        )
        return self.down_proj(self.apply_gate(combined))


class _UngatedExpert(nn.Module):
    def __init__(
        self,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        activation: Callable[[torch.Tensor], torch.Tensor],
        up_bias: torch.Tensor | None = None,
        down_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.up_proj = _linear(up_weight, up_bias)
        self.down_proj = _linear(down_weight, down_bias)
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.up_proj(hidden_states)))


class LinearExperts2D(nn.ModuleList):
    def __init__(self, original: nn.Module):
        self.num_experts = int(original.down_proj.shape[0])
        self.has_gate = isinstance(getattr(original, "gate_up_proj", None), nn.Parameter)
        self.has_bias = isinstance(getattr(original, "down_proj_bias", None), nn.Parameter)
        down = original.down_proj
        source_up = original.gate_up_proj if self.has_gate else original.up_proj

        # Standard Transformers experts are either [E, out, in] or [E, in, out].
        hidden_dim = int(getattr(original, "hidden_dim", 0))
        if not hidden_dim:
            hidden_dim = int(down.shape[1])
        non_transposed = down.shape[1] == hidden_dim
        intermediate = int(down.shape[2] if non_transposed else down.shape[1])
        apply_gate = _gate_without_expert_weights(original)
        activation = getattr(original, "act_fn", nn.Identity())

        experts: list[nn.Module] = []
        for index in range(self.num_experts):
            down_weight = down[index] if non_transposed else down[index].t()
            down_bias = (
                original.down_proj_bias[index] if self.has_bias else None
            )
            if self.has_gate:
                fused = source_up[index] if non_transposed else source_up[index].t()
                gate_weight = fused[:intermediate]
                up_weight = fused[intermediate:]
                gate_bias = up_bias = None
                if self.has_bias:
                    fused_bias = original.gate_up_proj_bias[index]
                    gate_bias = fused_bias[:intermediate]
                    up_bias = fused_bias[intermediate:]
                if apply_gate is None:
                    apply_gate = _default_apply_gate
                experts.append(
                    _GatedExpert(
                        gate_weight,
                        up_weight,
                        down_weight,
                        apply_gate,
                        gate_bias,
                        up_bias,
                        down_bias,
                    )
                )
            else:
                up_weight = source_up[index] if non_transposed else source_up[index].t()
                up_bias = original.up_proj_bias[index] if self.has_bias else None
                experts.append(
                    _UngatedExpert(
                        up_weight,
                        down_weight,
                        activation,
                        up_bias,
                        down_bias,
                    )
                )
        super().__init__(experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        ).permute(2, 1, 0)
        for expert_index, expert in enumerate(self):
            top_k_position, token_indices = torch.where(expert_mask[expert_index])
            if token_indices.numel() == 0:
                continue
            # Calibration hooks must observe the tokens selected by the router,
            # not every token in the batch. Otherwise every expert accumulates
            # the same Hessian/AWQ statistics and loses its routed distribution.
            expert_output = expert(hidden_states[token_indices])
            weighted = expert_output * top_k_weights[
                token_indices, top_k_position, None
            ]
            final.index_add_(0, token_indices, weighted.to(final.dtype))
        return final


def _is_fused_experts(module: nn.Module) -> bool:
    return isinstance(getattr(module, "down_proj", None), nn.Parameter) and (
        isinstance(getattr(module, "gate_up_proj", None), nn.Parameter)
        or isinstance(getattr(module, "up_proj", None), nn.Parameter)
    )


def linearize_fused_experts(model: nn.Module) -> list[str]:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name and _is_fused_experts(module)
    ]
    for name, module in targets:
        model.set_submodule(
            name,
            LinearExperts2D(module),
        )
    return [name for name, _ in targets]


__all__ = ["LinearExperts2D", "linearize_fused_experts"]
