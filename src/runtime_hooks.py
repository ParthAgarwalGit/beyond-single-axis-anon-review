"""Runtime hooks for T04 validation and later production integration.

The hook target is ``model.model.layers[L]``. For Llama-style decoder
blocks, the first tensor in the module output is the post-block residual
stream. The controller can operate as a read-only hook, a zero-vector
steering hook, or a non-zero additive steering hook.

Frozen timing rule:
- prompt-prefill positions are never changed;
- generated-token decode positions are changed;
- therefore the first sampled response token is unsteered, and steering can
  affect the second and later sampled tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


def first_tensor(output: Any) -> torch.Tensor:
    """Return the first hidden-state tensor from a decoder-layer output."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        "Expected decoder-layer output to be a Tensor or tuple/list whose "
        "first element is a Tensor."
    )


def replace_first_tensor(output: Any, hidden: torch.Tensor) -> Any:
    """Return ``output`` with its first tensor replaced by ``hidden``."""
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


@dataclass
class HookDiagnostics:
    """Small, JSON-serialisable diagnostics collected by a hook."""

    prefill_calls: int = 0
    decode_calls: int = 0
    prefill_max_abs_delta: float = 0.0
    decode_max_abs_delta: float = 0.0
    observed_shapes: list[list[int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prefill_calls": self.prefill_calls,
            "decode_calls": self.decode_calls,
            "prefill_max_abs_delta": self.prefill_max_abs_delta,
            "decode_max_abs_delta": self.decode_max_abs_delta,
            "observed_shapes": self.observed_shapes,
        }


class GeneratedTokenHook:
    """Read or steer one decoder block.

    Parameters
    ----------
    mode:
        ``"read_only"``, ``"zero"``, or ``"steer"``.
    unit_direction:
        One-dimensional unit vector with hidden-size entries. Required only
        for ``mode="steer"``.
    injection_scale:
        Scalar norm of the additive vector.
    modify_prefill:
        Must stay ``False`` for the frozen project protocol.
    capture_tensors:
        When true, detached CPU copies of observed hidden states are stored.
        Use only for short validation inputs.
    """

    VALID_MODES = {"read_only", "zero", "steer"}

    def __init__(
        self,
        mode: str,
        unit_direction: Optional[torch.Tensor] = None,
        injection_scale: float = 0.0,
        *,
        modify_prefill: bool = False,
        capture_tensors: bool = False,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown hook mode {mode!r}")
        if mode == "steer" and unit_direction is None:
            raise ValueError("unit_direction is required for non-zero steering")
        if mode != "steer" and abs(float(injection_scale)) > 0.0:
            raise ValueError("Non-zero injection_scale is valid only in steer mode")
        if modify_prefill:
            raise ValueError("T04 frozen timing forbids prompt-prefill modification")

        self.mode = mode
        self.unit_direction = unit_direction
        self.injection_scale = float(injection_scale)
        self.modify_prefill = modify_prefill
        self.capture_tensors = capture_tensors
        self.diagnostics = HookDiagnostics()
        self.captured: list[torch.Tensor] = []
        self._first_call = True

    def __call__(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any):
        hidden = first_tensor(output)
        if hidden.ndim != 3:
            raise ValueError(
                f"Expected hidden shape [batch, sequence, hidden], got {tuple(hidden.shape)}"
            )

        seq_len = int(hidden.shape[1])

        # The first model call is prompt prefill even when the prompt contains
        # exactly one token. Later one-token calls are autoregressive decode.
        if self._first_call:
            phase = "prefill"
            self._first_call = False
        else:
            phase = "decode"

        self.diagnostics.observed_shapes.append(list(hidden.shape))

        if phase == "prefill":
            self.diagnostics.prefill_calls += 1
        else:
            self.diagnostics.decode_calls += 1

        if self.capture_tensors:
            self.captured.append(hidden.detach().float().cpu())

        if self.mode == "read_only":
            return None

        # Frozen timing: prompt prefill is never modified.
        if phase == "prefill":
            return None

        if self.mode == "zero":
            delta = torch.zeros_like(hidden)
        else:
            direction = self.unit_direction.to(device=hidden.device, dtype=hidden.dtype)
            if direction.ndim != 1 or direction.numel() != hidden.shape[-1]:
                raise ValueError(
                    "unit_direction must be one-dimensional and match hidden size"
                )
            direction = direction / torch.linalg.vector_norm(direction.float()).to(
                direction.dtype
            )
            delta = direction.view(1, 1, -1) * torch.as_tensor(
                self.injection_scale,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            delta = delta.expand_as(hidden)

        changed = hidden + delta
        observed_delta = float(
            (changed.float() - hidden.float()).abs().max().detach().cpu()
        )
        if phase == "prefill":
            self.diagnostics.prefill_max_abs_delta = max(
                self.diagnostics.prefill_max_abs_delta, observed_delta
            )
        else:
            self.diagnostics.decode_max_abs_delta = max(
                self.diagnostics.decode_max_abs_delta, observed_delta
            )
        return replace_first_tensor(output, changed)


class CaptureLayerOutput:
    """Minimal read-only hook used for layer-index mapping checks."""

    def __init__(self) -> None:
        self.hidden: Optional[torch.Tensor] = None

    def __call__(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any):
        self.hidden = first_tensor(output).detach()
        return None
