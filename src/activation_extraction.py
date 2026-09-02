"""Canonical activation site, pooling, and layer convention (METHOD_FREEZE §9).

Hook ``model.model.layers[L]`` and take the first tensor of the block output
(before the model's final norm); ``layers[L] output == output_hidden_states[L+1]``.
The primary middle block is 16, reported as "block 16 output / hidden_states[17]".
Production execution stays gated on T04's independent hook validation.
"""

import numpy as np

from .config import load_frozen_config


def hook_module_path(cfg, layer):
    hook = cfg["activation_extraction"]["hook"]
    if not 0 <= layer < hook["n_blocks"]:
        raise ValueError(f"block index {layer} outside 0..{hook['n_blocks'] - 1}")
    return hook["module_path_template"].replace("[L]", f"[{layer}]")


def hidden_states_index(layer):
    """Index into output_hidden_states equivalent to block ``layer`` output."""
    return layer + 1


def primary_block(cfg=None):
    if cfg is None:
        cfg, _ = load_frozen_config()
    return cfg["activation_extraction"]["middle_layer"]["primary_block_index"]


def pool_response_vector(residuals, span, prompt_len):
    """Arithmetic mean of residual vectors over one frozen token pool.

    ``residuals`` has shape (sequence_length, hidden) over the concatenated
    prompt+output positions; ``span`` is an absolute half-open (start, end)
    span from :class:`src.generation.Segmentation`. Prompt positions are
    excluded from every pool, so ``span`` must start at or after
    ``prompt_len``.
    """
    if span is None:
        raise ValueError("pool unavailable for this segmentation case")
    start, end = span
    if start < prompt_len:
        raise ValueError("pool span includes prompt positions")
    if end <= start:
        raise ValueError("empty pool span")
    if end > len(residuals):
        raise ValueError("pool span extends past stored residuals")
    return np.asarray(residuals[start:end], dtype=np.float64).mean(axis=0)
