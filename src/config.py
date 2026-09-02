"""Versioned frozen-method configuration loaders.

Two config files have different provenance roles:

- ``configs/method_frozen.yaml`` is the historical V3 file whose SHA-256 is
  already stamped into executed T09/T10/T12 artifacts. Its bytes are immutable.
- ``configs/method_frozen_v4.yaml`` is the declared downstream configuration
  for new decisions and new artifacts.

``load_frozen_config()`` therefore means "load the active downstream freeze".
Legacy artifact validation can explicitly call
``load_historical_frozen_config()`` without mutating or re-stamping old rows.
Runtime overrides of frozen fields remain forbidden.
"""

import hashlib
from pathlib import Path
from types import MappingProxyType

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_CONFIG_PATH = REPO_ROOT / "configs" / "method_frozen.yaml"
DOWNSTREAM_CONFIG_PATH = REPO_ROOT / "configs" / "method_frozen_v4.yaml"

# Backwards-compatible name for callers that mean the active configuration.
FROZEN_CONFIG_PATH = DOWNSTREAM_CONFIG_PATH

_cache = {}


def _freeze(obj):
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def _load_config(path):
    path = Path(path)
    if path not in _cache:
        raw = path.read_bytes()
        _cache[path] = (_freeze(yaml.safe_load(raw)), hashlib.sha256(raw).hexdigest())
    return _cache[path]


def load_frozen_config():
    """Return ``(immutable_config, sha256_hex)`` for downstream V4."""
    return _load_config(DOWNSTREAM_CONFIG_PATH)


def load_historical_frozen_config():
    """Return the immutable historical V3 config used by stamped legacy rows."""
    return _load_config(HISTORICAL_CONFIG_PATH)


def known_frozen_config_shas():
    """Return the config SHA-256 values accepted for provenance validation.

    This does not make the historical config active for new work; it only lets
    validators recognize artifacts that were legitimately stamped before V4.
    """
    _, historical_sha = load_historical_frozen_config()
    _, downstream_sha = load_frozen_config()
    return frozenset((historical_sha, downstream_sha))


def block_question_ids(cfg):
    """Frozen question-ID tuples per block, e.g. {'E80': (...), ...}."""
    return {
        name: block["question_ids"]
        for name, block in cfg["question_design"]["blocks"].items()
    }


def prompt_index_for(cfg, block_ids, position):
    """Frozen prompt index for the question at ``position`` in its block.

    Rule (frozen): zero-based position within the block, mod the number of
    prompt conditions. The mapping is identical for every role, model, and
    arm.
    """
    n_prompts = len(cfg["prompt_rendering"]["default_conditions"])
    if not 0 <= position < len(block_ids):
        raise ValueError(f"position {position} outside block of {len(block_ids)}")
    return position % n_prompts
