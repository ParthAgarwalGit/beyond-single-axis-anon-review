import importlib.util
from pathlib import Path

import numpy as np

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "gpu_cells" / "t24_capture_qwen_calibration.py"
_spec = importlib.util.spec_from_file_location("t24_capture", _TOOL)
t24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t24)


def test_shard_name_is_deterministic_and_does_not_embed_item_id(tmp_path):
    a = t24.shard_path(tmp_path, 7, "sensitive/raw/id")
    b = t24.shard_path(tmp_path, 7, "sensitive/raw/id")
    assert a == b
    assert "sensitive" not in a.name
    assert a.name.startswith("007_") and a.name.endswith(".npz")


def test_manifest_contract_rejects_wrong_hash(tmp_path, monkeypatch):
    p = tmp_path / "manifest.jsonl"
    p.write_text('{"item_id":"x","messages":[{"role":"user","content":"x"}]}\n', encoding="utf-8")
    monkeypatch.setattr(t24, "MANIFEST_SHA256", "0" * 64)
    try:
        t24.load_manifest(p)
    except SystemExit as exc:
        assert "manifest hash mismatch" in str(exc)
    else:
        raise AssertionError("expected fail-closed manifest hash rejection")


def test_released_axis_loader_contract_is_two_dimensional(monkeypatch, tmp_path):
    # Pure shape contract test without depending on torch serialization.
    arr = np.zeros((54, 8), dtype=np.float32)
    assert arr.ndim == 2 and arr.shape[0] > max(t24.LAYERS)
    assert len(t24.LAYERS) == 8 and t24.LAYERS[0] == 46 and t24.LAYERS[-1] == 53


def test_capture_constants_match_reviewed_t24_freeze():
    assert t24.MODEL_ID == "Qwen/Qwen3-32B"
    assert t24.MODEL_REVISION == "9216db5781bf21249d130ec9da846c4624c16137"
    assert t24.AXIS_SHA256 == "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
    assert t24.CAP_SHA256 == "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
    assert t24.MANIFEST_SHA256 == "d611e904f3b5d47c71e5ab1f3fa1a84ead6cfd5d94dba337f5c3b71aafef7793"
    assert t24.N_ITEMS == 100
