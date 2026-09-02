"""Confound-audit metadata binding: altered/wrong metadata must be rejected."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "confound_audit", REPO_ROOT / "tools" / "run_deepseek_confound_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _blob(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\x00" % len(data))
    h.update(data)
    return h.hexdigest()


def make_tree(tmp_path, files):
    for rel, data in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return {rel: (len(data), _blob(data)) for rel, data in files.items()}


GOOD = {
    "t12-c80/C80-A/meta_part0000.jsonl": b'{"role": "pirate"}\n',
    "t12-c80/C80-B/meta_part0000.jsonl": b'{"role": "sage"}\n',
}


def test_matching_tree_verifies_and_hashes(tmp_path):
    expected = make_tree(tmp_path, GOOD)
    hashes = audit.verify_meta_tree(tmp_path, expected)
    assert set(hashes) == set(GOOD)
    for rel, data in GOOD.items():
        assert hashes[rel] == hashlib.sha256(data).hexdigest()


def test_modified_content_rejected(tmp_path):
    expected = make_tree(tmp_path, GOOD)
    target = tmp_path / "t12-c80/C80-A/meta_part0000.jsonl"
    # same length, different bytes — size check alone would miss this
    target.write_bytes(b'{"role": "PIRATE"}\n')
    with pytest.raises(SystemExit, match="does not match the pinned revision"):
        audit.verify_meta_tree(tmp_path, expected)


def test_missing_file_rejected(tmp_path):
    expected = make_tree(tmp_path, GOOD)
    (tmp_path / "t12-c80/C80-B/meta_part0000.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing 1"):
        audit.verify_meta_tree(tmp_path, expected)


def test_extra_file_rejected(tmp_path):
    expected = make_tree(tmp_path, GOOD)
    (tmp_path / "t12-c80/C80-A/meta_part9999.jsonl").write_bytes(b"{}\n")
    with pytest.raises(SystemExit, match="extra 1"):
        audit.verify_meta_tree(tmp_path, expected)


def test_resized_file_rejected(tmp_path):
    expected = make_tree(tmp_path, GOOD)
    (tmp_path / "t12-c80/C80-A/meta_part0000.jsonl").write_bytes(
        b'{"role": "pirate", "x": 1}\n')
    with pytest.raises(SystemExit, match="does not match the pinned revision"):
        audit.verify_meta_tree(tmp_path, expected)
