"""Tests for tools/fetch_lu_et_al_source.py.

Covers blocker 3 from PR #76 review: data/lu_et_al/questions.json,
role_prompts.json, roles.json and causal_evaluation.json embed verbatim text
from a GitHub repository with no LICENSE file, so their redistribution rights
independent of the upstream paper's CC BY 4.0 are unresolved. This script is
the "pinned fetch script + committed hashes" resolution - it clones the
pinned upstream commit and rebuilds those files, and is only meaningful if it
actually reproduces what's committed.

The full clone+rebuild+compare only runs when explicitly requested (network
access, ~50MB clone, ~20-40s) via RUN_LU_ET_AL_FETCH_TEST=1, matching the
project's convention of not making the default test run depend on network
access. The always-on tests below check what can be verified without a
network call: the module imports cleanly, and the pinned commit/remote this
script targets match what THIRD_PARTY_NOTICES.md documents.
"""
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "fetch_lu_et_al_source_test", ROOT / "tools" / "fetch_lu_et_al_source.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def test_pinned_commit_matches_third_party_notices():
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert TOOL.EXPECTED_COMMIT in notice, (
        "the script's pinned commit must match the commit THIRD_PARTY_NOTICES.md "
        "documents as the source of data/lu_et_al/ - a drift here would mean the "
        "notice and the fetch mechanism disagree about which upstream state was used"
    )


def test_pinned_commit_matches_other_lu_et_al_provenance_docs():
    for rel in ("docs/LU_COMPARABILITY_MATRIX.md", "docs/T21_SOURCE_CAPPING_EXECUTION.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert TOOL.EXPECTED_COMMIT in text, f"{rel} does not reference the pinned commit"


def test_expected_remote_is_the_public_upstream_repository():
    assert TOOL.EXPECTED_REMOTE == "https://github.com/safety-research/assistant-axis"


def test_run_utc_is_frozen_not_live_clock():
    # Byte-identical reproduction requires a frozen timestamp - a live clock
    # would make every rebuilt artifact differ from the committed copy on
    # generated_utc alone, even with identical scientific content.
    assert TOOL.RUN_UTC == "2026-07-31T00:00:00Z"


def test_vendor_dir_is_gitignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "vendor/" in gitignore.splitlines(), (
        "vendor/ (the fetch script's clone target) must stay gitignored - "
        "committing a full clone of the upstream repo would defeat the point "
        "of fetching it at run time instead of vendoring a copy"
    )


NETWORK_TEST_ENV_VAR = "RUN_LU_ET_AL_FETCH_TEST"


@pytest.mark.skipif(
    os.environ.get(NETWORK_TEST_ENV_VAR) != "1",
    reason=f"network-dependent (clones ~50MB upstream repo); set {NETWORK_TEST_ENV_VAR}=1 to run",
)
def test_fetch_reproduces_committed_artifacts_byte_for_byte():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "fetch_lu_et_al_source.py")],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert result.returncode == 0, (
        f"fetch_lu_et_al_source.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    for name in ("roles.json", "role_prompts.json", "questions.json", "causal_evaluation.json"):
        assert re.search(rf"\[MATCH\] {re.escape(name)} sha256=", result.stdout), (
            f"{name} was not confirmed byte-identical to the committed copy - "
            f"full output:\n{result.stdout}"
        )
