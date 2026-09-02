"""Report provenance stamping.

Every new report carries ``source_git_sha`` — the HEAD commit of the code
that GENERATED the report — plus the frozen-config SHA-256 (T03 step 7 and
the YAML ``implementation_contract``). When a generated report is committed
afterwards, ``source_git_sha`` is by construction the report commit's
parent: a report cannot contain the SHA of the commit that contains it.

Dirty production trees fail by default. Passing ``allow_dirty=True`` (for
development runs only) stamps ``working_tree_diff_sha256`` so the exact
uncommitted state is still identified.
"""

import hashlib
import subprocess

from .config import REPO_ROOT, load_frozen_config


def _git(*args):
    out = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout


def source_git_sha():
    """HEAD SHA of the repository the code is running from."""
    return _git("rev-parse", "HEAD").strip()


def git_dirty():
    """True when the working tree differs from HEAD (including untracked files)."""
    return bool(_git("status", "--porcelain").strip())


def working_tree_diff_sha256():
    """SHA-256 over the tracked diff plus untracked file list."""
    payload = _git("diff", "HEAD") + _git("ls-files", "--others", "--exclude-standard")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stamp_report(report, allow_dirty=False, dirty=None):
    """Return ``report`` with mandatory provenance fields added.

    Raises when the working tree is dirty unless ``allow_dirty=True``, in
    which case the diff hash is included. Refuses to overwrite provenance
    already present, so a report cannot be silently re-attributed to
    different code or configuration.

    ``dirty`` lets a caller pass a dirty-state snapshot taken *before* it
    wrote its own output artifacts, instead of re-checking live. Without
    this, a tool that writes untracked outputs into the repo tree before
    stamping (e.g. a worksheet + private key) would see its own just-written
    files and report a false positive — or, without ``allow_dirty``, fail on
    every run regardless of whether the code that produced the outputs was
    itself clean.
    """
    if dirty is None:
        dirty = git_dirty()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to stamp a report from a dirty working tree; commit first "
            "or pass allow_dirty=True (development only)"
        )
    _, config_sha = load_frozen_config()
    stamp = {
        "source_git_sha": source_git_sha(),
        "source_git_dirty": dirty,
        "config_sha256": config_sha,
    }
    if dirty:
        stamp["working_tree_diff_sha256"] = working_tree_diff_sha256()
    for key in stamp:
        if key in report and report[key] != stamp[key]:
            raise ValueError(f"report already stamped with different {key}")
    return {**report, **stamp}
