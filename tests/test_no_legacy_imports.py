"""Static check: production code never imports archived or legacy paths."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DIRS = ("src", "tools")

# Import statements only: docstrings may legitimately mention the archive.
FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+[\w.]*(?:archive|replications|worker)[\w.]*",
    re.MULTILINE,
)


def test_production_code_has_no_archive_or_legacy_imports():
    offenders = []
    for directory in PRODUCTION_DIRS:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            if path.name == "validate_method_frozen.py":
                continue  # pre-existing T02 validator, owned by that task
            source = path.read_text()
            for match in FORBIDDEN.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}: {match.group(0).strip()}")
    assert not offenders, "production code references legacy paths:\n" + "\n".join(offenders)


def test_no_worker_modules_exist_in_production_dirs():
    workers = [
        str(p.relative_to(REPO_ROOT))
        for d in PRODUCTION_DIRS
        for p in (REPO_ROOT / d).rglob("worker*.py")
    ]
    assert workers == []
