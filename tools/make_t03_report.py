"""Write the T03 canonicalisation report, stamped with Git SHA and config hash.

Usage: python tools/make_t03_report.py
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.provenance import stamp_report  # noqa: E402


def main():
    pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    summary = pytest.stdout.strip().splitlines()[-1] if pytest.stdout.strip() else ""

    # Stamped from a clean tree; the report is committed afterwards, so
    # source_git_sha is the report commit's parent (the code it describes).
    report = stamp_report({
        "task": "T03",
        "title": "Canonicalise code and archive duplicates",
        "scope": (
            "Canonical core library only. Executable production adapters "
            "(generation runner, activation hook runner, judge runner, "
            "mixed-effects analysis) are deferred; T03 remains open until "
            "each active artifact type has one production command path."
        ),
        "canonical_modules": sorted(
            p.name for p in (REPO_ROOT / "src").glob("*.py") if p.name != "__init__.py"
        ),
        "archived_files": sorted(
            p.name for p in (REPO_ROOT / "code" / "archive").iterdir()
            if p.name != "README.md"
        ),
        "tests_exit_code": pytest.returncode,
        "tests_summary": summary,
    })

    out = REPO_ROOT / "docs" / "T03_CANONICALISATION_REPORT.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return pytest.returncode


if __name__ == "__main__":
    raise SystemExit(main())
