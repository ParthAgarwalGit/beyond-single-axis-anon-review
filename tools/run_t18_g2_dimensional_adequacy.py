"""G2-compatible T18 production entry point.

This wrapper preserves the frozen T18 estimators, grids, folds, bootstrap,
multiplicity adjustment, adequacy margin, and verdict implementation from
``tools/run_t18_dimensional_adequacy.py``. It changes only the production row
loader so T18 diagnoses the same label-independent, prompt-conditioned C80
geometry frozen by PR #44 and G2.

The original PR #30 loader predates that confirmatory membership decision and
filters role rows through an automatic score-3 judge. T07 failed the frozen
judge-validation gate, so applying that filter here would diagnose a different
population from T15/T16 and T17. See
``docs/deviations/2026-08-25-t18-g2-membership-compatibility.md``.

Usage is otherwise unchanged::

    python tools/run_t18_g2_dimensional_adequacy.py --self-test
    python tools/run_t18_g2_dimensional_adequacy.py \
        --run-dir runs/confirmatory \
        --out results/t18/t18_report.json

The real run still requires ``docs/G2_GEOMETRY_FREEZE.json`` with
``freeze_status == FROZEN`` and should be executed only after this compatibility
amendment is independently reviewed.
"""

import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.run_t18_dimensional_adequacy as base  # noqa: E402
from src.config import known_frozen_config_shas  # noqa: E402
from src.schemas import read_jsonl  # noqa: E402
from src.t18_cached import analyse_cached  # noqa: E402
from src.t18_parallel import analyse_parallel_cached  # noqa: E402
from src.t18_readouts import (  # noqa: E402
    DEFAULT_ROLE_PREFIX,
    REQUIRED_DEFAULT_CONDITIONS,
    ReadoutRows,
)


REQUIRED_G2_FILES = (
    "generation.jsonl",
    "activations.jsonl",
    "vectors.npz",
    "eligible_roles.json",
)
MEMBERSHIP = "label_independent_technical_validity"
MEMBERSHIP_AUTHORITY = "docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md"
COMPATIBILITY_AUTHORITY = (
    "docs/deviations/2026-08-25-t18-g2-membership-compatibility.md"
)


def _load_g2_record(path=base.DEFAULT_FREEZE_FILE):
    """Load and validate the G2 authority used by the compatibility loader."""
    import json

    path = Path(path)
    if not path.exists():
        base.die(f"G2 freeze record is missing: {path}")
    raw = path.read_bytes()
    record = json.loads(raw.decode("utf-8"))
    if record.get("freeze_status") != "FROZEN":
        base.die(f"{path} is not FROZEN")
    geom = record.get("geometry", {})
    if geom.get("membership") != MEMBERSHIP:
        base.die(
            "G2 membership disagrees with the T18 compatibility contract: "
            f"{geom.get('membership')!r} != {MEMBERSHIP!r}"
        )
    compat = record.get("t18_compatibility", {})
    if compat.get("judge_filter_for_primary_rows") != "FORBIDDEN":
        base.die("G2 does not explicitly forbid primary judge filtering for T18")
    return record, hashlib.sha256(raw).hexdigest()


def _check_known_provenance(row_groups):
    """Reject unknown config successions while preserving historical strata.

    C80 generation and downstream reductions span the immutable V3 -> approved
    V4 method succession. Treating only the current V4 SHA as valid would
    incorrectly reject legitimate historical rows. We therefore accept only
    SHAs returned by ``known_frozen_config_shas()`` and record every observed
    config/code stratum in the T18 report.
    """
    known = set(known_frozen_config_shas())
    summary = {}
    for name, rows in row_groups:
        configs = sorted({r["config_sha256"] for r in rows})
        unknown = set(configs) - known
        if unknown:
            base.die(
                f"{name} contains unknown frozen-config SHA(s): "
                f"{sorted(unknown)[:2]}"
            )
        summary[name] = {
            "config_sha256": configs,
            "git_sha": sorted({r["git_sha"] for r in rows}),
        }
    return summary


def load_rows_g2(run_dir, cfg, pool=None, block_index=None, arm=None, blocks=None,
                 eligible_roles_file=None):
    """Assemble the frozen G2 slice without behaviour-label filtering.

    Role rows must be technically valid, belong to the frozen eligible-role set,
    and have the frozen primary activation present. No automatic role score is
    read. Default rows are technically valid rows from the five frozen default
    conditions. This matches the population used by the approved T15/T16
    confirmatory geometry.
    """
    g2, g2_sha = _load_g2_record()
    geom = cfg["confirmatory_geometry"]
    pool = pool or geom["primary_pool"]
    block_index = geom["primary_layer"] if block_index is None else block_index
    arm = arm or geom["primary_prompt_arm"]
    blocks = set(blocks or geom["primary_blocks"])

    expected = g2["geometry"]
    if pool != expected["pool"]:
        base.die(f"requested pool {pool!r} disagrees with G2 {expected['pool']!r}")
    if block_index != expected["block_index"]:
        base.die(
            f"requested block index {block_index!r} disagrees with G2 "
            f"{expected['block_index']!r}"
        )
    if arm != expected["prompt_arm"]:
        base.die(f"requested arm {arm!r} disagrees with G2 {expected['prompt_arm']!r}")
    if blocks != set(expected["blocks"]):
        base.die(
            f"requested blocks {sorted(blocks)!r} disagree with G2 "
            f"{sorted(expected['blocks'])!r}"
        )

    run_dir = Path(run_dir)
    for name in REQUIRED_G2_FILES:
        if not (run_dir / name).exists():
            base.die(
                f"{run_dir / name} is missing; G2-compatible T18 requires "
                f"{', '.join(REQUIRED_G2_FILES)}"
            )

    eligible, eligibility = base.load_eligible_roles(
        Path(eligible_roles_file) if eligible_roles_file
        else run_dir / "eligible_roles.json"
    )
    if len(eligible) != int(expected["retained_roles"]):
        base.die(
            f"eligible-role artifact declares {len(eligible)} roles but G2 freezes "
            f"{expected['retained_roles']}"
        )

    generation = read_jsonl(run_dir / "generation.jsonl", "generation")
    activations = read_jsonl(run_dir / "activations.jsonl", "activation")
    store = np.load(run_dir / "vectors.npz")

    provenance_strata = _check_known_provenance((
        ("generation.jsonl", generation),
        ("activations.jsonl", activations),
    ))
    gen_by_id = base._index_unique(generation, "row_id", "generation row_id")
    base._index_unique(activations, "row_id", "activation row_id")

    vectors, roles, questions, labels = [], [], [], []
    counts = Counter({
        "considered": 0,
        "wrong_slice": 0,
        "invalid": 0,
        "role_not_eligible": 0,
        "no_vector": 0,
        "kept_role": 0,
        "kept_default": 0,
    })
    seen_default_conditions, seen_roles = set(), set()

    for act in activations:
        counts["considered"] += 1
        if act["pool"] != pool or act["block_index"] != block_index:
            counts["wrong_slice"] += 1
            continue

        gen = gen_by_id.get(act["generation_row_id"])
        if gen is None:
            base.die(
                f"activation row {act['row_id']} references unknown generation "
                f"row {act['generation_row_id']}"
            )
        if gen["block"] not in blocks:
            counts["wrong_slice"] += 1
            continue

        is_default = gen["arm"] == "DEFAULT"
        if not is_default and gen["arm"] != arm:
            counts["wrong_slice"] += 1
            continue
        if gen["technical_validity"] != "valid":
            counts["invalid"] += 1
            continue

        if is_default:
            condition = gen["default_condition_index"]
            group = f"{DEFAULT_ROLE_PREFIX}{condition}"
            label = 1
            seen_default_conditions.add(condition)
        else:
            role = str(gen["role_id"])
            if role not in eligible:
                counts["role_not_eligible"] += 1
                continue
            group, label = role, 0
            seen_roles.add(role)

        if act["row_id"] not in store:
            counts["no_vector"] += 1
            continue
        vec = np.asarray(store[act["row_id"]], dtype=np.float64)
        digest = hashlib.sha256(
            np.ascontiguousarray(vec, dtype=np.float32).tobytes()
        ).hexdigest()
        if digest != act["vector_sha256"]:
            base.die(
                f"vector for activation row {act['row_id']} does not match its "
                "stored vector_sha256; the vector store and manifest disagree"
            )

        vectors.append(vec)
        roles.append(group)
        questions.append(int(gen["question_id"]))
        labels.append(label)
        counts["kept_default" if label else "kept_role"] += 1

    if not vectors:
        base.die("no rows survived the frozen G2 slice filters")
    if counts["kept_default"] == 0 or counts["kept_role"] == 0:
        base.die(f"one class is empty after filtering: {dict(counts)}")
    if len(seen_default_conditions) != REQUIRED_DEFAULT_CONDITIONS:
        base.die(
            f"expected exactly {REQUIRED_DEFAULT_CONDITIONS} default conditions, "
            f"found {sorted(seen_default_conditions)}"
        )
    missing = sorted(eligible - seen_roles)
    if missing:
        base.die(
            f"{len(missing)} roles in the frozen eligible set have no retained "
            f"rows in this slice (e.g. {missing[:5]})"
        )

    rows = ReadoutRows(np.vstack(vectors), roles, questions, labels)
    return rows, {
        "slice": {
            "pool": pool,
            "block_index": block_index,
            "arm": arm,
            "blocks": sorted(blocks),
        },
        "membership": MEMBERSHIP,
        "membership_authority": MEMBERSHIP_AUTHORITY,
        "compatibility_authority": COMPATIBILITY_AUTHORITY,
        "judge_filter_applied": False,
        "target_wording": "default-Assistant versus role-prompted responses",
        "counts": dict(counts),
        "eligibility": eligibility,
        "provenance_strata": provenance_strata,
        "n_roles": len(rows.role_roles),
        "n_default_conditions": len(rows.default_roles),
        "n_questions": len(set(rows.questions)),
        "g2_freeze_sha256": g2_sha,
        "g2_canonical_result_path": g2["authority"]["canonical_result_path"],
        "g2_membership_authority_pr": g2["authority"]["membership_authority_pr"],
    }


_frozen_analyse = base.analyse
_frozen_self_test = base.self_test


def _production_analyse(*args, **kwargs):
    """Serial cache by default; bounded fork parallelism when requested."""
    workers = int(os.environ.get("T18_OUTER_WORKERS", "1"))
    if workers <= 1:
        return analyse_cached(*args, **kwargs)
    return analyse_parallel_cached(*args, outer_workers=workers, **kwargs)


def _self_test_on_frozen_analyse(*args, **kwargs):
    """Run the real ``self_test`` with the frozen ``analyse``, not the cache.

    ``self_test`` calls the bare module-global ``analyse(...)``, so patching
    ``base.analyse`` below would otherwise silently make ``--self-test``
    validate ``analyse_cached`` instead of the frozen reference it is
    documented (CI, docs/T18_G2_PRODUCTION_RUN.md) to validate. Swap the
    frozen implementation back in only for the duration of the self-test
    call, then restore the cache for the production ``--run-dir`` path.
    """
    base.analyse = _frozen_analyse
    try:
        return _frozen_self_test(*args, **kwargs)
    finally:
        base.analyse = _production_analyse


def main(argv=None):
    """Run frozen T18 with G2 loading and computation-equivalent PCA caching."""
    base.load_rows = load_rows_g2
    base.analyse = _production_analyse
    base.self_test = _self_test_on_frozen_analyse
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
