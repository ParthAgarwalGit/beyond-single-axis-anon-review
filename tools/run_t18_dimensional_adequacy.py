"""T18 runner: is one direction an adequate summary?

Two modes.

``--self-test`` runs the pipeline against synthetic worlds whose answers are
known and writes a validation report. It needs no activations and no freeze,
so it can be run now, and it is what establishes that the machinery would
detect multidimensional or nonlinear structure if it were there. Every family
that can determine a verdict is covered.

``--run-dir DIR`` is the real analysis. It refuses to start until the
confirmatory geometry has been frozen, because T18 is a diagnostic that must
not feed back into the geometry it is diagnosing, and because running it early
would compete with T06/T07 for the same artifacts. ``--allow-unfrozen`` exists
only for rehearsals and stamps every output as non-reportable.

    python tools/run_t18_dimensional_adequacy.py --self-test
    python tools/run_t18_dimensional_adequacy.py --run-dir runs/confirmatory

Expected ``--run-dir`` layout (all five are required):

    generation.jsonl      GENERATION_ROW   arm, block, role_id, question_id,
                                           default_condition_index, validity
    judge.jsonl           JUDGE_ROW+T13    role_score, judge_region
    activations.jsonl     ACTIVATION_ROW   pool, block_index, vector_sha256
    vectors.npz                            row_id -> pooled vector
    eligible_roles.json                    frozen T14/T15 retained-role set

Fail-closed properties, each with a loader test:

* Role eligibility is **consumed** from the frozen artifact, never recomputed
  from raw score-3 counts, so T18 cannot diagnose a different role set from
  the one T15/G2 describe. The artifact's hash is bound into the report.
* Only the frozen primary judge region (``final_answer``) is read. T13 stores
  a paired ``full_visible`` sensitivity row per generation; keying by
  generation row alone would let one region silently overwrite the other.
* Duplicate primary measurements, duplicate row IDs, and config-hash
  disagreements are errors, not last-write-wins.
* Every activation vector is re-hashed against its stored ``vector_sha256``.
* The slice and the default-condition count are checked before any fitting.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_frozen_config  # noqa: E402
from src.provenance import stamp_report  # noqa: E402
from src.schemas import read_jsonl  # noqa: E402
from src.t18_readouts import (  # noqa: E402
    ADEQUACY_MARGIN,
    BOOTSTRAP_REPLICATES,
    DEFAULT_ROLE_PREFIX,
    INFERENCE_CLUSTER,
    INNER_FOLDS,
    OUTER_FOLDS,
    PERMUTATION_REPLICATES,
    PRIMARY_JUDGE_REGION,
    PRIMARY_METRIC,
    REQUIRED_DEFAULT_CONDITIONS,
    SPLIT_SEED,
    T18_CONFIG_SHA256,
    VERDICT_FAMILIES,
    ReadoutRows,
    adequacy_verdict,
    cluster_bootstrap,
    fold_aggregate_auroc,
    permute_labels_by_role,
    run_family,
    simultaneous_family_bootstrap,
)

BASELINE_FAMILY = "axis_1d"
DEFAULT_FREEZE_FILE = Path("docs/G2_GEOMETRY_FREEZE.json")
REQUIRED_FILES = ("generation.jsonl", "judge.jsonl", "activations.jsonl",
                  "vectors.npz", "eligible_roles.json")


def die(message):
    raise SystemExit(f"T18 aborted: {message}")


# ---------------------------------------------------------------------------
# Freeze gate
# ---------------------------------------------------------------------------

def check_geometry_freeze(freeze_file, allow_unfrozen):
    """Refuse to produce a reportable T18 result before the G2 freeze."""
    if allow_unfrozen:
        return {"frozen": False, "reportable": False,
                "reason": "run with --allow-unfrozen; rehearsal only"}
    if not freeze_file.exists():
        die(
            f"confirmatory geometry is not frozen ({freeze_file} not found).\n"
            "  T18 is a diagnostic on the frozen geometry and must not run "
            "before G2.\n"
            "  Use --self-test to validate the pipeline now, or "
            "--allow-unfrozen for a non-reportable rehearsal."
        )
    record = json.loads(freeze_file.read_text(encoding="utf-8"))
    if record.get("freeze_status") != "FROZEN":
        die(f"{freeze_file} exists but freeze_status is "
            f"{record.get('freeze_status')!r}, not 'FROZEN'")
    return {"frozen": True, "reportable": True,
            "freeze_file": str(freeze_file),
            "freeze_record_sha256": hashlib.sha256(
                freeze_file.read_bytes()).hexdigest()}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_eligible_roles(path):
    """Read the frozen T14/T15 retained-role artifact and bind its hash.

    T18 must diagnose exactly the role set the geometry was built on. Deriving
    eligibility here from raw score-3 counts would silently admit roles that
    fail the frozen per-block thresholds, so the set is consumed rather than
    recomputed and the artifact hash travels into the report.
    """
    if not path.exists():
        die(f"{path} is missing; T18 consumes the frozen eligible-role "
            "artifact and will not recompute eligibility from score-3 counts")
    raw = path.read_bytes()
    record = json.loads(raw.decode("utf-8"))
    roles = record.get("eligible_roles_primary")
    if not isinstance(roles, list) or not roles:
        die(f"{path} has no non-empty 'eligible_roles_primary' list")
    if len(set(roles)) != len(roles):
        die(f"{path} lists duplicate roles in 'eligible_roles_primary'")
    return set(map(str, roles)), {
        "eligible_roles_file": str(path),
        "eligible_roles_sha256": hashlib.sha256(raw).hexdigest(),
        "n_eligible_roles_declared": len(roles),
        "threshold": record.get("threshold"),
        "blocks": record.get("blocks"),
        "source_task": record.get("source_task"),
    }


def _primary_scores(judge_rows):
    """Role scores from the frozen primary judge region only.

    T13 writes one ``final_answer`` row and one paired ``full_visible``
    sensitivity row per generation. Keying by ``generation_row_id`` across
    both regions would let whichever row parsed last win, so the sensitivity
    region is filtered out and a repeated primary measurement is an error.
    """
    scores, seen_regions = {}, Counter()
    for row in judge_rows:
        region = row.get("judge_region")
        if region is None:
            die("judge rows carry no 'judge_region'; T18 requires the T13 "
                f"region tag to select the frozen primary region "
                f"{PRIMARY_JUDGE_REGION!r}")
        seen_regions[region] += 1
        if region != PRIMARY_JUDGE_REGION:
            continue
        gen_id = row["generation_row_id"]
        if gen_id in scores:
            die(f"generation row {gen_id} has more than one "
                f"{PRIMARY_JUDGE_REGION!r} judge measurement; refusing to pick "
                "one silently")
        # A parse failure, an abstention, or a missing region is a missing
        # measurement and is never coerced to a score.
        scores[gen_id] = (None if row["abstained"] else row["role_score"])
    if not scores:
        die(f"no judge rows in the frozen primary region "
            f"{PRIMARY_JUDGE_REGION!r} (regions present: {dict(seen_regions)})")
    return scores, dict(seen_regions)


def _index_unique(rows, key, what):
    """Index rows by a key, refusing duplicates rather than overwriting."""
    out = {}
    for row in rows:
        if row[key] in out:
            die(f"duplicate {what} {row[key]!r}; refusing to overwrite silently")
        out[row[key]] = row
    return out


def _check_config_agreement(*row_groups):
    """Every row must have been produced under one frozen config and commit."""
    _, config_sha = load_frozen_config()
    for name, rows in row_groups:
        mismatched = {r["config_sha256"] for r in rows} - {config_sha}
        if mismatched:
            die(f"{name} contains rows stamped with a different frozen config "
                f"({sorted(mismatched)[:2]}); expected {config_sha}")
        if len({r["git_sha"] for r in rows}) > 1:
            die(f"{name} mixes rows from more than one code commit; "
                "T18 requires a single generation stratum")


def load_rows(run_dir, cfg, pool=None, block_index=None, arm=None, blocks=None,
              eligible_roles_file=None):
    """Assemble the frozen primary slice into a ``ReadoutRows``.

    Defaults come from ``confirmatory_geometry`` in the frozen YAML rather
    than from constants here, so a slice can never silently disagree with the
    freeze. Role rows require membership of the frozen eligible-role set and a
    primary-region score of 3; default rows are filtered on technical validity
    only, matching ``role_vectors_and_axis.default_vector``.
    """
    geom = cfg["confirmatory_geometry"]
    pool = pool or geom["primary_pool"]
    block_index = geom["primary_layer"] if block_index is None else block_index
    arm = arm or geom["primary_prompt_arm"]
    blocks = set(blocks or geom["primary_blocks"])

    run_dir = Path(run_dir)
    for name in REQUIRED_FILES:
        if not (run_dir / name).exists():
            die(f"{run_dir / name} is missing; see this file's docstring for "
                "the expected run-directory layout")

    eligible, eligibility = load_eligible_roles(
        Path(eligible_roles_file) if eligible_roles_file
        else run_dir / "eligible_roles.json")

    generation = read_jsonl(run_dir / "generation.jsonl", "generation")
    judge = read_jsonl(run_dir / "judge.jsonl", "judge")
    activations = read_jsonl(run_dir / "activations.jsonl", "activation")
    store = np.load(run_dir / "vectors.npz")

    _check_config_agreement(("generation.jsonl", generation),
                            ("judge.jsonl", judge),
                            ("activations.jsonl", activations))
    gen_by_id = _index_unique(generation, "row_id", "generation row_id")
    _index_unique(activations, "row_id", "activation row_id")
    scores, regions_seen = _primary_scores(judge)

    vectors, roles, questions, labels = [], [], [], []
    counts = {"considered": 0, "wrong_slice": 0, "invalid": 0,
              "role_not_eligible": 0, "not_score_3": 0, "no_vector": 0,
              "kept_role": 0, "kept_default": 0}
    seen_default_conditions, seen_roles = set(), set()

    for act in activations:
        counts["considered"] += 1
        if act["pool"] != pool or act["block_index"] != block_index:
            counts["wrong_slice"] += 1
            continue
        gen = gen_by_id.get(act["generation_row_id"])
        if gen is None:
            die(f"activation row {act['row_id']} references unknown generation "
                f"row {act['generation_row_id']}")
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
            group = f"{DEFAULT_ROLE_PREFIX}{gen['default_condition_index']}"
            label = 1
            seen_default_conditions.add(gen["default_condition_index"])
        else:
            if str(gen["role_id"]) not in eligible:
                counts["role_not_eligible"] += 1
                continue
            if scores.get(act["generation_row_id"]) != 3:
                counts["not_score_3"] += 1
                continue
            group, label = str(gen["role_id"]), 0
            seen_roles.add(group)

        if act["row_id"] not in store:
            counts["no_vector"] += 1
            continue
        vec = np.asarray(store[act["row_id"]], dtype=np.float64)
        digest = hashlib.sha256(
            np.ascontiguousarray(vec, dtype=np.float32).tobytes()).hexdigest()
        if digest != act["vector_sha256"]:
            die(f"vector for activation row {act['row_id']} does not match its "
                "stored vector_sha256; the vector store and the manifest "
                "disagree")

        vectors.append(vec)
        roles.append(group)
        questions.append(int(gen["question_id"]))
        labels.append(label)
        counts["kept_default" if label else "kept_role"] += 1

    if not vectors:
        die("no rows survived the frozen slice filters")
    if counts["kept_default"] == 0 or counts["kept_role"] == 0:
        die(f"one class is empty after filtering: {counts}")
    if len(seen_default_conditions) != REQUIRED_DEFAULT_CONDITIONS:
        die(f"expected exactly {REQUIRED_DEFAULT_CONDITIONS} default "
            f"conditions, found {sorted(seen_default_conditions)}; the frozen "
            "default vector weights all five equally")
    missing = sorted(eligible - seen_roles)
    if missing:
        die(f"{len(missing)} roles in the frozen eligible set have no retained "
            f"rows in this slice (e.g. {missing[:5]}); the eligibility "
            "artifact and the run directory describe different data")

    rows = ReadoutRows(np.vstack(vectors), roles, questions, labels)
    return rows, {
        "slice": {"pool": pool, "block_index": block_index, "arm": arm,
                  "blocks": sorted(blocks)},
        "counts": counts,
        "judge_region": PRIMARY_JUDGE_REGION,
        "judge_regions_present": regions_seen,
        "eligibility": eligibility,
        "n_roles": len(rows.role_roles),
        "n_default_conditions": len(rows.default_roles),
        "n_questions": len(set(rows.questions)),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(rows, replicates=BOOTSTRAP_REPLICATES, seed=SPLIT_SEED,
            families=None, builders_by_family=None, folds=None,
            inner_folds=None, permutations=PERMUTATION_REPLICATES):
    """Run every family, compare simultaneously against the Axis, and decide."""
    families = tuple(families or VERDICT_FAMILIES)
    builders_by_family = builders_by_family or {}
    folds = folds or OUTER_FOLDS
    inner_folds = inner_folds or INNER_FOLDS

    predictions, fold_ids, selections = {}, {}, {}
    for family in (BASELINE_FAMILY, *families):
        pred, fold_of_row, sel = run_family(
            rows, family, folds=folds, inner_folds=inner_folds, seed=seed,
            builders=builders_by_family.get(family))
        predictions[family], fold_ids[family], selections[family] = (
            pred, fold_of_row, sel)

    fold_of_row = fold_ids[BASELINE_FAMILY]
    performance = {
        family: {
            **cluster_bootstrap(rows, pred, fold_ids[family], replicates, seed,
                                INFERENCE_CLUSTER),
            "question_clustered_diagnostic": cluster_bootstrap(
                rows, pred, fold_ids[family], replicates, seed, "question"),
        }
        for family, pred in predictions.items()
    }

    simultaneous = simultaneous_family_bootstrap(
        rows, {f: predictions[f] for f in families}, predictions[BASELINE_FAMILY],
        fold_of_row, replicates, seed, INFERENCE_CLUSTER)

    # Group-level label permutation. Descriptive only: it locates the null
    # roughly and would expose a gross leak, but no p-value is derived from it.
    nulls = []
    for i in range(permutations):
        permuted = permute_labels_by_role(rows, seed + 1000 + i)
        pred, permuted_folds, _ = run_family(
            permuted, BASELINE_FAMILY, folds=folds, inner_folds=inner_folds,
            seed=seed)
        scored = ~np.isnan(pred)
        nulls.append(fold_aggregate_auroc(
            permuted.y[scored], pred[scored], permuted_folds[scored],
            permuted.weights[scored]))

    return {
        "primary_metric": PRIMARY_METRIC,
        "estimand": "mean within-fold role-balanced AUROC",
        "inference_cluster": INFERENCE_CLUSTER,
        "adequacy_margin": ADEQUACY_MARGIN,
        "t18_config_sha256": T18_CONFIG_SHA256,
        "performance": performance,
        "simultaneous_vs_axis": simultaneous,
        "verdict": adequacy_verdict(simultaneous),
        "selections": selections,
        "permutation_null_diagnostic": {
            "status": "descriptive only; not an inferential test",
            "replicates": nulls,
            "mean": float(np.mean(nulls)) if nulls else float("nan"),
            "max_abs_deviation_from_chance": (
                float(np.max(np.abs(np.asarray(nulls) - 0.5))) if nulls
                else float("nan")),
        },
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test(seed=SPLIT_SEED):
    """Synthetic worlds with known answers, reported as pass/fail.

    This is the evidence that the verdict means something: the same pipeline,
    unchanged, must return ADEQUATE where one direction really is enough and
    INADEQUATE where it is not. Every family that can determine a verdict is
    included, so none of them can move a real result without having been
    validated here first.
    """
    from tests.test_t18_readouts import (
        FAST_AXIS, FAST_FULL, FAST_LINEAR, FAST_NONLINEAR,
        multidimensional_linear_world, one_dimensional_world, radial_world,
        role_identity_only_world,
    )

    worlds = [
        ("one_dimensional", one_dimensional_world(), "ADEQUATE",
         "one direction is genuinely sufficient"),
        ("multidimensional_linear", multidimensional_linear_world(),
         "INADEQUATE", "several linear directions are needed"),
        ("radial_nonlinear", radial_world(), "INADEQUATE",
         "no linear summary works; distance from the origin does"),
    ]
    fast = {"axis_1d": FAST_AXIS, "pca_linear": FAST_LINEAR,
            "full_linear": FAST_FULL, "small_nonlinear": FAST_NONLINEAR}
    verdict_families = ("pca_linear", "full_linear", "small_nonlinear")

    results, failures = [], []
    for name, rows, expected, why in worlds:
        out = analyse(rows, replicates=400, seed=seed,
                      families=verdict_families, builders_by_family=fast,
                      folds=(4, 3), inner_folds=(2, 2), permutations=3)
        got = out["verdict"]["verdict"]
        ok = got == expected
        results.append({
            "world": name, "expected": expected, "got": got, "passed": ok,
            "why": why,
            "families_validated": sorted(verdict_families),
            "axis_auroc": out["performance"]["axis_1d"]["point"],
            "best_alternative": out["verdict"]["best_alternative"],
            "max_family_difference": out["verdict"]["max_family_difference"],
            "simultaneous_ci": out["verdict"]["simultaneous_ci"],
            "permutation_null_mean": out["permutation_null_diagnostic"]["mean"],
        })
        if not ok:
            failures.append(f"{name}: expected {expected}, got {got}")

    # Null world: reported separately because the target is an unbiased
    # estimator, not a verdict label, and checked for every verdict family.
    #
    # Averaging over independent null datasets rather than reading one
    # interval is deliberate. A 95% interval misses one time in twenty by
    # construction, and with only five default groups the per-dataset estimate
    # genuinely ranges from about 0.2 to 0.75, so a single-dataset coverage
    # check would be a coin flip rather than a test. Interval coverage is
    # measured over many datasets in
    # tests/test_t18_readouts.py::test_question_clustering_is_anticonservative.
    null_seeds = [4 + 11 * i for i in range(6)]
    for family, builders in (("axis_1d", FAST_AXIS),
                             ("pca_linear", FAST_LINEAR),
                             ("full_linear", FAST_FULL),
                             ("small_nonlinear", FAST_NONLINEAR)):
        scores = []
        for null_seed in null_seeds:
            null_rows = role_identity_only_world(seed=null_seed)
            pred, fold_of_row, _ = run_family(
                null_rows, family, folds=(4, 3), inner_folds=(2, 2), seed=seed,
                builders=builders)
            scored = ~np.isnan(pred)
            scores.append(fold_aggregate_auroc(
                null_rows.y[scored], pred[scored], fold_of_row[scored],
                null_rows.weights[scored]))
        mean = float(np.mean(scores))
        unbiased = abs(mean - 0.5) < 0.10
        results.append({
            "world": f"role_identity_only[{family}]",
            "expected": "mean over null datasets within 0.10 of chance",
            "got": f"{mean:.3f} over {len(null_seeds)} datasets",
            "passed": bool(unbiased),
            "per_dataset": scores,
            "why": "label information tied to role identity must not transfer "
                   "to unseen roles",
        })
        if not unbiased:
            failures.append(
                f"role_identity_only[{family}]: mean {mean:.3f} is biased away "
                "from chance")

    return {"worlds": results, "all_passed": not failures,
            "failures": failures,
            "t18_config_sha256": T18_CONFIG_SHA256}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="validate the pipeline on synthetic worlds; no data needed")
    p.add_argument("--run-dir", help="directory holding the frozen run artifacts")
    p.add_argument("--eligible-roles",
                   help="override the path to the frozen eligible-role artifact")
    p.add_argument("--out", default="results/t18/t18_report.json")
    p.add_argument("--freeze-file", type=Path, default=DEFAULT_FREEZE_FILE)
    p.add_argument("--allow-unfrozen", action="store_true",
                   help="rehearse before G2; output is stamped non-reportable")
    p.add_argument("--allow-dirty", action="store_true",
                   help="stamp a report from a dirty working tree (development)")
    p.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = p.parse_args(argv)

    if bool(args.self_test) == bool(args.run_dir):
        die("choose exactly one of --self-test or --run-dir")

    cfg, config_sha = load_frozen_config()
    if args.self_test:
        report = {"task": "T18", "mode": "self_test",
                  "validation": self_test(args.seed)}
    else:
        gate = check_geometry_freeze(args.freeze_file, args.allow_unfrozen)
        rows, provenance = load_rows(args.run_dir, cfg,
                                     eligible_roles_file=args.eligible_roles)
        report = {
            "task": "T18",
            "mode": "analysis",
            "geometry_freeze": gate,
            "reportable": gate.get("reportable", False),
            "data": provenance,
            **analyse(rows, args.replicates, args.seed),
        }

    report = stamp_report(report, allow_dirty=args.allow_dirty)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=float)
        f.write("\n")

    if args.self_test:
        v = report["validation"]
        for w in v["worlds"]:
            print(f"  [{'PASS' if w['passed'] else 'FAIL'}] {w['world']:34s} "
                  f"expected {w['expected']:24s} got {w['got']}")
        print("\n" + ("all worlds passed" if v["all_passed"]
                      else "FAILURES: " + "; ".join(v["failures"])))
    else:
        v = report["verdict"]
        print(f"  verdict: {v['verdict']} (margin {v['margin']} {PRIMARY_METRIC})")
        print(f"  best alternative: {v['best_alternative']} "
              f"{v['max_family_difference']:+.4f}, simultaneous CI "
              f"{v['simultaneous_ci']} over {v['simultaneous_over']}")
        print(f"  permutation null (descriptive): "
              f"{report['permutation_null_diagnostic']['mean']:.4f}")
    print(f"  report: {out}  (method {config_sha[:12]}, "
          f"t18 {T18_CONFIG_SHA256[:12]})")
    return 0 if report.get("validation", {}).get("all_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
