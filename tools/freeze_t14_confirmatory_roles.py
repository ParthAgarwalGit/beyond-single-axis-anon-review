#!/usr/bin/env python3
"""Freeze the T14 C80 retained-role manifest for the measurement-limited branch.

Input is the T15/T16 reduction counts JSON produced before geometry statistics are
interpreted. Under the 2026-08-17 deviation, membership is label-independent
technical validity and the >=10 threshold is applied to technically eligible outputs.

This tool does not score outputs and does not inspect geometry. It only freezes the
role set implied by the already-declared membership rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_MEMBERSHIP = "label_independent_technical_validity"
EXPECTED_AUTHORITY_NAME = "DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md"
EXPECTED_THRESHOLD = 10
EXPECTED_ROLES = 275


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", required=True, help="t15_confirmatory_bundle.counts.json")
    ap.add_argument("--authority", required=True, help="merged 2026-08-17 membership deviation record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=int, default=EXPECTED_THRESHOLD)
    a = ap.parse_args(argv)

    cp, apath = Path(a.counts), Path(a.authority)
    if not cp.exists():
        raise SystemExit(f"counts file missing: {cp}")
    if not apath.exists() or apath.name != EXPECTED_AUTHORITY_NAME:
        raise SystemExit(f"membership authority missing/wrong: {apath}")
    if a.threshold != EXPECTED_THRESHOLD:
        raise SystemExit(f"primary threshold must remain {EXPECTED_THRESHOLD}, got {a.threshold}")

    d = json.loads(cp.read_text("utf-8"))
    prov = d.get("provenance", {})
    if prov.get("membership") != EXPECTED_MEMBERSHIP:
        raise SystemExit(
            f"counts membership {prov.get('membership')!r} != {EXPECTED_MEMBERSHIP!r}"
        )
    if prov.get("block_index") != 16 or prov.get("pool") != "ALL_RESPONSE_TOKENS":
        raise SystemExit("counts are not the frozen block-16 all-response construction")
    if prov.get("arm") != "USER_TRANSLATED_LU":
        raise SystemExit("counts are not the frozen USER_TRANSLATED_LU arm")

    ca, cb = d.get("a", {}), d.get("b", {})
    if not isinstance(ca, dict) or not isinstance(cb, dict):
        raise SystemExit("counts JSON must contain role-count dictionaries a and b")
    retained = sorted(
        {r for r, n in ca.items() if int(n) >= a.threshold}
        & {r for r, n in cb.items() if int(n) >= a.threshold}
    )
    if len(retained) != EXPECTED_ROLES:
        raise SystemExit(f"expected {EXPECTED_ROLES} retained roles, got {len(retained)}")
    if set(ca) != set(cb) or set(retained) != set(ca):
        raise SystemExit("primary measurement-limited branch should retain the full common 275-role set")

    eligible_a, eligible_b = sum(map(int, ca.values())), sum(map(int, cb.values()))
    if (eligible_a, eligible_b) != (21998, 22000):
        raise SystemExit(
            f"eligible totals differ from the frozen reduction: {eligible_a}/{eligible_b}"
        )

    out = {
        "task": "T14_confirmatory_retained_role_freeze",
        "status": "FROZEN",
        "membership": EXPECTED_MEMBERSHIP,
        "membership_authority": str(apath),
        "membership_authority_sha256": sha256(apath),
        "threshold_per_block": a.threshold,
        "block_index": 16,
        "pool": "ALL_RESPONSE_TOKENS",
        "arm": "USER_TRANSLATED_LU",
        "eligible_outputs": {"C80-A": eligible_a, "C80-B": eligible_b},
        "retained_role_count": len(retained),
        "retained_roles": retained,
        "counts_sha256": sha256(cp),
        "t12_manifest_sha256": prov.get("t12_manifest_sha256"),
        "claim_scope": (
            "technical-validity retained-role set; does not assert successful behavioural role enactment"
        ),
    }
    op = Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"T14 frozen: {len(retained)} roles -> {op}")


if __name__ == "__main__":
    main()
