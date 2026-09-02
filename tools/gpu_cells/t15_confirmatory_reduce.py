# ============================================================================
# T15/T16 REDUCTION CELL — build the confirmatory split-half bundle
# ============================================================================
# One self-contained notebook cell. Paste your HF token below. No GPU and no model
# are needed: this only pools activations T12 already extracted. It streams part by
# part, so peak memory is one 67 MB shard plus the running means.
#
# Verified against the real repository layout (2026-08-17):
#   t12-c80/C80-A/            meta_part{0000..0085}.jsonl + all_response_part*.safetensors
#   t12-c80/C80-B/            same, 22,000 rollouts per block, 275 roles x 80 questions
#   t12-c80/DEFAULT-C80-A/    block-specific default activations (400 = 80 q x 5 conds)
#   t12-c80/DEFAULT-C80-B/    same
#   t11-defaults/C80-A/defaults.jsonl   carries default_condition_index (0..4)
#
# Two layout facts this cell depends on, both confirmed:
#   * safetensors are keyed by uid, each value shape [32, 4096] float16 = all 32
#     blocks; the frozen middle block is index 16.
#   * the T12 default metadata does NOT record which of the five default conditions
#     a row belongs to (role/prompt_index are null), so the condition index must be
#     joined from t11-defaults/<block>/defaults.jsonl via uid == rollout_id.
#
# MEMBERSHIP (per docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md): primary is
# label-independent — technical validity plus a present all-response pool. There is
# no score-3 filter, because no validated judge scores exist for C80 at all.
#
# Outputs t15_confirmatory_bundle.npz + .counts.json, then uploads them. Analyse on
# CPU afterwards:
#   python tools/run_t15_confirmatory.py analyze \
#       --bundle t15_confirmatory_bundle.npz \
#       --counts t15_confirmatory_bundle.counts.json \
#       --out results/t15/confirmatory_reliability.json
# ============================================================================
HF_TOKEN = ""  # <-- paste, do not commit
REPO = "[Author-A-HF]/persona-artifacts"
DEST_PREFIX = "t15-confirmatory"
BLOCK_INDEX = 16          # frozen middle block
THRESHOLD = 10            # frozen per-80-ID-block eligibility, counted on valid outputs
DEFAULT_VALID_FLOOR = 0.95

import json, os, hashlib
import numpy as np

get_ipython().run_line_magic("pip", "install -q huggingface_hub safetensors numpy")  # noqa: F821
from huggingface_hub import hf_hub_download, list_repo_files, upload_folder
from safetensors.numpy import load_file

work = "/tmp/t15"; os.makedirs(work, exist_ok=True)
# Pin an IMMUTABLE dataset revision for every read. Reading "current main" makes the
# reduction unreproducible: the tree can be re-uploaded or metadata-patched underneath
# it and nothing in the output would show which bytes were actually consumed. Leave
# HF_REVISION empty only to discover the current sha, then paste it back and re-run.
HF_REVISION = ""   # <-- paste the 40-hex dataset revision, e.g. from HfApi().repo_info(...).sha

from huggingface_hub import HfApi
if not HF_REVISION:
    _sha = HfApi().repo_info(REPO, repo_type="dataset", token=HF_TOKEN).sha
    raise SystemExit(
        f"HF_REVISION is empty. Current {REPO} revision is {_sha}. Paste it into "
        f"HF_REVISION and re-run so the reduction is pinned to immutable bytes.")

FILES = list_repo_files(REPO, repo_type="dataset", token=HF_TOKEN, revision=HF_REVISION)


def _parts(prefix):
    metas = sorted(f for f in FILES
                   if f.startswith(f"{prefix}/meta_part") and f.endswith(".jsonl"))
    for m in metas:
        yield m, m.replace("meta_part", "all_response_part").replace(".jsonl", ".safetensors")


def _stream(prefix):
    """Yield (meta_row, block-16 vector float64) for every rollout under a prefix."""
    for meta_path, tensor_path in _parts(prefix):
        mp = hf_hub_download(REPO, meta_path, repo_type="dataset", token=HF_TOKEN,
                            revision=HF_REVISION)
        tp = hf_hub_download(REPO, tensor_path, repo_type="dataset", token=HF_TOKEN,
                            revision=HF_REVISION)
        tensors = load_file(tp)
        with open(mp, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                t = tensors.get(row["uid"])
                if t is None:
                    continue
                yield row, np.asarray(t[BLOCK_INDEX], dtype=np.float64)
        del tensors


def _eligible(row):
    """Label-independent membership: technically valid + the frozen pool present."""
    return row.get("validity") == "valid" and bool(row.get("has_all_response_pool"))


class _Mean:
    __slots__ = ("s", "n")

    def __init__(self):
        self.s = None; self.n = 0

    def add(self, v):
        self.s = v.copy() if self.s is None else self.s + v
        self.n += 1

    def mean(self):
        return self.s / self.n


def reduce_roles(block):
    acc, seen = {}, 0
    for row, vec in _stream(f"t12-c80/{block}"):
        seen += 1
        if not _eligible(row):
            continue
        acc.setdefault(row["role"], _Mean()).add(vec)
    counts = {r: m.n for r, m in acc.items()}
    print(f"  {block}: {seen} rollouts, {len(acc)} roles, "
          f"{sum(counts.values())} eligible outputs")
    return {r: m.mean() for r, m in acc.items()}, counts


def reduce_default(block):
    """Frozen default vector: mean within each of the five conditions, then equal
    0.2 weighting, with a 95% valid floor enforced per condition."""
    d = hf_hub_download(REPO, f"t11-defaults/{block}/defaults.jsonl",
                        repo_type="dataset", token=HF_TOKEN, revision=HF_REVISION)
    cond_of, total = {}, {}
    for line in open(d, encoding="utf-8"):
        r = json.loads(line)
        c = int(r["default_condition_index"])
        cond_of[r["rollout_id"]] = c
        total[c] = total.get(c, 0) + 1

    acc = {c: _Mean() for c in range(5)}
    for row, vec in _stream(f"t12-c80/DEFAULT-{block}"):
        c = cond_of.get(row["uid"])
        if c is None:
            raise SystemExit(f"{block}: default uid {row['uid']} missing from T11 records")
        if _eligible(row):
            acc[c].add(vec)

    fracs = {}
    for c in range(5):
        if acc[c].n == 0:
            raise SystemExit(f"{block}: default condition {c} has no eligible outputs")
        fracs[c] = acc[c].n / total[c]
        if fracs[c] < DEFAULT_VALID_FLOOR:
            raise SystemExit(f"{block}: default condition {c} valid fraction "
                             f"{fracs[c]:.4f} below the frozen {DEFAULT_VALID_FLOOR} floor")
    print(f"  {block} defaults: per-condition valid fractions "
          f"{ {c: round(f, 4) for c, f in fracs.items()} }")
    return np.mean([acc[c].mean() for c in range(5)], axis=0), fracs


print("reducing role means ...")
rm_a, ca = reduce_roles("C80-A")
rm_b, cb = reduce_roles("C80-B")
print("reducing block-specific defaults ...")
mu_a, fa = reduce_default("C80-A")
mu_b, fb = reduce_default("C80-B")

retained = {r for r in ca if ca[r] >= THRESHOLD} & {r for r in cb if cb[r] >= THRESHOLD}
print(f"retained (>= {THRESHOLD} eligible in BOTH blocks): {len(retained)} of "
      f"{len(set(ca) | set(cb))} roles")

bundle = {"mu_default_a": mu_a.astype(np.float32), "mu_default_b": mu_b.astype(np.float32)}
for r in sorted(set(rm_a) & set(rm_b)):
    bundle[f"role__{r}_a"] = rm_a[r].astype(np.float32)
    bundle[f"role__{r}_b"] = rm_b[r].astype(np.float32)
np.savez(os.path.join(work, "t15_confirmatory_bundle.npz"), **bundle)

man = {}
for blk in ("C80-A", "C80-B"):
    p = hf_hub_download(REPO, f"t12-c80/{blk}/manifest.json", repo_type="dataset",
                        token=HF_TOKEN, revision=HF_REVISION)
    man[blk] = hashlib.sha256(open(p, "rb").read()).hexdigest()

counts = {
    "a": ca, "b": cb,
    "provenance": {
        "t12_manifest_sha256": man, "arm": "USER_TRANSLATED_LU",
        "block_index": BLOCK_INDEX, "pool": "ALL_RESPONSE_TOKENS",
        "membership": "label_independent_technical_validity",
        "membership_authority": "docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md",
        "default_valid_fractions": {"C80-A": fa, "C80-B": fb},
        "source_repo": REPO,
        "source_hf_revision": HF_REVISION,
    },
}
open(os.path.join(work, "t15_confirmatory_bundle.counts.json"), "w").write(
    json.dumps(counts, indent=1))

upload_folder(folder_path=work, repo_id=REPO, repo_type="dataset", token=HF_TOKEN,
              path_in_repo=DEST_PREFIX, commit_message="T15/T16 confirmatory bundle")
print("uploaded ->", f"{REPO}:{DEST_PREFIX}/  (download both files, then run analyze)")
