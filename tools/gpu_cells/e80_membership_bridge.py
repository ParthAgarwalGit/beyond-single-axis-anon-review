# ============================================================================
# MEMBERSHIP BRIDGE CELL — cos(v_score3, v_all_valid) on E80
# ============================================================================
# Pre-declared sensitivity 1 of docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md.
#
# WHY: the confirmatory C80 Axis is now built label-independently, because no
# validated judge scores exist for C80. E80 is the one corpus where the archived
# production scores exist, so it is the only place we can measure how far the
# membership rule actually moves the Axis.
#
# No GPU and no model needed - pure pooling of archived activations. Downloads about
# 5.8 GB of pooled tensors, so run it where bandwidth is cheap.
# ============================================================================
HF_TOKEN = ""  # <-- paste, do not commit
REPO = "[Author-A-HF]/persona-artifacts"
BLOCK_INDEX = 16
THRESHOLD = 10
HF_REVISION = ""  # <-- required immutable 40-hex dataset revision

import json, os, hashlib
import numpy as np

get_ipython().run_line_magic("pip", "install -q huggingface_hub safetensors numpy")  # noqa: F821
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, upload_folder
from safetensors.numpy import load_file

_HEX = set("0123456789abcdef")
if not HF_REVISION:
    _sha = HfApi().repo_info(REPO, repo_type="dataset", token=HF_TOKEN).sha
    raise SystemExit(
        f"HF_REVISION is empty. Current {REPO} revision is {_sha}. Paste that 40-hex "
        "commit SHA into HF_REVISION and rerun so the bridge is pinned to immutable bytes."
    )
HF_REVISION = HF_REVISION.strip().lower()
if len(HF_REVISION) != 40 or any(ch not in _HEX for ch in HF_REVISION):
    raise SystemExit("HF_REVISION must be an immutable 40-hex HF dataset commit SHA")

FILES = list_repo_files(REPO, repo_type="dataset", token=HF_TOKEN, revision=HF_REVISION)
BASE = "lu-replication/run-003"


def _dl(path):
    return hf_hub_download(
        REPO, path, repo_type="dataset", token=HF_TOKEN, revision=HF_REVISION
    )


def _jsonl(path):
    return [json.loads(l) for l in open(_dl(path), encoding="utf-8")]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Mean:
    __slots__ = ("s", "n")

    def __init__(self):
        self.s = None; self.n = 0

    def add(self, v):
        self.s = v.copy() if self.s is None else self.s + v
        self.n += 1

    def mean(self):
        return self.s / self.n


meta_path = _dl(f"{BASE}/translated/pooled_meta.jsonl")
scores_path = _dl(f"{BASE}/translated/scores.jsonl")
meta = {r["uid"]: r for r in _jsonl(f"{BASE}/translated/pooled_meta.jsonl")}
scores = {r["uid"]: r for r in _jsonl(f"{BASE}/translated/scores.jsonl")}
print(f"E80 translated: {len(meta)} pooled outputs, {len(scores)} archived scores")

shards = sorted(
    f for f in FILES
    if f.startswith(f"{BASE}/translated/pooled_primary") and f.endswith(".safetensors")
)
all_valid, score3 = {}, {}
seen = 0
shard_hashes = {}
for sh in shards:
    local = _dl(sh)
    shard_hashes[sh] = _sha256(local)
    tensors = load_file(local)
    for uid, t in tensors.items():
        m = meta.get(uid)
        if m is None:
            continue
        seen += 1
        v = np.asarray(t[BLOCK_INDEX], dtype=np.float64)
        role = m["role"]
        all_valid.setdefault(role, _Mean()).add(v)
        s = scores.get(uid)
        if s is not None and s.get("lu_score") == 3:
            score3.setdefault(role, _Mean()).add(v)
    del tensors
    print(f"  {sh.split('/')[-1]}: cumulative {seen} outputs")

# ---- default mean: 5 conditions (prompt_idx), equal 0.2 weighting ----------------
dmeta_path = _dl(f"{BASE}/default/pooled_meta.jsonl")
default_tensor_path = _dl(f"{BASE}/default/pooled_primary_00.safetensors")
dmeta = _jsonl(f"{BASE}/default/pooled_meta.jsonl")
dt = load_file(default_tensor_path)
cond = {}
for r in dmeta:
    t = dt.get(r["uid"])
    if t is None:
        continue
    cond.setdefault(int(r["prompt_idx"]), _Mean()).add(np.asarray(t[BLOCK_INDEX], dtype=np.float64))
if sorted(cond) != [0, 1, 2, 3, 4]:
    raise SystemExit(f"expected 5 default conditions, got {sorted(cond)}")
mu_default = np.mean([cond[c].mean() for c in range(5)], axis=0)
print("default conditions:", {c: cond[c].n for c in range(5)})


def axis(role_means, roles):
    grand = np.mean([role_means[r].mean() for r in roles], axis=0)
    v = mu_default - grand
    unit = v / np.linalg.norm(v)
    if float(np.dot(mu_default, unit)) <= float(
            np.mean([np.dot(role_means[r].mean(), unit) for r in roles])):
        raise SystemExit("sign check failed")
    return unit


roles = sorted(
    {r for r in all_valid if all_valid[r].n >= THRESHOLD}
    & {r for r in score3 if score3[r].n >= THRESHOLD}
)
print(f"roles clearing >= {THRESHOLD} under BOTH rules: {len(roles)} "
      f"(all_valid-only: {sum(1 for r in all_valid if all_valid[r].n >= THRESHOLD)})")
if len(roles) < 3:
    raise SystemExit("too few roles for a meaningful bridge")

v_all = axis(all_valid, roles)
v_s3 = axis(score3, roles)
cosine = float(np.dot(v_all, v_s3))
p_all = np.array([np.dot(all_valid[r].mean(), v_all) for r in roles])
p_s3 = np.array([np.dot(score3[r].mean(), v_s3) for r in roles])
pearson = float(np.corrcoef(p_all, p_s3)[0, 1])

report = {
    "task": "E80_membership_bridge",
    "purpose": "pre-declared sensitivity 1 of the 2026-08-17 deviation record",
    "status": "SENSITIVITY_ONLY - not a confirmatory statistic",
    "corpus": "E80 run-003 translated",
    "block_index": BLOCK_INDEX,
    "cos_v_score3_v_all_valid": round(cosine, 6),
    "role_projection_pearson_r": round(pearson, 6),
    "n_roles_common": len(roles),
    "n_outputs": {
        "all_valid": sum(m.n for m in all_valid.values()),
        "score3": sum(m.n for m in score3.values()),
    },
    "input_provenance": {
        "source_repo": REPO,
        "source_hf_revision": HF_REVISION,
        "translated_meta_sha256": _sha256(meta_path),
        "translated_scores_sha256": _sha256(scores_path),
        "default_meta_sha256": _sha256(dmeta_path),
        "default_tensor_sha256": _sha256(default_tensor_path),
        "translated_tensor_sha256": shard_hashes,
    },
    "caveats": {
        "default_arm": "archived E80 default arm is USER_EXPLICIT while roles are "
                       "USER_TRANSLATED_LU; both axes share the same default mean, so "
                       "the comparison isolates membership rather than a prompt-arm effect",
        "corpus_differs_from_confirmatory": "E80, not C80-A/C80-B; E80 does not "
                                            "determine confirmatory eligibility",
    },
}
os.makedirs("/tmp/bridge", exist_ok=True)
open("/tmp/bridge/e80_membership_bridge.json", "w").write(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
upload_folder(
    folder_path="/tmp/bridge", repo_id=REPO, repo_type="dataset", token=HF_TOKEN,
    path_in_repo="t15-confirmatory", commit_message="E80 membership bridge (pinned sensitivity)"
)
print("\ncos(v_score3, v_all_valid) =", round(cosine, 4),
      "-> higher values mean the membership switch moves the Axis less")
