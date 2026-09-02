# ============================================================================
# T13 APPENDIX JUDGE CELL — fresh role scoring over C80  [UNVALIDATED]
# ============================================================================
# Pre-declared sensitivity 2 of docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md.
#
# ***  READ THIS BEFORE USING THE OUTPUT  ***
# The scores this cell produces come from an instrument that has NEVER been
# validated. T07 validated the ARCHIVED production scores, whose judge revision is
# unrecoverable (configs/t07_judge_frozen.json: model_revision null,
# NOT_RECORDED_IN_RECOVERABLE_ARCHIVED_ARTIFACTS), and the freeze forbids
# substituting the current Hub revision for it. A fresh run therefore inherits
# NONE of T07's characterisation - not even its failure numbers.
# These scores are for DESCRIPTIVE APPENDIX USE ONLY. They must not define role
# retention, must not build any Axis, and must not be reported as a confirmatory or
# robustness result. Every output file is stamped accordingly.
#
# ---------------------------------------------------------------------------
# INPUTS YOU MUST SUPPLY
#   1. roles.json  - upload the repo's data/lu_et_al/roles.json (275 roles, each with
#      role_id + eval_prompt). Set ROLES_JSON to its local path.
#   2. Generation records with output_text, per block.
#
#      AVAILABILITY WARNING (verified 2026-08-17): the HF dataset contains
#      results/generation/c80_role_C80_B.final.jsonl (338 MB) but there is NO
#      equivalent file for C80-A. The t12-c80 tree stores activations and metadata
#      only - no response text - so C80-A CANNOT be judged from what is published.
#      To cover C80-A, upload its generation .jsonl and add it to LOCAL_BLOCKS.
#      The cell judges whatever blocks it can find and records the coverage gap.
# ---------------------------------------------------------------------------
HF_TOKEN = ""            # <-- paste, do not commit
ROLES_JSON = "roles.json"
LOCAL_BLOCKS = {}        # e.g. {"C80-A": "/content/c80_role_C80_A.final.jsonl"}
REPO = "[Author-A-HF]/persona-artifacts"
DEST_PREFIX = "t13-appendix-unvalidated"
JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"

import json, os, re, hashlib, datetime

get_ipython().run_line_magic("pip", "install -q vllm huggingface_hub")  # noqa: F821
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from huggingface_hub import hf_hub_download, list_repo_files, upload_folder
from vllm import LLM, SamplingParams

# ---- frozen judge procedure, verbatim from -------------------------------------
# replications/assistant-axis-8b/production_judge_reference.py
THINK = "</think>"


def answer_region(completion):
    i = completion.rfind(THINK)
    return completion[i + len(THINK):].strip() if i >= 0 else ""


def judge_body(tmpl, answer_text):
    tmpl = tmpl or ""
    if "{response}" in tmpl:
        return tmpl.replace("{response}", answer_text)
    return f"{tmpl}\n\nResponse:\n{answer_text}"


def parse_score(judge_raw, answer_text, closed_think):
    m = re.search(r"[0-3]", (judge_raw or "").strip())
    return int(m.group()) if (m and answer_text and closed_think) else None


LABEL_MAP = {0: "no_role_playing", 1: "no_role_playing",
             2: "somewhat_role_playing", 3: "fully_role_playing"}
# --------------------------------------------------------------------------------

rubric = {r["role_id"]: r["eval_prompt"] for r in json.load(open(ROLES_JSON))["roles"]}
print(f"loaded {len(rubric)} role rubrics")

FILES = list_repo_files(REPO, repo_type="dataset", token=HF_TOKEN)
sources = dict(LOCAL_BLOCKS)
for blk in ("C80-A", "C80-B"):
    if blk in sources:
        continue
    cand = f"results/generation/c80_role_{blk}.final.jsonl"
    if cand in FILES:
        sources[blk] = hf_hub_download(REPO, cand, repo_type="dataset", token=HF_TOKEN)
missing = [b for b in ("C80-A", "C80-B") if b not in sources]
print(f"blocks to judge: {sorted(sources)}")
if missing:
    print(f"!! NO generation text for {missing} - these blocks will be absent from the "
          f"appendix. Upload their .jsonl and set LOCAL_BLOCKS to include them.")

llm = LLM(model=JUDGE_MODEL, quantization="awq", dtype="float16",
          gpu_memory_utilization=0.90, max_model_len=4096)
sp = SamplingParams(temperature=0.0, max_tokens=8)

os.makedirs("/tmp/t13", exist_ok=True)
summary = {}
for blk, path in sorted(sources.items()):
    recs, prompts = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        # judge only what the frozen rules allow to reach a judge at all
        if r.get("technical_validity") != "valid" or not r.get("role_judgeable"):
            continue
        txt = r.get("output_text") or ""
        a = answer_region(txt)
        recs.append({"rollout_id": r["rollout_id"], "role": r["role_id"],
                     "question_id": r.get("question_id"), "arm": r.get("arm"),
                     "block": blk, "retry": r.get("retry", 0),
                     "_answer": a, "_closed": THINK in txt})
        prompts.append(judge_body(rubric.get(r["role_id"], ""), a))
    print(f"{blk}: judging {len(recs)} outputs ...")
    outs = llm.generate(prompts, sp)

    out_path = f"/tmp/t13/scores_{blk}.jsonl"
    dist = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for rec, o in zip(recs, outs):
            raw = o.outputs[0].text.strip()
            sc = parse_score(raw, rec["_answer"], rec["_closed"])
            dist[str(sc)] = dist.get(str(sc), 0) + 1
            f.write(json.dumps({
                "rollout_id": rec["rollout_id"], "role": rec["role"],
                "question_id": rec["question_id"], "arm": rec["arm"],
                "block": rec["block"], "retry": rec["retry"],
                "lu_score": sc, "judge_raw": raw,
                "plan_label": LABEL_MAP.get(sc, "degenerate"),
                "validation_status": "UNVALIDATED_FRESH_JUDGE",
            }) + "\n")
    summary[blk] = {"n_judged": len(recs), "score_distribution": dist,
                    "abstained_none": dist.get("None", 0)}
    print(f"  -> {out_path}  distribution={dist}")

report = {
    "task": "T13_appendix_judge",
    "status": "UNVALIDATED - DESCRIPTIVE APPENDIX ONLY",
    "must_not": ["define role retention", "build any Axis",
                 "be reported as a confirmatory or robustness result"],
    "why_unvalidated": (
        "T07 validated archived production scores whose judge revision is "
        "unrecoverable (model_revision null). This is a fresh run on the current "
        "Hub revision and inherits none of T07's characterisation."),
    "judge_model": JUDGE_MODEL,
    "decoding": {"temperature": 0.0, "max_tokens": 8},
    "procedure_source": "replications/assistant-axis-8b/production_judge_reference.py",
    "blocks_judged": sorted(sources),
    "blocks_missing_generation_text": missing,
    "summary": summary,
    "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "authority": "docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md sensitivity 2",
}
open("/tmp/t13/t13_appendix_report.json", "w").write(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
upload_folder(folder_path="/tmp/t13", repo_id=REPO, repo_type="dataset",
              token=HF_TOKEN, path_in_repo=DEST_PREFIX,
              commit_message="T13 appendix judge scores (UNVALIDATED)")
print(f"\nuploaded -> {REPO}:{DEST_PREFIX}/  [UNVALIDATED - appendix use only]")
