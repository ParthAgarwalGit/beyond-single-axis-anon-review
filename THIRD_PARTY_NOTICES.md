# Third-party notices

This repository's `LICENSE` (MIT) covers original code, tests, tools, and documentation written for this study. It does not grant rights over third-party research materials.

## Lu et al. / Assistant Axis source material

The study uses source material from Lu, Gallagher, Michala, Fish, and Lindsey, *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models* (arXiv:2601.10387) and the reference repository `safety-research/assistant-axis`, pinned at commit `a98961956072224eaf244eb289d6c01700b63795`.

The paper PDF stored under `data/lu_et_al/paper/` is released under CC BY 4.0 according to the paper's arXiv listing. The separate GitHub reference repository does not contain a LICENSE file at the pinned revision.

Four project-created JSON containers embed verbatim text drawn from that unlicensed repository:

- `data/lu_et_al/questions.json`
- `data/lu_et_al/role_prompts.json`
- `data/lu_et_al/roles.json`
- `data/lu_et_al/causal_evaluation.json`

The container serialization is original to this study, but the embedded upstream role/instruction/question text has no explicit redistribution grant from the repository itself. The paper's CC BY 4.0 license is not treated here as a grant covering the separate repository assets.

**Release handling:** these four JSON containers and the vendored paper PDF are excluded from the anonymous reviewer/public-release snapshot produced by `tools/build_anonymous_review_snapshot.py`. Reproducibility is preserved through `tools/fetch_lu_et_al_source.py`, which clones the pinned public upstream commit and reconstructs the four containers. The working research repository retains the committed copies to preserve the hashes used by historical frozen analyses; this notice does not claim that retention settles public redistribution rights.

`data/lu_et_al/KNOWN_SOURCE_DEFECTS.md`, `VERIFICATION.md`, and `artifact_manifest.json` are original project analysis/provenance documents rather than verbatim copies of upstream repository files and are covered by this repository's MIT license.

## Archived replications

Everything under `replications/` is historical/exploratory and superseded for the paper's scientific claims. The directory is excluded from the anonymous review snapshot.

- `replications/persona-vectors/` contains original reproduction code/artifacts derived by this study; it does not vendor the Chen et al. paper or upstream released weights/data.
- `replications/engels-irreducibility/` contains original reproduction code, figures, and tables; it does not vendor the Engels et al. repository.
- `replications/assistant-axis/` and `replications/assistant-axis-8b/` are project-internal historical explorations.

## Remaining boundary

Public redistribution permission for the four source-derived Lu JSON containers remains unresolved unless the upstream authors provide an explicit license or permission. The reviewer/public-release artifact therefore does not include them. Users who need to reconstruct the exact source containers should use the pinned fetch script against the public upstream repository.
