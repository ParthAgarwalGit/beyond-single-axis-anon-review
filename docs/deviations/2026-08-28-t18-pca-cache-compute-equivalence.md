# T18 PCA-cache computation equivalence — 28 Aug 2026

Outcome-blind computation-only optimization. No real T18 outcome may be inspected
before this change is independently reviewed and the equivalence tests pass.

The reference implementation recomputes the same fold-local full SVD for every
PCA-based candidate. For a fixed training split the PCA basis depends only on
the training rows, not on C, hidden width, or alpha. The cache therefore fits
one full SVD per unique training split, computes each requested k-dimensional
projection once, shares it across `pca_linear` and `small_nonlinear`, and shares
full-dimensional centring/scaling across `full_linear` C values.

No change is made to G2 membership, rows, folds, seeds, weights, Axis
construction, PCA definition, candidate grids, solvers, `max_iter`, AUROC,
fold aggregation, bootstrap, multiplicity adjustment, 0.02 adequacy margin, or
verdict rule. `src/t18_readouts.py` remains unchanged as the numerical reference.

Before production, require legacy-vs-cache agreement on fold IDs, selected
candidates, out-of-fold probabilities within numerical tolerance, family
AUROCs, simultaneous interval, and verdict on deterministic synthetic worlds.
