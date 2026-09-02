# T18 outer-fold parallel execution — 29 Aug 2026

## Scope

Computation-only scheduling change layered on top of the reviewed PCA-cache
implementation in PR #79. The 20 frozen outer folds are independent fits and
are dispatched to a bounded Linux `fork` process pool.

## Unchanged scientific contract

No changes to G2 membership, frozen rows, grouped fold assignment, inner folds,
seeds, candidate grids, role balancing, PCA definition, scaling, model classes,
solvers, max_iter, AUROC, bootstrap, multiplicity correction, adequacy margin,
permutation diagnostic, or verdict rule.

Each worker inherits the exact same read-only input rows at fork, fits one outer
fold with the reviewed cache, and returns only that fold's predictions and
selection metadata. The parent restores results in deterministic fold-id order
before inference.

## Production setting

On the quota-limited `n2-highmem-32` benchmark VM:

- outer workers: 4
- BLAS threads per worker: 8
- total requested numerical threads: 32
- single-fold benchmark max RSS: 9.17 GiB
- VM memory: 256 GiB
- serial one-fold wall time: 1798.086 seconds

## Gate

Before real output is inspected:

1. serial-vs-parallel synthetic predictions/fold IDs/selections agree;
2. serial-vs-parallel inference/verdict agrees;
3. reviewed T18/cache/G2 focused tests remain green;
4. G2 `--self-test` remains on the frozen reference analyse path;
5. production runs from a clean committed source tree.
