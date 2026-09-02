# T08 question-artifact hash clarification

## Date

2026-08-05

## Issue

The frozen field `questions_artifact_sha256_in_review_package` was the byte
hash of the T02 review-package copy. The T08 builder incorrectly used this
value to validate `data/lu_et_al/questions.json`.

The repository artifact is identical between the T08 branch and `origin/main`,
but its byte hash is different from the review-package hash. No historical
repository version matches the review-package hash.

## Correction

The review-package hash was preserved.

Two provenance fields were added:

- `questions_artifact_sha256_repository`
- `questions_artifact_canonical_json_sha256`

The T08 builder now validates the repository file using
`questions_artifact_sha256_repository`.

## Scientific impact

No question IDs, question texts, block assignments, prompt assignments,
model outputs, activations, geometry, or outcomes were changed or inspected.
This is an implementation and provenance correction only.
