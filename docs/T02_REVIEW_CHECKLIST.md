# T02 Reviewer Checklist

Reviewer: [Reviewer]

- [ ] Read `docs/METHOD_FREEZE.md`.
- [ ] Compare source claims with the merged T01 verification package.
- [ ] Run:

```bash
python tools/validate_method_frozen.py configs/method_frozen.yaml
```

- [ ] Confirm the validator prints `status: PASS`.
- [ ] Confirm C80-A and C80-B were created before their outputs or geometry were inspected.
- [ ] Confirm T03 imports the YAML rather than duplicating constants.
- [ ] Confirm T04 independently validates block 16, response boundaries, and all three token pools.
- [ ] Confirm the T05 gold set matches the frozen final-answer rule and does not expose machine validity hints.
- [ ] Confirm runtime model, tokenizer, chat-template, and code hashes are stamped before production.
- [ ] Confirm no causal generation begins until `a`, P50, held-out Axis, random vector, and causal judge rubrics are separately frozen.
- [ ] Change `freeze_status` from `REVIEW_CANDIDATE` to `FROZEN` only after review.

Current YAML SHA-256: `febbcbfc10ec40e98373c88a43fc31ba1c6ef2486eedeacca38d81301937e77b`
