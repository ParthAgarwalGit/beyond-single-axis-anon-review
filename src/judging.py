"""Canonical role-expression judge input and score parsing (METHOD_FREEZE §8).

The judge sees only role, role description/rubric, question, and the
final-answer text. It never sees prompt arm, channel, wrapper wording,
earlier scores, retention status, or steering condition. A missing or
malformed judge input is a missing measurement, never score 0.
"""

import re

_SCORE_RE = re.compile(r"\A[0-3]\Z")


def build_judge_input(role, role_description, question, segmentation):
    """Judge payload for one output, or None when the judge must abstain."""
    if segmentation.judge_input_text is None:
        return None
    return {
        "role": role,
        "role_description": role_description,
        "question": question,
        "final_answer_text": segmentation.judge_input_text,
    }


def parse_role_score(raw):
    """Parse a judge response into a score in {0,1,2,3}, else None.

    Frozen rule: after whitespace stripping, accept only one standalone
    integer in {0,1,2,3}. Any explanation, sign, extra token, or missing
    value is a parse failure (missing measurement, never converted to 0).
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if _SCORE_RE.match(stripped):
        return int(stripped)
    return None


def judge_validation_gates_pass(cfg, metrics):
    """Check the frozen human-validation gates for the automatic judge.

    ``metrics`` needs: score_3_precision, score_3_recall, macro_f1,
    cohen_kappa. Abstentions must be counted in coverage upstream, not
    silently dropped.
    """
    gates = cfg["role_judge"]["validation_gates"]
    return (
        metrics["score_3_precision"] >= gates["score_3_precision_min"]
        and metrics["score_3_recall"] >= gates["score_3_recall_min"]
        and metrics["macro_f1"] >= gates["macro_f1_min"]
        and metrics["cohen_kappa"] >= gates["cohen_kappa_min"]
    )
