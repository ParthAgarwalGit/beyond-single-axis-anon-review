"""Regression coverage for T18 nonlinear role/default balancing."""

from collections import Counter

import numpy as np

from src.t18_readouts import DEFAULT_ROLE_PREFIX, _balanced_index


def test_balanced_index_equalises_groups_and_classes_when_lengths_do_not_divide():
    """No role/default group may gain weight from a partial class-prefix repeat.

    The fixture deliberately creates the failure shape from review: after the old
    within-class equalisation, class 0 had 9 sampled rows and class 1 had 10. Resizing
    the 9-row class to 10 repeated its first element, so one role received four copies
    while its peers received three. The corrected sampler must choose a common class
    total divisible by both group counts and then repeat each group independently.
    """
    roles = np.array(
        ["role_a"] * 2
        + ["role_b"] * 3
        + ["role_c"] * 5
        + [f"{DEFAULT_ROLE_PREFIX}0"] * 4
        + [f"{DEFAULT_ROLE_PREFIX}1"] * 7,
        dtype=object,
    )
    y = np.array([0] * 10 + [1] * 11, dtype=int)

    idx = _balanced_index(y, roles)
    selected_roles = roles[idx]
    selected_y = y[idx]

    class_totals = Counter(selected_y.tolist())
    assert class_totals[0] == class_totals[1]

    role_counts = Counter(selected_roles[selected_y == 0].tolist())
    default_counts = Counter(selected_roles[selected_y == 1].tolist())

    assert set(role_counts) == {"role_a", "role_b", "role_c"}
    assert set(default_counts) == {
        f"{DEFAULT_ROLE_PREFIX}0",
        f"{DEFAULT_ROLE_PREFIX}1",
    }
    assert len(set(role_counts.values())) == 1
    assert len(set(default_counts.values())) == 1

    # Minimal common target at or above the old 10-row maximum is 12:
    # 12 / 3 role groups = 4 each; 12 / 2 default groups = 6 each.
    assert set(role_counts.values()) == {4}
    assert set(default_counts.values()) == {6}
    assert class_totals == Counter({0: 12, 1: 12})
