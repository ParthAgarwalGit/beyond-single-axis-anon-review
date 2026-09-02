import importlib.util
from pathlib import Path

import numpy as np
import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_t24_source_control_freeze.py"
_spec = importlib.util.spec_from_file_location("t24_freeze", _TOOL)
t24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t24)


def test_matched_control_quantiles_follow_source_engagement():
    random_q, sphere_q = t24.matched_control_quantiles(0.5959)
    assert np.isclose(random_q, 0.5959)
    assert np.isclose(sphere_q, 0.4041)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_matched_control_quantiles_fail_closed_outside_open_unit_interval(bad):
    with pytest.raises(SystemExit, match="source engagement must lie strictly between 0 and 1"):
        t24.matched_control_quantiles(bad)


def test_matched_quantiles_recover_target_engagement_on_continuous_data():
    rng = np.random.default_rng(123)
    source_rate = 0.61
    natural = rng.normal(size=(50000, 8))
    direction = rng.normal(size=8)

    random_q, sphere_q = t24.matched_control_quantiles(source_rate)
    random_threshold = t24.calibrate_lower_tail_threshold(
        natural, direction, quantile=random_q
    )
    _, random_engaged = t24.lower_tail_cap(natural, direction, random_threshold)

    center = natural[:1000].mean(axis=0)
    sphere_radius = t24.calibrate_sphere_radius(
        natural, center, quantile=sphere_q
    )
    _, sphere_engaged = t24.spherical_distance_cap(natural, center, sphere_radius)

    assert abs(t24.engagement_rate(random_engaged) - source_rate) < 0.01
    assert abs(t24.engagement_rate(sphere_engaged) - source_rate) < 0.01


def test_mismatch_larger_than_frozen_tolerance_is_detectable():
    source_rate = 0.60
    control_rate = 0.50
    assert abs(control_rate - source_rate) > t24.ENGAGEMENT_TOL


def test_original_q25_q75_constants_are_retained_as_sensitivity():
    assert t24.ORIGINAL_RANDOM_LOWER_QUANTILE == 0.25
    assert t24.ORIGINAL_SPHERE_RADIUS_QUANTILE == 0.75
    assert t24.DEVIATION_ID == "T24_ENGAGEMENT_MATCHING_2026-08-21"



def test_freeze_aborts_on_genuine_control_engagement_mismatch(
    tmp_path, monkeypatch
):
    """Exercise freeze() itself and prove the engagement gate fails closed."""
    rng = np.random.default_rng(20260821)
    hidden = 8

    axis = np.zeros(hidden, dtype=np.float64)
    axis[0] = 1.0

    released = np.zeros((54, hidden), dtype=np.float64)
    released[:] = axis

    payload = {
        "calibration_source_id": np.asarray(
            t24.CALIBRATION_SOURCE_ID
        ),
        "calibration_manifest_sha256": np.asarray(
            t24.CALIBRATION_MANIFEST_SHA256
        ),
        "capture_source_git_sha": np.asarray("a" * 40),
        "capture_schema": np.asarray("test-fixture/1.0"),
    }

    source_caps = {}

    for layer in t24.LAYERS:
        natural = rng.normal(size=(400, hidden))
        default = rng.normal(size=(40, hidden))

        payload[f"axis_{layer}"] = axis.copy()
        payload[f"natural_{layer}"] = natural
        payload[f"default_{layer}"] = default

        source_vector = -axis.copy()
        source_projection = natural @ source_vector

        # Source engages ~50% of rows.
        source_threshold = float(
            np.quantile(source_projection, 0.50)
        )

        source_caps[layer] = (
            source_vector,
            source_threshold,
        )

    calibration = tmp_path / "calibration.npz"
    np.savez_compressed(calibration, **payload)

    axis_pt = tmp_path / "assistant_axis.pt"
    cap_pt = tmp_path / "capping_config.pt"

    axis_pt.write_bytes(b"fixture-axis")
    cap_pt.write_bytes(b"fixture-cap")

    # This test exercises freeze() without depending on the real Git tree.
    monkeypatch.setattr(
        t24,
        "require_clean_source_tree",
        lambda: "b" * 40,
    )

    def fake_sha(path):
        path = Path(path)

        if path == axis_pt:
            return t24.SOURCE_ASSISTANT_AXIS_SHA256

        if path == cap_pt:
            return t24.SOURCE_ARTIFACT_SHA256

        return "0" * 64

    monkeypatch.setattr(
        t24,
        "sha256_file",
        fake_sha,
    )

    monkeypatch.setattr(
        t24,
        "load_released_axis",
        lambda _: released,
    )

    monkeypatch.setattr(
        t24,
        "load_source_capping_interventions",
        lambda _: source_caps,
    )

    # Deliberately force the random control to engage ~99% of rows.
    # Source engagement is ~50%, so the real >0.03 gate must abort.
    def deliberately_bad_threshold(
        natural,
        direction,
        quantile,
    ):
        return float(
            np.quantile(
                natural @ direction,
                0.99,
            )
        )

    monkeypatch.setattr(
        t24,
        "calibrate_lower_tail_threshold",
        deliberately_bad_threshold,
    )

    with pytest.raises(
        SystemExit,
        match="random/source mismatch",
    ):
        t24.freeze(
            calibration,
            axis_pt,
            cap_pt,
            tmp_path / "controls.npz",
            tmp_path / "freeze.json",
        )


def test_clean_tree_gate_fails_closed(monkeypatch):
    monkeypatch.setattr(
        t24,
        "source_git_state",
        lambda: ("c" * 40, True),
    )

    with pytest.raises(
        SystemExit,
        match="working tree is dirty",
    ):
        t24.require_clean_source_tree()
