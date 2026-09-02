"""T15/T17 canonical provenance must fail closed on dirty working trees."""
import inspect
import re

import pytest

from tools import run_t15_confirmatory, run_t17_c160_geometry, run_t17_pool_sensitivity


def _default(func, name):
    return inspect.signature(func).parameters[name].default


def test_t15_analyze_defaults_to_fail_closed():
    assert _default(run_t15_confirmatory.analyze, "allow_dirty") is False


def test_t15_dirty_snapshot_refuses_without_opt_in(monkeypatch):
    monkeypatch.setattr(run_t15_confirmatory, "git_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="dirty working tree"):
        run_t15_confirmatory._provenance_dirty_snapshot(False)
    assert run_t15_confirmatory._provenance_dirty_snapshot(True) is True


@pytest.mark.parametrize(
    "module,helper",
    [
        (run_t17_pool_sensitivity, "_dirty_snapshot"),
        (run_t17_c160_geometry, "_dirty_snapshot"),
    ],
)
def test_t17_dirty_snapshots_refuse_without_opt_in(monkeypatch, module, helper):
    monkeypatch.setattr(module.provenance, "git_dirty", lambda: True)
    fn = getattr(module, helper)
    with pytest.raises(RuntimeError, match="dirty working tree"):
        fn(False)
    assert fn(True) is True


@pytest.mark.parametrize(
    "module",
    [run_t15_confirmatory, run_t17_pool_sensitivity, run_t17_c160_geometry],
)
def test_production_stamp_sites_do_not_hardcode_allow_dirty_true(module):
    """Regression guard for issue #56's exact bypass pattern."""
    source = inspect.getsource(module)
    assert not re.search(r"stamp_report\([^)]*allow_dirty=True", source, re.DOTALL)


@pytest.mark.parametrize(
    "module,argv",
    [
        (run_t15_confirmatory, ["analyze", "--help"]),
        (run_t17_pool_sensitivity, ["reduce", "--help"]),
        (run_t17_pool_sensitivity, ["analyze", "--help"]),
        (run_t17_c160_geometry, ["--help"]),
    ],
)
def test_cli_exposes_development_allow_dirty_escape_hatch(module, argv, capsys):
    with pytest.raises(SystemExit):
        module.main(argv)
    assert "--allow-dirty" in capsys.readouterr().out
