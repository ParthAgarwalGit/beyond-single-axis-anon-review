"""T26/T27 production stamps must fail closed on a dirty working tree.

``src/provenance.stamp_report`` documents ``allow_dirty=True`` as development
only, but both production stamp sites passed it unconditionally. A dirty stamp
means "the approved code drew this sample" is asserted rather than provable —
and for T26 that matters twice over, because the sample manifest is the
authority a T27 gating run binds itself to.
"""
import inspect
import re

import pytest

import src.provenance as provenance
from tools import run_t26_sample, run_t27_causal_judge_validation


def _default(func, name):
    return inspect.signature(func).parameters[name].default


def test_t26_draw_defaults_to_fail_closed():
    assert _default(run_t26_sample.draw, "allow_dirty") is False


def test_t27_run_defaults_to_fail_closed():
    assert _default(run_t27_causal_judge_validation.run, "allow_dirty") is False


def test_stamp_report_raises_on_dirty_tree(monkeypatch):
    monkeypatch.setattr(provenance, "git_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="dirty working tree"):
        provenance.stamp_report({"task": "probe"})


def test_stamp_report_allows_dirty_only_when_opted_in(monkeypatch):
    monkeypatch.setattr(provenance, "git_dirty", lambda: True)
    stamped = provenance.stamp_report({"task": "probe"}, allow_dirty=True)
    assert stamped["source_git_dirty"] is True


@pytest.mark.parametrize("module", [run_t26_sample, run_t27_causal_judge_validation])
def test_production_stamp_sites_do_not_hardcode_allow_dirty(module):
    """The unconditional stamp_report(..., allow_dirty=True) call must not come back.

    Matches the actual defect pattern - a stamp_report(...) call whose closing
    allow_dirty argument is the literal True - rather than any occurrence of
    the substring "allow_dirty=True", which also appears in this module's own
    --allow-dirty help text and error messages.
    """
    source = inspect.getsource(module)
    assert not re.search(r"stamp_report\([^)]*allow_dirty=True", source, re.DOTALL)


@pytest.mark.parametrize(
    "module,argv",
    [
        (run_t26_sample, ["draw", "--help"]),
        (run_t27_causal_judge_validation, ["run", "--help"]),
    ],
)
def test_cli_exposes_the_development_escape_hatch(module, argv, capsys):
    with pytest.raises(SystemExit):
        module.main(argv)
    assert "--allow-dirty" in capsys.readouterr().out
