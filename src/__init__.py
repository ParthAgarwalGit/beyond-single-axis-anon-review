"""Canonical implementations for the NeurReps Assistant-Axis study (T03).

One active module per concern. New frozen method fields come from
``configs/method_frozen_v4.yaml`` via :mod:`src.config`; no module may
redefine them locally. ``configs/method_frozen.yaml`` is retained only as the
immutable historical provenance file for already-stamped artifacts.
Superseded variants live in ``code/archive/``.

Scope: this package is the canonical CORE LIBRARY. Executable production
adapters (generation runner, model hook runner, judge runner, mixed-effects
analysis entry points) are deliberately not included yet; they follow after
T04 hook validation and judge validation, and T03 stays open until each
active artifact type has one production command path. Production code must
never import from ``code/archive/`` or ``replications/``.
"""
