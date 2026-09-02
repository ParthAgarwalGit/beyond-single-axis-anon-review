#!/usr/bin/env python3
"""Render one T31 figure from its saved source data (no fabrication).

Contract (TODO T31: "Every figure must have saved source data"):
  - a figure renders only from the exact columns its manifest entry declares;
  - if a panel's source CSV is header-only (no rows), rendering is refused - a
    figure is never produced from absent or invented data;
  - each source CSV's header must match the frozen column contract exactly;
  - a multi-panel figure is ready only when EVERY panel is ready.

Every declared chart type is implemented and drawn from the panel's
``render_spec``, which names the column playing each role. There is **no
generic fallback**: an unimplemented chart type raises rather than drawing a
bar chart from whatever column happens to be second and reporting success. That
matters because e.g. F2's second column is ``channel``, a category, and a
fallback bar path would have tried to read it as a number while still returning
``rendered: true``.

Plotting uses matplotlib when available; the column/row contract is enforced
regardless, so this validates figure-readiness on any machine. This tool reads
data and draws; it never writes into the source CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


class UnsupportedChartType(ValueError):
    """The contract declares a chart type the renderer does not implement."""


def load_manifest(root: Path) -> dict[str, Any]:
    return json.loads(
        (root / "design" / "figures_manifest.json").read_text(encoding="utf-8")
    )


def figure_entry(manifest: dict[str, Any], figure_id: str) -> dict[str, Any]:
    entry = next(
        (f for f in manifest["figures"] if f["figure_id"] == figure_id), None
    )
    if entry is None:
        raise ValueError(f"unknown figure {figure_id!r}")
    return entry


def read_source(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def _panel_name(panel: dict[str, Any]) -> str:
    return panel.get("panel_id") or "main"


def check_panel_ready(root: Path, panel: dict[str, Any]) -> dict[str, Any]:
    name = _panel_name(panel)
    src = root / panel["source_csv"]
    if panel["chart_type"] not in CHART_RENDERERS:
        return {"panel": name, "ready": False,
                "reason": f"chart type {panel['chart_type']!r} is not implemented"}
    if not src.exists():
        return {"panel": name, "ready": False, "reason": "source CSV missing"}
    header, rows = read_source(src)
    if header != panel["required_columns"]:
        return {
            "panel": name, "ready": False,
            "reason": f"header {header} does not match contract {panel['required_columns']}",
        }
    if not rows:
        return {
            "panel": name, "ready": False,
            "reason": "source CSV is header-only (AWAITING_RESULTS); refusing to fabricate",
        }
    return {"panel": name, "ready": True, "n_rows": len(rows), "columns": header}


def check_ready(root: Path, figure_id: str) -> dict[str, Any]:
    """Verify a figure is ready to render, without drawing. Returns status."""
    entry = figure_entry(load_manifest(root), figure_id)
    panels = [check_panel_ready(root, p) for p in entry["panels"]]
    ready = all(p["ready"] for p in panels)
    status: dict[str, Any] = {
        "figure_id": figure_id, "ready": ready, "panels": panels,
    }
    if not ready:
        blocking = [p for p in panels if not p["ready"]]
        status["reason"] = "; ".join(
            f"panel {p['panel']}: {p['reason']}" for p in blocking
        )
    else:
        status["n_rows"] = sum(p["n_rows"] for p in panels)
        status["columns"] = panels[0]["columns"] if len(panels) == 1 else None
    return status


# --------------------------------------------------------------------------- #
# Chart implementations. Each reads only the columns its render_spec names.
# --------------------------------------------------------------------------- #

def _floats(rows, column):
    try:
        return [float(r[column]) for r in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} is not numeric: {exc}") from exc


def _errorbars(rows, spec, values):
    if "err_low" not in spec or "err_high" not in spec:
        return None
    low = _floats(rows, spec["err_low"])
    high = _floats(rows, spec["err_high"])
    return [
        [v - lo for v, lo in zip(values, low)],
        [hi - v for v, hi in zip(values, high)],
    ]


def _render_scatter(ax, rows, spec, panel):
    xs = _floats(rows, spec["x"])
    ys = _floats(rows, spec["y"])
    ax.scatter(xs, ys, s=12)
    ax.set_xlabel(spec["x"])
    ax.set_ylabel(spec["y"])


def _render_bar(ax, rows, spec, panel):
    labels = [r[spec["category"]] for r in rows]
    values = _floats(rows, spec["value"])
    ax.bar(range(len(values)), values, yerr=_errorbars(rows, spec, values), capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(spec["value"])


def _render_grouped_bar(ax, rows, spec, panel):
    groups, categories = [], []
    for r in rows:
        if r[spec["group"]] not in groups:
            groups.append(r[spec["group"]])
        if r[spec["category"]] not in categories:
            categories.append(r[spec["category"]])
    width = 0.8 / max(1, len(groups))
    for gi, group in enumerate(groups):
        sub = [r for r in rows if r[spec["group"]] == group]
        by_cat = {r[spec["category"]]: r for r in sub}
        values, err_low, err_high, positions = [], [], [], []
        for ci, cat in enumerate(categories):
            if cat not in by_cat:
                continue
            r = by_cat[cat]
            values.append(float(r[spec["value"]]))
            positions.append(ci - 0.4 + width * (gi + 0.5))
            if "err_low" in spec:
                err_low.append(float(r[spec["err_low"]]))
                err_high.append(float(r[spec["err_high"]]))
        yerr = None
        if err_low:
            yerr = [[v - lo for v, lo in zip(values, err_low)],
                    [hi - v for v, hi in zip(values, err_high)]]
        ax.bar(positions, values, width=width, label=str(group), yerr=yerr, capsize=3)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel(spec["value"])
    ax.legend(title=spec["group"], fontsize="small")


def _render_dose_response(ax, rows, spec, panel):
    series_names = []
    for r in rows:
        if r[spec["series"]] not in series_names:
            series_names.append(r[spec["series"]])
    for name in series_names:
        sub = sorted(
            (r for r in rows if r[spec["series"]] == name),
            key=lambda r: float(r[spec["x"]]),
        )
        xs = _floats(sub, spec["x"])
        ys = _floats(sub, spec["y"])
        yerr = _errorbars(sub, spec, ys)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=str(name))
    ax.set_xlabel(spec["x"])
    ax.set_ylabel(spec["y"])
    ax.legend(title=spec["series"], fontsize="small")


def _render_heatmap(ax, rows, spec, panel):
    row_keys, col_keys = [], []
    for r in rows:
        if r[spec["row"]] not in row_keys:
            row_keys.append(r[spec["row"]])
        if r[spec["col"]] not in col_keys:
            col_keys.append(r[spec["col"]])
    grid = [[0.0] * len(col_keys) for _ in row_keys]
    for r in rows:
        grid[row_keys.index(r[spec["row"]])][col_keys.index(r[spec["col"]])] = float(
            r[spec["value"]]
        )
    im = ax.imshow(grid, aspect="auto")
    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(col_keys, rotation=45, ha="right")
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_keys)
    ax.set_xlabel(spec["col"])
    ax.set_ylabel(spec["row"])
    ax.figure.colorbar(im, ax=ax, label=spec["value"])


def _render_dag(ax, rows, spec, panel):
    """Layered DAG drawn from declared dependencies, not a hand-placed diagram."""
    nodes = [r[spec["node"]] for r in rows]
    deps = {
        r[spec["node"]]: [
            d.strip() for d in (r[spec["depends_on"]] or "").split(";") if d.strip()
        ]
        for r in rows
    }
    depth: dict[str, int] = {}

    def resolve(node, seen=()):
        if node in depth:
            return depth[node]
        if node in seen or node not in deps:
            return 0
        d = 1 + max((resolve(p, seen + (node,)) for p in deps[node]), default=-1)
        depth[node] = d
        return d

    for node in nodes:
        resolve(node)
    layers: dict[int, list[str]] = {}
    for node in nodes:
        layers.setdefault(depth.get(node, 0), []).append(node)
    pos = {}
    for d, members in sorted(layers.items()):
        for i, node in enumerate(members):
            pos[node] = (d, i - (len(members) - 1) / 2)
    label_by_node = {r[spec["node"]]: r[spec["label"]] for r in rows}
    for node, (x, y) in pos.items():
        ax.annotate(label_by_node.get(node, node), (x, y), ha="center",
                    va="center", fontsize="x-small",
                    bbox={"boxstyle": "round", "fc": "white", "ec": "black"})
    for node, parents in deps.items():
        for parent in parents:
            if parent in pos and node in pos:
                ax.annotate(
                    "", xy=pos[node], xytext=pos[parent],
                    arrowprops={"arrowstyle": "->", "alpha": 0.5},
                )
    if pos:
        ax.set_xlim(min(p[0] for p in pos.values()) - 0.6,
                    max(p[0] for p in pos.values()) + 0.6)
        ax.set_ylim(min(p[1] for p in pos.values()) - 0.8,
                    max(p[1] for p in pos.values()) + 0.8)
    ax.axis("off")


CHART_RENDERERS = {
    "scatter": _render_scatter,
    "bar": _render_bar,
    "grouped_bar": _render_grouped_bar,
    "dose_response": _render_dose_response,
    "heatmap": _render_heatmap,
    "dag": _render_dag,
}


def render(root: Path, figure_id: str, out_path: Path) -> dict[str, Any]:
    entry = figure_entry(load_manifest(root), figure_id)

    # Fail closed on an unimplemented chart type BEFORE reporting any success.
    for panel in entry["panels"]:
        if panel["chart_type"] not in CHART_RENDERERS:
            raise UnsupportedChartType(
                f"{figure_id}[{_panel_name(panel)}]: chart type "
                f"{panel['chart_type']!r} is not implemented; refusing to draw a "
                f"placeholder. Implemented types: {sorted(CHART_RENDERERS)}"
            )

    status = check_ready(root, figure_id)
    if not status["ready"]:
        raise SystemExit(f"cannot render {figure_id}: {status['reason']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        return {**status, "rendered": False,
                "reason": "matplotlib unavailable; contract verified but not drawn"}

    import matplotlib.pyplot as plt

    panels = entry["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ax, panel in zip(axes, panels):
        _, rows = read_source(root / panel["source_csv"])
        CHART_RENDERERS[panel["chart_type"]](ax, rows, panel["render_spec"], panel)
        ax.set_title(
            entry["title"] if len(panels) == 1
            else f"{entry['title']} - {_panel_name(panel)}",
            fontsize="medium",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {**status, "rendered": True, "output": out_path.name}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--figure", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.check_only:
        print(json.dumps(check_ready(root, args.figure), indent=2))
        return 0
    out = root / (args.out or f"paper/figures/{args.figure}.png")
    print(json.dumps(render(root, args.figure, out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
