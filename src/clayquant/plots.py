"""The result figures.

Four plots, built as Plotly figures so that the GUI shows them interactively and
:func:`save_report` writes the same objects to self-contained HTML:

:func:`plot_mounts`
    The three oriented mounts overlaid - air-dried, heated and glycol-solvated.

:func:`plot_fit`
    The measurement with the fitted curve and the difference.

:func:`plot_components`
    The measurement with every fitted phase drawn separately.

:func:`plot_clay_components`
    The same for the clay minerals alone, with the accompanying minerals left
    out.

Plotly is an optional dependency (``pip install clayquant[gui]``); the curves
behind every figure are also written as CSV by :func:`save_report`, so the data
is available without it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .nnls import FitResult
from .pattern import Pattern
from .quantification import Quantification, quantify

__all__ = [
    "plot_mounts",
    "plot_fit",
    "plot_components",
    "plot_clay_components",
    "save_report",
    "MOUNT_COLORS",
]

MOUNT_COLORS = {"air": "#1f77b4", "glycol": "#2ca02c", "heated": "#d62728"}
MOUNT_LABELS = {"air": "Air-dried", "glycol": "Ethylene glycol", "heated": "Heated (500 C)"}

_PHASE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
)


def _figure():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the result figures need plotly: pip install 'clayquant[gui]'"
        ) from exc
    return go.Figure()


def _style(figure, title: str, y_title: str = "Intensity (counts)"):
    figure.update_layout(
        title=title,
        height=520,
        margin={"l": 70, "r": 20, "t": 50, "b": 55},
        legend={"orientation": "h", "y": -0.18, "yanchor": "top"},
        hovermode="x unified",
        template="plotly_white",
    )
    figure.update_xaxes(title="2θ (degrees, Cu Kα)")
    figure.update_yaxes(title=y_title)
    return figure


def plot_mounts(mounts: dict[str, Pattern], title: str = "Oriented mounts", normalise: bool = False):
    """Overlay the air-dried, heated and glycolated mounts.

    With ``normalise`` each mount is scaled to unit maximum, which makes the
    peak *positions* comparable when the mounts hold different amounts of clay;
    otherwise the counts are shown as measured.
    """
    figure = _figure()
    for key in ("air", "glycol", "heated"):
        pattern = mounts.get(key)
        if pattern is None:
            continue
        intensity = pattern.intensity
        if normalise:
            peak = float(np.max(intensity))
            intensity = intensity / peak if peak > 0 else intensity
        figure.add_scatter(
            x=pattern.two_theta,
            y=intensity,
            name=MOUNT_LABELS[key],
            line={"color": MOUNT_COLORS[key], "width": 1.2},
        )
    return _style(
        figure, title, "Intensity (normalised)" if normalise else "Intensity (counts)"
    )


def plot_fit(result: FitResult, title: str = "Measured and fitted pattern"):
    """The measurement, the fit, and the difference curve below it."""
    figure = _figure()
    inside = result.mask
    two_theta = result.two_theta[inside]
    observed = result.observed[inside]
    calculated = result.calculated[inside]
    offset = 0.15 * float(np.max(observed)) if observed.size else 0.0

    figure.add_scatter(x=two_theta, y=observed, name="measured",
                       line={"color": "#1f77b4", "width": 1})
    figure.add_scatter(x=two_theta, y=calculated, name="fit",
                       line={"color": "#d62728", "width": 1.6})
    figure.add_scatter(x=two_theta, y=observed - calculated - offset, name="difference",
                       line={"color": "#7f7f7f", "width": 1})
    figure.add_hline(y=-offset, line={"color": "#cccccc", "width": 1})
    return _style(
        figure,
        f"{title} — Rwp = {100.0 * result.r_wp:.2f}%",
    )


def _plot_component_set(
    result: FitResult,
    phases: dict[str, np.ndarray],
    title: str,
    stack_note: str = "",
):
    figure = _figure()
    inside = result.mask
    two_theta = result.two_theta[inside]
    figure.add_scatter(x=two_theta, y=result.observed[inside], name="measured",
                       line={"color": "#333333", "width": 1.2})
    total = np.zeros_like(result.observed)
    for index, (phase, curve) in enumerate(
        sorted(phases.items(), key=lambda item: float(np.trapezoid(item[1])), reverse=True)
    ):
        total = total + curve
        figure.add_scatter(
            x=two_theta,
            y=curve[inside],
            name=phase,
            line={"color": _PHASE_COLORS[index % len(_PHASE_COLORS)], "width": 1.2},
        )
    if phases:
        figure.add_scatter(x=two_theta, y=total[inside], name="sum of these phases",
                           line={"color": "#d62728", "width": 1.4, "dash": "dash"})
    return _style(figure, f"{title}{stack_note}")


def plot_components(result: FitResult, title: str = "Measured pattern and fitted phases"):
    """The measurement with every fitted phase drawn separately."""
    return _plot_component_set(result, result.phase_components(), title)


def plot_clay_components(
    result: FitResult,
    quantification: Quantification | None = None,
    title: str = "Measured pattern and clay mineral phases",
):
    """The measurement with only the clay mineral phases drawn separately."""
    quantification = quantification or quantify(result)
    clay_names = {share.phase for share in quantification.clays}
    components = {
        phase: curve
        for phase, curve in result.phase_components().items()
        if phase in clay_names
    }
    note = (
        f" — clays are {100.0 * quantification.clay_total_scattering:.1f}% of the pattern"
        if quantification.clays
        else ""
    )
    return _plot_component_set(result, components, title, note)


def save_report(
    directory: str | Path,
    result: FitResult,
    mounts: dict[str, Pattern] | None = None,
    quantification: Quantification | None = None,
    prefix: str = "clayquant",
) -> list[Path]:
    """Write the four figures, the two quantification tables and the curves.

    Figures go to self-contained HTML, which needs no plotting stack to open and
    keeps the zoom and hover of the interactive versions.  The curves behind the
    component plot are also written as CSV, one column per phase, so the fit can
    be replotted anywhere.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    quantification = quantification or quantify(result)
    written: list[Path] = []

    figures = [
        ("fit", plot_fit(result)),
        ("phases", plot_components(result)),
        ("clay_phases", plot_clay_components(result, quantification)),
    ]
    if mounts:
        figures.insert(0, ("mounts", plot_mounts(mounts)))
    for name, figure in figures:
        path = directory / f"{prefix}_{name}.html"
        figure.write_html(str(path), include_plotlyjs="inline")
        written.append(path)

    written.append(quantification.to_csv(directory / f"{prefix}_all_phases.csv"))
    written.append(
        quantification.to_csv(directory / f"{prefix}_clays_only.csv", clay_basis=True)
    )

    curves = directory / f"{prefix}_curves.csv"
    components = result.phase_components()
    inside = result.mask
    with curves.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["two_theta", "measured", "fit", *components])
        for index in np.flatnonzero(inside):
            writer.writerow(
                [
                    f"{result.two_theta[index]:.4f}",
                    f"{result.observed[index]:.4f}",
                    f"{result.calculated[index]:.4f}",
                    *[f"{curve[index]:.4f}" for curve in components.values()],
                ]
            )
    written.append(curves)
    return written
