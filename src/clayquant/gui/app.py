"""The ClayQuant GUI.

A local Dash application that walks the analysis in the order it is done at the
bench:

1. load the three oriented mounts (air-dried, glycol-solvated, heated);
2. find the 2theta zero error on the quartz 100 reflection, with a slider that
   steps in 0.01 deg - half the usual measurement step;
3. choose and tune a background, with the ``1/x`` term accumulable on top of
   any of the other components;
4. search for the accompanying ("main") minerals and confirm which of them to
   fit alongside the clays;
5. compare air-dried with heated to find kaolinite by its collapse;
6. compare air-dried with glycolated to find expandable clay by its swelling;
7. fit the glycolated mount against the calculated library by non-negative least
   squares, and read the quantification on a clay-only and an all-phase basis,
   with the three mounts, the fit, and the phases separated, as four plots.

A last tab is the pattern simulator, which drives the interstratification model
directly for any composition, stacking order and crystallite thickness.

Start it with ``clayquant-gui`` or ``python -m clayquant.gui.app``.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate

from ..background import BackgroundModel
from ..bern import is_clay_phase
from ..detection import detect_phases, screen_phases
from ..calibration import (
    QUARTZ_100_D,
    estimate_zero_error,
    reference_two_theta,
    zero_error_profile,
)
from ..diagnostics import (
    EXPANDABLE_001_WINDOW,
    KAOLINITE_001_WINDOW,
    kaolinite_collapse,
    smectite_swelling,
)
from ..emission import CU_KA_5LINE
from ..library import (
    CHLORITE_SMECTITE_FRACTIONS,
    ILLITE_SMECTITE_FRACTIONS,
    PREFERRED_ORIENTATIONS,
    PatternLibrary,
    build_library,
)
from ..mixed_layer import MixedLayerStack, lognormal_csds, markov_transition, random_transition
from ..models import CIF_SOURCES, available_phases, eg_smectite_layer, load_crystal, load_layer
from ..nnls import nnls_fit
from ..optics import Divergence
from ..pairing import find_siblings, sample_key
from ..pattern import Instrument, mixed_layer_pattern, powder_pattern, two_theta_grid
from ..plots import (
    plot_clay_components,
    plot_components,
    plot_fit,
    plot_mounts,
    save_report,
)
from ..profile import PeakShape
from ..quantification import quantify
from .state import MOUNT_LABELS, MOUNTS, STATE

COLORS = {"air": "#1f77b4", "glycol": "#2ca02c", "heated": "#d62728"}
_PHASE_PALETTE = (
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#17becf",
    "#1f77b4",
    "#2ca02c",
)
GRAPH_HEIGHT = 480


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font={"size": 14})
    figure.update_layout(
        height=GRAPH_HEIGHT,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 60, "r": 20, "t": 30, "b": 50},
    )
    return figure


def style_axes(figure: go.Figure, y_title: str = "Intensity") -> go.Figure:
    figure.update_layout(
        height=GRAPH_HEIGHT,
        margin={"l": 60, "r": 20, "t": 30, "b": 50},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom"},
        hovermode="x unified",
    )
    figure.update_xaxes(title="2θ (degrees, Cu Kα)")
    figure.update_yaxes(title=y_title)
    return figure


def background_model_from_controls(
    kind: str, degree: int, decay: float, use_inverse: bool, inverse_offset: float
) -> BackgroundModel:
    """Build a :class:`BackgroundModel` from the control values."""
    return BackgroundModel(
        polynomial_degree=degree if kind == "polynomial" else None,
        chebyshev_degree=degree if kind == "chebyshev" else None,
        exponential_decay=decay if kind == "exponential" else None,
        inverse=bool(use_inverse),
        inverse_offset=inverse_offset,
    )


def background_report(pattern, fit, background, model) -> str:
    """Describe the fitted background in the terms the operator has to judge it by.

    A single R\u00b2 against the stripped estimate is close to 1 for almost any
    model - the estimate is smooth and the fit has several free terms - so it
    reads as success even when the background is a thousand counts below the
    measurement at the low-angle end, which is where the choice actually
    matters.  What is reported instead is the size of the disagreement in
    counts, separately for the low-angle end, and what the model leaves behind
    there as signal.
    """
    two_theta = pattern.two_theta
    residual = fit.target - background
    overall = float(np.sqrt(np.mean(residual ** 2)))
    low = two_theta <= two_theta[0] + 1.0
    low_rms = float(np.sqrt(np.mean(residual[low] ** 2))) if low.any() else float("nan")

    start = float(two_theta[0])
    measured_start = float(pattern.intensity[0])
    model_start = float(background[0])
    left = max(0.0, measured_start - model_start)
    share = 100.0 * left / measured_start if measured_start > 0 else 0.0

    text = (
        f"{' + '.join(model.components)}, {model.n_terms} terms. "
        f"Follows the peak-stripped estimate to {overall:.0f} counts RMS "
        f"({low_rms:.0f} counts over the lowest 1\u00b0). "
        f"At {start:.2f}\u00b0 the model reads {model_start:.0f} of {measured_start:.0f} "
        f"measured counts, leaving {left:.0f} ({share:.0f}%) as signal."
    )
    if share > 25.0:
        text += (
            " Most of the low-angle counts are therefore kept as signal; a narrower "
            "stripping width keeps less, at the cost of a higher background under "
            "the peaks."
        )
    above = float(np.mean(background > pattern.intensity))
    if above > 0.02:
        text += f" Warning: the model sits above the data at {100.0 * above:.0f}% of points."
    return text


def gui_instrument(
    specimen_length: float = 20.0,
    goniometer_radius: float = 280.0,
    divergence_slit: float = 0.5,
    size_ab: float = 400.0,
) -> Instrument:
    return Instrument(
        emission=CU_KA_5LINE,
        peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=size_ab),
        lp_mode="powder",
        divergence=Divergence(
            specimen_length=specimen_length,
            goniometer_radius=goniometer_radius,
            divergence=divergence_slit,
        ),
    )


def _with_main_minerals(library):
    """Return the library with the confirmed main minerals appended.

    Each selected phase is calculated on the library's own grid and added as one
    entry.  They are deliberately calculated without preferred orientation:
    quartz, feldspar and carbonate grains that survive the separation are
    scattered through the mount rather than plated onto the glass, so there is no
    reason to apply the clay texture to them.
    """
    if not STATE.selected_main or not STATE.phase_database:
        return library, 0

    from ..library import PatternLibrary

    extended = PatternLibrary(
        two_theta=library.two_theta,
        entries=list(library.entries),
        metadata=dict(library.metadata),
    )
    instrument = gui_instrument()
    added = 0
    for name in STATE.selected_main:
        crystal = STATE.phase_database.get(name)
        if crystal is None:
            continue
        pattern = powder_pattern(
            crystal, library.two_theta, instrument, r_march_dollase=1.0, name=name
        )
        if float(np.max(pattern.intensity)) <= 0.0:
            continue
        extended.add(pattern, phase=name, march_dollase=1.0)
        added += 1
    return extended, added


def _quantification_table(quantification, result) -> html.Div:
    """The clay-basis and all-phase tables, side by side."""

    def build(rows, caption: str, note: str) -> html.Div:
        return html.Div(
            [
                html.H4(caption, style={"marginBottom": "4px"}),
                html.Div(note, style={"fontSize": "0.78rem", "color": "#555",
                                      "marginBottom": "6px"}),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Phase"),
                                    html.Th("Group"),
                                    html.Th("Scattering %"),
                                    html.Th("Amplitude %"),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(row["phase"]),
                                        html.Td(row["group"]),
                                        html.Td(f"{row['scattering_percent']:.2f}"),
                                        html.Td(f"{row['amplitude_percent']:.2f}"),
                                    ]
                                )
                                for row in rows
                            ]
                        ),
                    ],
                    style={"fontSize": "0.85rem", "borderCollapse": "collapse"},
                ),
            ],
            style={"flex": "1", "minWidth": "0"},
        )

    clay_note = (
        f"Accompanying minerals removed and the clays renormalised to 100%. "
        f"The clay minerals are {100.0 * quantification.clay_total_scattering:.1f}% "
        f"of the whole pattern."
    )
    return html.Div(
        [
            html.Div(
                [
                    build(quantification.table(clay_basis=True), "Clay minerals", clay_note),
                    build(
                        quantification.table(),
                        "All fitted phases",
                        "Shares of the whole pattern, clays and accompanying minerals together.",
                    ),
                ],
                style={"display": "flex", "gap": "24px"},
            ),
            html.H4("Contributing patterns"),
            html.Table(
                [
                    html.Thead(
                        html.Tr([html.Th(text) for text in
                                 ["Pattern", "Coefficient", "Scattering %"]])
                    ),
                    html.Tbody(
                        [
                            html.Tr([html.Td(name), html.Td(f"{coefficient:.4g}"),
                                     html.Td(f"{100.0 * share:.2f}")])
                            for name, coefficient, share in result.active(threshold=1e-3)
                        ]
                    ),
                ],
                style={"fontSize": "0.85rem"},
            ),
            html.P(result.metrics_note, style={"fontSize": "0.78rem", "color": "#555"}),
        ]
    )


def error_message(exc: Exception) -> html.Div:
    return html.Div(
        [html.B("Could not complete that step. "), html.Span(str(exc))],
        style={"color": "#b00020", "whiteSpace": "pre-wrap"},
    )


def label(text: str) -> html.Label:
    return html.Label(text, style={"fontWeight": 600, "fontSize": "0.85rem"})


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

CONTROL_PANEL = {
    "width": "320px",
    "minWidth": "320px",
    "padding": "12px",
    "background": "#f7f7f9",
    "borderRadius": "6px",
    "fontSize": "0.9rem",
}
ROW = {"display": "flex", "gap": "16px", "alignItems": "flex-start"}
GRAPH_BOX = {"flex": "1", "minWidth": "0"}


def load_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    label("Data folder"),
                    dcc.Input(
                        id="directory",
                        type="text",
                        placeholder=r"e.g. C:\Users\you\Documents\HM",
                        style={"width": "100%"},
                        debounce=True,
                    ),
                    html.Button("Scan folder", id="scan", n_clicks=0, style={"marginTop": "8px"}),
                    html.Hr(),
                    *[
                        html.Div(
                            [
                                label(MOUNT_LABELS[mount]),
                                dcc.Dropdown(id=f"file-{mount}", options=[], placeholder="file"),
                            ],
                            style={"marginBottom": "8px"},
                        )
                        for mount in MOUNTS
                    ],
                    html.Div(id="file-pairing",
                             style={"fontSize": "11px", "color": "#666", "marginBottom": "8px"}),
                    html.Button("Load patterns", id="load", n_clicks=0),
                    html.Div(id="load-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div([dcc.Graph(id="load-graph", figure=empty_figure("Load patterns to begin"))],
                     style=GRAPH_BOX),
        ],
        style=ROW,
    )


def zero_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    label("Mount"),
                    dcc.RadioItems(
                        id="zero-mount",
                        options=[{"label": MOUNT_LABELS[m], "value": m} for m in MOUNTS],
                        value="air",
                    ),
                    html.Hr(),
                    label("Calibration reflection"),
                    dcc.Dropdown(
                        id="zero-reference",
                        options=[
                            {"label": "Quartz 100, 4.2551 Å (20.86°)", "value": "quartz100"},
                            {"label": "Quartz 101, 3.3435 Å (26.64°)", "value": "quartz101"},
                        ],
                        value="quartz100",
                        clearable=False,
                    ),
                    html.Div(
                        [
                            label("Zero error (°2θ, steps of 0.01)"),
                            dcc.Slider(
                                id="zero-slider",
                                min=-0.5,
                                max=0.5,
                                step=0.01,
                                value=0.0,
                                marks={v: f"{v:g}" for v in (-0.5, -0.25, 0, 0.25, 0.5)},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        style={"marginTop": "12px"},
                    ),
                    html.Button("Estimate automatically", id="zero-auto", n_clicks=0),
                    html.Div(id="zero-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div([dcc.Graph(id="zero-graph", figure=empty_figure("Load patterns first"))],
                     style=GRAPH_BOX),
        ],
        style=ROW,
    )


def background_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    label("Mount"),
                    dcc.RadioItems(
                        id="bg-mount",
                        options=[{"label": MOUNT_LABELS[m], "value": m} for m in MOUNTS],
                        value="air",
                    ),
                    html.Hr(),
                    label("Main component"),
                    dcc.Dropdown(
                        id="bg-kind",
                        options=[
                            {"label": "Polynomial", "value": "polynomial"},
                            {"label": "Chebyshev", "value": "chebyshev"},
                            {"label": "Exponential", "value": "exponential"},
                            {"label": "None (1/x only)", "value": "none"},
                        ],
                        value="chebyshev",
                        clearable=False,
                    ),
                    label("Degree"),
                    dcc.Slider(id="bg-degree", min=0, max=12, step=1, value=4,
                               marks={0: "0", 4: "4", 8: "8", 12: "12"},
                               tooltip={"placement": "bottom"}),
                    label("Exponential decay (1/°)"),
                    dcc.Slider(id="bg-decay", min=0.02, max=1.0, step=0.02, value=0.2,
                               marks={0.02: "0.02", 0.5: "0.5", 1.0: "1"},
                               tooltip={"placement": "bottom"}),
                    html.Hr(),
                    dcc.Checklist(
                        id="bg-inverse",
                        options=[{"label": "  Accumulate 1/x term", "value": "on"}],
                        value=["on"],
                    ),
                    label("1/x offset (°2θ)"),
                    dcc.Slider(id="bg-offset", min=0.0, max=10.0, step=0.25, value=1.0,
                               marks={0: "0", 5: "5", 10: "10"},
                               tooltip={"placement": "bottom"}),
                    html.Hr(),
                    label("Peak-stripping width (°2θ)"),
                    dcc.Slider(id="bg-snip", min=0.5, max=10.0, step=0.5, value=4.0,
                               marks={0.5: "0.5", 5: "5", 10: "10"},
                               tooltip={"placement": "bottom"}),
                    html.Div(
                        "The blue dashed curve is what the model is fitted to. "
                        "Narrow settings leave peak wings in the background and eat "
                        "into the reflections; wide ones strip the direct-beam tail "
                        "as well. On the test measurements the fit improves up to "
                        "about 4\u00b0 and is flat beyond it, which is the default.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
                    html.Button("Apply to this mount", id="bg-apply", n_clicks=0,
                                style={"marginTop": "10px"}),
                    html.Button("Apply to all mounts", id="bg-apply-all", n_clicks=0,
                                style={"marginTop": "6px"}),
                    html.Div(id="bg-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div([dcc.Graph(id="bg-graph", figure=empty_figure("Load patterns first"))],
                     style=GRAPH_BOX),
        ],
        style=ROW,
    )


def kaolinite_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.P(
                        "Kaolinite dehydroxylates on heating and its 7.15 Å reflection "
                        "disappears; chlorite 002 at 7.13 Å survives. The intensity lost "
                        "measures kaolinite, what remains measures chlorite.",
                        style={"fontSize": "0.82rem", "color": "#444"},
                    ),
                    label("7.15 Å window (°2θ)"),
                    dcc.RangeSlider(id="kao-window", min=10.0, max=15.0, step=0.1,
                                    value=list(KAOLINITE_001_WINDOW),
                                    tooltip={"placement": "bottom", "always_visible": True}),
                    label("Scaling reference"),
                    dcc.Dropdown(
                        id="kao-reference",
                        options=[
                            {"label": "Quartz 100 (20.86°)", "value": "quartz"},
                            {"label": "Illite 002 (17.7°)", "value": "illite"},
                            {"label": "Manual factor", "value": "manual"},
                        ],
                        value="quartz",
                        clearable=False,
                    ),
                    label("Manual scale factor"),
                    dcc.Input(id="kao-scale", type="number", value=1.0, step=0.05,
                              style={"width": "100%"}),
                    html.Button("Analyse", id="kao-run", n_clicks=0, style={"marginTop": "10px"}),
                    html.Div(id="kao-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div([dcc.Graph(id="kao-graph", figure=empty_figure("Load the air-dried and heated mounts"))],
                     style=GRAPH_BOX),
        ],
        style=ROW,
    )


def smectite_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.P(
                        "Smectitic interlayers take up glycol and expand to about 16.9 Å, "
                        "so the low-angle reflection moves to higher d. The shift and the "
                        "intensity arriving there measure expandable content.",
                        style={"fontSize": "0.82rem", "color": "#444"},
                    ),
                    label("Search window (°2θ)"),
                    dcc.RangeSlider(id="sme-window", min=2.0, max=14.0, step=0.1,
                                    value=list(EXPANDABLE_001_WINDOW),
                                    tooltip={"placement": "bottom", "always_visible": True}),
                    label("Scaling reference"),
                    dcc.Dropdown(
                        id="sme-reference",
                        options=[
                            {"label": "Quartz 100 (20.86°)", "value": "quartz"},
                            {"label": "Manual factor", "value": "manual"},
                        ],
                        value="quartz",
                        clearable=False,
                    ),
                    label("Manual scale factor"),
                    dcc.Input(id="sme-scale", type="number", value=1.0, step=0.05,
                              style={"width": "100%"}),
                    html.Button("Analyse", id="sme-run", n_clicks=0, style={"marginTop": "10px"}),
                    html.Div(id="sme-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div([dcc.Graph(id="sme-graph", figure=empty_figure("Load the air-dried and glycolated mounts"))],
                     style=GRAPH_BOX),
        ],
        style=ROW,
    )


def main_minerals_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.P(
                        "A clay separate carries quartz, feldspars and carbonates through "
                        "the separation. This searches the measurement for them so they can "
                        "be fitted alongside the clays and then normalised out of the clay "
                        "percentages.",
                        style={"fontSize": "0.82rem", "color": "#444"},
                    ),
                    label("Phase database (.json)"),
                    dcc.Input(id="db-path", type="text", value="structures/phases.json",
                              style={"width": "100%"}, debounce=True),
                    html.Button("Load database", id="db-load", n_clicks=0,
                                style={"marginTop": "6px"}),
                    html.Div(id="db-status", style={"marginTop": "8px", "fontSize": "0.82rem"}),
                    html.Hr(),
                    label("Mount to search"),
                    dcc.RadioItems(
                        id="detect-mount",
                        options=[{"label": MOUNT_LABELS[m], "value": m} for m in MOUNTS],
                        value="air",
                    ),
                    label("Unit cell allowance (%)"),
                    dcc.Slider(id="detect-allowance", min=0.0, max=5.0, step=0.25, value=2.0,
                               marks={0: "0", 2: "2", 5: "5"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    label("Minimum share (‰) or score (%)"),
                    dcc.Slider(id="detect-score", min=5, max=95, step=5, value=10,
                               marks={5: "5", 50: "50", 95: "95"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    label("Minimum peak S/N"),
                    dcc.Slider(id="detect-snr", min=2, max=20, step=1, value=5,
                               marks={2: "2", 10: "10", 20: "20"},
                               tooltip={"placement": "bottom"}),
                    label("Search range (°2θ)"),
                    dcc.RangeSlider(id="detect-range", min=2.0, max=70.0, step=0.5,
                                    value=[4.0, 40.0],
                                    tooltip={"placement": "bottom", "always_visible": True}),
                    html.Button("Search for main minerals", id="detect-run", n_clicks=0,
                                style={"marginTop": "10px"}),
                    html.Div(id="detect-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div(
                [
                    html.Div(id="detect-dialog"),
                    dcc.Graph(id="detect-graph",
                              figure=empty_figure("Load a phase database and search")),
                    html.Div(id="detect-selection", style={"marginTop": "8px"}),
                ],
                style=GRAPH_BOX,
            ),
        ],
        style=ROW,
    )


def fit_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    label("Library file (.npz)"),
                    dcc.Input(id="lib-path", type="text", value="library/clayquant_library.npz",
                              style={"width": "100%"}, debounce=True),
                    html.Button("Load library", id="lib-load", n_clicks=0,
                                style={"marginTop": "6px"}),
                    html.Button("Build library now", id="lib-build", n_clicks=0,
                                style={"marginTop": "6px"}),
                    html.Div(id="lib-status", style={"marginTop": "8px", "fontSize": "0.82rem"}),
                    html.Hr(),
                    label("Mount to fit"),
                    dcc.RadioItems(
                        id="fit-mount",
                        options=[{"label": MOUNT_LABELS[m], "value": m} for m in MOUNTS],
                        value="glycol",
                    ),
                    label("Fit range (°2θ)"),
                    dcc.RangeSlider(id="fit-range", min=2.0, max=45.0, step=0.5, value=[4.0, 34.0],
                                    tooltip={"placement": "bottom", "always_visible": True}),
                    label("Restrict orientation parameters"),
                    dcc.Dropdown(
                        id="fit-orientations",
                        options=[{"label": f"PO = {r:g}", "value": r} for r in PREFERRED_ORIENTATIONS],
                        value=[],
                        multi=True,
                        placeholder="all",
                    ),
                    dcc.Checklist(
                        id="fit-subtract",
                        options=[{"label": "  Subtract the fitted background", "value": "on"}],
                        value=["on"],
                    ),
                    html.Button("Run NNLS fit", id="fit-run", n_clicks=0,
                                style={"marginTop": "10px"}),
                    html.Div(id="fit-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div(
                [
                    dcc.Tabs(
                        id="result-tabs",
                        value="fit",
                        children=[
                            dcc.Tab(
                                label="Fit",
                                value="fit",
                                children=[
                                    dcc.Graph(
                                        id="fit-graph",
                                        figure=empty_figure("Load a library and a mount"),
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="All phases",
                                value="phases",
                                children=[dcc.Graph(id="fit-components-graph",
                                                    figure=empty_figure("Run the fit"))],
                            ),
                            dcc.Tab(
                                label="Clay phases",
                                value="clays",
                                children=[dcc.Graph(id="fit-clay-graph",
                                                    figure=empty_figure("Run the fit"))],
                            ),
                            dcc.Tab(
                                label="Three mounts",
                                value="mounts",
                                children=[dcc.Graph(id="fit-mounts-graph",
                                                    figure=empty_figure("Load the mounts"))],
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            label("Export folder"),
                            dcc.Input(id="export-path", type="text", value="results",
                                      style={"width": "60%"}),
                            html.Button("Export tables and plots", id="export-run", n_clicks=0,
                                        style={"marginLeft": "8px"}),
                            html.Div(id="export-status", style={"marginTop": "6px",
                                                                "fontSize": "0.82rem"}),
                        ],
                        style={"marginTop": "10px"},
                    ),
                    html.Div(id="fit-table"),
                ],
                style=GRAPH_BOX,
            ),
        ],
        style=ROW,
    )


def simulator_tab() -> html.Div:
    phase_options = [{"label": key, "value": key} for key in CIF_SOURCES]
    return html.Div(
        [
            html.Div(
                [
                    label("Model"),
                    dcc.Dropdown(
                        id="sim-model",
                        options=[
                            {"label": "Discrete phase (hkl pattern)", "value": "phase"},
                            {"label": "Illite / smectite", "value": "IS"},
                            {"label": "Chlorite / smectite", "value": "CS"},
                            {"label": "Pure glycolated smectite", "value": "S"},
                        ],
                        value="IS",
                        clearable=False,
                    ),
                    label("Phase / host"),
                    dcc.Dropdown(id="sim-phase", options=phase_options, value="illite",
                                 clearable=False),
                    label("Host layer fraction"),
                    dcc.Slider(id="sim-fraction", min=0.0, max=1.0, step=0.01, value=0.8,
                               marks={0: "0", 0.5: "0.5", 1: "1"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    label("Junction probability P(host→smectite)  — 0 uses random stacking"),
                    dcc.Slider(id="sim-pab", min=0.0, max=1.0, step=0.01, value=0.0,
                               marks={0: "random", 1: "1"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Hr(),
                    label("Mean crystallite thickness (layers)"),
                    dcc.Slider(id="sim-csds", min=2, max=60, step=1, value=10,
                               marks={2: "2", 20: "20", 40: "40", 60: "60"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    label("CSDS width β (in ln N)"),
                    dcc.Slider(id="sim-beta", min=0.0, max=1.0, step=0.05, value=0.35,
                               marks={0: "0", 0.5: "0.5", 1: "1"},
                               tooltip={"placement": "bottom"}),
                    label("March-Dollase orientation r"),
                    dcc.Slider(id="sim-po", min=0.1, max=1.0, step=0.05, value=0.2,
                               marks={0.1: "0.1", 0.5: "0.5", 1.0: "1 (random)"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    label("Glycolated smectite d(001) (Å)"),
                    dcc.Slider(id="sim-dsmectite", min=15.0, max=18.0, step=0.02, value=16.86,
                               marks={15: "15", 16.86: "16.86", 18: "18"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    label("2θ range (degrees)"),
                    dcc.RangeSlider(id="sim-range", min=2.0, max=70.0, step=1.0, value=[2.0, 40.0],
                                    tooltip={"placement": "bottom"}),
                    dcc.Checklist(
                        id="sim-options",
                        options=[
                            {"label": "  Include non-basal (hkl) reflections", "value": "hkl"},
                            {"label": "  Overlay the pure end members", "value": "ends"},
                        ],
                        value=["hkl", "ends"],
                    ),
                    html.Div(id="sim-status", style={"marginTop": "10px", "fontSize": "0.82rem"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div([dcc.Graph(id="sim-graph", figure=empty_figure("Adjust the controls"))],
                     style=GRAPH_BOX),
        ],
        style=ROW,
    )


def build_layout() -> html.Div:
    missing = [key for key, present in available_phases().items() if not present]
    banner = []
    if missing:
        banner = [
            html.Div(
                [
                    html.B("Structure files missing: "),
                    html.Span(
                        f"{', '.join(missing)}. ClayQuant does not redistribute ICSD data. "
                        f"Export the CIFs listed in the README and put them in a 'structures' "
                        f"folder, or set $CLAYQUANT_STRUCTURE_DIR. Simulation and library "
                        f"building need them; loading, calibration, background and the "
                        f"diagnostics do not."
                    ),
                ],
                style={
                    "background": "#fff4e5",
                    "border": "1px solid #ffb74d",
                    "padding": "10px",
                    "borderRadius": "6px",
                    "marginBottom": "12px",
                    "fontSize": "0.85rem",
                },
            )
        ]
    return html.Div(
        [
            html.H2("ClayQuant", style={"marginBottom": "0"}),
            html.P(
                "Quantification of clay mineral assemblages from oriented mounts",
                style={"marginTop": "2px", "color": "#666"},
            ),
            *banner,
            dcc.Tabs(
                id="tabs",
                value="load",
                children=[
                    dcc.Tab(label="1. Load", value="load", children=load_tab()),
                    dcc.Tab(label="2. Zero error", value="zero", children=zero_tab()),
                    dcc.Tab(label="3. Background", value="background", children=background_tab()),
                    dcc.Tab(label="4. Main minerals", value="main", children=main_minerals_tab()),
                    dcc.Tab(label="5. Kaolinite", value="kaolinite", children=kaolinite_tab()),
                    dcc.Tab(label="6. Expandable", value="smectite", children=smectite_tab()),
                    dcc.Tab(label="7. Fit and results", value="fit", children=fit_tab()),
                    dcc.Tab(label="Simulator", value="simulator", children=simulator_tab()),
                ],
            ),
        ],
        style={"fontFamily": "system-ui, sans-serif", "margin": "18px", "maxWidth": "1500px"},
    )


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #


def register_callbacks(app: Dash) -> None:
    @app.callback(
        [Output(f"file-{mount}", "options") for mount in MOUNTS],
        Output("load-status", "children", allow_duplicate=True),
        Input("scan", "n_clicks"),
        State("directory", "value"),
        prevent_initial_call=True,
    )
    def scan_directory(_clicks, directory):
        if not directory:
            return [], [], [], error_message(ValueError("Type the folder holding the scans."))
        try:
            files = STATE.list_files(directory)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            return [], [], [], error_message(exc)
        options = [{"label": name, "value": name} for name in files]
        return (
            options,
            options,
            options,
            html.Div(f"Found {len(files)} files in {STATE.directory}."),
        )

    @app.callback(
        *[Output(f"file-{mount}", "value", allow_duplicate=True) for mount in MOUNTS],
        Output("file-pairing", "children"),
        *[Input(f"file-{mount}", "value") for mount in MOUNTS],
        State("file-air", "options"),
        prevent_initial_call=True,
    )
    def complete_triplet(*arguments):
        """Fill the other two mounts once one file of a sample is chosen.

        The three mounts of a sample differ in one token of the file name, so
        choosing ``DBB_17_PW_air.xrdml`` is enough to identify the glycol and
        heated scans.  Only the two boxes the analyst did not just touch are
        written to, and a box already holding a mount of the same sample is left
        as it is - so a deliberate choice of, say, the XRDML export where the
        others are .xy files is not undone, while changing to a different sample
        takes one selection rather than three.
        """
        *values, options = arguments
        chosen = dict(zip(MOUNTS, values))
        triggered = callback_context.triggered_id
        if not triggered or not str(triggered).startswith("file-"):
            raise PreventUpdate
        source = str(triggered).removeprefix("file-")
        selected = chosen.get(source)
        if not selected:
            raise PreventUpdate

        available = [option["value"] for option in (options or [])]
        siblings = find_siblings(selected, available)
        if not siblings:
            return *(no_update for _ in MOUNTS), ""

        key = sample_key(selected)
        updates: list[object] = []
        filled: list[str] = []
        missing: list[str] = []
        for mount in MOUNTS:
            if mount == source:
                updates.append(no_update)
                continue
            match = siblings.get(mount)
            current = chosen.get(mount)
            if match is None:
                missing.append(MOUNT_LABELS[mount])
                updates.append(no_update)
                continue
            if current == match or (current and sample_key(current) == key):
                updates.append(no_update)
                continue
            updates.append(match)
            filled.append(f"{MOUNT_LABELS[mount]}: {match}")

        note = []
        if filled:
            note.append("Matched " + "; ".join(filled) + ".")
        if missing:
            note.append(
                f"No {' or '.join(missing)} scan with a matching name is in this folder."
            )
        return *updates, " ".join(note)

    @app.callback(
        Output("load-graph", "figure"),
        Output("load-status", "children"),
        Input("load", "n_clicks"),
        [State(f"file-{mount}", "value") for mount in MOUNTS],
        prevent_initial_call=True,
    )
    def load_patterns(_clicks, *filenames):
        chosen = {mount: name for mount, name in zip(MOUNTS, filenames) if name}
        if not chosen:
            return no_update, error_message(ValueError("Select at least one file."))
        messages = []
        try:
            for mount, filename in chosen.items():
                pattern = STATE.load(mount, filename)
                messages.append(
                    f"{MOUNT_LABELS[mount]}: {filename} — "
                    f"{len(pattern.two_theta)} points, "
                    f"{pattern.two_theta[0]:.2f}–{pattern.two_theta[-1]:.2f}°, "
                    f"step {np.mean(np.diff(pattern.two_theta)):.3f}°"
                )
        except Exception as exc:  # noqa: BLE001
            return no_update, error_message(exc)

        figure = go.Figure()
        for mount in STATE.loaded_mounts():
            pattern = STATE.mounts[mount].raw
            figure.add_scatter(
                x=pattern.two_theta, y=pattern.intensity, name=MOUNT_LABELS[mount],
                line={"color": COLORS[mount], "width": 1},
            )
        return style_axes(figure, "Counts"), html.Ul([html.Li(text) for text in messages])

    @app.callback(
        Output("zero-slider", "value"),
        Output("zero-status", "children", allow_duplicate=True),
        Input("zero-auto", "n_clicks"),
        State("zero-mount", "value"),
        State("zero-reference", "value"),
        prevent_initial_call=True,
    )
    def auto_zero(_clicks, mount, reference):
        pattern = STATE.mounts[mount].raw
        if pattern is None:
            return no_update, error_message(ValueError(f"{MOUNT_LABELS[mount]} is not loaded."))
        target = _reference_angle(reference)
        try:
            result = estimate_zero_error(pattern, reference=target, window=0.8)
        except Exception as exc:  # noqa: BLE001
            return no_update, error_message(exc)
        return result.shift, html.Div(
            f"Peak found at {result.observed_two_theta:.3f}°, reference {target:.3f}°: "
            f"zero error {result.shift:+.2f}° (unsnapped {result.raw_shift:+.3f}°)."
        )

    @app.callback(
        Output("zero-graph", "figure"),
        Output("zero-status", "children"),
        Input("zero-slider", "value"),
        Input("zero-mount", "value"),
        Input("zero-reference", "value"),
        # As for the background tab: the layout is built once, so the view has
        # to be told that patterns have since been loaded.
        Input("load-status", "children"),
    )
    def update_zero(shift, mount, reference, _loaded):
        state = STATE.mounts[mount]
        if state.raw is None:
            return empty_figure(f"{MOUNT_LABELS[mount]} is not loaded"), ""
        state.zero_error = float(shift or 0.0)
        target = _reference_angle(reference)
        corrected = state.corrected()

        figure = go.Figure()
        window = (corrected.two_theta > target - 2.0) & (corrected.two_theta < target + 2.0)
        figure.add_scatter(
            x=state.raw.two_theta[window], y=state.raw.intensity[window],
            name="measured", line={"color": "#999", "width": 1, "dash": "dot"},
        )
        figure.add_scatter(
            x=corrected.two_theta[window], y=corrected.intensity[window],
            name=f"corrected ({state.zero_error:+.2f}°)",
            line={"color": COLORS[mount], "width": 2},
        )
        figure.add_vline(x=target, line={"color": "#d62728", "dash": "dash"},
                         annotation_text=f"{target:.2f}°")
        style_axes(figure, "Counts")

        shifts, merit = zero_error_profile(state.raw, reference=target)
        figure.add_scatter(
            x=target + shifts, y=merit * float(np.max(corrected.intensity[window]) or 1.0),
            name="alignment merit", line={"color": "#ff7f0e", "width": 1}, opacity=0.4,
            yaxis="y",
        )
        best = float(shifts[int(np.argmax(merit))])
        return figure, html.Div(
            f"Slider at {state.zero_error:+.2f}°. Merit curve peaks at {best:+.2f}°. "
            f"Applies to {MOUNT_LABELS[mount]} only — each mount has its own displacement."
        )

    @app.callback(
        Output("bg-graph", "figure"),
        Output("bg-status", "children"),
        Input("bg-mount", "value"),
        # The tab is built once at start-up, so without these the view still
        # shows "not loaded" after the patterns are read and keeps a background
        # fitted to the uncorrected angles after the zero error is set - which
        # reads as the model not being applied at all.
        Input("load-status", "children"),
        Input("zero-status", "children"),
        Input("bg-kind", "value"),
        Input("bg-degree", "value"),
        Input("bg-decay", "value"),
        Input("bg-inverse", "value"),
        Input("bg-offset", "value"),
        Input("bg-snip", "value"),
        Input("bg-apply", "n_clicks"),
        Input("bg-apply-all", "n_clicks"),
    )
    def update_background(mount, _loaded, _zeroed, kind, degree, decay, inverse, offset,
                          snip, _apply, _apply_all):
        state = STATE.mounts[mount]
        if state.raw is None:
            return empty_figure(f"{MOUNT_LABELS[mount]} is not loaded"), ""
        use_inverse = bool(inverse) and "on" in inverse
        if kind == "none" and not use_inverse:
            return empty_figure("Select a component or enable the 1/x term"), error_message(
                ValueError("A background model needs at least one component.")
            )
        model = background_model_from_controls(kind, int(degree), float(decay), use_inverse,
                                              float(offset))
        pattern = state.corrected()
        try:
            fit = model.fit(pattern.two_theta, pattern.intensity, snip_window=float(snip))
        except Exception as exc:  # noqa: BLE001
            return empty_figure("Fit failed"), error_message(exc)

        triggered = [item["prop_id"] for item in callback_context.triggered]
        applied = ""
        if any("bg-apply-all" in prop for prop in triggered):
            for other in STATE.loaded_mounts():
                other_state = STATE.mounts[other]
                other_pattern = other_state.corrected()
                other_state.background_model = model
                other_state.background_fit = model.fit(
                    other_pattern.two_theta, other_pattern.intensity, snip_window=float(snip)
                )
            applied = " Applied to all loaded mounts."
        elif any("bg-apply" in prop for prop in triggered):
            state.background_model = model
            state.background_fit = fit
            applied = f" Applied to {MOUNT_LABELS[mount]}."

        background = fit(pattern.two_theta)
        figure = go.Figure()
        # Fixed colours, not the mount colour: for the glycol mount COLORS[mount]
        # is the same green as the subtracted trace, and the two curves then lie
        # on top of each other indistinguishably.
        figure.add_scatter(x=pattern.two_theta, y=pattern.intensity, name="measured",
                           line={"color": "#444444", "width": 1})
        figure.add_scatter(x=pattern.two_theta, y=background, name="background model",
                           line={"color": "#ff7f0e", "width": 3})
        # Drawn after the model, because a good model lies on top of it: dashes
        # over the orange line are what shows the two agree.
        figure.add_scatter(
            x=pattern.two_theta,
            y=fit.target,
            name=f"peak-stripped estimate (width {float(snip):g}\u00b0)",
            line={"color": "#1f77b4", "width": 2, "dash": "dash"},
        )
        figure.add_scatter(x=pattern.two_theta, y=fit.subtract(pattern.two_theta, pattern.intensity),
                           name="subtracted", line={"color": "#2ca02c", "width": 1})
        style_axes(figure, "Counts")

        return figure, html.Div(background_report(pattern, fit, background, model) + applied)

    @app.callback(
        Output("kao-graph", "figure"),
        Output("kao-status", "children"),
        Input("kao-run", "n_clicks"),
        State("kao-window", "value"),
        State("kao-reference", "value"),
        State("kao-scale", "value"),
        prevent_initial_call=True,
    )
    def run_kaolinite(_clicks, window, reference, manual_scale):
        air = STATE.mounts["air"].subtracted()
        heated = STATE.mounts["heated"].subtracted()
        if air is None or heated is None:
            return no_update, error_message(
                ValueError("Both the air-dried and the heated mount must be loaded.")
            )
        reference_window = {"quartz": None, "illite": (17.0, 18.4)}.get(reference)
        scale = float(manual_scale) if reference == "manual" else None
        try:
            result = kaolinite_collapse(
                air, heated, window=tuple(window), reference_window=reference_window, scale=scale
            )
        except Exception as exc:  # noqa: BLE001
            return no_update, error_message(exc)

        figure = go.Figure()
        figure.add_scatter(x=air.two_theta, y=air.intensity, name="air-dried",
                           line={"color": COLORS["air"], "width": 1})
        figure.add_scatter(x=heated.two_theta, y=result.scale * heated.intensity,
                           name=f"heated × {result.scale:.3f}",
                           line={"color": COLORS["heated"], "width": 1})
        figure.add_vrect(x0=window[0], x1=window[1], fillcolor="#ffd54f", opacity=0.25,
                         line_width=0, annotation_text="7.15 Å")
        figure.update_xaxes(range=[max(air.two_theta[0], window[0] - 6.0),
                                   min(air.two_theta[-1], window[1] + 12.0)])
        style_axes(figure, "Counts")

        conclusion = []
        if result.kaolinite_detected:
            conclusion.append(f"Kaolinite present: {100.0 * result.collapse_fraction:.1f}% of the "
                              f"7.15 Å area collapsed.")
        else:
            conclusion.append("No significant collapse: no kaolinite detected.")
        if result.chlorite_indicated:
            conclusion.append(f"{100.0 * result.residual_fraction:.1f}% survived, indicating "
                              f"chlorite 002.")
        return figure, html.Div(
            [
                html.Div(result.summary()),
                html.Ul([html.Li(text) for text in conclusion]),
                html.Div(
                    f"Air area {result.air.area:.4g} at {result.air.d_spacing:.3f} Å; "
                    f"heated area {result.heated.area:.4g} (scaled "
                    f"{result.scale * result.heated.area:.4g}); lost height "
                    f"{result.lost_height:.4g}.",
                    style={"fontSize": "0.8rem", "color": "#555"},
                ),
            ]
        )

    @app.callback(
        Output("sme-graph", "figure"),
        Output("sme-status", "children"),
        Input("sme-run", "n_clicks"),
        State("sme-window", "value"),
        State("sme-reference", "value"),
        State("sme-scale", "value"),
        prevent_initial_call=True,
    )
    def run_smectite(_clicks, window, reference, manual_scale):
        air = STATE.mounts["air"].subtracted()
        glycol = STATE.mounts["glycol"].subtracted()
        if air is None or glycol is None:
            return no_update, error_message(
                ValueError("Both the air-dried and the glycolated mount must be loaded.")
            )
        scale = float(manual_scale) if reference == "manual" else None
        try:
            result = smectite_swelling(air, glycol, window=tuple(window), scale=scale)
        except Exception as exc:  # noqa: BLE001
            return no_update, error_message(exc)

        figure = go.Figure()
        figure.add_scatter(x=air.two_theta, y=air.intensity, name="air-dried",
                           line={"color": COLORS["air"], "width": 1})
        figure.add_scatter(x=glycol.two_theta, y=result.scale * glycol.intensity,
                           name=f"glycolated × {result.scale:.3f}",
                           line={"color": COLORS["glycol"], "width": 1})
        for metrics, color, name in (
            (result.air, COLORS["air"], "air 001"),
            (result.glycol, COLORS["glycol"], "glycol 001"),
        ):
            if metrics.is_present:
                figure.add_vline(x=metrics.position, line={"color": color, "dash": "dash"},
                                 annotation_text=f"{name}: {metrics.d_spacing:.2f} Å")
        figure.add_vrect(x0=window[0], x1=window[1], fillcolor="#a5d6a7", opacity=0.2, line_width=0)
        figure.update_xaxes(range=[air.two_theta[0], min(air.two_theta[-1], window[1] + 10.0)])
        style_axes(figure, "Counts")

        verdict = (
            "Expandable clay present." if result.expandable_detected else "No expansion detected."
        )
        return figure, html.Div([html.B(verdict), html.Div(result.summary())])

    @app.callback(
        Output("db-status", "children"),
        Input("db-load", "n_clicks"),
        State("db-path", "value"),
        prevent_initial_call=True,
    )
    def load_database(_clicks, path):
        try:
            count = STATE.load_phase_database(resolve_user_path(path))
        except Exception as exc:  # noqa: BLE001
            return error_message(exc)
        clays = sum(1 for name in STATE.phase_database if is_clay_phase(name))
        return html.Div(
            f"{count} phases loaded ({clays} phyllosilicates, which are left to the clay "
            f"library and not offered here)."
        )

    @app.callback(
        Output("detect-graph", "figure"),
        Output("detect-dialog", "children"),
        Output("detect-status", "children"),
        Input("detect-run", "n_clicks"),
        State("detect-mount", "value"),
        State("detect-allowance", "value"),
        State("detect-score", "value"),
        State("detect-snr", "value"),
        State("detect-range", "value"),
        prevent_initial_call=True,
    )
    def run_detection(_clicks, mount, allowance, min_score, snr, search_range):
        if not STATE.phase_database:
            return no_update, None, error_message(
                ValueError("Load a phase database first.")
            )
        pattern = STATE.mounts[mount].subtracted()
        if pattern is None:
            return no_update, None, error_message(
                ValueError(f"{MOUNT_LABELS[mount]} is not loaded.")
            )
        # With a clay library loaded the candidates can be fitted in competition
        # with the clays, which discriminates far better than matching peak
        # positions; without one, fall back to position matching.
        screened = STATE.library is not None
        try:
            if screened:
                findings = screen_phases(
                    STATE.mounts[mount].corrected(),
                    STATE.phase_database,
                    STATE.library.select(march_dollase=[0.1, 0.5, 1.0]),
                    background=STATE.mounts[mount].background_fit,
                    two_theta_range=tuple(search_range),
                    instrument=gui_instrument(),
                    min_share=float(min_score) / 1000.0,
                    cell_allowance=float(allowance) / 100.0,
                )
            else:
                findings = detect_phases(
                    pattern,
                    STATE.phase_database,
                    cell_allowance=float(allowance) / 100.0,
                    min_score=float(min_score) / 100.0,
                    min_signal_to_noise=float(snr),
                    two_theta_range=tuple(search_range),
                    instrument=gui_instrument(),
                    subtract_background=STATE.mounts[mount].background_fit is None,
                )
        except Exception as exc:  # noqa: BLE001
            return no_update, None, error_message(exc)
        STATE.detected = findings

        figure = go.Figure()
        figure.add_scatter(x=pattern.two_theta, y=pattern.intensity, name="measured",
                           line={"color": "#333333", "width": 1})
        peak_height = float(np.max(pattern.intensity)) or 1.0
        for index, evidence in enumerate(findings[:8]):
            colour = _PHASE_PALETTE[index % len(_PHASE_PALETTE)]
            matched = [match for match in evidence.matches if match.matched]
            if not matched:
                continue
            figure.add_scatter(
                x=[match.found_two_theta for match in matched],
                y=[match.height for match in matched],
                mode="markers",
                marker={"color": colour, "size": 9, "symbol": "triangle-down"},
                name=f"{evidence.name} ({100.0 * evidence.score:.0f}%)",
            )
        figure.update_yaxes(range=[0, 1.15 * peak_height])
        style_axes(figure, "Counts (background subtracted)")

        if not findings:
            return figure, None, html.Div(
                "No accompanying minerals reached the score threshold. Lower the minimum "
                "score or widen the search range if you expect some."
            )
        dialog = html.Div(
            [
                html.Div(
                    [
                        html.H4(
                            "Evidence for the following main minerals has been found in the data:",
                            style={"marginTop": 0},
                        ),
                        dcc.Checklist(
                            id="detect-choices",
                            options=[
                                {
                                    "label": f"  {evidence.name} — "
                                    + (
                                        f"{100.0 * evidence.score:.1f}% of the pattern, "
                                        if screened
                                        else f"score {100.0 * evidence.score:.0f}%, "
                                    )
                                    + f"{evidence.n_matched}/{evidence.n_expected} expected "
                                    f"lines found, best S/N "
                                    f"{evidence.best_signal_to_noise:.0f}",
                                    "value": evidence.name,
                                }
                                for evidence in findings[:25]
                            ],
                            # Ticked by default only where the share is clear.  The
                            # screen's tail at one or two percent is largely
                            # least-squares slack rather than real mineralogy, so
                            # it is listed but left for the analyst to judge.
                            value=[
                                evidence.name
                                for evidence in findings
                                if evidence.score >= (0.03 if screened else 0.9)
                            ],
                            style={"maxHeight": "260px", "overflowY": "auto"},
                            labelStyle={"display": "block", "fontSize": "0.85rem"},
                        ),
                        html.P(
                            "Phases with similar cells cannot be separated on peak positions "
                            "alone, so check the list against what the sample can plausibly "
                            "contain. Selected phases are fitted with the clays and then "
                            "normalised out of the clay percentages.",
                            style={"fontSize": "0.78rem", "color": "#555"},
                        ),
                        html.Button("Include selected in the fit", id="detect-accept", n_clicks=0),
                        html.Button("Dismiss", id="detect-dismiss", n_clicks=0,
                                    style={"marginLeft": "8px"}),
                    ],
                    style={
                        "background": "#fffaf0",
                        "border": "2px solid #ffb74d",
                        "borderRadius": "8px",
                        "padding": "14px",
                        "marginBottom": "10px",
                        "boxShadow": "0 2px 10px rgba(0,0,0,0.12)",
                    },
                )
            ]
        )
        method = (
            "fitted in competition with the clay library"
            if screened
            else "matched on peak positions (load a clay library for the sharper test)"
        )
        return figure, dialog, html.Div(f"{len(findings)} candidate phases, {method}.")

    @app.callback(
        Output("detect-selection", "children"),
        Output("detect-dialog", "children", allow_duplicate=True),
        Input("detect-accept", "n_clicks"),
        Input("detect-dismiss", "n_clicks"),
        State("detect-choices", "value"),
        prevent_initial_call=True,
    )
    def accept_detection(accept_clicks, dismiss_clicks, chosen):
        # The dialog and its buttons are created by the search callback, and Dash
        # fires callbacks once for newly added components.  Without this guard the
        # dialog accepts itself the moment it appears, taking whatever was checked
        # by default and never letting the analyst choose.
        if not accept_clicks and not dismiss_clicks:
            raise PreventUpdate
        triggered = callback_context.triggered[0]["prop_id"]
        if "detect-dismiss" in triggered:
            STATE.selected_main = []
            return html.Div("No main minerals will be included in the fit."), None
        STATE.selected_main = list(chosen or [])
        if not STATE.selected_main:
            return html.Div("Nothing selected; the fit will use the clay library only."), None
        return (
            html.Div(
                [
                    html.B("Included in the fit: "),
                    html.Span(", ".join(STATE.selected_main)),
                    html.Div(
                        "These are fitted together with the clays and reported separately, "
                        "then removed when the clay percentages are normalised.",
                        style={"fontSize": "0.8rem", "color": "#555"},
                    ),
                ]
            ),
            None,
        )

    @app.callback(
        Output("lib-status", "children"),
        Input("lib-load", "n_clicks"),
        Input("lib-build", "n_clicks"),
        State("lib-path", "value"),
        prevent_initial_call=True,
    )
    def manage_library(_load, _build, path):
        triggered = callback_context.triggered[0]["prop_id"]
        try:
            if "lib-build" in triggered:
                library = build_library(instrument=gui_instrument())
                library.save(path)
            else:
                library = PatternLibrary.load(resolve_user_path(path))
        except Exception as exc:  # noqa: BLE001
            return error_message(exc)
        STATE.library = library
        STATE.library_path = Path(path)
        return html.Pre(library.describe(), style={"fontSize": "0.75rem", "margin": 0})

    @app.callback(
        Output("fit-graph", "figure"),
        Output("fit-components-graph", "figure"),
        Output("fit-clay-graph", "figure"),
        Output("fit-mounts-graph", "figure"),
        Output("fit-table", "children"),
        Output("fit-status", "children"),
        Input("fit-run", "n_clicks"),
        State("fit-mount", "value"),
        State("fit-range", "value"),
        State("fit-orientations", "value"),
        State("fit-subtract", "value"),
        prevent_initial_call=True,
    )
    def run_fit(_clicks, mount, fit_range, orientations, subtract):
        blank = (no_update,) * 5
        if STATE.library is None:
            return (*blank, error_message(ValueError("Load or build a library first.")))
        state = STATE.mounts[mount]
        if state.raw is None:
            return (*blank, error_message(ValueError(f"{MOUNT_LABELS[mount]} is not loaded.")))

        library = STATE.library
        if orientations:
            library = library.select(march_dollase=orientations)
            if len(library) == 0:
                return (
                    *blank,
                    error_message(
                        ValueError("No library entries match the selected orientation parameters.")
                    ),
                )
        try:
            library, added = _with_main_minerals(library)
        except Exception as exc:  # noqa: BLE001
            return (*blank, error_message(exc))

        use_background = bool(subtract) and "on" in subtract
        try:
            result = nnls_fit(
                state.corrected(),
                library,
                background=state.background_fit if use_background else None,
                range_two_theta=tuple(fit_range),
            )
            quantification = quantify(result)
        except Exception as exc:  # noqa: BLE001
            return (*blank, error_message(exc))

        STATE.fit_result = result
        STATE.quantification = quantification

        mounts = {
            key: STATE.mounts[key].corrected()
            for key in MOUNTS
            if STATE.mounts[key].is_loaded
        }
        try:
            mounts_figure = (
                plot_mounts(mounts) if mounts else empty_figure("No mounts loaded")
            )
        except Exception:  # noqa: BLE001
            mounts_figure = empty_figure("Could not draw the mounts")

        table = _quantification_table(quantification, result)
        status = html.Div(
            f"Rwp = {100.0 * result.r_wp:.2f}%, Rp = {100.0 * result.r_p:.2f}% over "
            f"{result.metadata['n_points']} points and {len(library)} patterns"
            + (f" (including {added} main mineral{'s' if added != 1 else ''})" if added else "")
            + ("" if use_background else ". Fitted without background subtraction")
            + "."
        )
        return (
            plot_fit(result),
            plot_components(result),
            plot_clay_components(result, quantification),
            mounts_figure,
            table,
            status,
        )

    @app.callback(
        Output("export-status", "children"),
        Input("export-run", "n_clicks"),
        State("export-path", "value"),
        prevent_initial_call=True,
    )
    def export_results(_clicks, folder):
        if STATE.fit_result is None:
            return error_message(ValueError("Run the fit first."))
        mounts = {
            key: STATE.mounts[key].corrected()
            for key in MOUNTS
            if STATE.mounts[key].is_loaded
        }
        try:
            written = save_report(
                resolve_user_path(folder or "results"),
                STATE.fit_result,
                mounts=mounts,
                quantification=STATE.quantification,
            )
        except Exception as exc:  # noqa: BLE001
            return error_message(exc)
        return html.Div(
            [html.B(f"{len(written)} files written:")]
            + [html.Div(str(path), style={"fontSize": "0.78rem"}) for path in written]
        )

    @app.callback(
        Output("sim-graph", "figure"),
        Output("sim-status", "children"),
        Input("sim-model", "value"),
        Input("sim-phase", "value"),
        Input("sim-fraction", "value"),
        Input("sim-pab", "value"),
        Input("sim-csds", "value"),
        Input("sim-beta", "value"),
        Input("sim-po", "value"),
        Input("sim-dsmectite", "value"),
        Input("sim-range", "value"),
        Input("sim-options", "value"),
    )
    def simulate(model, phase, fraction, p_ab, csds_mean, beta, po, d_smectite, angle_range,
                 options):
        options = options or []
        instrument = gui_instrument()
        grid = two_theta_grid(float(angle_range[0]), float(angle_range[1]), 0.02)
        csds = lognormal_csds(float(csds_mean), float(beta))
        try:
            smectite = eg_smectite_layer(float(d_smectite))
            figure = go.Figure()
            notes: list[str] = []

            if model == "phase":
                pattern = powder_pattern(load_crystal(phase), grid, instrument,
                                         r_march_dollase=float(po))
                figure.add_scatter(x=pattern.two_theta, y=pattern.normalized().intensity,
                                   name=pattern.name, line={"color": "#1f77b4"})
                notes.append(f"{pattern.metadata['n_reflections']} reflections, "
                             f"{CIF_SOURCES[phase].description}")
            elif model == "S":
                stack = MixedLayerStack(smectite, smectite, 1.0, csds=csds, name="smectite (EG)")
                from ..pattern import basal_pattern

                pattern = basal_pattern(stack, grid, instrument, r_march_dollase=float(po))
                figure.add_scatter(x=pattern.two_theta, y=pattern.normalized().intensity,
                                   name=pattern.name, line={"color": "#2ca02c"})
                notes.append(f"d(001) = {d_smectite:.2f} Å, Reynolds (1965) layer model")
            else:
                host_key = "illite" if model == "IS" else "chlorite"
                host = load_crystal(host_key)
                host_layer = load_layer(host_key)
                layers = CIF_SOURCES[host_key].layers_per_cell
                transition = (
                    random_transition(float(fraction))
                    if float(p_ab) <= 0.0
                    else markov_transition(float(fraction), float(p_ab))
                )
                stack = MixedLayerStack(
                    host_layer, smectite, float(fraction), transition=transition, csds=csds,
                    name=f"{host_key}/smectite {fraction:.2f}/{1 - float(fraction):.2f}",
                )
                pattern = mixed_layer_pattern(
                    stack, host, layers, grid, instrument, r_march_dollase=float(po),
                    include_non_basal="hkl" in options,
                )
                figure.add_scatter(x=pattern.two_theta, y=pattern.normalized().intensity,
                                   name=pattern.name, line={"color": "#1f77b4", "width": 2})
                notes.append(
                    f"d({host_key}) = {host_layer.thickness:.3f} Å, d(smectite) = "
                    f"{d_smectite:.2f} Å, mean {csds.mean:.1f} layers per crystallite"
                )
                if float(p_ab) > 0.0:
                    notes.append(
                        f"Markov stacking: P(host→smectite) = {p_ab:.2f} "
                        f"(random would be {1 - float(fraction):.2f})"
                    )
                else:
                    notes.append("Random interstratification (Reichweite R = 0)")
                if "ends" in options:
                    for end_fraction, color, name in (
                        (1.0, "#999999", f"pure {host_key}"),
                        (0.0, "#2ca02c", "pure smectite"),
                    ):
                        end_stack = MixedLayerStack(host_layer, smectite, end_fraction, csds=csds)
                        end_pattern = mixed_layer_pattern(
                            end_stack, host, layers, grid, instrument,
                            r_march_dollase=float(po), include_non_basal="hkl" in options,
                        )
                        figure.add_scatter(
                            x=end_pattern.two_theta, y=end_pattern.normalized().intensity,
                            name=name, line={"color": color, "width": 1, "dash": "dot"},
                        )
        except Exception as exc:  # noqa: BLE001
            return empty_figure("Simulation failed"), error_message(exc)

        style_axes(figure, "Intensity (normalised)")
        return figure, html.Ul([html.Li(text) for text in notes])


def _reference_angle(reference: str) -> float:
    from ..calibration import QUARTZ_101_D

    return reference_two_theta(QUARTZ_100_D if reference == "quartz100" else QUARTZ_101_D)


def create_app() -> Dash:
    # The main mineral dialog and its checkboxes are created by a callback, so
    # callbacks referring to them must be allowed before they exist.
    app = Dash(__name__, title="ClayQuant", suppress_callback_exceptions=True)
    app.layout = build_layout
    register_callbacks(app)
    return app


def main(argv: list[str] | None = None) -> int:
    """Command line entry point for ``clayquant-gui``."""
    parser = argparse.ArgumentParser(prog="clayquant-gui", description="Start the ClayQuant GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args(argv)

    app = create_app()
    print(f"ClayQuant GUI at http://{arguments.host}:{arguments.port}")
    app.run(host=arguments.host, port=arguments.port, debug=arguments.debug)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
