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
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import replace
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate

from ..background import (
    CLAYFIT_ANCHOR_RADIUS,
    CLAYFIT_ANCHOR_STRIDE,
    CLAYFIT_INVERSE_X_AMPLITUDE,
    CLAYFIT_MODELS,
    CLAYFIT_MODEL_LABELS,
    CLAYFIT_ORDER,
    CLAYFIT_ORDER_LIMITS,
    QPA_BASELINE_ORDER,
    QPA_BASELINE_SMOOTH,
    QPA_FINAL_ORDER,
    QPA_FINAL_SMOOTH,
    QPA_MODEL_NAME,
    QPA_PERCENTILE_WINDOW,
    ClayfitBackground,
    clayfit_background,
    SONNEVELD_VISSER_ITERATIONS,
    sonneveld_visser_reach,
    suggest_sonneveld_visser,
)
from ..bern import is_clay_phase
from ..detection import (
    evidence_score,
    screen_treatments,
    COMMON_IN_CLAY_SEPARATES,
    detect_phases,
    screen_phases,
    stable_phases,
)
from ..calibration import (
    QUARTZ_100_D,
    QUARTZ_101_D,
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
from ..io import resolve_user_path
from ..library import (
    DEFAULT_PEAK_SHAPE,
    PREFERRED_ORIENTATIONS,
    PatternLibrary,
    build_library,
    describe_instrument_mismatch,
    instrument_from_measurement,
)
from ..mixed_layer import MixedLayerStack, lognormal_csds, markov_transition, random_transition
from ..models import CIF_SOURCES, available_phases, eg_smectite_layer, load_crystal, load_layer
from ..nnls import (
    _library_subset,
    screen_diagnostic_peaks,
    select_with_lattice_scaling,
    clay_families,
    nnls_fit,
    orientation_families,
    select_one_per_family,
)
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
from ..quantification import Calibration, quantify
from ..treatment import expandable_bound, shift_evidence
from . import folder_dialog
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


def background_settings_from_controls(
    kind, order, inverse, amplitude, radius, stride,
    qpa_window, qpa_baseline_smooth, qpa_baseline_order,
    qpa_final_smooth, qpa_final_order, granularity, bending,
) -> dict:
    """The keyword arguments the Background tab's controls describe.

    Gathered into one dictionary so that the same settings can be applied to
    another mount without the caller reading thirteen values off the screen
    again.
    """
    return {
        "kind": str(kind),
        "order": int(order),
        "inverse_x_amplitude": (float(amplitude)
                                if bool(inverse) and "on" in inverse else 0.0),
        "radius": float(radius),
        "stride": int(stride),
        "qpa_window": float(qpa_window),
        "qpa_baseline_smooth": float(qpa_baseline_smooth),
        "qpa_baseline_order": int(qpa_baseline_order),
        "qpa_final_smooth": float(qpa_final_smooth),
        "qpa_final_order": int(qpa_final_order),
        "granularity": int(granularity),
        "bending": float(bending),
    }


def fit_background_from_controls(pattern, **settings) -> ClayfitBackground:
    """The background the Background tab's controls describe.

    One call, because each of the six models is a complete method in its own
    right rather than a component to be combined with the others - which is the
    respect in which this follows Clayfit and the earlier design here did not.
    """
    return clayfit_background(pattern.two_theta, pattern.intensity, **settings)


def working(*children) -> dcc.Loading:
    """Wrap a pane so a spinner covers it while a callback is computing it.

    Several steps take seconds rather than milliseconds - the mineral search
    fits every candidate phase, the quantification solves a non-negative least
    squares over more than a thousand patterns - and Dash shows nothing at all
    while a callback runs.  A button that appears to do nothing for ten seconds
    is indistinguishable from a button that does nothing, and was reported as
    exactly that.
    """
    return dcc.Loading(
        children=list(children),
        type="circle",
        color="#6b46c1",
        # Shown only once the step has taken long enough to need explaining, so
        # the quick ones do not flicker; the pane underneath is dimmed rather
        # than blanked, which reads as "this is being recalculated" instead of
        # "this is gone".
        delay_show=200,
        overlay_style={"visibility": "visible", "opacity": 0.35,
                       "filter": "grayscale(60%)"},
    )


STABLE_TICK_COVERAGE = 0.5
"""Least coverage a phase needs to arrive ticked in the three-mount search.

Half a phase's own strong reflections standing on peaks that every treatment
shares.  On one real separate it ticks quartz and rutile at 100 per cent and
leaves albite at 43 per cent for the analyst, which is the right way round: a
feldspar's dozen reflections overlap so much of the pattern that its coverage is
not the thing to decide it on.
"""

MIN_TICK_AGREEMENT = 0.25
"""Least intensity agreement a phase needs to arrive ticked.

Low enough to admit a genuinely overlapped mineral and high enough to exclude a
phase whose line set the pattern contradicts.  Measured on one clay separate the
phases it excludes are graphite at 0.00 and gypsum at 0.25; the ones it admits
include quartz at 0.54, microcline at 0.76 and albite at 0.78.
"""


def phases_to_tick(findings, screened: bool, tick_above: float,
                   stable: bool = False) -> list[str]:
    """Which found phases arrive already ticked in the main-mineral dialog.

    Only the competitive screen pre-ticks anything.  A position-matching score
    says that a phase's expected lines fall where the data has peaks, which in a
    crowded pattern a candidate with a dozen expected lines manages whether or
    not it is there - scores of 95 to 100 % are normal for phases that are
    absent.  There is no honest threshold to pre-tick on, and pre-ticking the top
    of that list put galena, otavite and cassiterite in front of an operator of a
    clay separate, one click from entering the fit.

    With a library loaded the score is the share of the pattern the phase
    accounts for, chosen in rounds against what the clays and the phases already
    chosen explain, and ``tick_above`` is that share in per cent.

    A share is still not sufficient on its own, and ``min_agreement`` is the
    second condition.  ``intensity_agreement`` asks whether the measured heights
    at a phase's own line positions stand in that phase's own proportions; a
    phase can take a share of the pattern without that being true, and the case
    that forced this is graphite, whose two reflections include an 002 within a
    tenth of a degree of the quartz 101.  It takes a share, it reports two of
    two lines found and a signal-to-noise of 153, and its intensity agreement is
    zero, because the pattern has nothing at all where its other line should be.
    Every number in that row reads as confirmation except the one that matters.
    Such a phase is still listed - it is the analyst's call, and a real mineral
    can have an overlapped line set - but it does not arrive ticked.
    """
    if stable:
        # In the three-mount search the score is coverage, and the threshold is
        # its own: half a phase's strong reflections standing on peaks that
        # every treatment shares.  ``tick_above`` is a share of the pattern and
        # means nothing here.
        return [
            evidence.name for evidence in findings
            if evidence.score >= STABLE_TICK_COVERAGE and evidence.n_matched >= 2
        ]
    if not screened:
        return []
    return [
        evidence.name for evidence in findings
        if evidence.score >= float(tick_above) / 100.0
        and evidence.intensity_agreement >= MIN_TICK_AGREEMENT
    ]


def instrument_strip():
    """The instrument in use and the one the library holds, side by side.

    A plain function rather than only a callback body, so what it says in each
    of its states can be tested: nothing loaded, a measurement with no library,
    a library that records nothing, and a library that disagrees.
    """
    from ..library import DEFAULT_PEAK_SHAPE

    def widths(shape) -> str:
        return ", ".join(
            f"{angle:.0f}\u00b0: {float(shape.fwhm(np.array([angle]), 1.540596)[0]):.3f}\u00b0"
            for angle in (8.0, 20.0, 26.0, 36.0)
        )

    rows = []
    if STATE.instrument is None:
        rows.append(html.Span(
            "Measurement: none loaded \u2014 calculations use the 280 mm / 0.5\u00b0 "
            "default and a generic peak width.",
            style={"color": "#a15c00"},
        ))
    else:
        divergence = STATE.instrument.divergence
        rows.append(html.Span(
            f"Measurement: {divergence.goniometer_radius:.0f} mm radius, "
            f"{divergence.divergence:g}\u00b0 slit, "
            f"{divergence.specimen_length:g} mm specimen \u2014 FWHM "
            f"{widths(STATE.instrument.peak_shape)}"
        ))
    library = STATE.library
    if library is None:
        rows.append(html.Span("Library: none loaded.", style={"color": "#666"}))
    else:
        stored = library.metadata.get("peak_shape")
        geometry = library.metadata.get("geometry")
        if not stored or not geometry:
            rows.append(html.Span(
                "Library: does not record what it was calculated with, so it cannot "
                "be checked against this measurement.",
                style={"color": "#a15c00"},
            ))
        else:
            shape = replace(
                DEFAULT_PEAK_SHAPE,
                u=float(stored.get("u", 0.0)), v=float(stored.get("v", 0.0)),
                w=float(stored.get("w", 0.01)), eta=float(stored.get("eta", 0.5)),
                size_c=stored.get("size_c"), size_ab=stored.get("size_ab"),
                strain=float(stored.get("strain", 0.0) or 0.0),
            )
            complaint = (
                describe_instrument_mismatch(library, STATE.instrument)
                if STATE.instrument is not None else ""
            )
            rows.append(html.Span(
                f"Library: {geometry['goniometer_radius']:.0f} mm radius, "
                f"{geometry['divergence']:g}\u00b0 slit \u2014 FWHM {widths(shape)}",
                style={"color": "#a15c00" if complaint else "#1a7f37"},
            ))
            if complaint:
                rows.append(html.Span(
                    "These do not match. Rebuild the library with a scan from this "
                    "instrument \u2014 the desktop icon asks for one \u2014 or the fit "
                    "will be short of intensity at every strong peak.",
                    style={"color": "#a15c00", "fontWeight": "600"},
                ))
    return html.Div([html.Div(row) for row in rows])


def between_the_peaks(pattern, fit) -> float:
    """What the subtracted pattern still holds where there is no reflection.

    The one number that says whether a background is in the right place, and it
    is not Rwp.  Rwp prefers a lower baseline whatever the specimen - subtracting
    too little raises the observed intensity in its denominator as much as its
    numerator - so it cannot choose between two background estimates, and on one
    real scan it ranked the worst of four first.  Between the peaks the answer
    is known instead: the subtracted pattern should be at zero.

    "Between the peaks" is taken as the lowest two fifths of the subtracted
    points, which needs no list of where the reflections are and is what that
    region is: on the same scan the four estimates leave 110, 27, 17 and 18
    counts there, and the default left the most.
    """
    left = fit.subtract(pattern.two_theta, pattern.intensity)
    if left.size == 0:
        return float("nan")
    quiet = left <= np.quantile(left, 0.4)
    return float(np.median(left[quiet])) if quiet.any() else float("nan")


def background_report(pattern, fit, background) -> str:
    """Describe the background in the terms the operator has to judge it by.

    Three numbers, in the order in which they decide whether the curve is in
    the right place.

    How well it passes through the anchor points, in counts, for the models that
    are fitted to them.  R² is reported too but is close to 1 for almost any of
    them, so it reads as success even where the curve is a thousand counts
    below the measurement at the low-angle end, which is where the choice
    actually matters.

    What is left as signal at the start of the scan, where an oriented mount
    carries the direct-beam tail *and* the strongest basal reflections, and a
    background a few hundred counts too high removes the thing being measured.

    And what the subtracted pattern still holds between the peaks, where it
    should hold nothing.  That is the number to compare two models by;
    R_wp is not, because a lower baseline raises the observed intensity in its
    denominator as much as in its numerator, so it prefers the model that
    subtracts least whatever the specimen.
    """
    two_theta = pattern.two_theta
    left_over = between_the_peaks(pattern, fit)

    start = float(two_theta[0])
    measured_start = float(pattern.intensity[0])
    model_start = float(background[0])
    left = max(0.0, measured_start - model_start)
    share = 100.0 * left / measured_start if measured_start > 0 else 0.0

    heading = f"{fit.model}. "
    anchors = getattr(fit, "anchor_two_theta", None)
    if anchors is not None and anchors.size and fit.n_terms:
        residual = fit.anchor_intensity - fit(anchors)
        agreement = (
            f"Passes the {anchors.size:d} anchor points to "
            f"{float(np.sqrt(np.mean(residual ** 2))):.0f} counts RMS "
            f"(R² {fit.r_squared(two_theta):.3f}). "
        )
    elif anchors is not None and anchors.size:
        # The Chebyshev shape is fitted to the pattern, not to the anchors, so
        # its distance from them is a property of the method and not a misfit.
        agreement = f"Anchored on {anchors.size:d} rolling-ball points. "
    else:
        agreement = "Nothing is fitted: the curve is the estimate itself. "

    text = (
        heading
        + agreement
        + (
        f"At {start:.2f}\u00b0 the background reads {model_start:.0f} of "
        f"{measured_start:.0f} measured counts, leaving {left:.0f} ({share:.0f}%) "
        f"as signal. Between the peaks the subtracted pattern still holds "
        f"{left_over:.0f} counts, where it should hold none - that is the number "
        f"to judge a background by, and Rwp is not, because Rwp prefers "
        f"a lower baseline whatever the specimen."
        )
    )
    if getattr(fit, "note", ""):
        text += " " + fit.note
    if share > 25.0:
        text += (
            " That is a large share to carry into the fit. Unless a real reflection "
            "lies at the start of the scan, the background should be following the "
            "direct-beam tail there: check the orange curve against the measurement."
        )
    # How far above the measurement the curve runs, judged against the counting
    # noise rather than by counting the points: a background fitted to a lower
    # envelope crosses above the noise somewhere on any real scan, and a
    # warning that fires on that is a warning that fires always.  What matters
    # is whether it sits above *systematically*, because that intensity is
    # signal being removed.
    excess = background - pattern.intensity
    over = excess > 0.0
    if over.any():
        noise = np.sqrt(np.maximum(pattern.intensity[over], 1.0))
        sigmas = float(np.median(excess[over] / noise))
        counts = float(np.median(excess[over]))
        text += (f" It runs above the measurement over {100.0 * float(over.mean()):.0f}% "
                 f"of the scan, by {counts:.0f} counts where it does "
                 f"({sigmas:.1f} times the counting noise).")
        if sigmas > 1.0:
            text += (" More than the noise, so that is signal being removed rather "
                     "than the curve threading through it: compare it against the "
                     "other models in the plot before using it.")
    return text


LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"})

REMOTE_CHOOSER = (
    "This page is open on a different computer from the one running ClayQuant, so "
    "a chooser here would appear on that other machine. Type the path as the "
    "computer running ClayQuant sees it."
)

FOLDER_DIALOG_TIMEOUT = 600.0


def served_locally() -> bool:
    """Whether the page asking is on the same computer as this process.

    The folder chooser opens on the *server*, so offering it to a browser
    somewhere else would pop a dialog on an empty desk and hang the request
    until it timed out.  ClayQuant is started by the analyst on their own
    machine, so this is normally true; it is checked rather than assumed
    because the alternative fails in a way nobody could diagnose from the
    browser.
    """
    try:
        from flask import has_request_context, request
    except ImportError:  # pragma: no cover - Dash brings Flask with it
        return True
    if not has_request_context():
        return True
    return (request.remote_addr or "") in LOOPBACK


def ask_for_file(initial: str | None,
                 filetypes: list[str] | None = None) -> tuple[str | None, str | None]:
    """Open the operating system's file chooser.  See :func:`ask_for_folder`."""
    return _ask(["--file"] + ([initial] if initial else [""]) + list(filetypes or []))


def ask_for_folder(initial: str | None) -> tuple[str | None, str | None]:
    """Open the operating system's folder chooser.

    Returns the chosen path and ``None``, or ``None`` and something to tell the
    analyst.  Both being ``None`` means the dialog was dismissed, which needs no
    message.

    The chooser runs as a child process - see
    :mod:`clayquant.gui.folder_dialog` for why - started by file path rather
    than with ``-m`` so that the child does not import this module and drag
    numpy, dash and plotly in behind it just to draw a dialog.
    """
    return _ask([initial] if initial else [])


def _ask(arguments: list[str]) -> tuple[str | None, str | None]:
    """Run the chooser child with these arguments and interpret how it ended."""
    script = Path(folder_dialog.__file__)
    command = [sys.executable, str(script)] + [str(item) for item in arguments]
    try:
        finished = subprocess.run(
            command, capture_output=True, text=True, timeout=FOLDER_DIALOG_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return None, (
            "The chooser was still open after "
            f"{FOLDER_DIALOG_TIMEOUT / 60:.0f} minutes, so it was closed. "
            "Type the path instead, or press Browse again."
        )
    except OSError as exc:
        return None, f"Could not start the chooser: {exc}"

    if finished.returncode == 0:
        chosen = finished.stdout.strip()
        return (chosen or None), None

    detail = finished.stderr.strip().splitlines()
    detail = detail[-1] if detail else f"exit code {finished.returncode}"
    if finished.returncode == 2:
        return None, (
            "This Python has no tkinter, so it cannot open a file or folder chooser. "
            "Type the path into the box instead. On Debian or Ubuntu, "
            "installing python3-tk and reinstalling ClayQuant adds the chooser; "
            "the Windows and macOS installers from python.org include it."
        )
    if finished.returncode == 3:
        return None, (
            "There is no desktop session for a dialog to open on, so the path "
            f"has to be typed into the box. ({detail})"
        )
    return None, f"The chooser failed: {detail}"


def gui_instrument(
    specimen_length: float = 35.0,
    goniometer_radius: float = 280.0,
    divergence_slit: float = 0.5,
    size_ab: float = 400.0,
) -> Instrument:
    """The instrument to calculate with.

    Once a mount has been loaded this returns the one derived from it by
    :func:`instrument_from_measurement`, so the calculated patterns, the library
    and the fit all use the same geometry and the same widths.  The arguments
    are the fallback for before that, and are also how a caller can override the
    derived values deliberately.

    That it used to return the defaults unconditionally was a real defect, and
    the first one to look for when a fit is short of intensity at every strong
    peak at once: the goniometer radius and the divergence slit are recorded in
    the data file, and calculating with 280 mm on a 240 mm instrument mis-states
    how much of the beam the specimen intercepts at low angle, which lands
    directly on the 001 reflections that the whole method rests on.
    """
    if STATE.instrument is not None:
        return STATE.instrument
    return Instrument(
        emission=CU_KA_5LINE,
        peak_shape=replace(DEFAULT_PEAK_SHAPE, size_ab=size_ab),
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
        extended.add(pattern, phase=name, march_dollase=1.0,
                     unit_mass=crystal.cell_mass, unit_volume=crystal.volume)
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
                                    html.Th("Weight %"),
                                    html.Th("Expandable %"),
                                    html.Th("r (texture)"),
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
                                        html.Td(
                                            f"{row['weight_percent']:.1f}"
                                            if row["weight_percent"] != "" else "-",
                                            style={"fontWeight": "600"},
                                        ),
                                        html.Td(
                                            f"{100.0 * (1.0 - row['host_fraction']):.0f}"
                                            if row["group"] == "clay" else "-",
                                        ),
                                        html.Td(f"{row['march_dollase']:.2f}"),
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

    # What proportion of the clay minerals is expandable *layers*, which is not
    # the same number as the weight of the phases that carry them and is the
    # one an operator means by "swelling clay".  An "I/S" at 5 % expandable is
    # a stack of 95 % illite layers, and reporting its weight without saying so
    # reads as though most of the specimen swelled.
    expandable = sum(share.clay_weight * share.expandable_fraction
                     for share in quantification.shares if share.is_clay)
    clay_note = (
        f"Accompanying minerals removed and the clays renormalised to 100%. "
        f"The clay minerals are {100.0 * quantification.clay_total_scattering:.1f}% "
        f"of the whole pattern, and {expandable:.1f}% of them is expandable "
        f"layers \u2014 the Expandable % column is each phase's own proportion, so "
        f"an I/S at 5% is a stack of 95% illite layers however much of it there is."
    )
    if quantification.weights_available:
        weight_note = html.Div(
            [
                html.B("Weight % is calculated, not calibrated. "),
                "It comes from the fitted scale factors and the mass of the unit each pattern "
                "was calculated for, and assumes the calculated pattern accounts for everything "
                "the phase contributes. In an oriented mount that includes how strongly the "
                "phase orients, which differs between minerals and between preparations: the "
                "fit cannot tell a well-oriented phase from an abundant one, and the effect is "
                "large: a basal series is enhanced by r to the power -3, so a phase fitted at "
                "r = 0.2 is calculated to scatter 125 times more per gram than the same phase at "
                "r = 1. Read the r column beside each weight. Treat these as estimates until the "
                "preparation has been calibrated against mixtures of known composition. "
                f"Calibration in use: {quantification.calibration.source}.",
            ]
            + (
                [
                    html.Br(),
                    html.Br(),
                    html.B("Smectite has been put on the texture the other clays were "
                           "fitted at. "),
                    "; ".join(
                        f"{phase} was stored at r = {stored:g} and is weighed at "
                        f"r = {target:g}, a factor of {(stored / target) ** 3:.0f} "
                        f"less mass for the same fitted intensity"
                        for phase, (stored, target) in
                        quantification.rebased_orientation.items()
                    )
                    + ". Every reflection of a glycolated smectite is basal, so its "
                    "pattern has the same shape at every r and the fit cannot measure "
                    "its texture - the library has to assume one. Leaving that "
                    "assumption at the library's value beside clays the fit put at a "
                    "tenth is what turns a per cent of the scattering into a fifth of "
                    "the clay weight.",
                ]
                if quantification.rebased_orientation
                else []
            ),
            style={"background": "#fff4e5", "border": "1px solid #ffb74d", "padding": "8px",
                   "borderRadius": "6px", "fontSize": "0.8rem", "margin": "10px 0"},
        )
    else:
        weight_note = html.Div(
            "Weight % is not available: this library does not record the mass of the units its "
            "patterns were calculated for. Rebuild the library to get it.",
            style={"background": "#fff4e5", "border": "1px solid #ffb74d", "padding": "8px",
                   "borderRadius": "6px", "fontSize": "0.8rem", "margin": "10px 0"},
        )
    return html.Div(
        [
            weight_note,
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
                    html.Div(
                        [
                            html.Button("Browse\u2026", id="browse", n_clicks=0),
                            html.Button("Scan folder", id="scan", n_clicks=0),
                        ],
                        style={"display": "flex", "gap": "6px", "marginTop": "8px"},
                    ),
                    html.Div(
                        "Browse opens the folder chooser of this computer, and scans "
                        "what you pick. The path can also be typed or pasted; press "
                        "Enter to scan it.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
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
            html.Div(
                [
                    working(dcc.Graph(id="load-graph",
                                      figure=empty_figure("Load patterns to begin"))),
                    working(html.Div(id="load-screen")),
                ],
                style=GRAPH_BOX,
            ),
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
                            ),
                        ],
                        style={"marginTop": "12px"},
                    ),
                    html.Button("Estimate automatically", id="zero-auto", n_clicks=0),
                    html.Div(id="zero-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div([working(dcc.Graph(id="zero-graph",
                                        figure=empty_figure("Load patterns first")))],
                     style=GRAPH_BOX),
        ],
        style=ROW,
    )


def background_tab() -> html.Div:
    """Clayfit's background step: six models, one order, one A/x amplitude.

    The models are not components to be added together and they do not agree
    with one another, which is the point of showing all six curves at once: the
    polynomial passes through the anchor points, the Chebyshev shape is fitted
    to the whole pattern including the peaks and then rescaled, the exponential
    follows the direct-beam tail, the asymmetric least squares and the erosion
    follow the measurement, and the percentile model is the one meant for a
    quantitative fit.  Choosing between them is a judgement about the specimen,
    so the program shows what each one does rather than picking for you.
    """
    low, high = CLAYFIT_ORDER_LIMITS
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
                    label("Background model"),
                    dcc.Dropdown(
                        id="bg-kind",
                        options=[{"label": CLAYFIT_MODEL_LABELS[name], "value": name}
                                 for name in CLAYFIT_MODELS],
                        value="exponential",
                        clearable=False,
                    ),
                    html.Div(
                        "The five models above Sonneveld–Visser are Clayfit's, and the "
                        "same choice here gives the same curve there. They are separate "
                        "methods, not two bases for one fit: Polynomial is a least-squares "
                        "polynomial through the anchor points in raw 2θ, Chebyshev is a "
                        "series fitted to I/2θ over the whole pattern - peaks included - "
                        "and then rescaled by one amplitude, so it is not obliged to pass "
                        "through the anchors and usually sits below them. The order moves "
                        "both.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
                    html.Div(
                        [
                            label(f"Order ({low}–{high})"),
                            dcc.Slider(id="bg-order", min=low, max=high, step=1,
                                       value=CLAYFIT_ORDER,
                                       marks={value: str(value)
                                              for value in range(low, high + 1, 2)}),
                        ],
                        id="bg-order-box",
                    ),
                    html.Div(
                        [
                            label("Granularity (points per interval)"),
                            dcc.Slider(id="bg-granularity", min=0, max=50, step=1, value=20,
                                       marks={0: "0", 10: "10", 20: "20", 30: "30",
                                              40: "40", 50: "50"}),
                            label("Bending factor"),
                            dcc.Slider(id="bg-bending", min=0.0, max=100.0, step=1.0, value=0.0,
                                       marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"}),
                            html.Div(id="bg-sv-note",
                                     style={"fontSize": "11px", "color": "#666"}),
                            html.Div(id="bg-suggestion",
                                     style={"fontSize": "11px", "color": "#1f77b4",
                                            "marginTop": "4px"}),
                        ],
                        id="bg-sv-box",
                    ),
                    html.Div(
                        [
                            label("Percentile window (°2θ)"),
                            dcc.Slider(id="bg-qpa-window", min=0.2, max=6.0, step=0.1,
                                       value=QPA_PERCENTILE_WINDOW,
                                       marks={0.2: "0.2", 1.5: "1.5", 3: "3", 6: "6"}),
                            label("Background smoothing (°2θ) and order"),
                            dcc.Slider(id="bg-qpa-baseline-smooth", min=0.0, max=1.5, step=0.02,
                                       value=QPA_BASELINE_SMOOTH,
                                       marks={0: "0", 0.3: "0.30", 0.75: "0.75", 1.5: "1.5"}),
                            dcc.Slider(id="bg-qpa-baseline-order", min=0, max=6, step=1,
                                       value=QPA_BASELINE_ORDER,
                                       marks={value: str(value) for value in range(7)}),
                            label("Corrected-pattern smoothing (°2θ) and order"),
                            dcc.Slider(id="bg-qpa-final-smooth", min=0.0, max=0.5, step=0.01,
                                       value=QPA_FINAL_SMOOTH,
                                       marks={0: "0", 0.08: "0.08", 0.25: "0.25", 0.5: "0.5"}),
                            dcc.Slider(id="bg-qpa-final-order", min=0, max=6, step=1,
                                       value=QPA_FINAL_ORDER,
                                       marks={value: str(value) for value in range(7)}),
                            html.Div(
                                "This is the one model that also smooths the corrected "
                                "pattern, which is what Clayfit's \"QPA\" means: the "
                                "pattern is being prepared for a quantitative fit rather "
                                "than displayed. Set the percentile window wider than the "
                                "broadest reflection that has to survive - a fifteenth "
                                "percentile follows a smectite band as readily as it "
                                "follows the background.",
                                style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                            ),
                        ],
                        id="bg-qpa-box",
                    ),
                    html.Hr(),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="bg-inverse",
                                options=[{"label": "  Add A/2θ term", "value": "on"}],
                                value=[],
                            ),
                            label("Amplitude A (counts × °2θ)"),
                            dcc.Input(id="bg-amplitude", type="number",
                                      value=CLAYFIT_INVERSE_X_AMPLITUDE, min=0.0, step=50.0,
                                      style={"width": "100%"}),
                            html.Div(
                                "Set, not fitted, and that is deliberate: the term is "
                                "there to be turned up until the low-angle rise is "
                                "accounted for, and a fitted A would be free to eat the "
                                "001 reflections that stand on that rise instead. At the "
                                "default it adds A/3 ≈ 167 counts at 3° and a tenth of "
                                "that at 30°.",
                                style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                            ),
                        ],
                        id="bg-inverse-box",
                    ),
                    html.Hr(),
                    html.Div(
                        [
                            label("Rolling-ball radius"),
                            dcc.Slider(id="bg-radius", min=0.05, max=1.0, step=0.05,
                                       value=CLAYFIT_ANCHOR_RADIUS,
                                       marks={0.05: "0.05", 0.5: "0.5", 1.0: "1"}),
                            label("Points between ball samples"),
                            dcc.Slider(id="bg-stride", min=1, max=20, step=1,
                                       value=CLAYFIT_ANCHOR_STRIDE,
                                       marks={1: "1", 5: "5", 10: "10", 20: "20"}),
                            html.Div(
                                "The anchor points the polynomial and the exponential are "
                                "fitted to, and the amplitude of the Chebyshev shape. A "
                                "ball of radius 0.5 - half the width of the pattern, the "
                                "pattern being scaled into a unit square - rolling on "
                                "every fifth point is Clayfit's setting and finds twenty "
                                "to thirty points on a clay mount. A smaller ball enters "
                                "the gaps between reflections and follows the measurement "
                                "more closely; a larger one spans them and cuts under.",
                                style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                            ),
                        ],
                        id="bg-anchor-box",
                    ),
                    html.Button("Apply to this mount", id="bg-apply", n_clicks=0,
                                style={"marginTop": "10px"}),
                    html.Button("Apply to all mounts", id="bg-apply-all", n_clicks=0,
                                style={"marginTop": "6px"}),
                    html.Div(id="bg-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div([working(dcc.Graph(id="bg-graph",
                                        figure=empty_figure("Load patterns first")))],
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
                                    value=list(KAOLINITE_001_WINDOW)),
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
            html.Div([working(dcc.Graph(
                id="kao-graph",
                figure=empty_figure("Load the air-dried and heated mounts")))],
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
                                    value=list(EXPANDABLE_001_WINDOW)),
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
            html.Div([working(dcc.Graph(
                id="sme-graph",
                figure=empty_figure("Load the air-dried and glycolated mounts")))],
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
                    html.Div(
                        [
                            html.Button("Browse\u2026", id="db-browse", n_clicks=0),
                            html.Button("Load database", id="db-load", n_clicks=0),
                        ],
                        style={"display": "flex", "gap": "6px", "marginTop": "6px"},
                    ),
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
                               marks={0: "0", 2: "2", 5: "5"}),
                    label("Minimum share (‰) or score (%)"),
                    dcc.Slider(id="detect-score", min=5, max=95, step=5, value=10,
                               marks={5: "5", 50: "50", 95: "95"}),
                    label("Minimum peak S/N"),
                    dcc.Slider(id="detect-snr", min=2, max=20, step=1, value=5,
                               marks={2: "2", 10: "10", 20: "20"}),
                    label("Tick phases above (% of the pattern)"),
                    dcc.Slider(id="detect-tick", min=0.5, max=10.0, step=0.5, value=3.0,
                               marks={0.5: "0.5", 3: "3", 10: "10"}),
                    html.Div(
                        "Every phase found is listed; this decides only which arrive "
                        "already ticked, and it applies to the competitive screen "
                        "alone \u2014 with no clay library loaded the search falls back "
                        "to matching peak positions, whose score is not a share of "
                        "anything, and nothing is pre-ticked. A share depends on what "
                        "else is in the database competing for the same intensity, so "
                        "a phase can drop below the line because a different candidate "
                        "was added, not because the evidence for it changed. If a "
                        "mineral you expect is listed but unticked, lower this rather "
                        "than assuming it was not found.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
                    label("Search range (°2θ)"),
                    dcc.RangeSlider(id="detect-range", min=2.0, max=70.0, step=0.5,
                                    value=[4.0, 40.0]),
                    label("Minimum coverage (%), three-mount search"),
                    dcc.Slider(id="detect-coverage", min=5, max=95, step=5, value=30,
                               marks={5: "5", 30: "30", 60: "60", 95: "95"}),
                    html.Div(
                        "With two or more mounts loaded the search uses the one thing "
                        "three treatments give that no fit of a single scan can: the "
                        "clays move and nothing else does, so a peak at the same angle "
                        "in every mount belongs to a non-clay phase. Coverage is the "
                        "share of a phase's own strong reflections that stand on such a "
                        "peak, and it does not change because another candidate was "
                        "added. It is what finds a minor phase whose strongest line is "
                        "overlapped: rutile is 1.5 % of one separate and covers 100 %.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
                    label("Restrict the search to"),
                    dcc.Dropdown(id="detect-only", options=[], value=[], multi=True,
                                 placeholder="the whole database"),
                    html.Div(
                        [
                            html.Button("Common in clay separates", id="detect-common",
                                        n_clicks=0, style={"marginRight": "6px"}),
                            html.Button("Clear", id="detect-clear", n_clicks=0),
                        ],
                        style={"marginTop": "6px"},
                    ),
                    html.Div(
                        "An unrestricted search over two hundred candidates on one "
                        "scan is genuinely ambiguous, and naming the phases the "
                        "specimen can contain is the information the data does not "
                        "hold. Measured on one separate: unrestricted, rutile is not "
                        "selected at all, because a dozen phases the sample cannot "
                        "contain fit its peaks and are taken first, having more lines "
                        "to spread a claim over. Restricted to quartz, albite, "
                        "sekaninaite and rutile, all four come out in order with "
                        "rutile's own lines agreeing to 0.95.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
                    html.Button("Search for main minerals", id="detect-run", n_clicks=0,
                                style={"marginTop": "10px"}),
                    working(html.Div(id="detect-status", style={"marginTop": "10px"})),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div(
                [
                    working(
                        html.Div(id="detect-dialog"),
                        dcc.Graph(id="detect-graph",
                                  figure=empty_figure("Load a phase database and search")),
                    ),
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
                    html.Div(
                        [
                            html.Button("Browse\u2026", id="lib-browse", n_clicks=0),
                            html.Button("Load library", id="lib-load", n_clicks=0),
                        ],
                        style={"display": "flex", "gap": "6px", "marginTop": "6px"},
                    ),
                    html.Button("Build library now", id="lib-build", n_clicks=0,
                                style={"marginTop": "6px"}),
                    working(html.Div(id="lib-status",
                                     style={"marginTop": "8px", "fontSize": "0.82rem"})),
                    html.Hr(),
                    label("Mount to fit"),
                    dcc.RadioItems(
                        id="fit-mount",
                        options=[{"label": MOUNT_LABELS[m], "value": m} for m in MOUNTS],
                        value="glycol",
                    ),
                    label("Fit range (°2θ)"),
                    dcc.RangeSlider(id="fit-range", min=2.0, max=45.0, step=0.5, value=[4.0, 34.0]),
                    label("Orientation"),
                    dcc.Dropdown(
                        id="fit-exclusive",
                        options=[
                            {"label": "One per composition (recommended)",
                             "value": "composition"},
                            {"label": "One per phase (as Clayfit)", "value": "phase"},
                            {"label": "Every entry free", "value": "free"},
                        ],
                        value="composition",
                        clearable=False,
                    ),
                    label("Layer spacing"),
                    dcc.Dropdown(
                        id="fit-lattice",
                        options=[
                            {"label": "As calculated (recommended)", "value": "0"},
                            {"label": "Refine per phase, up to \u00b10.5 %",
                             "value": "0.005"},
                            {"label": "Refine per phase, up to \u00b11 %", "value": "0.01"},
                            {"label": "Refine per phase, up to \u00b12 % (as Clayfit)",
                             "value": "0.02"},
                        ],
                        value="0",
                        clearable=False,
                    ),
                    html.Div(
                        "The library spans the layer spacing on a grid and a specimen's "
                        "spacing is not on the grid, so refining one linked spacing per "
                        "phase covers what lies between. It fits better and, on the one "
                        "mount with an independent refinement to answer against, it "
                        "answers worse: the freedom is spent moving kaolinite onto the "
                        "chlorite 002 rather than on finding kaolinite's own spacing, "
                        "because the calculated intensities are not yet right enough to "
                        "object. Left off until they are.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
                    html.Div(
                        "The March–Dollase parameter r is a distribution: P(α;r) "
                        "already spans aligned to random, and how much of each it holds "
                        "is what r says. So a fit that takes half a composition at "
                        "r = 0.5 and half at r = 1 counts the randomly oriented "
                        "crystallites twice, and the r it reports is the mean of two "
                        "values — the orientation of nothing. One per composition "
                        "forbids that while leaving the compositions free; one per "
                        "phase is Clayfit's stricter rule.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
                    label("Expandable clays"),
                    dcc.Checklist(
                        id="fit-air-dried",
                        options=[{"label": " Use the air-dried mount as evidence",
                                  "value": "on"}],
                        value=["on"],
                    ),
                    dcc.Checklist(
                        id="fit-diagnostic",
                        options=[{"label": " Require a diagnostic reflection",
                                  "value": "on"}],
                        value=["on"],
                        style={"marginTop": "6px"},
                    ),
                    html.Div(
                        "A non-negative fit will take a little of a broad mixed-layer "
                        "pattern because it improves a strong reflection the pattern "
                        "overlaps, and predict a low-angle reflection the measurement "
                        "does not contain. That reflection is the whole evidence for the "
                        "expandable clay, so each one is made to justify it: the fit is "
                        "run again without that entry, and what the measurement leaves "
                        "there must clear both the noise and a quarter of what the entry "
                        "predicts.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
                    html.Div(
                        "An air-dried mount cannot be fitted \u2014 its interlayer may hold "
                        "zero to three layers of water depending on the cation and the "
                        "humidity, which is what glycol solvation exists to remove \u2014 so "
                        "it is read as evidence instead: whether the basal reflections "
                        "moved. A specimen whose 001 did not move on glycolation has no "
                        "expandable clay, and the size of the movement it could have "
                        "hidden puts a ceiling on how much it may carry.",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
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
                    label("Weight % calibration"),
                    dcc.Dropdown(
                        id="fit-calibration",
                        options=[
                            {"label": "As calculated (every factor 1)", "value": "none"},
                            {"label": "Clayfit family calibration", "value": "clayfit"},
                        ],
                        value="none",
                        clearable=False,
                    ),
                    html.Button("Run NNLS fit", id="fit-run", n_clicks=0,
                                style={"marginTop": "10px"}),
                    html.Div(id="fit-status", style={"marginTop": "10px"}),
                ],
                style=CONTROL_PANEL,
            ),
            html.Div(
                [
                    working(dcc.Tabs(
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
                    )),
                    working(html.Div(
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
                    )),
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
                               marks={0: "0", 0.5: "0.5", 1: "1"}),
                    label("Junction probability P(host→smectite)  — 0 uses random stacking"),
                    dcc.Slider(id="sim-pab", min=0.0, max=1.0, step=0.01, value=0.0,
                               marks={0: "random", 1: "1"}),
                    html.Hr(),
                    label("Mean crystallite thickness (layers)"),
                    dcc.Slider(id="sim-csds", min=2, max=60, step=1, value=10,
                               marks={2: "2", 20: "20", 40: "40", 60: "60"}),
                    label("CSDS width β (in ln N)"),
                    dcc.Slider(id="sim-beta", min=0.0, max=1.0, step=0.05, value=0.35,
                               marks={0: "0", 0.5: "0.5", 1: "1"}),
                    label("March-Dollase orientation r"),
                    dcc.Slider(id="sim-po", min=0.1, max=1.0, step=0.05, value=0.2,
                               marks={0.1: "0.1", 0.5: "0.5", 1.0: "1 (random)"}),
                    label("Glycolated smectite d(001) (Å)"),
                    dcc.Slider(id="sim-dsmectite", min=15.0, max=18.0, step=0.02, value=16.86,
                               marks={15: "15", 16.86: "16.86", 18: "18"}),
                    label("2θ range (degrees)"),
                    dcc.RangeSlider(id="sim-range", min=2.0, max=70.0, step=1.0, value=[2.0, 40.0]),
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
            html.Div([working(dcc.Graph(id="sim-graph",
                                        figure=empty_figure("Adjust the controls")))],
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
            # The instrument, on every tab.  It decides the width and so the
            # height of every calculated peak, it comes from two different
            # places - the measurement and the library - and when those two
            # disagree the fit is short of intensity at every strong peak with
            # no other symptom.  A value that matters that much does not belong
            # in a status line under a button that has to be pressed to see it.
            html.Div(id="instrument-strip", style={
                "margin": "6px 0 10px 0", "padding": "6px 10px",
                "border": "1px solid #dddddd", "borderRadius": "6px",
                "background": "#fafafa", "fontSize": "0.8rem", "color": "#444",
            }),
            dcc.Interval(id="instrument-tick", interval=1500, n_intervals=0),
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
        Input("directory", "value"),
        prevent_initial_call=True,
    )
    def scan_directory(_clicks, directory):
        if not directory:
            # An empty box is someone clearing it to type or paste something
            # else, not a request to scan nothing; only the button complains.
            if callback_context.triggered_id == "directory":
                raise PreventUpdate
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
        Output("directory", "value"),
        Output("load-status", "children", allow_duplicate=True),
        Input("browse", "n_clicks"),
        State("directory", "value"),
        prevent_initial_call=True,
    )
    def browse_for_folder(_clicks, current):
        """Put the operating system's folder chooser in front of the analyst.

        Setting the path also scans it, because ``scan_directory`` listens to
        the box rather than only to its button - so Browse is one click rather
        than two.
        """
        if not served_locally():
            return no_update, error_message(RuntimeError(REMOTE_CHOOSER))
        chosen, problem = ask_for_folder(current)
        if problem:
            return no_update, error_message(RuntimeError(problem))
        if chosen is None:
            raise PreventUpdate  # dismissed; leave everything as it was
        return chosen, no_update

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
        if STATE.instrument_note:
            messages.append(STATE.instrument_note)

        figure = go.Figure()
        for mount in STATE.loaded_mounts():
            pattern = STATE.mounts[mount].raw
            figure.add_scatter(
                x=pattern.two_theta, y=pattern.intensity, name=MOUNT_LABELS[mount],
                line={"color": COLORS[mount], "width": 1},
            )
        return style_axes(figure, "Counts"), html.Ul([html.Li(text) for text in messages])

    @app.callback(
        Output("load-screen", "children"),
        Input("load-status", "children"),
        Input("db-status", "children"),
    )
    def screen_on_load(_loaded, _database):
        """Identify the accompanying minerals as soon as the scans are read.

        Before the zero error, before a background and without a reference
        library, which is where Clayfit does it and is possible for the reason
        set out in :func:`clayquant.detection.screen_treatments`: each mount is
        aligned on its own quartz lines, so nothing the operator has yet to set
        is needed, and the test is which peaks the three treatments have in
        common rather than how well one of them can be explained.

        It fires on either input because it needs both, and which arrives first
        is up to the operator: the structure database can be loaded before the
        scans or after them.
        """
        loaded = STATE.loaded_mounts()
        if len(loaded) < 2 or not STATE.phase_database:
            STATE.screen = None
            if len(loaded) >= 2:
                return html.Div(
                    "Load the structure database in step 4 and the accompanying "
                    "minerals will be identified here, from the three scans alone - "
                    "no zero error, no background and no reference library needed.",
                    style={"fontSize": "0.82rem", "color": "#666", "marginTop": "8px"},
                )
            return ""

        mounts = {name: STATE.mounts[name].raw for name in loaded}
        try:
            screen = screen_treatments(
                mounts, STATE.phase_database, instrument=gui_instrument(),
                only={name for name in COMMON_IN_CLAY_SEPARATES
                      if name in STATE.phase_database} or None,
            )
        except Exception as exc:  # noqa: BLE001 - a screen must not stop the workflow
            STATE.screen = None
            return error_message(exc)
        STATE.screen = screen
        if not screen.findings:
            return html.Div([html.B("No accompanying mineral was found. "), screen.note],
                            style={"fontSize": "0.82rem", "color": "#666",
                                   "marginTop": "8px"})

        rows = [
            html.Tr([
                html.Td(evidence.name),
                html.Td(f"{100.0 * evidence.score:.0f}%"),
                html.Td(f"{evidence.n_matched} of {evidence.n_expected}"),
                html.Td(f"{evidence.cell_deviation_percent:+.1f}%"),
                html.Td(f"{evidence_score(evidence):.2f}"),
            ])
            for evidence in screen.findings[:15]
        ]
        width = ("" if not np.isfinite(screen.width)
                 else f" The quartz lines are {screen.width:.3f}\u00b0 wide.")
        return html.Div(
            [
                html.H4(f"{len(screen.findings)} accompanying minerals stand in all "
                        f"{len(mounts)} mounts",
                        style={"marginBottom": "4px"}),
                html.Div(
                    "Found before anything has been set, from the peaks the mounts have "
                    "in common: the clays move between air, glycol and heat and nothing "
                    "else does. Step 4 is where they are chosen and fitted; this is what "
                    "the specimen supports." + width,
                    style={"fontSize": "0.8rem", "color": "#555", "marginBottom": "6px"},
                ),
                html.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Phase"), html.Th("Coverage"),
                            html.Th("Stable lines"), html.Th("Cell"), html.Th("Score"),
                        ])),
                        html.Tbody(rows),
                    ],
                    style={"fontSize": "0.85rem"},
                ),
                html.Div(screen.note, style={"fontSize": "0.78rem", "color": "#555",
                                             "marginTop": "6px"}),
                html.Div(
                    "Coverage is the share of the phase's own calculated intensity that "
                    "stands on a stable peak, and the score is that rewarded for resting "
                    "on several lines. Cell is the coherent scaling the phase needed: at "
                    "0% it fits the database entry as written, and one near the 2% "
                    "allowance is weak evidence.",
                    style={"fontSize": "0.78rem", "color": "#666", "marginTop": "4px"},
                ),
            ],
            style={"marginTop": "10px"},
        )

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
        Output("bg-order-box", "style"),
        Output("bg-sv-box", "style"),
        Output("bg-qpa-box", "style"),
        Output("bg-anchor-box", "style"),
        Output("bg-inverse-box", "style"),
        Input("bg-kind", "value"),
    )
    def show_the_controls_that_apply(kind):
        """Hide the parameters the chosen model does not have.

        Each model has its own; leaving all of them on screen invites someone to
        move a slider that does nothing, which then reads as a defect in the
        program.  The order belongs to the two polynomial models, the
        granularity and bending factor to the erosion, the five windows to the
        percentile model, and the rolling ball to the three models that are
        fitted to its anchor points.

        The A/2theta term is offered on every model except the erosion, which is
        where Clayfit offers it.  It is withheld there because the erosion
        returns a convex background exactly as it found it, and the low-angle
        rise from air scatter and the direct beam is convex, so it is already in
        the curve: adding A/2theta on top would count it twice.
        """
        hidden = {"display": "none"}
        shown = {}
        return (
            shown if kind in ("polynomial", "chebyshev") else hidden,
            shown if kind == "sonneveld-visser" else hidden,
            shown if kind == QPA_MODEL_NAME else hidden,
            shown if kind in ("exponential", "polynomial", "chebyshev") else hidden,
            hidden if kind == "sonneveld-visser" else shown,
        )

    @app.callback(
        Output("bg-granularity", "value"),
        Output("bg-bending", "value"),
        Output("bg-suggestion", "children"),
        Input("load-status", "children"),
        prevent_initial_call=True,
    )
    def suggest_the_starting_values(_loaded):
        """Read starting values for the two parameters off the glycol mount.

        The glycol mount, because that is the one the quantification is made on
        (Sec. 2.1) and the one whose broad interstratified intensity the
        background most easily eats.  Any of the three would do for the
        arithmetic; this is the one whose answer matters.

        It fires on loading and not afterwards, so an adjustment by hand is not
        undone by the program a moment later.
        """
        loaded = STATE.loaded_mounts()
        if not loaded:
            raise PreventUpdate
        mount = "glycol" if "glycol" in loaded else loaded[0]
        pattern = STATE.mounts[mount].corrected()
        try:
            suggestion = suggest_sonneveld_visser(pattern.two_theta, pattern.intensity)
        except Exception as exc:  # noqa: BLE001 - a starting value is not worth failing over
            return no_update, no_update, f"Could not read a starting value: {exc}"
        source = MOUNT_LABELS[mount]
        if mount != "glycol":
            source += " (the glycol mount is not loaded)"
        return (
            suggestion.granularity,
            suggestion.bending,
            f"Starting values read from the {source} scan. {suggestion.note}",
        )

    @app.callback(
        Output("bg-sv-note", "children"),
        Input("bg-granularity", "value"),
        Input("bg-mount", "value"),
        Input("load-status", "children"),
    )
    def report_the_reach(granularity, mount, _loaded):
        """Say what a granularity in points comes to in degrees on this scan.

        Granularity is counted in points because that is what the paper and
        HighScore both mean by it, and the consequence is that the same number
        reaches different distances on scans of different step size - 20 points
        is 2.4 deg on a 0.01 deg film scan and 4 deg on a 0.0167 deg one. The
        number that matters is the one in degrees, so it is shown.
        """
        state = STATE.mounts[mount]
        if state.raw is None or len(state.raw.two_theta) < 2:
            return "Load a pattern to see what this granularity reaches in degrees."
        step = float(np.mean(np.diff(state.raw.two_theta)))
        reach = sonneveld_visser_reach(int(granularity), step, SONNEVELD_VISSER_ITERATIONS)
        return (
            f"{int(granularity)} points is {int(granularity) * step:.3f}\u00b0 between samples, "
            f"which half-removes features up to about {reach:.1f}\u00b0 wide."
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
        Input("bg-order", "value"),
        Input("bg-inverse", "value"),
        Input("bg-amplitude", "value"),
        Input("bg-radius", "value"),
        Input("bg-stride", "value"),
        Input("bg-qpa-window", "value"),
        Input("bg-qpa-baseline-smooth", "value"),
        Input("bg-qpa-baseline-order", "value"),
        Input("bg-qpa-final-smooth", "value"),
        Input("bg-qpa-final-order", "value"),
        Input("bg-granularity", "value"),
        Input("bg-bending", "value"),
        Input("bg-apply", "n_clicks"),
        Input("bg-apply-all", "n_clicks"),
    )
    def update_background(mount, _loaded, _zeroed, kind, order, inverse, amplitude,
                          radius, stride, qpa_window, qpa_baseline_smooth,
                          qpa_baseline_order, qpa_final_smooth, qpa_final_order,
                          granularity, bending, _apply, _apply_all):
        state = STATE.mounts[mount]
        if state.raw is None:
            return empty_figure(f"{MOUNT_LABELS[mount]} is not loaded"), ""
        pattern = state.corrected()
        try:
            settings = background_settings_from_controls(
                kind, order, inverse, amplitude, radius, stride,
                qpa_window, qpa_baseline_smooth, qpa_baseline_order,
                qpa_final_smooth, qpa_final_order, granularity, bending,
            )
        except (TypeError, ValueError) as exc:
            return empty_figure("Check the background settings"), error_message(exc)

        try:
            fit = fit_background_from_controls(pattern, **settings)
        except Exception as exc:  # noqa: BLE001
            return empty_figure("Fit failed"), error_message(exc)

        triggered = [item["prop_id"] for item in callback_context.triggered]
        applied = ""
        if any("bg-apply-all" in prop for prop in triggered):
            for other in STATE.loaded_mounts():
                other_state = STATE.mounts[other]
                other_pattern = other_state.corrected()
                other_fit = fit_background_from_controls(other_pattern, **settings)
                other_state.background_fit = other_fit
            applied = " Applied to all loaded mounts."
        elif any("bg-apply" in prop for prop in triggered):
            state.background_fit = fit
            applied = f" Applied to {MOUNT_LABELS[mount]}."
        if applied:
            # Re-measure the peak widths now that the background is gone.  A
            # width read off a pattern that still carries its background is read
            # at a half maximum that is too high up the peak, so it comes out
            # too narrow, and a calculated pattern too narrow is one the fit
            # cannot match at the peak top either.
            STATE.take_instrument_from(pattern, background=fit)

        background = fit(pattern.two_theta)
        figure = go.Figure()
        # Fixed colours, not the mount colour: for the glycol mount COLORS[mount]
        # is the same green as the subtracted trace, and the two curves then lie
        # on top of each other indistinguishably.
        figure.add_scatter(x=pattern.two_theta, y=pattern.intensity, name="measured",
                           line={"color": "#444444", "width": 1})
        # Every model that was not chosen, faintly.  The choice between them is
        # the choice that matters and it cannot be made without seeing them:
        # on a clay mount they differ by hundreds of counts in the middle of
        # the range, which is more than any order changes anything.
        for name, colour in (("exponential", "#1f77b4"), ("polynomial", "#2ca02c"),
                             ("chebyshev", "#9467bd"), ("als", "#8c564b"),
                             (QPA_MODEL_NAME, "#e377c2"),
                             ("sonneveld-visser", "#17becf")):
            if name == kind:
                continue
            try:
                other = fit_background_from_controls(
                    pattern, **{**settings, "kind": name}
                )(pattern.two_theta)
            except Exception:  # noqa: BLE001 - a comparison must not break the view
                continue
            figure.add_scatter(
                x=pattern.two_theta, y=other,
                name=CLAYFIT_MODEL_LABELS[name],
                line={"color": colour, "width": 1, "dash": "dot"},
                opacity=0.55,
            )
        figure.add_scatter(x=pattern.two_theta, y=background,
                           name=CLAYFIT_MODEL_LABELS[kind],
                           line={"color": "#ff7f0e", "width": 3})
        # The anchor points, because they are what three of the six models are
        # fitted to and the only thing the operator can move them by is the
        # rolling ball: a polynomial that misses the measurement usually misses
        # it where the ball found no point to hold it down.
        if fit.anchor_two_theta.size:
            figure.add_scatter(
                x=fit.anchor_two_theta, y=fit.anchor_intensity,
                name=f"anchor points ({fit.anchor_two_theta.size})",
                mode="markers",
                marker={"color": "#d62728", "size": 7, "symbol": "circle-open",
                        "line": {"width": 2}},
            )
        figure.add_scatter(x=pattern.two_theta,
                           y=fit.subtract(pattern.two_theta, pattern.intensity),
                           name="subtracted", line={"color": "#2ca02c", "width": 1})
        style_axes(figure, "Counts")

        return figure, html.Div(background_report(pattern, fit, background) + applied)

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
        # The whole collected range, not the search window and a margin.  What
        # the eye needs here is a reflection that cannot move between the two
        # treatments: if every peak in view has shifted, that is a zero error
        # and not swelling, and there is no way to tell the two apart without
        # something invariant in the picture.  Quartz is that something, and its
        # 101 at 26.6 deg was outside the old range.
        low = min(air.two_theta[0], glycol.two_theta[0])
        high = max(air.two_theta[-1], glycol.two_theta[-1])
        figure.update_xaxes(range=[low, high])
        marks = []
        for spacing, label_text in ((QUARTZ_100_D, "quartz 100"), (QUARTZ_101_D, "quartz 101")):
            position = reference_two_theta(spacing)
            if not low <= position <= high:
                continue
            figure.add_vline(x=position, line={"color": "#777777", "dash": "dot", "width": 1},
                             annotation_text=label_text, annotation_position="top")
            found = []
            for mount, pattern in (("air", air), ("glycol", glycol)):
                near = np.abs(pattern.two_theta - position) < 0.35
                if not near.any() or pattern.intensity[near].max() <= 0:
                    continue
                peak = float(pattern.two_theta[near][int(np.argmax(pattern.intensity[near]))])
                found.append((mount, peak))
            if len(found) == 2:
                marks.append(
                    f"{label_text}: air {found[0][1]:.3f}\u00b0, glycol {found[1][1]:.3f}\u00b0, "
                    f"difference {found[1][1] - found[0][1]:+.3f}\u00b0"
                )
        style_axes(figure, "Counts")

        verdict = (
            "Expandable clay present." if result.expandable_detected else "No expansion detected."
        )
        note = (
            html.Div(
                [html.Div(line) for line in marks]
                + [html.Div(
                    "Quartz does not respond to glycol, so a difference here is a zero error "
                    "between the two scans and not swelling. Correct it in step 2 before "
                    "reading the 001 shift.",
                    style={"color": "#666"},
                )],
                style={"marginTop": "8px", "fontSize": "0.8rem"},
            )
            if marks else None
        )
        return figure, html.Div([html.B(verdict), html.Div(result.summary()), note])

    @app.callback(
        Output("instrument-strip", "children"),
        Input("instrument-tick", "n_intervals"),
    )
    def show_instrument(_tick):
        return instrument_strip()

    @app.callback(
        Output("detect-only", "value"),
        Input("detect-common", "n_clicks"),
        Input("detect-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def preset_restriction(_common, _clear):
        if "detect-clear" in callback_context.triggered[0]["prop_id"]:
            return []
        return [name for name in COMMON_IN_CLAY_SEPARATES if name in STATE.phase_database]

    @app.callback(
        Output("db-status", "children"),
        Output("detect-only", "options"),
        Input("db-load", "n_clicks"),
        State("db-path", "value"),
        prevent_initial_call=True,
    )
    def load_database(_clicks, path):
        try:
            count = STATE.load_phase_database(resolve_user_path(path))
        except Exception as exc:  # noqa: BLE001
            return error_message(exc), []
        clays = sum(1 for name in STATE.phase_database if is_clay_phase(name))
        options = [
            {"label": name, "value": name}
            for name in sorted(STATE.phase_database)
            if not is_clay_phase(name)
        ]
        return html.Div(
            f"{count} phases loaded ({clays} phyllosilicates, which are left to the clay "
            f"library and not offered here)."
        ), options

    @app.callback(
        Output("db-path", "value"),
        Output("db-status", "children", allow_duplicate=True),
        Input("db-browse", "n_clicks"),
        State("db-path", "value"),
        prevent_initial_call=True,
    )
    def browse_for_database(_clicks, current):
        if not served_locally():
            return no_update, error_message(RuntimeError(REMOTE_CHOOSER))
        chosen, problem = ask_for_file(current, ["Phase database|*.json"])
        if problem:
            return no_update, error_message(RuntimeError(problem))
        if chosen is None:
            raise PreventUpdate
        return chosen, no_update

    @app.callback(
        Output("lib-path", "value"),
        Output("lib-status", "children", allow_duplicate=True),
        Input("lib-browse", "n_clicks"),
        State("lib-path", "value"),
        prevent_initial_call=True,
    )
    def browse_for_library(_clicks, current):
        if not served_locally():
            return no_update, error_message(RuntimeError(REMOTE_CHOOSER))
        chosen, problem = ask_for_file(current, ["Pattern library|*.npz"])
        if problem:
            return no_update, error_message(RuntimeError(problem))
        if chosen is None:
            raise PreventUpdate
        return chosen, no_update

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
        State("detect-tick", "value"),
        State("detect-only", "value"),
        State("detect-coverage", "value"),
        prevent_initial_call=True,
    )
    def run_detection(_clicks, mount, allowance, min_score, snr, search_range, tick_above,
                      only, min_coverage):
        """Search for accompanying minerals, three ways, in order of preference.

        With two or more mounts loaded the search asks which peaks stand at the
        same angle in all of them.  That is a different measurement rather than
        a better statistic, and it is the only one of the three that can find a
        minor phase whose strongest line is overlapped: rutile is 1.5 per cent
        of one real separate, its 110 falls on a feldspar reflection, and no
        amount of fitting one scan recovers it - thirty-five phases explain more
        of that pattern than it does.  Its two lines are at 27.47 and 36.07
        degrees in the air-dried, glycolated and heated scans alike, and on that
        test it comes second of eight with 100 per cent coverage while
        anatase, faujasite, graphite and calciolangbeinite have no stable line
        at all.

        With one mount and a clay library, the candidates are fitted in
        competition with the clays.  With one mount and no library, peak
        positions are matched, which is much the weakest of the three: it ranked
        quartz 101st of 169 on the same separate.  The dialog says which test
        produced the list it is showing.
        """
        if not STATE.phase_database:
            return no_update, None, error_message(
                ValueError("Load a phase database first.")
            )
        pattern = STATE.mounts[mount].subtracted()
        if pattern is None:
            return no_update, None, error_message(
                ValueError(f"{MOUNT_LABELS[mount]} is not loaded.")
            )

        loaded = [
            STATE.mounts[name].subtracted() for name in STATE.loaded_mounts()
            if STATE.mounts[name].subtracted() is not None
        ]
        stable = len(loaded) >= 2
        # With a clay library loaded the candidates can be fitted in competition
        # with the clays, which discriminates far better than matching peak
        # positions; without one, fall back to position matching.
        screened = STATE.library is not None
        peaks: list = []
        try:
            if stable:
                findings, peaks = stable_phases(
                    loaded,
                    STATE.phase_database,
                    two_theta_range=tuple(search_range),
                    instrument=gui_instrument(),
                    min_coverage=float(min_coverage) / 100.0,
                    min_signal_to_noise=float(snr),
                    only=set(only) if only else None,
                )
            elif screened:
                findings = screen_phases(
                    STATE.mounts[mount].corrected(),
                    STATE.phase_database,
                    STATE.library.select(march_dollase=[0.1, 0.5, 1.0]),
                    background=STATE.mounts[mount].background_fit,
                    two_theta_range=tuple(search_range),
                    instrument=gui_instrument(),
                    min_share=float(min_score) / 1000.0,
                    cell_allowance=float(allowance) / 100.0,
                    only=set(only) if only else None,
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
                            "Phases whose reflections stand at the same angle in every mount:"
                            if stable else
                            "Evidence for the following main minerals has been found in the data:"
                            if screened else
                            "Phases whose expected lines fall where the data has peaks:",
                            style={"marginTop": 0},
                        ),
                        # Without a clay library this is position matching, which
                        # is a far weaker test - and it is weak in a way that looks
                        # strong, because a phase with a dozen expected lines in a
                        # crowded pattern scores highly whether or not it is there.
                        # Saying so where the list is read, rather than only in the
                        # status line under the button, is the difference between a
                        # shortlist and ten spurious phases entering a fit.
                        None if (screened or stable) else html.Div(
                            [
                                html.Strong("This is the weaker test. "),
                                "No clay library is loaded, so these are matches on peak "
                                "positions alone, not on how much of the pattern each phase "
                                "explains. Scores of 95–100 % are normal here for phases "
                                "that are not present: a candidate with a dozen expected "
                                "lines will find them all in a crowded pattern. Nothing is "
                                "ticked for that reason. Load a clay library on the ",
                                html.Em("Fit and results"),
                                " tab and search again, and the list becomes each phase's "
                                "share of the pattern, fitted in competition with the clays "
                                "and with the other candidates.",
                            ],
                            style={"background": "#fdf3f2", "border": "1px solid #d9a9a2",
                                   "borderRadius": "4px", "padding": "8px 10px",
                                   "margin": "6px 0", "fontSize": "0.85rem"},
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
                            # Ticked by default only where the share is clear, at a
                            # level the analyst can move.  The screen's tail at one
                            # or two percent is largely least-squares slack rather
                            # than real mineralogy, so it is listed but left to be
                            # judged - but the level was fixed at 3 %, invisible,
                            # and a share is not a property of one phase: it is what
                            # is left after every other candidate in the database has
                            # taken what it can explain.  Add a candidate that happens
                            # to cover the same reflections and a real mineral drops
                            # below the line with no change in the evidence for it.
                            # A number that decides what the operator sees should not
                            # be one they cannot see.
                            # Only the competitive screen pre-ticks anything.  A
                            # position-matching score is not a measure of how much
                            # of the pattern a phase accounts for, so there is no
                            # honest threshold to pre-tick on - and pre-ticking the
                            # top of that list put galena, otavite and cassiterite
                            # in front of an operator of a clay separate, one click
                            # from entering the fit.
                            value=phases_to_tick(findings, screened, tick_above, stable),
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
        if stable:
            method = (
                f"scored on the {len(peaks)} peaks that stand at the same angle in all "
                f"{len(loaded)} loaded mounts. The clays move between treatments and "
                f"nothing else does, so this is the one test that finds a minor phase "
                f"whose strongest line is overlapped"
            )
        elif screened:
            method = "fitted in competition with the clay library, on one mount"
        else:
            method = "matched on peak positions (load a clay library for the sharper test)"
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
        built_blind = False
        try:
            if "lib-build" in triggered:
                # The library has to be calculated with the instrument the
                # measurement was made on, and the only way to know that is to
                # have a measurement.  Building before loading one is the easy
                # mistake here and it is not a small one: it used the 280 mm
                # default on a 240 mm instrument and a generic peak width, and
                # the whole library then has reference peaks of the wrong width,
                # which the fit can only absorb as missing intensity.
                built_blind = STATE.instrument is None
                library = build_library(instrument=gui_instrument())
                library.save(path)
            else:
                library = PatternLibrary.load(resolve_user_path(path))
        except Exception as exc:  # noqa: BLE001
            return error_message(exc)
        STATE.library = library
        STATE.library_path = Path(path)
        children = [html.Pre(library.describe(), style={"fontSize": "0.75rem", "margin": 0})]
        if built_blind:
            children.append(html.Div(
                "Built before any measurement was loaded, so it uses the default "
                "geometry (280 mm goniometer radius, 0.5\u00b0 divergence slit) and a "
                "generic peak width rather than yours. Load a mount in step 1 and "
                "build again if you want the reference peaks to have the width of "
                "your diffractometer.",
                style={
                    "marginTop": "8px", "padding": "8px", "background": "#fff4e5",
                    "border": "1px solid #f0ad4e", "borderRadius": "6px",
                    "fontSize": "0.8rem",
                },
            ))
        elif "lib-build" in triggered:
            children.append(html.Div(
                STATE.instrument_note,
                style={"marginTop": "8px", "fontSize": "0.78rem", "color": "#555"},
            ))
        if STATE.instrument is not None:
            warning = describe_instrument_mismatch(library, STATE.instrument)
            if warning:
                children.append(html.Div(warning, style={
                    "marginTop": "8px", "padding": "8px",
                    "background": "#fff4e5", "border": "1px solid #f0ad4e",
                    "borderRadius": "6px", "fontSize": "0.8rem",
                }))
        return html.Div(children)

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
        State("fit-calibration", "value"),
        State("fit-exclusive", "value"),
        State("fit-air-dried", "value"),
        State("fit-diagnostic", "value"),
        State("fit-lattice", "value"),
        prevent_initial_call=True,
    )
    def run_fit(_clicks, mount, fit_range, orientations, subtract, calibration_choice,
                exclusive, use_air_dried, use_diagnostic, lattice):
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

        # The air-dried mount is evidence, not a second pattern to fit: its
        # smectite interlayer may hold zero to three layers of water depending
        # on the cation and the humidity, so there is no air-dried structure to
        # calculate.  What it gives is whether the basal reflections *moved*,
        # measured off the two scans, and that puts a ceiling on how much
        # expandable layer the specimen can contain without having shown it.
        air_note = ""
        air_state = STATE.mounts.get("air")
        wanted = bool(use_air_dried) and "on" in (use_air_dried or [])
        if wanted and mount == "glycol" and air_state is not None and air_state.is_loaded:
            try:
                evidence = shift_evidence(
                    air_state.subtracted() if use_background else air_state.corrected(),
                    state.subtracted() if use_background else state.corrected(),
                )
                bound = expandable_bound(evidence, library)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                air_note = f"The air-dried mount could not be read as evidence: {exc}"
            else:
                air_note = bound.status
                if not bound.unrestricted and bound.excluded:
                    library = _library_subset(library, bound.allowed)
        elif wanted:
            air_note = (
                "The air-dried mount was not used: it restrains the glycolated mount, and "
                + ("no air-dried mount is loaded."
                   if mount == "glycol" else f"{MOUNT_LABELS[mount]} is being fitted.")
            )
        constraints: list = []

        try:
            selection = None
            lattice_note = ""
            if exclusive == "free":
                result = nnls_fit(
                    state.corrected(),
                    library,
                    background=state.background_fit if use_background else None,
                    range_two_theta=tuple(fit_range),
                    constraints=constraints,
                    treatment=mount,
                )
            else:
                families = (clay_families(library) if exclusive == "phase"
                            else orientation_families(library))
                deviation = float(lattice or 0.0)
                arguments = dict(
                    background=state.background_fit if use_background else None,
                    range_two_theta=tuple(fit_range),
                    constraints=constraints,
                    treatment=mount,
                )
                if deviation > 0.0:
                    selection = select_with_lattice_scaling(
                        state.corrected(), library, families,
                        maximum_deviation=deviation, **arguments,
                    )
                    # The result is a fit of the *moved* patterns, so everything
                    # after this has to read them rather than the originals.
                    library = selection.library or library
                    lattice_note = "Layer spacing refined: " + ", ".join(
                        f"{phase} \u00d7{scale:.4f}"
                        for phase, scale in sorted(selection.lattice_scales.items())
                        if abs(scale - 1.0) > 1e-5
                    )
                else:
                    selection = select_one_per_family(
                        state.corrected(), library, families=families, **arguments,
                    )
                result = selection.result
            # Every expandable entry in the fit now has to justify the
            # reflection that identifies it, which is a thing the residual
            # cannot ask for itself.
            diagnostic_note = ""
            if bool(use_diagnostic) and "on" in (use_diagnostic or []):
                fitted = (
                    library if selection is None
                    else _library_subset(
                        library,
                        np.asarray([library.names.index(name) for name in result.names],
                                   dtype=int),
                    )
                )
                screened = screen_diagnostic_peaks(
                    state.corrected(),
                    fitted,
                    result=result,
                    background=state.background_fit if use_background else None,
                    range_two_theta=tuple(fit_range),
                    constraints=constraints,
                    treatment=mount,
                )
                result = screened.result
                diagnostic_note = screened.status
            calibration = (
                Calibration.from_clayfit(library) if calibration_choice == "clayfit" else None
            )
            quantification = quantify(result, calibration=calibration)
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
        if selection is not None:
            undetermined = [
                family for family, margin in selection.margins.items()
                if margin < 0.02
            ]
            status = html.Div([
                html.Div(status),
                html.Div(
                    f"One pattern per "
                    f"{'composition' if exclusive == 'composition' else 'phase'}: "
                    f"{len(selection.chosen)} chosen in {selection.evaluations} fits"
                    + (f", {selection.screened_out} families set aside by a screening "
                       f"fit" if selection.screened_out else "")
                    + (f" and {len(selection.reinstated)} of those reinstated"
                       if selection.reinstated else "")
                    + ". "
                    + (f"{len(undetermined)} of the choices are not determined by the "
                       f"data - the next best fitted within 2% - so read those "
                       f"compositions and orientations as a range."
                       if undetermined else
                       "Every choice beat its runner-up by more than 2%."),
                    style={"marginTop": "8px", "fontSize": "0.8rem", "color": "#555"},
                ),
            ])
        if lattice_note:
            status = html.Div([
                html.Div(status),
                html.Div(lattice_note, style={
                    "marginTop": "8px", "fontSize": "0.8rem", "color": "#555",
                }),
            ])
        if diagnostic_note:
            status = html.Div([
                html.Div(status),
                html.Div(diagnostic_note, style={
                    "marginTop": "8px", "fontSize": "0.8rem", "color": "#555",
                }),
            ])
        if air_note:
            # Never silent: whether the air-dried mount restrained this fit
            # changes what the expandable clays in the table mean.
            used = bool(constraints)
            status = html.Div([
                html.Div(status),
                html.Div(air_note, style={
                    "marginTop": "8px", "padding": "8px",
                    "background": "#eef6ee" if used else "#fff4e5",
                    "border": f"1px solid {'#8bbf8b' if used else '#f0ad4e'}",
                    "borderRadius": "6px", "fontSize": "0.8rem",
                }),
            ])
        if result.metadata.get("crowded"):
            status = html.Div([
                html.Div(status),
                html.Div(result.metadata["crowded"], style={
                    "marginTop": "8px", "padding": "8px", "background": "#fff4e5",
                    "border": "1px solid #f0ad4e", "borderRadius": "6px",
                    "fontSize": "0.8rem",
                }),
            ])
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


def free_port(host: str, preferred: int) -> int:
    """``preferred`` if it can be bound, otherwise a port that can be.

    Someone who starts ClayQuant from a desktop icon has no terminal to read an
    "address already in use" traceback from, and starting it twice is an easy
    thing to do.  The second copy moves aside instead of failing.
    """
    for candidate in [preferred] + list(range(preferred + 1, preferred + 20)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
            return candidate
    return preferred


def open_when_ready(host: str, port: int, timeout: float = 20.0) -> None:
    """Open the interface in the default browser once the server answers.

    In a background thread, because the server has not started yet: it is
    started by the call that follows, on this thread, and does not return until
    it is stopped.  Opening the browser before then shows a connection error, so
    the thread waits for the port to accept a connection first.
    """

    def wait_and_open() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.5)
                if probe.connect_ex((host, port)) == 0:
                    break
            time.sleep(0.2)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=wait_and_open, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    """Command line entry point for ``clayquant-gui``."""
    parser = argparse.ArgumentParser(prog="clayquant-gui", description="Start the ClayQuant GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument(
        "--browser", dest="browser", action="store_true", default=None,
        help="open the interface in the default browser (the default when there is no terminal)",
    )
    browser.add_argument(
        "--no-browser", dest="browser", action="store_false",
        help="do not open a browser; just serve",
    )
    arguments = parser.parse_args(argv)

    port = free_port(arguments.host, arguments.port)
    url = f"http://{arguments.host}:{port}"

    # Started from a desktop icon there is no terminal to read the address from
    # and no one to type it, so the browser is opened unless asked not to.  From
    # a terminal it is opened too, which is what every local notebook-style tool
    # does; --no-browser is there for a server or a script.
    if arguments.browser is None:
        arguments.browser = True
    if arguments.browser:
        open_when_ready(arguments.host, port)

    app = create_app()
    print(f"ClayQuant GUI at {url}")
    if port != arguments.port:
        print(f"(port {arguments.port} was in use)")
    print("Press Ctrl-C to stop it.")
    app.run(host=arguments.host, port=port, debug=arguments.debug)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
