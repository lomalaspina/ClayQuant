"""Searching a measurement for evidence of accompanying ("main") minerals.

A clay separate is never pure: quartz, feldspars, carbonates and sulphates come
through the separation in varying amounts, and quartz is wanted anyway as the
zero-error and intensity reference.  Before the full-pattern fit it is therefore
worth asking which of the phases in a structure database leave visible traces in
the measurement, so that those - and only those - are offered to the fit.

Two methods
-----------
:func:`screen_phases` is the one to use.  It calculates every candidate onto the
clay library's grid and fits them all *in competition* with the clay patterns and
with each other in a single non-negative least-squares solve, ranking each phase
by the share of the pattern it takes.  A phase wins a share only by explaining
intensity that nothing else can.

:func:`detect_phases` is the fallback for when no clay library is loaded: it
predicts each candidate's strongest reflections and searches the measurement for
them.  Position matching is much the weaker test - on a real clay separate it
ranked quartz 56th of 205 phases, behind ilmenite, pyrite, cementite and
faujasite, while the competitive fit put quartz and albite first and second with
a clean gap to everything below.  Its per-line evidence is still worth showing,
so :func:`screen_phases` attaches it to each result.

Cell tolerance
--------------
Most accompanying minerals are solid solutions - dolomite towards ankerite,
albite towards anorthite, calcite carrying magnesium - so their cell parameters,
and with them their peak positions, differ from the database entry.  A
``cell_allowance`` of 2% by default absorbs that.

Crucially the allowance is applied as one *coherent* scaling of all the phase's
d-spacings, not as a free window around each peak.  A real cell deviation
stretches the whole lattice together, so every line of the phase must move by
the same relative amount; letting each line find its own best position instead
makes the test meaningless.  Applied per peak, a 2% allowance is a window up to
1.6 deg wide at high angle, and on a peak-rich pattern essentially every phase
in a 200-phase database then "matches" - measured here, 190 of 205 scored 100%.
Searching a single shared scale factor and keeping only near-instrumental
agreement at each line removes that: a phase that is not present cannot line all
of its reflections up at one scale.

Scoring
-------
Two things must both hold for a phase to be credible, and the score is their
product:

*presence* - the fraction of the phase's expected intensity that is actually
found above the local noise, so a missing strong line counts far more than a
missing weak one;

*intensity agreement* - the cosine similarity between the expected intensities
and the measured heights at those positions.  If the phase is really there the
heights track the expected intensities; if the lines only happen to fall on
other phases' peaks, they do not.

The result is a ranked shortlist offered for confirmation, never applied
automatically: near-degenerate phases (the feldspars, the spinels, dolomite
against ankerite) cannot be separated on peak positions alone, and that judgment
belongs to the analyst.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .background import snip_baseline
from .bern import is_clay_phase
from .calibration import QUARTZ_100_D, QUARTZ_101_D, reference_two_theta
from .crystal import Crystal
from .pattern import Instrument, Pattern, peak_list, powder_pattern

__all__ = [
    "COMMON_IN_CLAY_SEPARATES",
    "PeakMatch",
    "StablePeak",
    "stable_peaks",
    "stable_phases",
    "PhaseEvidence",
    "detect_phases",
    "screen_phases",
    "DEFAULT_CELL_ALLOWANCE",
    "CLAYFIT_CLUSTER",
    "CLAYFIT_INTENSITY_FLOOR",
    "CLAYFIT_MATCH_TOLERANCE",
    "CLAYFIT_MAX_ZERO_SHIFT",
    "CLAYFIT_MIN_COVERAGE",
    "CLAYFIT_MIN_MATCHES",
    "CLAYFIT_QUARTZ_STRENGTH",
    "CLAYFIT_SHIFT_AGREEMENT",
    "QUARTZ_CALIBRATION",
    "TreatmentPeaks",
    "TripletScreen",
    "match_across_treatments",
    "quartz_zero_shift",
    "evidence_score",
    "screen_treatments",
    "treatment_peaks",
]

QUARTZ_CALIBRATION: tuple[float, float] = (
    reference_two_theta(QUARTZ_100_D),
    reference_two_theta(QUARTZ_101_D),
)
"""The two quartz lines the load-time screen aligns each mount on.

The 100 at 20.86 and the 101 at 26.64 degrees, from the same spacings the zero
error step calibrates on, so a shift measured here and one measured there are
the same quantity.
"""

COMMON_IN_CLAY_SEPARATES: tuple[str, ...] = (
    # Silica
    "Quartz", "Tridymite", "Cristobalite",
    # Feldspars
    "Albite", "Anorthite", "Labradorite", "Microcline", "Orthoclase", "Sanidine",
    # Carbonates
    "Calcite", "Dolomite", "Ankerite", "Aragonite", "Siderite", "Magnesite",
    # Sulphates
    "Gypsum", "Bassanite", "Anhydrite", "Barite", "Celestine",
    # Oxides and hydroxides
    "Rutile", "Anatase", "Ilmenite", "Hematite", "Magnetite", "Goethite",
    "Gibbsite", "Boehmite", "Diaspore", "Corundum", "Spinel",
    # Sulphides, halides, phosphates
    "Pyrite", "Halite", "Sylvite", "Apatite-OH", "Fluorapatite",
    # Silicates that survive a separation
    "Zircon", "Sekaninaite", "Cordierite", "Riebeckite", "Hornblende", "Titanite",
    "Epidote", "Pyrope", "Almandine", "Staurolite", "Tourmaline", "Monazite",
)
"""Phases a clay separate plausibly carries, as a starting restriction.

Not a claim about any specimen - it is a list to edit, and the point of it is
that editing a list of forty is possible and editing a list of two hundred is
not.  A general-purpose structure library holds lead and copper sulphates,
metals and cements, and on one real separate thirty-five of them explained more
of the pattern than rutile did, which on a single scan is not a mistake the
ranking makes but a fact about what one scan determines.  Names absent from the
loaded database are simply skipped.
"""

DEFAULT_CELL_ALLOWANCE = 0.02
"""Relative cell deviation allowed when matching peak positions (2%)."""


def tolerance_window(two_theta: float, cell_allowance: float) -> float:
    """Half-width in degrees of the search window at ``two_theta``.

    From ``d = lambda / (2 sin(theta))``, a relative change in ``d`` of
    ``cell_allowance`` moves the reflection by ``2 tan(theta) * cell_allowance``
    radians.
    """
    theta = math.radians(two_theta) / 2.0
    return float(np.degrees(2.0 * math.tan(theta) * cell_allowance))


@dataclass
class PeakMatch:
    """One expected reflection and what was found near it."""

    expected_two_theta: float
    expected_intensity: float
    window: float
    found_two_theta: float | None
    height: float
    noise: float

    @property
    def signal_to_noise(self) -> float:
        return self.height / self.noise if self.noise > 0 else 0.0

    @property
    def matched(self) -> bool:
        return self.found_two_theta is not None

    @property
    def offset(self) -> float | None:
        if self.found_two_theta is None:
            return None
        return self.found_two_theta - self.expected_two_theta


@dataclass
class PhaseEvidence:
    """How much evidence a measurement holds for one phase."""

    name: str
    is_clay: bool
    score: float
    matches: list[PeakMatch] = field(default_factory=list)
    cell_allowance: float = DEFAULT_CELL_ALLOWANCE
    cell_scale: float = 1.0
    presence: float = 0.0
    intensity_agreement: float = 0.0
    explains_alone: float = 0.0
    """Share of the pattern this phase accounts for against the clays alone.

    The companion to :attr:`score`, which is what it adds once the other phases
    are in.  A phase high here and low there is one whose lines are shared, and
    that is a fact about the pattern rather than a verdict on the phase: rutile
    on one real separate explains 1.4 per cent alone and 0.13 per cent after
    albite, because albite has a reflection at 14 per cent of its own maximum
    sitting on the rutile 110.  An unrestricted search cannot recover such a
    phase, and the two numbers together are what say so.
    """

    position_offset: float = 0.0
    """Correction in degrees the screen applied to the measured angles.

    One number for the whole screen, not per phase, because a zero error moves
    every reflection of every phase alike; it comes back equal to the error in
    the measured angles, so ``-0.05`` means the peaks sat 0.05 deg low.  Large
    and the zero error is worth correcting in its own step, where it is measured
    against a reference instead of inferred from the whole pattern.  Distinct
    from :attr:`mean_offset`, which is what a phase's own lines have left over
    after both this and its cell scale.
    """

    @property
    def cell_deviation_percent(self) -> float:
        """The coherent cell deviation that best lines the phase up, in percent."""
        return 100.0 * (self.cell_scale - 1.0)

    @property
    def n_expected(self) -> int:
        return len(self.matches)

    @property
    def n_matched(self) -> int:
        return sum(1 for match in self.matches if match.matched)

    @property
    def best_signal_to_noise(self) -> float:
        return max((match.signal_to_noise for match in self.matches), default=0.0)

    @property
    def mean_offset(self) -> float:
        """Mean position offset of the matched peaks, in degrees.

        A systematic offset across all lines points at a cell that differs from
        the database entry, or at a residual zero error.
        """
        offsets = [match.offset for match in self.matches if match.offset is not None]
        return float(np.mean(offsets)) if offsets else 0.0

    def summary(self) -> str:
        """One line per phase, with the number that can contradict the others.

        The line count and the signal-to-noise can both look like confirmation
        for a phase that is absent: in a crowded pattern a candidate's lines
        land on peaks whether or not they are its peaks.  ``intensity
        agreement`` is the one figure here that can say no - it compares the
        measured heights at those positions against the phase's own relative
        intensities - so it is shown beside them rather than left in the object.
        """
        return (
            f"{self.name}: {100.0 * self.score:.1f}% "
            f"({self.n_matched}/{self.n_expected} expected lines found, "
            f"best S/N {self.best_signal_to_noise:.0f}, "
            f"intensity agreement {self.intensity_agreement:.2f}, "
            f"{100.0 * self.explains_alone:.1f}% on its own)"
        )


def _local_noise(intensity: np.ndarray, index: int, half_width: int = 40) -> float:
    """Noise level near ``index``, from counting statistics and local scatter."""
    start = max(0, index - half_width)
    stop = min(len(intensity), index + half_width + 1)
    window = intensity[start:stop]
    if window.size < 3:
        return 1.0
    # Median absolute deviation of the first difference: insensitive to peaks.
    differences = np.diff(window)
    mad = float(np.median(np.abs(differences - np.median(differences))))
    scatter = 1.4826 * mad / math.sqrt(2.0)
    return max(scatter, 1.0)


def _scale_positions(positions: np.ndarray, wavelength: float, scale: float) -> np.ndarray:
    """Move reflections to where they sit when every d-spacing is scaled by ``scale``."""
    sine = np.sin(np.radians(positions) / 2.0) / scale
    valid = sine < 1.0
    scaled = np.full(positions.shape, np.nan)
    scaled[valid] = np.degrees(2.0 * np.arcsin(sine[valid]))
    return scaled


def _score_at_scale(
    positions: np.ndarray,
    heights: np.ndarray,
    two_theta: np.ndarray,
    intensity: np.ndarray,
    wavelength: float,
    scale: float,
    match_window: float,
    min_signal_to_noise: float,
) -> tuple[float, float, float, list[PeakMatch]]:
    """Presence, intensity agreement and combined score for one cell scaling."""
    scaled = _scale_positions(positions, wavelength, scale)
    matches: list[PeakMatch] = []
    for expected, weight, target in zip(positions, heights, scaled):
        if not np.isfinite(target):
            continue
        inside = np.flatnonzero(np.abs(two_theta - target) <= match_window)
        if inside.size == 0:
            continue
        best = int(inside[int(np.argmax(intensity[inside]))])
        noise = _local_noise(intensity, best)
        height = float(intensity[best])
        found = float(two_theta[best]) if height >= min_signal_to_noise * noise else None
        matches.append(
            PeakMatch(
                expected_two_theta=float(expected),
                expected_intensity=float(weight),
                window=match_window,
                found_two_theta=found,
                height=height,
                noise=noise,
            )
        )
    if not matches:
        return 0.0, 0.0, 0.0, []

    total = sum(match.expected_intensity for match in matches)
    present = sum(match.expected_intensity for match in matches if match.matched)
    presence = present / total if total > 0 else 0.0

    expected = np.array([match.expected_intensity for match in matches])
    measured = np.array([max(match.height, 0.0) for match in matches])
    agreement = _proportionality(expected, measured)
    return presence * agreement, presence, agreement, matches


def _proportionality(expected: np.ndarray, measured: np.ndarray) -> float:
    """How well measured heights follow the expected intensities, from 0 to 1.

    Each line implies a scale ``k_i = measured_i / expected_i`` at which the
    phase would account for that line.  If the phase is present, every line
    implies a similar scale.  Overlapping reflections of other phases can only
    raise a line, so scatter towards *larger* ``k`` is expected; a line implying
    a much *smaller* scale than the rest is evidence against the phase, because
    its own reflection is missing there.

    The measure compares a low quantile of the implied scales with the median:

        agreement = quantile(k, 0.2) / median(k)

    It is near 1 when all lines agree, and near 0 when the phase needs a line
    the measurement does not have.  A cosine similarity cannot do this - it is
    dominated by whichever single line falls on the largest peak, so any phase
    with one reflection near a strong peak scores about 1 regardless of the
    rest; likewise a one-sided deficit maximised over ``k`` is trivially
    satisfied by the smallest ``k``, which explains nothing.
    """
    usable = expected > 0.05 * expected.max() if expected.size else np.array([], dtype=bool)
    if usable.sum() < 2:
        return 0.0
    ratios = measured[usable] / expected[usable]
    median = float(np.median(ratios))
    if median <= 0.0:
        return 0.0
    return float(np.clip(np.quantile(ratios, 0.2) / median, 0.0, 1.0))


def detect_phases(
    pattern: Pattern,
    crystals: dict[str, Crystal],
    cell_allowance: float = DEFAULT_CELL_ALLOWANCE,
    n_peaks: int = 12,
    min_signal_to_noise: float = 4.0,
    min_score: float = 0.5,
    min_lines: int = 5,
    match_window: float = 0.10,
    scale_steps: int = 41,
    two_theta_range: tuple[float, float] | None = None,
    instrument: Instrument | None = None,
    subtract_background: bool = True,
    include_clays: bool = False,
) -> list[PhaseEvidence]:
    """Rank the phases in ``crystals`` by the evidence for them in ``pattern``.

    Parameters
    ----------
    cell_allowance:
        Relative cell deviation allowed, applied as one coherent scaling of all
        the phase's d-spacings; see the module docstring.
    n_peaks:
        Number of strongest predicted reflections tested per phase.
    min_signal_to_noise:
        Height above the local noise a peak must reach to count as found.
    min_score:
        Smallest score a phase must reach to be reported, where the score is
        presence times intensity agreement.
    match_window:
        Half-width in degrees allowed between a scaled reflection and the
        measured peak, on top of the cell scaling.  This should be of the order
        of the peak width, not of the cell allowance.
    min_lines:
        Fewest reflections a phase must have in range to be judged at all.  A
        phase with three lines can match a peak-rich pattern by coincidence, so
        such phases are not reported.
    min_lines:
        Fewest reflections a phase must have in range to be judged at all.  A
        phase with three or four lines can match a peak-rich pattern by
        coincidence, so such phases are not reported.
    scale_steps:
        Number of cell scalings tried across the allowance.
    include_clays:
        Whether to test the phyllosilicates too.  They are excluded by default:
        the clay minerals are handled by the interstratification library, and
        their basal reflections would otherwise be attributed twice.
    subtract_background:
        Strip the background before searching.  Leave on unless ``pattern`` is
        already background-corrected.

    Returns
    -------
    Evidence for each phase reaching ``min_score``, strongest first.
    """
    if not 0.0 <= cell_allowance < 0.5:
        raise ValueError("cell_allowance must lie in [0, 0.5)")

    two_theta = pattern.two_theta
    intensity = pattern.intensity.astype(float)
    if subtract_background:
        intensity = np.clip(intensity - snip_baseline(two_theta, intensity, window=4.0), 0.0, None)
    if two_theta_range is None:
        two_theta_range = (float(two_theta[0]), float(two_theta[-1]))

    reference = instrument or Instrument()
    wavelength = reference.emission.principal_wavelength

    findings: list[PhaseEvidence] = []
    for name, crystal in crystals.items():
        clay = is_clay_phase(name)
        if clay and not include_clays:
            continue
        try:
            positions, heights = peak_list(crystal, two_theta_range, instrument)
        except Exception:  # noqa: BLE001 - a broken database entry must not stop the scan
            continue
        if len(positions) == 0:
            continue

        strongest = np.sort(np.argsort(heights)[::-1][:n_peaks])
        selected_positions = positions[strongest]
        selected_heights = heights[strongest]

        scales = (
            np.linspace(1.0 - cell_allowance, 1.0 + cell_allowance, scale_steps)
            if cell_allowance > 0
            else np.array([1.0])
        )
        best: tuple[float, float, float, float, list[PeakMatch]] = (0.0, 1.0, 0.0, 0.0, [])
        for scale in scales:
            score, presence, agreement, matches = _score_at_scale(
                selected_positions,
                selected_heights,
                two_theta,
                intensity,
                wavelength,
                float(scale),
                match_window,
                min_signal_to_noise,
            )
            if score > best[0]:
                best = (score, float(scale), presence, agreement, matches)

        score, scale, presence, agreement, matches = best
        if len(matches) >= min_lines and score >= min_score:
            findings.append(
                PhaseEvidence(
                    name=name,
                    is_clay=clay,
                    score=score,
                    matches=matches,
                    cell_allowance=cell_allowance,
                    cell_scale=scale,
                    presence=presence,
                    intensity_agreement=agreement,
                )
            )

    findings.sort(key=lambda evidence: (evidence.score, evidence.best_signal_to_noise), reverse=True)
    return findings


def _weighted_design(library, two_theta: np.ndarray, selection: np.ndarray,
                     root: np.ndarray) -> np.ndarray:
    """The library's patterns on the measured points, weighted, as columns."""
    return (library.matrix(two_theta[selection]).T) * root[:, None]


def step_of(grid: np.ndarray) -> float:
    """Median spacing of a two-theta grid, in degrees."""
    return float(np.median(np.diff(grid))) if grid.size > 1 else 0.02


def _rescaled(grid: np.ndarray, column: np.ndarray, scale: float) -> np.ndarray:
    """A calculated pattern as it would be with the cell scaled by ``scale``.

    A cell ``scale`` times larger puts every reflection at ``scale * d``, so the
    value wanted at angle ``x`` is the original pattern at the angle whose
    spacing is ``d(x) / scale``, and from Bragg's law that is
    ``2 arcsin(scale * sin(x / 2))``.  So one interpolation does it, and it is
    the right shape of freedom: a relative change in the cell, which is what the
    unit cell allowance asks for, not a constant shift in angle.

    The difference between the two matters here.  A cell scale moves a
    reflection by ``2 tan(theta) * scale``, so it is small at low angle and large
    at high; a zero error moves every reflection equally.  Neither can stand in
    for the other exactly, and a cell allowance wide enough to cover a zero
    error at the angles that carry the fit is what the allowance is for.
    """
    sine = scale * np.sin(np.radians(grid) / 2.0)
    angles = np.where(np.abs(sine) < 1.0, 2.0 * np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0))),
                      np.nan)
    resampled = np.interp(angles, grid, column, left=0.0, right=0.0)
    return np.nan_to_num(resampled)


def _scaled_bank(
    grid: np.ndarray,
    patterns: list[np.ndarray],
    points: np.ndarray,
    root: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Each candidate's weighted column at each trial cell scale.

    Shape ``(n_points, n_candidates, n_scales)``.  Allowing the cell to differ
    is what lets a phase be recognised when its published cell is not quite the
    specimen's, and it is also what keeps the sharpest phases in the running:
    quartz's reflections are 0.09 deg wide, and a fit that insists on the
    published cell to better than that has nothing to do with whether quartz is
    present.
    """
    return np.stack([
        np.stack([
            np.interp(points, grid, _rescaled(grid, column, scale)) for scale in scales
        ], axis=1)
        for column in patterns
    ], axis=1) * root[:, None, None]


def _best_shift_gain(bank: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each candidate, its best gain over the trial cell scales, and which one."""
    inner = np.einsum("ipo,i->po", bank, residual)
    norm = np.einsum("ipo,ipo->po", bank, bank)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where((inner > 0.0) & (norm > 0.0), inner**2 / np.maximum(norm, 1e-30), 0.0)
    gain = np.nan_to_num(gain)
    chosen = np.argmax(gain, axis=1)
    return gain[np.arange(gain.shape[0]), chosen], chosen


def _matched_filter_gain(columns: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Weighted least-squares gain of adding each column alone to the fit.

    For one column ``q`` and residual ``r``, the scale that best reduces the sum
    of squares is ``a = <q, r> / <q, q>`` and the reduction it achieves is
    ``<q, r>**2 / <q, q>``.  A negative ``a`` would mean subtracting the phase,
    which a non-negative fit cannot do, so those score zero.

    This is the first step of forward selection, and it is the whole reason for
    using it: it asks what each phase explains *that nothing already in the fit
    explains*, which is a question with one answer.  Ranking by a share taken
    from a single joint solve asks a question with many answers, because a
    library of a thousand broad clay patterns and two hundred minerals over two
    thousand points can distribute one peak among several columns in ways that
    fit equally well and rank quite differently.
    """
    inner = columns.T @ residual
    norm = np.einsum("ij,ij->j", columns, columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where((inner > 0.0) & (norm > 0.0), inner**2 / np.maximum(norm, 1e-30), 0.0)
    return np.nan_to_num(gain)


def screen_phases(
    pattern: Pattern,
    crystals: dict[str, Crystal],
    clay_library,
    background=None,
    two_theta_range: tuple[float, float] = (4.0, 34.0),
    instrument: Instrument | None = None,
    min_share: float = 0.005,
    max_phases: int = 25,
    cell_allowance: float = DEFAULT_CELL_ALLOWANCE,
    include_clays: bool = False,
    scale_steps: int = 9,
    align: float = 0.15,
    min_agreement: float = 0.15,
    only: set[str] | None = None,
    shortlist_size: int = 50,
) -> list[PhaseEvidence]:
    """Rank accompanying minerals by what each explains that nothing else does.

    The clay library is fitted first, on its own.  Then, in rounds, every
    remaining candidate is scored by how much of the *current residual* it can
    account for, the best one is taken into the fit, everything active is
    refitted together, and the residual is recomputed.  A phase's reported score
    is the share of the pattern's weighted sum of squares that its entry into
    the fit removed.

    Why it is done in rounds rather than in one solve.  The obvious method - put
    all two hundred candidates into one non-negative least squares beside the
    clay library and read off the shares - was what this did, and it is not
    identifiable: with a thousand clay patterns and two hundred minerals over a
    couple of thousand points, the columns span the data many times over, and
    which phase is credited with a peak depends on conditioning rather than on
    evidence.  Measured on one real clay separate it ranked quartz first; on
    another, with a library differing only in how many layer spacings it spanned,
    quartz took no share at all and graphite - two reflections, one of which
    lands on the quartz 101 peak - was credited with 8.9% of the pattern.  That
    is not a tuning problem, it is the wrong question.

    Selection in rounds cannot make that mistake, and the reason is worth
    stating: a phase with two reflections can win the first round only if it
    explains the residual better than everything else, and it can never win a
    later one, because once the phase that owns the peak is in the fit there is
    nothing left under those two lines.  A short line list stops being an
    advantage.

    The per-line evidence of :func:`detect_phases` is still attached to each
    result, so the analyst can see which reflections carry the phase, and
    ``intensity_agreement`` says whether the measured heights at those positions
    are in the phase's own proportions - which is what distinguishes a phase
    that is present from one whose lines merely have company.

    Parameters
    ----------
    clay_library:
        The clay :class:`~clayquant.library.PatternLibrary` the candidates are
        fitted against.  Restricting it to a few orientation parameters keeps
        the screen fast.
    min_share:
        Smallest share of the weighted sum of squares a phase must remove to be
        reported.  The rounds stop when nothing reaches it.
    max_phases:
        Most rounds to run, and so the most phases to return.
    align:
        Half-width in degrees of the bodily shift each candidate is allowed, to
        absorb a residual zero error or a coherent cell difference.  Zero fixes
        every phase at its calculated positions.  The shift taken is reported as
        each finding's ``mean_offset``; the same shift on every phase at once is
        a zero error that has not been applied, and is worth applying.
    """
    from scipy.optimize import nnls

    reference = instrument or Instrument()
    grid = clay_library.two_theta
    candidates: list[str] = []
    calculated: list[np.ndarray] = []
    for name, crystal in crystals.items():
        if is_clay_phase(name) and not include_clays:
            continue
        if only is not None and name not in only:
            continue
        try:
            one = powder_pattern(crystal, grid, reference, r_march_dollase=1.0, name=name)
        except Exception:  # noqa: BLE001 - a broken database entry must not stop the screen
            continue
        if float(np.max(one.intensity)) <= 0.0:
            continue
        candidates.append(name)
        calculated.append(one.intensity)
    if not candidates:
        return []

    two_theta = pattern.two_theta
    raw = pattern.intensity.astype(float)
    observed = background.subtract(two_theta, raw) if background is not None else raw
    wavelength = reference.emission.principal_wavelength
    low, high = two_theta_range
    selection = (two_theta >= low) & (two_theta <= high)
    selection &= (two_theta >= grid[0]) & (two_theta <= grid[-1])
    if selection.sum() < 10:
        return []

    # Counting statistics, as in nnls_fit: the variance of a point is the number
    # of photons recorded there, which background subtraction does not reduce.
    root = np.sqrt(1.0 / np.clip(raw[selection], 1.0, None))
    target = observed[selection] * root
    clays = _weighted_design(clay_library, two_theta, selection, root)
    scales = (
        np.array([1.0]) if cell_allowance <= 0.0
        else np.linspace(1.0 - cell_allowance, 1.0 + cell_allowance, scale_steps)
    )

    # Two different things move a reflection away from where it was calculated,
    # and one allowance cannot stand in for the other.  A cell that differs from
    # the database entry moves every reflection of that phase in proportion to
    # tan(theta), so it is small at low angle and large at high; a zero error
    # moves every reflection of every phase by the same amount.  The cell
    # allowance is the first and is per phase.  The second is fitted here, once,
    # as a shift shared by everything - which is what it physically is - because
    # the search has to work on a measurement whose zero error has not been
    # corrected yet, and a 2 % cell allowance does not rescue it: a scale that
    # lines up the low-angle lines throws the high-angle ones off.
    #
    # It is done by moving the measurement rather than the models, so one
    # interpolation per trial serves the clay library and all two hundred
    # candidates at once, and the criterion is the best gain any candidate can
    # make against the clay residual - the quantity the whole screen turns on.
    offset = 0.0
    if align > 0.0:
        trials = np.arange(-align, align + 1e-9, max(step_of(grid), 0.01))
        best_offset, best_value = 0.0, -np.inf
        for trial in trials:
            shifted_observed = np.interp(two_theta[selection] + trial, two_theta, observed)
            shifted_raw = np.interp(two_theta[selection] + trial, two_theta, raw)
            trial_root = np.sqrt(1.0 / np.clip(shifted_raw, 1.0, None))
            trial_target = shifted_observed * trial_root
            trial_clays = (clay_library.matrix(two_theta[selection]).T) * trial_root[:, None]
            trial_coefficients, _ = nnls(trial_clays, trial_target)
            trial_residual = trial_target - trial_clays @ trial_coefficients
            flat = np.column_stack([
                np.interp(two_theta[selection], grid, column) for column in calculated
            ]) * trial_root[:, None]
            value = float(np.max(_matched_filter_gain(flat, trial_residual)))
            if value > best_value:
                best_offset, best_value = float(trial), value
        offset = best_offset
        if offset:
            observed = np.interp(two_theta + offset, two_theta, observed)
            raw = np.interp(two_theta + offset, two_theta, raw)
            root = np.sqrt(1.0 / np.clip(raw[selection], 1.0, None))
            target = observed[selection] * root
            clays = _weighted_design(clay_library, two_theta, selection, root)

    bank = _scaled_bank(grid, calculated, two_theta[selection], root, scales)

    total = float(target @ target)
    if total <= 0.0:
        return []

    # How well each candidate's own line proportions are reproduced, computed
    # once and used as a floor on entry, not as a weight on the gain.
    #
    # Both were tried.  Multiplying the gain by the agreement is the obvious
    # thing and it is wrong, because this agreement is measured against the
    # pattern as it stands, where every line has company: rutile's three
    # reflections sit in a clean stretch and score 0.95, and quartz's six
    # include several standing on clay basal peaks that raise them, so quartz
    # scores 0.56.  Weighting by it therefore rewards phases in empty regions
    # and demoted albite from second to sixteenth.  As a floor it does the one
    # job it can do honestly: it excludes a phase the pattern contradicts
    # outright - graphite at 0.00, whose second reflection has nothing under it
    # at all - without ranking the rest.
    quality = np.ones(len(candidates))
    for index, name in enumerate(candidates):
        try:
            positions, heights = peak_list(crystals[name], two_theta_range, reference)
            if positions.size == 0:
                continue
            strongest = np.sort(np.argsort(heights)[::-1][:12])
            _, _, agreement, _ = _score_at_scale(
                positions[strongest], heights[strongest], two_theta, observed,
                wavelength, 1.0, 0.10, 4.0,
            )
        except Exception:  # noqa: BLE001 - a broken entry keeps its neutral weight
            continue
        quality[index] = float(agreement)

    def fit(active: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        design = (clays if not active else np.column_stack(
            [clays, *(bank[:, index, scale] for index, scale in active)]))
        coefficients, _ = nnls(design, target)
        return coefficients, target - design @ coefficients

    # Selection is on the gain a phase actually achieves when everything already
    # in the fit is free to readjust, not on its projection onto the residual.
    # The projection - the matched filter - is what this used to choose by, and
    # it over-credits: it is the reduction a phase would achieve if it could be
    # scaled freely with nothing else moving, so a phase whose many lines each
    # partly coincide with something is credited for all of them at once.  On a
    # real separate it gave langite 6.6 per cent of the pattern and second
    # place; its true gain, refitted, is 0.18 per cent and twelfth.
    #
    # The refit is done against the clay entries that are actually carrying
    # intensity rather than the whole library - eighteen of four hundred and
    # eighty-one on that separate - because a column at zero does not come off
    # zero for a small addition elsewhere, and the full solve is fifty times
    # dearer.  The full library is used for the residual between rounds and for
    # the shares at the end.
    coefficients, residual = fit([])
    live = np.flatnonzero(coefficients > 0.0)
    base = clays[:, live] if live.size else clays[:, :1] * 0.0

    def true_gain(design: np.ndarray, index: int) -> tuple[float, int]:
        """Best reduction in the weighted sum of squares this phase can add."""
        before = float(np.sum((target - design @ nnls(design, target)[0]) ** 2))
        best, best_scale = 0.0, 0
        for scale in range(bank.shape[2]):
            trial = np.column_stack([design, bank[:, index, scale]])
            after = float(np.sum((target - trial @ nnls(trial, target)[0]) ** 2))
            if before - after > best:
                best, best_scale = before - after, scale
        return best / total, best_scale

    # What each phase explains on its own, against the clays and nothing else.
    # Reported beside the incremental gain because the two together say what one
    # cannot: a phase high on its own and low afterwards is one whose lines are
    # shared, which is a fact about the pattern and not a verdict on the phase.
    # Rutile on that separate explains 1.4 per cent alone and 0.13 per cent once
    # albite is in, because albite has a reflection at 14 per cent of its own
    # maximum sitting on the rutile 110 - and labradorite one at 35 per cent.
    standalone = np.zeros(len(candidates))
    alone_scale = np.zeros(len(candidates), dtype=int)
    for index in range(len(candidates)):
        if quality[index] < min_agreement:
            continue
        standalone[index], alone_scale[index] = true_gain(base, index)

    remaining = {
        index for index in range(len(candidates))
        if quality[index] >= min_agreement and standalone[index] > 0.0
    }
    active: list[tuple[int, int]] = []
    gains: dict[int, float] = {}
    scale_of: dict[int, float] = {}
    floor = min(min_share, 1e-4) / 10.0
    for _ in range(max_phases):
        if not remaining:
            break
        design = (base if not active else np.column_stack(
            [base, *(bank[:, index, scale] for index, scale in active)]))
        # A cheap projection narrows the field; the true gain decides among the
        # survivors.  The narrowing is generous on purpose - the phase that wins
        # a later round on its true gain can sit well down the projection's
        # order, and rutile sits thirty-eighth on it.
        gain, _ = _best_shift_gain(bank, residual)
        shortlist = sorted(remaining, key=lambda index: -gain[index])[:shortlist_size]
        best, best_gain, best_scale = None, 0.0, 0
        for index in shortlist:
            value, scale = true_gain(design, index)
            if value > best_gain:
                best, best_gain, best_scale = index, value, scale
        if best is None or best_gain < floor:
            break
        active.append((best, best_scale))
        remaining.discard(best)
        gains[best] = best_gain
        scale_of[best] = float(scales[best_scale])
        _, residual = fit(active)

    if not active:
        return []
    coefficients, _ = fit(active)
    areas = np.array([
        np.trapezoid(column, two_theta[selection])
        for column in (*clays.T, *(bank[:, index, scale].T for index, scale in active))
    ])
    contributions = np.clip(coefficients, 0.0, None) * np.clip(areas, 0.0, None)
    scattering = float(np.sum(contributions))
    shares = {
        candidates[index]: (
            float(contributions[clays.shape[1] + position] / scattering)
            if scattering > 0.0 else 0.0
        )
        for position, (index, _scale) in enumerate(active)
    }

    intensity = observed
    findings: list[PhaseEvidence] = []
    for index, _scale in active:
        name = candidates[index]
        if shares[name] < min_share:
            continue
        matches: list[PeakMatch] = []
        presence = agreement = 0.0
        try:
            positions, heights = peak_list(crystals[name], two_theta_range, reference)
            positions = _scale_positions(positions, wavelength, scale_of[index])
            usable = np.isfinite(positions)
            positions, heights = positions[usable], heights[usable]
            if len(positions):
                strongest = np.sort(np.argsort(heights)[::-1][:12])
                _, presence, agreement, matches = _score_at_scale(
                    positions[strongest],
                    heights[strongest],
                    two_theta,
                    intensity,
                    wavelength,
                    1.0,
                    0.10,
                    4.0,
                )
        except Exception:  # noqa: BLE001 - evidence is for display only
            matches = []
        findings.append(
            PhaseEvidence(
                name=name,
                is_clay=is_clay_phase(name),
                score=shares[name],
                matches=matches,
                cell_allowance=cell_allowance,
                cell_scale=scale_of[index],
                presence=presence,
                intensity_agreement=agreement,
                explains_alone=float(standalone[index]),
                position_offset=offset,
            )
        )
    findings.sort(key=lambda evidence: evidence.score, reverse=True)
    return findings


# --- Treatment-stable phases -------------------------------------------------


@dataclass
class StablePeak:
    """A reflection that stands at the same angle in every mount."""

    two_theta: float
    heights: tuple[float, ...]
    signal_to_noise: float

    @property
    def height(self) -> float:
        return float(np.mean(self.heights))


def stable_peaks(
    patterns: list[Pattern],
    tolerance: float = 0.05,
    min_signal_to_noise: float = 8.0,
    minimum_height: float = 0.01,
    separation: float = 0.10,
) -> list[StablePeak]:
    """Peaks that stand at the same angle in every mount given.

    The point of the three mounts is that the clays move and nothing else does:
    glycol takes the smectite interlayer from about 15 to 17 A and heating
    collapses it to 10, while quartz, the feldspars, the carbonates and the
    oxides sit exactly where they were.  So a peak at the same angle in all
    three is evidence of a non-clay phase in a way that no amount of fitting one
    mount can be - it is a different measurement, not a better statistic.

    This is worth saying plainly because the alternative was tried at length.
    Ranking candidates by how much of one pattern each explains cannot recover a
    phase whose strong line is overlapped: rutile's 110 falls on a feldspar
    reflection, so with albite in the fit rutile explains 0.13 per cent of the
    pattern and thirty-five phases explain more.  Its two strong lines are
    nevertheless at 27.44 and 36.08 degrees in the air-dried, glycolated and
    heated scans alike, which settles it.

    ``tolerance`` is the angular agreement required between mounts,
    ``minimum_height`` a fraction of each pattern's strongest point, and
    ``separation`` the half-width over which a point must be the largest to
    count as a peak at all.

    The thresholds have to be strict and the first attempt was not.  Taking
    every local maximum at three sigma gave five hundred and twenty-seven
    stable peaks on one specimen, at which density a 0.05 degree window catches
    something near every calculated line, and forty phases came out at 100 per
    cent coverage - belite, cerussite, mayenite, troilite.  A peak list that
    matches everything distinguishes nothing.
    """
    if not patterns:
        return []
    found: list[list[tuple[float, float, float]]] = []
    for pattern in patterns:
        angles, heights, noises = [], [], []
        intensity = np.asarray(pattern.intensity, dtype=float)
        two_theta = np.asarray(pattern.two_theta, dtype=float)
        threshold = minimum_height * float(np.max(intensity)) if intensity.size else 0.0
        step = float(np.median(np.diff(two_theta))) if two_theta.size > 1 else 0.02
        reach = max(1, int(round(separation / step)))
        for index in range(reach, intensity.size - reach):
            here = intensity[index]
            if here < threshold:
                continue
            window = intensity[index - reach:index + reach + 1]
            if here < window.max() or here <= window.min():
                continue
            noise = _local_noise(intensity, index)
            if here < min_signal_to_noise * noise:
                continue
            angles.append(float(two_theta[index]))
            heights.append(float(here))
            noises.append(float(noise))
        found.append(list(zip(angles, heights, noises)))

    # A peak of the first mount is stable when every other mount has one within
    # the tolerance.  The first mount is only the bookkeeping order; requiring
    # agreement from all of them makes the result independent of which it is.
    stable: list[StablePeak] = []
    for angle, height, noise in found[0]:
        partners = [(angle, height, noise)]
        for others in found[1:]:
            near = [item for item in others if abs(item[0] - angle) <= tolerance]
            if not near:
                partners = []
                break
            partners.append(min(near, key=lambda item: abs(item[0] - angle)))
        if not partners:
            continue
        stable.append(StablePeak(
            two_theta=float(np.mean([item[0] for item in partners])),
            heights=tuple(item[1] for item in partners),
            signal_to_noise=float(min(item[1] / max(item[2], 1e-9) for item in partners)),
        ))
    return stable


def stable_phases(
    mounts: list[Pattern],
    crystals: dict[str, Crystal],
    two_theta_range: tuple[float, float] = (4.0, 40.0),
    instrument: Instrument | None = None,
    tolerance: float = 0.05,
    min_coverage: float = 0.30,
    min_matched: int = 2,
    min_intensity: float = 0.10,
    min_signal_to_noise: float = 5.0,
    minimum_height: float = 0.006,
    separation: float = 0.09,
    include_clays: bool = False,
    only: set[str] | None = None,
) -> tuple[list[PhaseEvidence], list[StablePeak]]:
    """Rank phases by how much of each stands on peaks common to all mounts.

    Each finding's ``score`` is its *coverage*: the share of the phase's own
    expected intensity, over the reflections above ``min_intensity`` of its
    strongest, that falls on a stable peak.  Coverage is a property of the phase
    and the stable peak list alone, so - unlike a share of the pattern - it does
    not change because a different candidate was added, and a minor phase whose
    few lines are all present scores as highly as a major one.  That is the
    whole point: rutile is 1.5 per cent of one real specimen and its coverage is
    100 per cent, while anatase, faujasite, graphite and calciolangbeinite,
    which an unrestricted fit of one scan ranked above it, have no stable line
    at all.

    ``min_intensity`` decides which of a phase's own reflections count, as a
    fraction of its strongest, and it matters more than it looks: counting every
    calculated line down to one per cent dilutes the coverage of a phase with a
    long tail of weak reflections, which is most of them, and then says more
    about the length of the tail than about the specimen.  At a tenth, quartz
    and rutile both come out at 100 per cent on two stable reflections each.

    ``min_matched`` guards the other way, because coverage alone would give a
    phase with one strong line a perfect score.

    Returns the findings, best coverage first, and the stable peaks themselves.
    """
    reference = instrument or Instrument()
    peaks = stable_peaks(
        mounts, tolerance=tolerance, min_signal_to_noise=min_signal_to_noise,
        minimum_height=minimum_height, separation=separation,
    )
    inside = [
        peak for peak in peaks
        if two_theta_range[0] <= peak.two_theta <= two_theta_range[1]
    ]
    if not inside:
        return [], peaks

    angles = np.array([peak.two_theta for peak in inside])
    findings: list[PhaseEvidence] = []
    for name, crystal in crystals.items():
        if is_clay_phase(name) and not include_clays:
            continue
        if only is not None and name not in only:
            continue
        try:
            positions, heights = peak_list(crystal, two_theta_range, reference)
        except Exception:  # noqa: BLE001 - a broken database entry must not stop the scan
            continue
        if positions.size == 0 or heights.max() <= 0.0:
            continue
        strong = heights >= min_intensity * heights.max()
        positions, heights = positions[strong], heights[strong]
        if positions.size == 0:
            continue
        explained = 0.0
        hit: set[int] = set()
        matches: list[PeakMatch] = []
        for position, weight in zip(positions, heights):
            index = int(np.argmin(np.abs(angles - position)))
            found = abs(angles[index] - position) <= tolerance
            peak = inside[index]
            if found:
                explained += float(weight)
                hit.add(index)
            matches.append(PeakMatch(
                expected_two_theta=float(position),
                expected_intensity=float(weight),
                window=tolerance,
                found_two_theta=float(peak.two_theta) if found else None,
                height=float(peak.height) if found else 0.0,
                noise=float(peak.height / peak.signal_to_noise)
                if found and peak.signal_to_noise > 0 else 1.0,
            ))
        coverage = explained / float(np.sum(heights))
        # Reflections, not distinct peaks: two lines of a phase can fall on one
        # stable peak, and both are reflections the measurement accounts for.
        # This is what "2 stable reflections" counts.
        if sum(1 for match in matches if match.matched) < min_matched:
            continue
        if not hit or coverage < min_coverage:
            continue
        findings.append(PhaseEvidence(
            name=name,
            is_clay=is_clay_phase(name),
            score=coverage,
            matches=matches,
            presence=coverage,
            intensity_agreement=float("nan"),
        ))
    findings.sort(key=lambda evidence: (-evidence.score, -evidence.n_matched))
    return findings, peaks


# --- Clayfit's triplet screen ------------------------------------------------
#
# Clayfit identifies the accompanying minerals as soon as the three scans are
# read, *before* the zero error is set, and the two things that let it are worth
# naming because neither is obvious.
#
# It does not need the zero error because it measures one for each scan from
# quartz and applies it there and then: the two strongest quartz lines are
# looked for within half a degree of where they belong, the pair whose implied
# shifts agree is taken, and every peak of that scan is moved by the mean of
# them.  A displacement of a few hundredths of a degree - which is what an
# oriented mount gives - is gone before any phase is matched, and the three
# scans are each aligned independently, so they need not share a zero.
#
# And it is a screen on *stability* rather than on fit: a reflection counts only
# when it is present in the air-dried, glycolated and heated scan alike.  That
# is a different measurement from explaining one pattern, and it is the one that
# finds a phase whose strongest line is overlapped.  Rutile's 110 sits on an
# albite reflection, so an unrestricted fit of one scan ranks thirty-five phases
# above it; its lines are nevertheless at the same angles in all three scans.

CLAYFIT_MATCH_TOLERANCE = 0.11
"""How far a measured peak may be from a calculated line, in degrees.

Clayfit's ``MATCH_TOLERANCE_DEGREES``.  It has to cover what the quartz shift
leaves behind - a specimen displacement is not constant in 2theta - plus the
cell difference between the database entry and the specimen's own solid
solution.  The screen also accepts a coherent cell scaling
(``cell_allowance``), which is the more honest way to write the second of those.
"""

CLAYFIT_MAX_ZERO_SHIFT = 0.50
"""Half-width of the window the quartz calibration lines are looked for in.

Clayfit's ``MAX_ZERO_SHIFT_DEGREES``.  Generous on purpose: this runs before
anything has been calibrated, and a mount that is half a degree out is a mount
whose zero error the operator has not yet had a chance to see.
"""

CLAYFIT_SHIFT_AGREEMENT = 0.10
"""How closely the two quartz lines must agree on the shift, in degrees.

A pair that disagrees by more than this is two different reflections, not the
quartz doublet, and taking their mean would calibrate on a coincidence.
"""

CLAYFIT_QUARTZ_STRENGTH = 0.03
"""How prominent a peak must be, against the strongest, to be taken for quartz.

Not Clayfit's - Clayfit takes the strongest agreeing pair whatever its absolute
size - and added because on real mounts that is not enough.  See
:func:`quartz_zero_shift` for the measurement it comes from.
"""

CLAYFIT_INTENSITY_FLOOR = 0.06
"""Weakest calculated line the screen will ask to be present.

Clayfit's ``detection_intensity_floor`` of 6 % of the phase's strongest, and it
does two things.  It keeps the coverage denominator from filling with lines too
weak to see, which would make every phase with a long tail look absent.  And it
keeps the numerator honest, because a weak line has more company within 0.11
degrees than a strong one and is correspondingly easier to match by accident.
"""

CLAYFIT_CLUSTER = 0.10
"""Calculated lines closer than this count once, at the strongest of them.

Clayfit's own comment names the case: albite has closely spaced line groups near
24 and 28 degrees, and a measured peak that answers three of them at once should
be three-quarters of the phase's evidence only if they are really resolved.
"""

CLAYFIT_MIN_MATCHES = 2
CLAYFIT_MIN_COVERAGE = 0.30
"""Clayfit's reporting thresholds: two stable reflections and 30 % coverage."""


@dataclass
class TreatmentPeaks:
    """The peaks of one scan, and what they are worth."""

    two_theta: np.ndarray
    prominence: np.ndarray
    width: np.ndarray

    def __len__(self) -> int:
        return int(self.two_theta.size)


def treatment_peaks(
    pattern: Pattern,
    window: float = 0.055,
    prominence_sigmas: float = 4.0,
    prominence_share: float = 0.0025,
) -> TreatmentPeaks:
    """Peaks of one scan, found the way Clayfit finds them.

    The pattern is background-corrected and smoothed by the percentile model of
    :func:`clayquant.background.qpa_percentile_baseline` - which is what that
    model is for - and the peaks are taken from the smoothed signal with a
    prominence of four times the noise or a quarter of a per cent of the
    strongest point, whichever is larger, and a minimum separation of ``window``
    degrees.

    The noise is measured from the first difference of the corrected signal,
    ``1.4826 * median|dI| / sqrt(2)``, which is a robust estimate that a broad
    feature does not inflate - the same reason the median absolute deviation is
    used to seed :func:`clayquant.background.noise_level`.
    """
    from scipy.signal import find_peaks, peak_widths

    from .background import qpa_percentile_baseline

    two_theta = np.asarray(pattern.two_theta, dtype=float)
    corrected = qpa_percentile_baseline(two_theta, pattern.intensity).smoothed
    empty = np.array([], dtype=float)
    if corrected.size < 8:
        return TreatmentPeaks(empty, empty, empty)
    step = float(np.median(np.diff(two_theta)))
    noise = 1.4826 * float(np.median(np.abs(np.diff(corrected)))) / math.sqrt(2.0)
    prominence = max(prominence_sigmas * noise,
                     prominence_share * float(np.max(corrected)), 1.0)
    indices, properties = find_peaks(
        corrected, prominence=prominence, distance=max(1, int(round(window / step)))
    )
    if indices.size == 0:
        return TreatmentPeaks(empty, empty, empty)
    widths = peak_widths(corrected, indices, rel_height=0.5)[0] * step
    return TreatmentPeaks(
        two_theta=np.asarray(two_theta[indices], dtype=float),
        prominence=np.asarray(properties["prominences"], dtype=float),
        width=np.asarray(widths, dtype=float),
    )


def quartz_zero_shift(
    peaks: TreatmentPeaks,
    calibration: tuple[float, float] = QUARTZ_CALIBRATION,
    reach: float = CLAYFIT_MAX_ZERO_SHIFT,
    agreement: float = CLAYFIT_SHIFT_AGREEMENT,
    strength: float = CLAYFIT_QUARTZ_STRENGTH,
) -> tuple[float | None, tuple[int, ...]]:
    """The zero shift one scan's quartz lines imply, and which peaks gave it.

    Every peak within ``reach`` of the 20.86 and 26.64 degree quartz lines is a
    candidate; each pair implies two shifts, and a pair is usable only when
    those agree to ``agreement``.  Among the usable pairs the strongest wins,
    with the disagreement as a tie-break::

        score = sqrt(prominence_low * prominence_high) / (0.02 + disagreement)

    The constant keeps a pair that agrees exactly from scoring infinitely, so
    strength decides between two pairs that both agree well - which is the right
    way round, because a strong pair that agrees to 0.01 degrees is better
    evidence than a weak one that agrees to 0.001.

    With no usable pair the isolated 20.86 line is returned on its own, as a
    starting value rather than a calibration.  ``None`` means quartz was not
    found at all.
    """
    if len(peaks) == 0:
        return None, ()
    low_reference, high_reference = calibration
    # A scan of a clay separate holds forty to a hundred peaks, so within half a
    # degree of each reference there are usually several candidates and a pair
    # of them can agree on a shift by coincidence.  Measured on nine real
    # mounts, the peak nearest the 26.64 line carries 1 to 100 per cent of the
    # strongest peak's prominence, and the ones at the bottom of that range are
    # noise that had been taken for quartz: on one specimen they aligned all
    # three mounts by +0.37 deg, which is not a displacement any oriented mount
    # has.  Requiring a candidate to carry `strength` of the strongest peak
    # leaves a modest quartz in - the weakest real one measured was 3.3 per cent
    # - and removes those.
    floor = strength * float(np.max(peaks.prominence)) if len(peaks) else 0.0
    strong = peaks.prominence >= floor
    low = np.flatnonzero((np.abs(peaks.two_theta - low_reference) <= reach) & strong)
    high = np.flatnonzero((np.abs(peaks.two_theta - high_reference) <= reach) & strong)

    best: tuple[float, float, int, int] | None = None
    for first in low:
        for second in high:
            low_shift = low_reference - float(peaks.two_theta[first])
            high_shift = high_reference - float(peaks.two_theta[second])
            disagreement = abs(low_shift - high_shift)
            if disagreement > agreement:
                continue
            strength = math.sqrt(max(peaks.prominence[first], 0.0)
                                 * max(peaks.prominence[second], 0.0))
            score = strength / (0.02 + disagreement)
            if best is None or score > best[0]:
                best = (score, 0.5 * (low_shift + high_shift), int(first), int(second))
    if best is not None:
        return float(best[1]), (best[2], best[3])
    if low.size:
        index = int(low[int(np.argmax(peaks.prominence[low]))])
        return low_reference - float(peaks.two_theta[index]), (index,)
    return None, ()


def _clustered_lines(
    positions: np.ndarray,
    heights: np.ndarray,
    floor: float,
    cluster: float,
) -> list[tuple[float, float]]:
    """Calculated lines above ``floor``, with near neighbours merged."""
    if positions.size == 0 or heights.max() <= 0.0:
        return []
    strong = heights >= floor * heights.max()
    lines = sorted(zip(positions[strong].tolist(), heights[strong].tolist()))
    merged: list[tuple[float, float]] = []
    for position, height in lines:
        if merged and position - merged[-1][0] <= cluster:
            if height > merged[-1][1]:
                merged[-1] = (position, height)
        else:
            merged.append((position, height))
    return merged


def match_across_treatments(
    name: str,
    positions: np.ndarray,
    heights: np.ndarray,
    peaks_by_mount: dict[str, TreatmentPeaks],
    shifts: dict[str, float],
    tolerance: float = CLAYFIT_MATCH_TOLERANCE,
    floor: float = CLAYFIT_INTENSITY_FLOOR,
    cluster: float = CLAYFIT_CLUSTER,
    scale: float = 1.0,
) -> PhaseEvidence | None:
    """How much of one phase stands on peaks present in every mount.

    Each calculated line, strongest first, is looked for in every scan at once,
    after that scan's own quartz shift; a line counts only when every scan has a
    peak within ``tolerance`` of it that no stronger line has already claimed.
    Coverage is the share of the phase's own calculated intensity that the
    counted lines carry.

    Claiming each measured peak once is what stops a phase with a dense
    calculated pattern from scoring by coincidence: without it, one strong
    measured peak answers every line that happens to lie near it.

    ``scale`` applies a coherent cell scaling to the calculated positions, in
    the sense of :func:`_scale_positions` - one number for the whole phase, not
    a window per line.  It is how a solid solution whose cell differs from the
    database entry is allowed for without widening the tolerance, which would
    let everything match everything.
    """
    lines = _clustered_lines(positions, heights, floor, cluster)
    if not lines:
        return None
    if scale != 1.0:
        lines = [(2.0 * math.degrees(math.asin(min(1.0, scale * math.sin(
            math.radians(0.5 * position))))), height) for position, height in lines]

    claimed: dict[str, set[int]] = {mount: set() for mount in peaks_by_mount}
    matches: list[PeakMatch] = []
    explained = 0.0
    for position, height in sorted(lines, key=lambda row: -row[1]):
        found: dict[str, int] = {}
        for mount, peaks in peaks_by_mount.items():
            corrected = peaks.two_theta + shifts[mount]
            if corrected.size == 0:
                found = {}
                break
            index = int(np.argmin(np.abs(corrected - position)))
            if abs(float(corrected[index]) - position) > tolerance:
                found = {}
                break
            if index in claimed[mount]:
                found = {}
                break
            found[mount] = index
        matched = len(found) == len(peaks_by_mount)
        if matched:
            for mount, index in found.items():
                claimed[mount].add(index)
            explained += height
        angle = float(np.mean([
            peaks_by_mount[mount].two_theta[index] + shifts[mount]
            for mount, index in found.items()
        ])) if matched else None
        strength = float(np.mean([
            peaks_by_mount[mount].prominence[index] for mount, index in found.items()
        ])) if matched else 0.0
        matches.append(PeakMatch(
            expected_two_theta=float(position),
            expected_intensity=float(height),
            window=tolerance,
            found_two_theta=angle,
            height=strength,
            noise=1.0,
        ))

    total = sum(height for _, height in lines)
    coverage = explained / total if total > 0 else 0.0
    return PhaseEvidence(
        name=name,
        is_clay=is_clay_phase(name),
        score=coverage,
        matches=matches,
        cell_scale=float(scale),
        presence=coverage,
        intensity_agreement=float("nan"),
    )


def evidence_score(evidence: PhaseEvidence) -> float:
    """Clayfit's ranking: coverage, rewarded for resting on several lines.

    ``coverage * sqrt(matched)``.  Coverage alone would put a phase whose one
    strong line happens to land on a peak level with one whose five lines are
    all there, and the second is the better evidence by more than coverage says:
    a coincidence at one line is ordinary, and five coincidences at the angles
    one phase predicts are not.  The square root rather than the count, because
    the lines of one phase are not independent tests - they come from one cell,
    so getting the cell wrong moves all of them together.
    """
    return float(evidence.score) * math.sqrt(max(evidence.n_matched, 0))


@dataclass
class TripletScreen:
    """What the load-time screen found, and what it calibrated itself on."""

    findings: list[PhaseEvidence]
    shifts: dict[str, float]
    """The zero shift measured from quartz on each mount, in degrees."""

    calibrated: bool
    """Whether every mount gave a full quartz pair rather than one line."""

    width: float
    """Median full width at half maximum of the quartz lines, in degrees."""

    note: str = ""


def screen_treatments(
    mounts: dict[str, Pattern],
    crystals: dict[str, Crystal],
    two_theta_range: tuple[float, float] = (4.0, 40.0),
    instrument: Instrument | None = None,
    tolerance: float = CLAYFIT_MATCH_TOLERANCE,
    min_coverage: float = CLAYFIT_MIN_COVERAGE,
    min_matched: int = CLAYFIT_MIN_MATCHES,
    floor: float = CLAYFIT_INTENSITY_FLOOR,
    cell_allowance: float = DEFAULT_CELL_ALLOWANCE,
    scale_steps: int = 5,
    include_clays: bool = False,
    only: set[str] | None = None,
) -> TripletScreen:
    """Identify the accompanying minerals from the mounts alone, before anything else.

    Clayfit's screen, which runs as soon as the three scans are read: each scan
    is aligned on its own quartz lines, every eligible structure is matched
    against the peaks the scans have in common, and a phase is reported when at
    least ``min_matched`` of its calculated lines are stable and they carry at
    least ``min_coverage`` of its calculated intensity.

    No reference library is needed and no background model has to have been
    chosen, which is the point: the answer is available before the operator has
    set anything, and the things it depends on - where quartz is, and which
    peaks the three treatments share - are not things the operator sets.

    ``cell_allowance`` additionally lets each phase's whole pattern be scaled
    coherently, by up to that fraction of its cell, and takes the best scaling.
    Most accompanying minerals are solid solutions whose cell differs from the
    database entry, and scaling the pattern is the right way to allow for that:
    a per-line window of the same size would be up to 1.6 degrees wide at high
    angle and would match almost anything (Sec. A.6 of the manual).

    Returns the findings by coverage, the shift measured on each mount, and the
    quartz line width - which is a usable starting value for the instrument's
    peak width before any fit has been made.
    """
    reference = instrument or Instrument()
    peaks_by_mount = {name: treatment_peaks(pattern) for name, pattern in mounts.items()}
    usable = {name: peaks for name, peaks in peaks_by_mount.items() if len(peaks)}
    if not usable:
        return TripletScreen([], {}, False, float("nan"),
                             "No peaks rose above the noise in any mount.")

    shifts: dict[str, float] = {}
    widths: list[float] = []
    paired = 0
    for name, peaks in usable.items():
        shift, indices = quartz_zero_shift(peaks)
        if shift is None:
            continue
        shifts[name] = shift
        if len(indices) == 2:
            paired += 1
        widths.extend(float(peaks.width[index]) for index in indices)
    calibrated = bool(shifts) and paired == len(usable)
    if not shifts:
        return TripletScreen(
            [], {}, False, float("nan"),
            "Quartz was not found in any mount, so the scans could not be put on a "
            "common angle scale and nothing was screened. Set the zero error by hand "
            "and use the main-mineral search instead.",
        )
    # Only the mounts that were aligned can take part: a scan left on its own
    # angle scale would fail every phase and drag every coverage to zero.
    peaks_by_mount = {name: usable[name] for name in shifts}

    scales = (np.linspace(1.0 - cell_allowance, 1.0 + cell_allowance, scale_steps)
              if cell_allowance > 0.0 and scale_steps > 1 else np.array([1.0]))

    findings: list[PhaseEvidence] = []
    for name, crystal in crystals.items():
        if is_clay_phase(name) and not include_clays:
            continue
        if only is not None and name not in only:
            continue
        try:
            positions, heights = peak_list(crystal, two_theta_range, reference)
        except Exception:  # noqa: BLE001 - one broken entry must not stop the screen
            continue
        best: PhaseEvidence | None = None
        for scale in scales:
            evidence = match_across_treatments(
                name, positions, heights, peaks_by_mount, shifts,
                tolerance=tolerance, floor=floor, scale=float(scale),
            )
            if evidence is None:
                continue
            if best is None or evidence.score > best.score:
                best = evidence
        if best is None:
            continue
        if best.n_matched < min_matched or best.score < min_coverage:
            continue
        findings.append(best)

    findings.sort(key=lambda evidence: (-evidence_score(evidence), evidence.name))
    width = float(np.median(widths)) if widths else float("nan")
    aligned = ", ".join(f"{name} {shift:+.3f}°" for name, shift in sorted(shifts.items()))
    if calibrated:
        note = (f"Quartz located in every mount and each aligned on its own pair: "
                f"{aligned}. {len(findings)} phases have {min_matched} or more lines "
                f"stable across them.")
    else:
        note = (f"Only the 20.86° quartz line was usable in at least one mount, so "
                f"the alignment is a starting value rather than a calibration: "
                f"{aligned}. Treat what follows as provisional and check it after the "
                f"zero error is set.")
    # Each mount is aligned on its own quartz, which is right - they are three
    # preparations and a displacement is a property of the preparation - but a
    # difference of more than a tenth of a degree between them is larger than a
    # displacement and is usually the calibration having taken the wrong pair of
    # peaks in one of them.  Said rather than corrected: the screen cannot tell
    # which mount is the odd one out, and the operator can.
    if len(shifts) > 1:
        spread = max(shifts.values()) - min(shifts.values())
        if spread > 3.0 * CLAYFIT_SHIFT_AGREEMENT:
            middle = float(np.median(list(shifts.values())))
            odd = max(shifts, key=lambda name: abs(shifts[name] - middle))
            note += (f" The mounts disagree by {spread:.2f}\u00b0 on where quartz is, "
                     f"which is more than a specimen displacement: {odd} is the outlier. "
                     f"Check that mount's 20.86 and 26.64\u00b0 peaks before believing "
                     f"this list.")
    if cell_allowance > 0.0:
        stretched = [item.name for item in findings
                     if abs(item.cell_scale - 1.0) > 0.8 * cell_allowance]
        if stretched:
            note += (f" {len(stretched)} of them needed almost the whole "
                     f"{100.0 * cell_allowance:.0f}% cell allowance to line up "
                     f"({', '.join(stretched[:4])}"
                     f"{', ...' if len(stretched) > 4 else ''}), which is weaker evidence: "
                     f"the allowance is there for a solid solution a per cent off its "
                     f"database entry, not to rescue a phase that does not fit.")
    return TripletScreen(findings, shifts, calibrated, width, note)
