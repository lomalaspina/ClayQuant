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
from .crystal import Crystal
from .pattern import Instrument, Pattern, peak_list, powder_pattern

__all__ = [
    "PeakMatch",
    "PhaseEvidence",
    "detect_phases",
    "screen_phases",
    "DEFAULT_CELL_ALLOWANCE",
]

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
            f"intensity agreement {self.intensity_agreement:.2f})"
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

    # Selection runs on the gain, reporting runs on the share, and the two are
    # not the same number.  A gain is what a phase adds *after* everything
    # already chosen, so the second and third phase of a real assemblage have
    # small gains however plainly they are present; stopping the rounds at the
    # reporting threshold therefore cut the list to the first one or two phases
    # and looked like the minerals had gone missing.  The rounds instead run to
    # a floor far below it, and the threshold is applied at the end to the share
    # each phase takes in one final joint fit - which is the quantity the
    # threshold has always meant and is comparable between phases.
    floor = min(min_share, 1e-4) / 10.0
    _, residual = fit([])
    remaining = {
        index for index in range(len(candidates)) if quality[index] >= min_agreement
    }
    active: list[tuple[int, int]] = []
    gains: dict[int, float] = {}
    scale_of: dict[int, float] = {}
    for _ in range(max_phases):
        if not remaining:
            break
        gain, chosen = _best_shift_gain(bank, residual)
        best = max(remaining, key=lambda index: gain[index])
        before = float(residual @ residual)
        _, trial = fit([*active, (best, int(chosen[best]))])
        removed = (before - float(trial @ trial)) / total
        if removed < floor:
            break
        active.append((best, int(chosen[best])))
        remaining.discard(best)
        gains[best] = removed
        scale_of[best] = float(scales[int(chosen[best])])
        residual = trial

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
                position_offset=offset,
            )
        )
    findings.sort(key=lambda evidence: evidence.score, reverse=True)
    return findings
