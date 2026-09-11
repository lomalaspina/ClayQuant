"""Searching a measurement for evidence of accompanying ("main") minerals.

A clay separate is never pure: quartz, feldspars, carbonates and sulphates come
through the separation in varying amounts, and quartz is wanted anyway as the
zero-error and intensity reference.  Before the full-pattern fit it is therefore
worth asking which of the phases in a structure database leave visible traces in
the measurement, so that those - and only those - are offered to the fit.

Method
------
For each candidate phase the strongest reflections in the measured angular range
are predicted, and the background-corrected measurement is searched for each one
inside a tolerance window.  A phase's score is the fraction of its *expected
intensity* that is actually found, so a missing strong line counts far more than
a missing weak one.

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
from .pattern import Instrument, Pattern, peak_list

__all__ = [
    "PeakMatch",
    "PhaseEvidence",
    "detect_phases",
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
        return (
            f"{self.name}: score {100.0 * self.score:.0f}% "
            f"({self.n_matched}/{self.n_expected} lines, presence {100.0 * self.presence:.0f}%, "
            f"intensity agreement {100.0 * self.intensity_agreement:.0f}%), "
            f"cell {self.cell_deviation_percent:+.2f}%, best S/N {self.best_signal_to_noise:.0f}"
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
