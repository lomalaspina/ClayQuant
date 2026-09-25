"""Diagnostic comparisons between the air-dried, heated and glycolated mounts.

Two classical tests are implemented, both of which compare two mounts of the
same clay separate:

Kaolinite, from air-dried against heated (500-550 C)
    Kaolinite dehydroxylates between about 500 and 550 C and its 7.15 A basal
    reflection disappears, while chlorite - whose 002 at 7.13 A is otherwise
    almost exactly superimposed on it - survives.  The intensity *lost* from the
    7.15 A peak on heating is therefore the kaolinite contribution, and what
    remains is chlorite.

Expandable clay, from air-dried against ethylene-glycol solvated
    Smectitic interlayers take up glycol and expand to about 16.9 A, so the
    low-angle reflection migrates from its air-dried position (12.4 A for one
    water layer, about 15 A for two) to higher d.  The size of the shift and the
    intensity that arrives at the glycolated position measure expandability.

Both comparisons need the two mounts on a common intensity scale, because they
are different glass slides with different amounts of clay.  A reflection from a
phase unaffected by the treatment is used for that: quartz 100 by default,
which is stable to 500 C and unaffected by glycol.

The results are explicitly *relative* measures.  Turning them into weight
fractions requires the full-pattern fit of :mod:`clayquant.nnls` together with
reference intensity ratios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .background import snip_baseline
from .calibration import QUARTZ_100_D, reference_two_theta
from .pattern import Pattern

__all__ = [
    "PeakMetrics",
    "measure_peak",
    "scale_to_reference",
    "KaoliniteResult",
    "kaolinite_collapse",
    "ExpandabilityResult",
    "smectite_swelling",
    "KAOLINITE_001_WINDOW",
    "EXPANDABLE_001_WINDOW",
    "CHLORITE_002_SURVIVAL",
    "MINIMUM_SIGMAS",
    "NOISE_SPAN",
    "PEAK_WIDTH",
]

CU_KA1 = 1.540596

KAOLINITE_001_WINDOW = (11.6, 13.2)
"""2theta window around the kaolinite 001 / chlorite 002 reflection at ~7.15 A."""

EXPANDABLE_001_WINDOW = (3.0, 10.5)
"""2theta window in which the expandable 001 reflection is sought."""

ILLITE_002_WINDOW = (17.0, 18.4)
"""2theta window around the illite 002 reflection at 5.01 A."""

MINIMUM_SIGMAS = 3.0
"""How far above the background a diagnostic reflection must stand to count.

Three standard deviations of the *matched-filter* estimate of its height, plus
an allowance for the window having been searched; see :func:`measure_peak`.
Below that the window holds noise, and a diagnostic that reads noise reports a
mineral the specimen does not contain: on an illite standard with nothing at all
at 7.15 A, the collapse test found 45 % of the area disappearing on heating and
called it kaolinite.
"""

NOISE_SPAN = 3.0
"""Width of the region the noise is estimated over, in window widths."""

CHLORITE_002_SURVIVAL = (0.32, 0.72)
"""What fraction of a chlorite 002 survives heating, from two real standards.

The classical test reads the 7.15 A intensity *lost* on heating as kaolinite and
what remains as chlorite 002, on the premise that chlorite's 002 survives.  It
does not survive intact.  Measured here on two chlorite standards prepared and
heated by the same procedure - a clinochloritic Chlorite and an iron-rich
Prochlorite, neither containing kaolinite - the 7.15 A area after 1.5 h at
550 C was 32 % and 72 % of the air-dried area.  A factor of 2.2 between two
chlorites, so no single correction serves both.

What follows is that on a chlorite-bearing specimen this test cannot report a
kaolinite percentage at all, only a range: with ``A`` the air-dried area and
``H`` the heated one, the chlorite accounts for ``H / f`` of ``A``, so the
kaolinite lies between ``1 - H/(f_low * A)`` and ``1 - H/(f_high * A)``.  On both
standards that range contains zero, which is the right answer for a chlorite.
"""

PEAK_WIDTH = 0.12
"""Expected FWHM of a diagnostic reflection in degrees, for the matched filter.

A single point is not a reflection, and testing the tallest point in a window
against the noise of one point asks the wrong question: the tallest of a hundred
noise points stands three standard deviations above zero all by itself.  A peak
is instead sought in the window averaged over this width, where noise falls as
the root of the number of points averaged and a real reflection does not, which
is the matched filter for a peak of that width.  The default is a well
collimated instrumental width; a broader specimen only makes the estimate
conservative, because averaging over less than the true width still gains on the
noise.
"""


def d_from_two_theta(two_theta: float, wavelength: float = CU_KA1) -> float:
    """d-spacing in A for a Bragg angle in degrees."""
    sine = np.sin(np.radians(two_theta) / 2.0)
    if sine <= 0:
        return float("inf")
    return float(wavelength / (2.0 * sine))


@dataclass
class PeakMetrics:
    """Height, area and position of a peak in a fixed angular window."""

    window: tuple[float, float]
    position: float
    d_spacing: float
    height: float
    area: float
    n_points: int
    noise: float = 0.0
    """Standard deviation of the background around the window, in counts."""

    minimum_sigmas: float = 0.0
    """Standard errors the height had to clear, search allowance included."""

    matched_height: float = 0.0
    """Height of the window averaged over one peak width, which is what is tested."""

    averaged_points: int = 1
    """How many points that average covers; the noise of it falls as their root."""

    @property
    def significance(self) -> float:
        """Matched-filter height in standard errors of its own estimate.

        ``inf`` where the noise could not be estimated, which keeps a peak
        measured on noise-free synthetic data present rather than absent.
        """
        if self.height <= 0.0 or self.matched_height <= 0.0:
            return 0.0
        if self.noise <= 0.0:
            return float("inf")
        return float(
            self.matched_height / (self.noise / math.sqrt(max(self.averaged_points, 1)))
        )

    @property
    def is_present(self) -> bool:
        """Whether there is a reflection here at all, rather than noise.

        It used to be ``height > 0 and area > 0``, which is true of any window:
        after a baseline is removed the tallest noise spike is positive, and the
        area integrates only the points above zero, so an empty window has a
        positive area by construction and can never be absent.  On a real
        chlorite standard with nothing at 7.15 A that read as 45 % of the area
        collapsing on heating - a kaolinite content invented out of the noise.
        A reflection now has to stand ``minimum_sigmas`` above the background.
        """
        return self.height > 0.0 and self.significance >= self.minimum_sigmas


def measure_peak(
    pattern: Pattern,
    window: tuple[float, float],
    baseline: str = "snip",
    wavelength: float = CU_KA1,
    minimum_sigmas: float = MINIMUM_SIGMAS,
    noise_span: float = NOISE_SPAN,
    peak_width: float = PEAK_WIDTH,
) -> PeakMetrics:
    """Measure the peak inside ``window``.

    The position is the centroid of the points above half the window maximum,
    which is robust against the asymmetry the Ka doublet introduces.

    Parameters
    ----------
    minimum_sigmas:
        How far above the background a peak must stand to count as a reflection
        at all; see :attr:`PeakMetrics.is_present`.  Zero measures whatever is
        in the window and asks no questions, which is what this did before.

        The test is made on the window averaged over ``peak_width``, because a
        single point is not a reflection: the noise of an average of ``n`` points
        is smaller by the root of ``n`` while a real peak is not, so the average
        is the matched filter for a peak of that width.  The threshold also
        allows for the window having been *searched* - the largest of ``n``
        independent samples of noise stands about ``sqrt(2 ln n)`` standard
        errors above zero with no peak present at all - so the height must clear
        ``minimum_sigmas + sqrt(2 ln n)`` standard errors, where ``n`` is how
        many independent peak-width positions the window holds.
    peak_width:
        Expected FWHM of the reflection in degrees; see :data:`PEAK_WIDTH`.
    noise_span:
        The noise is estimated over this multiple of the window's width, centred
        on it, from the median absolute successive difference - a robust
        estimate, so the peak itself does not inflate it, and taken over a wider
        span than the window so that background dominates the estimate even when
        the peak fills the window.
    baseline:
        How to remove the local background before measuring, which is what makes
        areas from two different mounts comparable once they are scaled.
        ``"snip"`` strips a peak-free continuum by iterative clipping and is the
        default because it copes with a window sitting on the steep low-angle
        rise, where a straight line through the window edges runs above the data
        and erases the peak.  ``"chord"`` uses that straight line, which is
        adequate for a narrow window on a flat background.  ``"none"`` measures
        the pattern as given, for data that is already background-corrected.
    """
    low, high = float(window[0]), float(window[1])
    if high <= low:
        raise ValueError("window must be (low, high) with high > low")
    inside = np.flatnonzero((pattern.two_theta >= low) & (pattern.two_theta <= high))
    if inside.size < 3:
        raise ValueError(
            f"the window {window} contains {inside.size} points; it may lie outside the scan range "
            f"({pattern.two_theta[0]:.2f} to {pattern.two_theta[-1]:.2f} deg)"
        )
    x = pattern.two_theta[inside]
    y = pattern.intensity[inside].astype(float)
    if baseline == "chord":
        y = y - np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    elif baseline == "snip":
        y = y - snip_baseline(x, y, window=0.75 * (high - low))
    elif baseline != "none":
        raise ValueError("baseline must be 'snip', 'chord' or 'none'")

    # The level and scatter are taken over a span wider than the window, so that
    # background sets them even when a reflection fills the window itself.
    half = 0.5 * max(noise_span, 1.0) * (high - low)
    centre = 0.5 * (low + high)
    around = np.flatnonzero(
        (pattern.two_theta >= centre - half) & (pattern.two_theta <= centre + half)
    )
    wide_x = pattern.two_theta[around]
    wide_y = pattern.intensity[around].astype(float)
    if baseline == "chord" and wide_x.size > 1:
        wide_y = wide_y - np.interp(wide_x, [wide_x[0], wide_x[-1]], [wide_y[0], wide_y[-1]])
    elif baseline == "snip" and wide_x.size > 1:
        wide_y = wide_y - snip_baseline(wide_x, wide_y, window=0.75 * (high - low))
    level, noise = _window_floor(wide_x, wide_y)
    y = y - level
    step = float(np.median(np.diff(x))) if x.size > 1 else 0.0
    averaged = max(1, int(round(float(peak_width) / step))) if step > 0.0 else 1
    if averaged > 1 and y.size >= averaged:
        kernel = np.ones(averaged) / averaged
        smoothed = np.convolve(y, kernel, mode="valid")
        matched = float(smoothed.max()) if smoothed.size else float(y.max())
    else:
        matched = float(y.max())
    # The noise of an average of `averaged` points, and the allowance for having
    # searched that many independent positions across the window.
    positions = max(2.0, (high - low) / max(float(peak_width), step or 1.0))
    required = float(minimum_sigmas) + math.sqrt(2.0 * math.log(positions))
    height = float(y.max())
    if height <= 0.0:
        return PeakMetrics(
            window=(low, high),
            position=float("nan"),
            d_spacing=float("nan"),
            height=0.0,
            area=0.0,
            n_points=int(inside.size),
            noise=noise,
            minimum_sigmas=required,
            matched_height=matched,
            averaged_points=averaged,
        )
    positive = np.clip(y, 0.0, None)
    keep = positive >= 0.5 * height
    position = float(np.sum(x[keep] * positive[keep]) / np.sum(positive[keep]))
    return PeakMetrics(
        window=(low, high),
        position=position,
        d_spacing=d_from_two_theta(position, wavelength),
        height=height,
        area=float(np.trapezoid(positive, x)),
        n_points=int(inside.size),
        noise=noise,
        minimum_sigmas=required,
        matched_height=matched,
        averaged_points=averaged,
    )


def _window_floor(
    x: np.ndarray, subtracted: np.ndarray
) -> tuple[float, float]:
    """What the window looks like where there is no peak: its level and scatter.

    Returned as ``(level, noise)``, both from the robust statistics of the
    baseline-subtracted signal itself rather than from counting statistics
    alone.  That is deliberate.  Counting noise is not what limits this test: a
    peak-stripped baseline under a curved low-angle background leaves a broad
    positive pedestal, which no amount of averaging removes and which a test
    against Poisson noise reads as a reflection.  Measured on nine real clay
    standards, a window with nothing in it stood six standard deviations above
    zero on counting noise alone, and four of the nine were reported as
    containing kaolinite.  The median of the subtracted signal is that pedestal
    and the peak is measured from it; ``1.4826 * MAD`` about it is the scatter a
    peak has to beat, and a peak occupying a minority of the span moves neither.
    """
    if subtracted.size < 8:
        return 0.0, 0.0
    level = float(np.median(subtracted))
    scatter = 1.4826 * float(np.median(np.abs(subtracted - level)))
    return level, (float(scatter) if np.isfinite(scatter) else 0.0)


def scale_to_reference(
    pattern: Pattern,
    other: Pattern,
    window: tuple[float, float] | None = None,
    wavelength: float = CU_KA1,
    tolerance: float = 0.4,
) -> float:
    """Factor that puts ``other`` on the intensity scale of ``pattern``.

    The factor is the ratio of the areas of a reflection that the treatment does
    not affect - by default quartz 100, whose window is derived from its
    d-spacing and the wavelength.
    """
    if window is None:
        center = reference_two_theta(QUARTZ_100_D, wavelength)
        window = (center - tolerance, center + tolerance)
    here = measure_peak(pattern, window, wavelength=wavelength)
    there = measure_peak(other, window, wavelength=wavelength)
    if not there.is_present:
        raise ValueError(
            f"no reference reflection found in {window} deg of {other.name!r}; "
            f"choose another reference window or scale the mounts manually"
        )
    return here.area / there.area


@dataclass
class KaoliniteResult:
    """Kaolinite content indicated by the collapse of the 7.15 A reflection."""

    air: PeakMetrics
    heated: PeakMetrics
    scale: float
    lost_area: float
    lost_height: float
    collapse_fraction: float
    residual_fraction: float
    reference_window: tuple[float, float] | None

    @property
    def kaolinite_bounds(self) -> tuple[float, float]:
        """Kaolinite's share of the 7.15 A area, as a range, or NaNs.

        The range comes from how much of a chlorite 002 survives heating, which
        two real chlorite standards put between 32 and 72 %
        (:data:`CHLORITE_002_SURVIVAL`).  With nothing left after heating the
        range collapses to a single value of 1: there was no chlorite under it.
        """
        if not self.air.is_present or self.air.area <= 0.0:
            return float("nan"), float("nan")
        remaining = (
            self.scale * self.heated.area / self.air.area
            if self.heated.is_present else 0.0
        )
        low_survival, high_survival = CHLORITE_002_SURVIVAL
        least = 1.0 - remaining / low_survival
        most = 1.0 - remaining / high_survival
        return (float(np.clip(least, 0.0, 1.0)), float(np.clip(most, 0.0, 1.0)))

    @property
    def kaolinite_detected(self) -> bool:
        """Whether kaolinite is there whatever the chlorite under it did.

        The lower bound, not the collapsed fraction: intensity lost from the
        7.15 A peak is kaolinite only if no chlorite 002 could have lost it, and
        a chlorite 002 can lose two thirds of itself.
        """
        if not self.air.is_present:
            return False
        least, _ = self.kaolinite_bounds
        return bool(least > 0.1)

    @property
    def chlorite_indicated(self) -> bool:
        """Whether intensity survives at 7.15 A, indicating chlorite 002."""
        return self.heated.is_present and self.residual_fraction > 0.1

    def summary(self) -> str:
        if not self.air.is_present:
            return (
                f"No reflection at 7.15 A in the air-dried mount - the tallest point in "
                f"the window stands {self.air.significance:.1f} standard deviations above "
                f"the background, short of the {self.air.minimum_sigmas:g} required - so "
                f"neither kaolinite nor chlorite is indicated here."
            )
        if not self.heated.is_present:
            return (
                f"The 7.15 A reflection, {self.air.significance:.0f} standard deviations "
                f"above the background in the air-dried mount, is gone after heating: all "
                f"of it collapsed, so it is kaolinite and there is no chlorite 002 under it."
            )
        least, most = self.kaolinite_bounds
        return (
            f"7.15 A peak area {self.air.area:.4g} (air) vs {self.scale * self.heated.area:.4g} "
            f"(heated, scaled): {100.0 * self.collapse_fraction:.1f}% of it collapsed on "
            f"heating. A chlorite 002 keeps between "
            f"{100.0 * CHLORITE_002_SURVIVAL[0]:.0f} and "
            f"{100.0 * CHLORITE_002_SURVIVAL[1]:.0f}% of itself through the same heating - "
            f"measured on two chlorite standards containing no kaolinite - so the kaolinite "
            f"is between {100.0 * least:.0f} and {100.0 * most:.0f}% of this peak and the "
            f"chlorite is the rest."
        )


def kaolinite_collapse(
    air: Pattern,
    heated: Pattern,
    window: tuple[float, float] = KAOLINITE_001_WINDOW,
    reference_window: tuple[float, float] | None = None,
    scale: float | None = None,
    wavelength: float = CU_KA1,
) -> KaoliniteResult:
    """Quantify the kaolinite 001 collapse between the air-dried and heated mounts.

    Parameters
    ----------
    reference_window:
        Window of the reflection used to put the two mounts on a common scale.
        ``None`` uses quartz 100.
    scale:
        Explicit scale factor for the heated pattern, bypassing the reference
        reflection.
    """
    if scale is None:
        scale = scale_to_reference(air, heated, reference_window, wavelength=wavelength)
    air_peak = measure_peak(air, window, wavelength=wavelength)
    heated_peak = measure_peak(heated, window, wavelength=wavelength)

    # A reflection that did not clear the noise is not there, and its area is
    # the positive half of the noise rather than a measurement.  Counting it
    # would mean an empty window reporting a collapse, and a fully collapsed
    # kaolinite reporting less than a full one.
    air_area = air_peak.area if air_peak.is_present else 0.0
    air_height = air_peak.height if air_peak.is_present else 0.0
    heated_area = heated_peak.area if heated_peak.is_present else 0.0
    heated_height = heated_peak.height if heated_peak.is_present else 0.0

    lost_area = air_area - scale * heated_area
    lost_height = air_height - scale * heated_height
    if air_area > 0:
        collapse = float(np.clip(lost_area / air_area, 0.0, 1.0))
    else:
        collapse = 0.0
    return KaoliniteResult(
        air=air_peak,
        heated=heated_peak,
        scale=float(scale),
        lost_area=float(lost_area),
        lost_height=float(lost_height),
        collapse_fraction=collapse,
        residual_fraction=1.0 - collapse,
        reference_window=reference_window,
    )


@dataclass
class ExpandabilityResult:
    """Expandable-clay behaviour indicated by the air-dried to glycol shift."""

    air: PeakMetrics
    glycol: PeakMetrics
    scale: float
    shift_two_theta: float
    shift_d: float
    glycol_area_ratio: float
    reference_window: tuple[float, float] | None

    @property
    def expandable_detected(self) -> bool:
        """Whether the low-angle reflection moved to higher d on glycolation."""
        return self.shift_d > 0.3 and self.glycol.is_present

    def summary(self) -> str:
        if not self.glycol.is_present:
            return (
                f"No low-angle reflection in the glycolated mount: the tallest point in "
                f"the window stands {self.glycol.significance:.1f} standard deviations "
                f"above the background, short of the "
                f"{self.glycol.minimum_sigmas:g} required."
            )
        if not self.air.is_present:
            return (
                f"Glycolated reflection at {self.glycol.d_spacing:.2f} A with no air-dried "
                f"counterpart in the window: fully expandable material."
            )
        return (
            f"Low-angle reflection moved from {self.air.d_spacing:.2f} A (air) to "
            f"{self.glycol.d_spacing:.2f} A (glycol), a shift of {self.shift_d:+.2f} A "
            f"({self.shift_two_theta:+.2f} deg 2theta); glycolated/air-dried area ratio "
            f"{self.glycol_area_ratio:.2f}."
        )


def smectite_swelling(
    air: Pattern,
    glycol: Pattern,
    window: tuple[float, float] = EXPANDABLE_001_WINDOW,
    reference_window: tuple[float, float] | None = None,
    scale: float | None = None,
    wavelength: float = CU_KA1,
) -> ExpandabilityResult:
    """Quantify the glycol-induced expansion of the low-angle reflection.

    Both mounts are measured in the same window; the shift in d-spacing and the
    ratio of the areas (after putting the mounts on a common scale) are the
    relative measures of expandable content.
    """
    if scale is None:
        scale = scale_to_reference(air, glycol, reference_window, wavelength=wavelength)
    air_peak = measure_peak(air, window, wavelength=wavelength)
    glycol_peak = measure_peak(glycol, window, wavelength=wavelength)

    if air_peak.is_present and glycol_peak.is_present:
        shift_two_theta = glycol_peak.position - air_peak.position
        shift_d = glycol_peak.d_spacing - air_peak.d_spacing
    else:
        shift_two_theta = float("nan")
        shift_d = float("nan")
    ratio = (
        scale * glycol_peak.area / air_peak.area
        if air_peak.is_present and glycol_peak.is_present
        else float("nan")
    )
    return ExpandabilityResult(
        air=air_peak,
        glycol=glycol_peak,
        scale=float(scale),
        shift_two_theta=float(shift_two_theta),
        shift_d=float(shift_d),
        glycol_area_ratio=float(ratio),
        reference_window=reference_window,
    )
