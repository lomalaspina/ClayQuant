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
    "BOUND_WIDTH_LIMIT",
    "PeakMetrics",
    "measure_peak",
    "common_scale",
    "scale_to_reference",
    "KaoliniteResult",
    "kaolinite_collapse",
    "ExpandabilityResult",
    "smectite_swelling",
    "KAOLINITE_001_WINDOW",
    "EXPANDABLE_001_WINDOW",
    "CHLORITE_001_ENHANCEMENT",
    "CHLORITE_001_WINDOW",
    "CHLORITE_002_OVER_003_SURVIVAL",
    "CHLORITE_002_SURVIVAL",
    "chlorite_002_survival_from_003",
    "CHLORITE_004_SURVIVAL",
    "CHLORITE_TYPE_CALIBRATION",
    "chlorite_survival_from_type",
    "CHLORITE_003_WINDOW",
    "CHLORITE_004_WINDOW",
    "chlorite_004_window",
    "MINIMUM_DOUBLET_SEPARATION",
    "CHLORITE_004_LEAKAGE",
    "KAOLINITE_002_TO_001",
    "CHLORITE_RATIOS",
    "ChloriteShare",
    "KAOLINITE_002_WINDOW",
    "KaoliniteEvidence",
    "kaolinite_evidence",
    "MINIMUM_SIGMAS",
    "NOISE_SPAN",
    "PEAK_WIDTH",
]

CU_KA1 = 1.540596
"""Cu K-alpha1 wavelength in A."""

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

CHLORITE_TYPE_CALIBRATION = ((0.814, 0.321), (1.373, 0.721))
"""``(003/001 area ratio, 7.15 A survival)`` for the two chlorite standards.

A clinochloritic chlorite and an iron-rich prochlorite keep 0.32 and 0.72 of
their 7.15 A reflection through heating - a factor of 2.2, and bounding every
specimen by the pair of them is what makes a kaolinite estimate a wide range
rather than a number.  They are the same mineral at different compositions, which
is why the pattern library treats them as one phase with an iron axis; but they
dehydroxylate differently, so the diagnostic cannot.

What distinguishes them without touching a window kaolinite reaches is the ratio
of the chlorite 003 to the chlorite 001 - two reflections kaolinite does not have
at all.  It is 0.814 on the clinochlore and 1.373 on the prochlorite, a factor of
1.7, and it moves the same way the survival does.  So a specimen's own 003/001
says which of the two its chlorite resembles, and the survival can be
interpolated instead of spanned.

Two points are a line and not a calibration, and this is stated as such wherever
it is used: within the bracket the interpolation is reported with the full range
beside it, and outside the bracket the value is clamped to the nearer standard
and the full range kept, because extrapolating a two-point fit is not a
measurement.
"""

CHLORITE_004_SURVIVAL = (0.53, 0.80)
"""What fraction of a chlorite 004 survives heating, from the same two standards.

The 3.58 A window is the second place kaolinite and chlorite overlap - kaolinite
002 against chlorite 004 - and it carries the same collapse test as the 7.15 A
window, measured against this range instead.  Using only the 7.15 A window threw
half the evidence away.

Measured on the same clinochloritic Chlorite and iron-rich Prochlorite, heated by
the same procedure: the 3.58 A area after heating was 53 % and 80 % of the
air-dried area, against 32 % and 72 % at 7.15 A.  The fourth order survives
better than the second and, more usefully, *varies less between the two
chlorites* - a spread of 1.5 against 2.2 - so the bound this window puts on
kaolinite is the tighter of the two even though the reflection is weaker.
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


CHLORITE_002_OVER_003_SURVIVAL = (0.89, 0.98)
"""How a chlorite 002's heat survival compares with its own 003's.

The absolute survival of a chlorite basal order varies hugely between chlorites
- :data:`CHLORITE_002_SURVIVAL` spans 32 to 72 % across two standards - so an
absolute bracket is a weak constraint, and on a chlorite that collapses harder
than either standard it is the wrong one.  The *ratio* between two orders of the
same chlorite is far better behaved: measured on the same two standards, the 002
and the 003 survive within a tenth of each other (0.89 and 0.98), because
dehydroxylation redistributes intensity into the 001 and takes the even and odd
orders above it down together.

That is what makes :func:`chlorite_002_survival_from_003` possible, and it is
the one route out of the circularity: the 003 is a chlorite reflection kaolinite
does not have, so measuring it in a kaolinite-bearing specimen is legitimate,
while the 002 and the 004 are both shared and measuring either tells you
nothing you did not assume.
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
    fallback: float | None = None,
) -> float:
    """Factor that puts ``other`` on the intensity scale of ``pattern``.

    The factor is the ratio of the areas of a reflection that the treatment does
    not affect - by default quartz 100, whose window is derived from its
    d-spacing and the wavelength.

    ``fallback`` is what to return when that reflection is not in the window.
    Without one this raises, which is right for a caller that cannot proceed
    without a measured scale; a caller that can - every diagnostic here, because
    two mounts of the same separate measured at the same settings are already on
    a common scale - passes 1.0 and reports that it did.  The reason this matters
    is that a pure mineral standard contains no quartz at all, and a pure dickite
    is exactly the specimen the kaolinite diagnostic should be easiest on.
    """
    if window is None:
        center = reference_two_theta(QUARTZ_100_D, wavelength)
        window = (center - tolerance, center + tolerance)
    here = measure_peak(pattern, window, wavelength=wavelength)
    there = measure_peak(other, window, wavelength=wavelength)
    if not (there.is_present and here.is_present) or there.area <= 0.0:
        if fallback is not None:
            return float(fallback)
        raise ValueError(
            f"no reference reflection found in {window} deg of {other.name!r}; "
            f"choose another reference window or scale the mounts manually"
        )
    return here.area / there.area


def common_scale(
    pattern: Pattern,
    other: Pattern,
    window: tuple[float, float] | None = None,
    wavelength: float = CU_KA1,
    tolerance: float = 0.4,
) -> tuple[float, str]:
    """The scale between two mounts, and a sentence saying where it came from.

    Returns ``(scale, note)``, the note empty when the reference reflection was
    measured and saying so when it was not and 1.0 was assumed instead.  Nothing
    here fails for want of a reference: the two mounts are usually the same
    specimen measured twice, so 1.0 is the honest default, and the one thing the
    reader has to know is whether it was measured or assumed.
    """
    if window is None:
        center = reference_two_theta(QUARTZ_100_D, wavelength)
        window = (center - tolerance, center + tolerance)
    here = measure_peak(pattern, window, wavelength=wavelength)
    there = measure_peak(other, window, wavelength=wavelength)
    if not (there.is_present and here.is_present) or there.area <= 0.0:
        return 1.0, (
            f"No reference reflection in {window[0]:.2f}-{window[1]:.2f} deg of either "
            f"mount, so the two were taken as already on a common scale (factor 1.0). "
            f"That is the right assumption for one specimen measured twice at the same "
            f"settings, and it is what a pure mineral standard needs, having no quartz "
            f"to scale on; set the factor by hand if the mounts differ."
        )
    return here.area / there.area, ""


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
    survival: tuple[float, float] = CHLORITE_002_SURVIVAL
    """How much of the chlorite order under this window survives heating.

    The 7.15 A window sits over a chlorite 002 and the 3.58 A window over a
    chlorite 004, and the two survive differently, so the range is carried with
    the measurement rather than assumed from the class.
    """

    label: str = "7.15 A"

    scale_note: str = ""
    """Empty when the two mounts were scaled on a measured reflection.

    Otherwise it says that no reference was found and 1.0 was assumed, which is
    what happens on a pure mineral standard: it has no quartz to scale on, and
    refusing to analyse it for that reason - as this used to - fails on exactly
    the specimens the diagnostic should be easiest on.
    """

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
        low_survival, high_survival = self.survival
        if low_survival <= 0.0 or high_survival <= 0.0:
            # A chlorite expected to survive nothing cannot be told from a
            # kaolinite by survival, so the window bounds nothing.
            return float("nan"), float("nan")
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
                f"No reflection at {self.label} in the air-dried mount - the tallest point in "
                f"the window stands {self.air.significance:.1f} standard deviations above "
                f"the background, short of the {self.air.minimum_sigmas:g} required - so "
                f"neither kaolinite nor chlorite is indicated here."
            )
        if not self.heated.is_present:
            return (
                f"The {self.label} reflection, {self.air.significance:.0f} standard deviations "
                f"above the background in the air-dried mount, is gone after heating: all "
                f"of it collapsed, so it is kaolinite and there is no chlorite 002 under it."
            )
        least, most = self.kaolinite_bounds
        low, high = self.survival
        measured = self.survival != CHLORITE_002_SURVIVAL
        basis = (
            "taken from this specimen's own chlorite 003, where no kaolinite "
            "contributes, scaled by the order ratio of two chlorite standards"
            if measured else
            "measured on two chlorite standards containing no kaolinite"
        )
        return (
            f"{self.label} peak area {self.air.area:.4g} (air) vs {self.scale * self.heated.area:.4g} "
            f"(heated, scaled): {100.0 * self.collapse_fraction:.1f}% of it collapsed on "
            f"heating. The chlorite order under this window was expected to keep between "
            f"{100.0 * low:.0f} and {100.0 * high:.0f}% of itself through the same heating - "
            f"{basis} - so the kaolinite "
            f"is between {100.0 * least:.0f} and {100.0 * most:.0f}% of this peak and the "
            f"chlorite is the rest."
        )


def chlorite_002_survival_from_003(
    air: Pattern,
    heated: Pattern,
    scale: float | None = None,
    ratio: tuple[float, float] = CHLORITE_002_OVER_003_SURVIVAL,
    fallback: tuple[float, float] = CHLORITE_002_SURVIVAL,
    wavelength: float = CU_KA1,
    minimum_sigmas: float = MINIMUM_SIGMAS,
) -> tuple[float, float, str]:
    """How much of *this* specimen's chlorite 002 survives heating, and how it is known.

    The 7.15 A window holds a kaolinite 001 and a chlorite 002 together, so what
    kaolinite's share of it is depends entirely on how much of the chlorite 002
    was expected to survive 550 C.  Assuming the standards' absolute bracket
    (:data:`CHLORITE_002_SURVIVAL`, 32 to 72 %) is what this used to do, and on a
    chlorite that collapses harder than either standard it asks kaolinite to
    account for a drop that the chlorite managed on its own.  On one real
    metabasite separate it put kaolinite at 55 to 80 % of the window when the
    truth was 18 to 26 %, and forcing the fit to honour that cost ten points of
    Rwp.

    The way out is the 003.  It is a chlorite reflection with no kaolinite under
    it, so measuring its collapse in a kaolinite-bearing specimen is not
    circular - unlike the 002 and the 004, which are both shared, and which is
    why a specimen's own 7.15 A or 3.58 A behaviour can never be used to
    characterise its chlorite.  The standards then supply only the *ratio*
    between the two orders' survival, which is the stable part
    (:data:`CHLORITE_002_OVER_003_SURVIVAL`).

    Returns ``(least, most, note)``.  Where the 003 cannot be measured in both
    mounts the ``fallback`` comes back and the note says why.
    """
    air_third = measure_peak(air, CHLORITE_003_WINDOW, wavelength=wavelength,
                             minimum_sigmas=minimum_sigmas)
    if not air_third.is_present or air_third.area <= 0.0:
        return (*fallback, "no chlorite 003 in the air-dried mount, so the chlorite's own "
                           "collapse could not be measured and the standards' range is kept")
    if scale is None:
        scale, _ = common_scale(air, heated, None, wavelength=wavelength)
    heated_third = measure_peak(heated, CHLORITE_003_WINDOW, wavelength=wavelength,
                                minimum_sigmas=minimum_sigmas)
    # A heated 003 that did not clear the noise has not survived nothing: it has
    # survived less than this measurement could see.  Taking it as zero would
    # make the expected chlorite 002 survival zero too, which divides by zero in
    # the share and, worse, asserts something the data cannot support.  The
    # detection limit is an upper bound on what survived, and an upper bound on
    # the chlorite's survival is exactly what gives a lower bound on kaolinite -
    # the direction the constraint needs.
    if heated_third.is_present:
        survived = scale * heated_third.area / air_third.area
        how = "measured"
    else:
        survived = scale * _detection_limit_area(heated_third) / air_third.area
        how = "bounded by the detection limit, the heated 003 being too weak to measure"
    survived = float(np.clip(survived, 0.0, 1.5))
    if survived <= 0.0:
        return (*fallback, "the chlorite 003 gave no usable survival, so the standards' "
                           "absolute range is kept")
    low, high = sorted(ratio)
    least, most = survived * low, survived * high
    return (
        float(np.clip(least, 1e-4, 1.0)),
        float(np.clip(most, 1e-4, 1.0)),
        f"this chlorite's 003 kept {100.0 * survived:.1f}% of its area through the "
        f"heating ({how}), in a window where no kaolinite contributes; on the two "
        f"standards a chlorite 002 survives {low:.2f} to {high:.2f} times as well as "
        f"its own 003, so this chlorite's 002 is expected to keep "
        f"{100.0 * least:.1f} to {100.0 * most:.1f}% rather than the "
        f"{100.0 * fallback[0]:.0f} to {100.0 * fallback[1]:.0f}% the standards span "
        f"in absolute terms"
    )


def kaolinite_collapse(
    air: Pattern,
    heated: Pattern,
    window: tuple[float, float] = KAOLINITE_001_WINDOW,
    reference_window: tuple[float, float] | None = None,
    scale: float | None = None,
    wavelength: float = CU_KA1,
    survival: tuple[float, float] = CHLORITE_002_SURVIVAL,
    label: str = "7.15 A",
) -> KaoliniteResult:
    """Quantify a kaolinite reflection's collapse between the air-dried and heated mounts.

    Parameters
    ----------
    reference_window:
        Window of the reflection used to put the two mounts on a common scale.
        ``None`` uses quartz 100.
    scale:
        Explicit scale factor for the heated pattern, bypassing the reference
        reflection.
    """
    scale_note = ""
    if scale is None:
        scale, scale_note = common_scale(air, heated, reference_window, wavelength=wavelength)
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
        survival=survival,
        label=label,
        scale_note=scale_note,
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
    scale_note: str = ""
    """Empty when the mounts were scaled on a measured reflection; see
    :attr:`KaoliniteResult.scale_note`."""

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
    scale_note = ""
    if scale is None:
        scale, scale_note = common_scale(air, glycol, reference_window, wavelength=wavelength)
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
        scale_note=scale_note,
    )


# --------------------------------------------------------------------------
# Separating the chlorite from the kaolinite without heating anything
# --------------------------------------------------------------------------

KAOLINITE_002_WINDOW = (24.60, 25.00)
"""2theta window around the kaolinite 002 at 3.579 A, to the low-angle side.

This used to span 24.0 to 25.8 degrees and so held the chlorite 004 as well,
which threw away the one place the two minerals are *separable*.  Kaolinite's
second order is at 3.579 A and a chlorite's fourth at 3.52-3.56, which is
0.3 degrees away - broad clay peaks overlap there but they are not one peak, and
reading them as one discards the classical kaolinite/chlorite deconvolution.
"""

CHLORITE_004_WINDOW = (25.00, 25.45)
"""2theta window around the chlorite 004 at ~3.54 A, to the high-angle side.

The third window, and the point of the split: with the chlorite 001 and 003 it
makes three places a chlorite can be measured where no kaolinite reaches, against
two where the two minerals overlap.
"""

KAOLINITE_002_TO_001 = (0.21, 0.29)
"""A kaolinite's 002 area as a share of its 001, from the two pure standards.

Measured the same way the diagnostic measures them, on Kaolinite_12 and
Kaolinite_43: 0.213 and 0.291.  It is what makes the second order a *test* of the
first rather than a second opinion about it - a 7.15 A reflection that collapses
on heating is kaolinite only if a 3.58 A reflection of about a quarter its area
is there too, and a specimen with a strong 001 and no 002 at all has something
else at 7.15 A.
"""

CHLORITE_004_LEAKAGE = (0.166, 0.169)
"""What share of its 004 window a kaolinite-free chlorite puts in the 002 window.

The two windows above are adjacent and clay reflections are broad, so splitting
them does not separate the minerals by itself: the chlorite 004's low-angle tail
reaches into the kaolinite window and would be read as kaolinite.  How much it
reaches is measurable, on the same two standards that carry no kaolinite, and it
is remarkably stable - 0.166 on the clinochloritic Chlorite and 0.169 on the
iron-rich Prochlorite, two specimens that differ by 2.2 times in how much of the
004 survives heating.  A tail is a peak shape rather than a composition, which is
why it transfers where the survival fractions do not.

So the kaolinite 002 area is the 002 window less this share of the 004 window,
and a specimen where that comes out near zero has no kaolinite second order
whatever its 7.15 A window is doing.
"""

CHLORITE_001_WINDOW = (5.6, 6.9)
"""Around the chlorite 001 at 14.2 A, which no kaolinite reflection reaches."""

CHLORITE_003_WINDOW = (18.1, 19.4)
"""Around the chlorite 003 at 4.73 A, the other reflection kaolinite cannot reach."""

CHLORITE_RATIOS: dict[tuple[str, str], tuple[float, float]] = {
    ("7.15", "001"): (1.862, 2.289),
    ("7.15", "003"): (1.691, 2.360),
}
"""How much a chlorite puts at 7.15 A per unit of its 001 or 003.

The 3.58 A entries are gone, and their absence is the point of the three-window
split rather than a loss.  They existed to *infer* the chlorite's contribution to
a 3.58 A window that held the chlorite 004 as well as the kaolinite 002.  Now
that the window is split at the position this specimen's own chlorite 001 puts
its fourth order (:func:`chlorite_004_window`), that contribution is measured
next door instead of inferred, which is the better of the two by the same
argument that prefers measuring a chlorite 001 to assuming one.  The 7.15 A
window has no such neighbour - kaolinite 001 and chlorite 002 are 0.10 deg
apart, inside a peak width - so there the inference stays.

Measured on two chlorite standards containing no kaolinite - a clinochloritic
Chlorite and an iron-rich Prochlorite - as integrated areas of the air-dried
mounts, on the instrument these samples are measured on.  Each pair is the range
the two of them span.

This is what makes the kaolinite test possible without heating anything, and it
is the better test.  Kaolinite has reflections at 12.36 and 24.85 deg and
chlorite has its 002 at 12.46 and its 004 at 25.06, within a peak width of each
other, so neither of those windows can be read alone.  But chlorite's 001 at
6.22 deg and its 003 at 18.73 deg are reflections kaolinite does not have at
all: measure either, multiply by the ratio, and that is the chlorite's share of
the overlapped window.  What is left is kaolinite.

The alternative - heating the specimen and reading what disappears - is much
weaker, because heating does not simply weaken a chlorite.  On these two
standards 1.5 h at 550 C *enhanced* the chlorite 001 by 2.2 and 3.5 times while
taking the 002/001 ratio from 1.86 to 0.17 and from 2.29 to 0.75: the hydroxide
sheet dehydroxylates, so the contrast reflection grows as the sum reflections
die.  A ratio that moves by a factor of 4.5 between two chlorites cannot bound
anything, where the air-dried 002/001 moves by 1.23.
"""

CHLORITE_001_ENHANCEMENT = (2.18, 3.51)
"""How much heating multiplies a chlorite 001, from the same two standards.

Its own diagnostic, and an unambiguous one: nothing else in a clay separate
grows several times stronger at 550 C.  A specimen whose 14.2 A reflection does
not grow on heating has no chlorite, whatever else the 7.15 A peak does.
"""


@dataclass
class ChloriteShare:
    """What a chlorite accounts for in an overlapped window, and what is left."""

    window: str
    reference: str
    reference_area: float
    window_area: float
    chlorite: tuple[float, float]
    """The chlorite's share of the window, as a range, from the two standards."""

    limited: bool = False
    """Whether the reference was below the noise, so this is an upper bound only.

    Then the chlorite's share runs from zero - it may be absent altogether - up
    to what a reflection at the detection limit would have accounted for.
    """

    @property
    def kaolinite(self) -> tuple[float, float]:
        """What is left for kaolinite, as a range, never below zero."""
        low, high = self.chlorite
        return (max(0.0, 1.0 - high), max(0.0, 1.0 - low))

    def describe(self) -> str:
        least, most = self.kaolinite
        if self.limited:
            return (
                f"{self.window} A window: no chlorite {self.reference} above the "
                f"noise, and one at the detection limit would account for at most "
                f"{100.0 * (1.0 - least):.0f}% of it, so kaolinite is at least "
                f"{100.0 * least:.0f}%"
            )
        return (
            f"{self.window} A window against the chlorite {self.reference}: "
            f"kaolinite is {100.0 * least:.0f} to {100.0 * most:.0f}% of it"
        )


BOUND_WIDTH_LIMIT = 0.5
"""How wide the kaolinite bound may be and still be worth applying.

Half the range.  A bound wider than that excludes less than it admits, and the
restraint built from it costs a fit its freedom without telling it anything: on
a montmorillonite, whose 7.15 A window holds 80 counts against a strongest line
of ten thousand, the bound comes back 0 to 1 and the constraint restrains
nothing while still asking the fit to satisfy an extra row.
"""


@dataclass
class KaoliniteEvidence:
    """Everything the three mounts say about kaolinite, and whether it agrees."""

    windows: dict[str, PeakMetrics]
    references: dict[str, PeakMetrics]
    shares: tuple[ChloriteShare, ...]
    collapse: "KaoliniteResult | None" = None
    second_collapse: "KaoliniteResult | None" = None
    """The same collapse test on the 3.58 A window, which is independent of it."""

    measured_collapse: "KaoliniteResult | None" = None
    """The 7.15 A collapse against this specimen's own chlorite 003 survival.

    :attr:`collapse` uses the two standards' absolute range, which is right only
    for a chlorite that behaves like them.  This one uses what the specimen's own
    chlorite did, measured where no kaolinite contributes.  Both are reported
    because they can disagree widely and the disagreement is informative; see
    :func:`chlorite_002_survival_from_003`.
    """

    survival_note: str = ""
    """How the chlorite 002's expected survival was arrived at.

    Either from this specimen's own chlorite 003 or, where that could not be
    measured, from the standards' absolute range; the two give very different
    kaolinite shares, so which one was used belongs in the report.  See
    :func:`chlorite_002_survival_from_003`.
    """

    second_order: tuple[float, float, bool] | None = None
    """``(expected, found, consistent)`` from :func:`second_order_check`.

    The third window's contribution.  A 7.15 A reflection that collapses on
    heating is kaolinite only if the 3.58 A second order is there in proportion,
    and the chlorite 004 window is what makes that second order measurable
    rather than blended with a chlorite's.
    """

    @property
    def usable(self) -> tuple[ChloriteShare, ...]:
        return tuple(
            share for share in self.shares
            if share.reference_area > 0.0 and share.window_area > 0.0
        )

    @property
    def measured(self) -> tuple[ChloriteShare, ...]:
        """Those resting on a chlorite reflection that was actually seen."""
        return tuple(share for share in self.usable if not share.limited)

    @property
    def bounds(self) -> tuple[float, float]:
        """The envelope of every usable estimate, or NaNs if there are none.

        The envelope rather than an average, because the estimates are not
        repeats of one measurement: each uses a different reflection of the
        chlorite, and where they disagree the disagreement is the uncertainty.
        """
        usable = self.usable
        if not usable:
            return float("nan"), float("nan")
        lows = [share.kaolinite[0] for share in usable]
        highs = [share.kaolinite[1] for share in usable]
        return float(min(lows)), float(max(highs))

    @property
    def best_bounds(self) -> tuple[float, float]:
        """The tightest bound the three mounts support, envelope or collapse route.

        The collapse route is a measurement where the envelope of
        :attr:`bounds` is the span of several, so the route is usually the
        tighter and preferring it is right - but not invariably, and the
        exception is not rare enough to ignore.  On a dickite standard the route
        gives 0.25 to 0.67 where the envelope gives 1.00 to 1.00, and applying
        the wider wrong one as a restraint at weight 50 drove R_wp to 582 per
        cent and halved the kaolinite the specimen is made of.  Taking whichever
        is narrower returns 1.00 to 1.00 and 99 per cent kaolinite.

        Narrower and not lower: a bound is only useful in proportion to how much
        it excludes, and between two honest measurements of the same quantity
        the tighter is the more informative.  Where they are equally wide the
        envelope is kept, it being the one that cannot be narrower than its own
        parts.
        """
        best = self.bounds
        width = best[1] - best[0]
        for route in self.collapse_routes:
            low, high = route.kaolinite_bounds
            if low != low or high != high:      # NaN: this route measured nothing
                continue
            if not (width == width) or (high - low) < width:
                best, width = (float(low), float(high)), high - low
        return best

    @property
    def informative(self) -> bool:
        """Whether :attr:`best_bounds` says anything a fit should be held to.

        A bound spanning half the range or more excludes almost nothing, and
        applying it as a restraint costs whatever the restraint costs while
        buying nothing.

        This judges the *width* of :attr:`best_bounds` and nothing else, which
        is all the evidence alone can judge.  A narrow bound is not the same as
        a sound one: on a montmorillonite, whose 7.15 A window holds no
        reflection, the envelope comes back 0 to 1 while a collapse route
        measured from the same noise says 1 to 1, and the narrower of those is
        narrow.  What catches that is the window itself, in
        :func:`clayquant.treatment.kaolinite_share_constraint`, which sees the
        pattern this does not and declines when the window stands below
        :data:`~clayquant.treatment.WINDOW_REFLECTION_SHARE` of the strongest
        line.  Both guards are needed and neither subsumes the other.
        """
        low, high = self.best_bounds
        if low != low or high != high:
            return False
        return (high - low) < BOUND_WIDTH_LIMIT

    @property
    def references_agree(self) -> bool:
        """Whether the chlorite 001 and 003 routes overlap for any window.

        They should.  When they do not, one of the two reference reflections is
        being mismeasured - the 001 sits on the steepest part of the low-angle
        air scatter, which is the usual culprit - and the answer is the wider
        bound rather than either of them.
        """
        by_reference: dict[str, list[tuple[float, float]]] = {}
        for share in self.measured:
            by_reference.setdefault(share.reference, []).append(share.kaolinite)
        if len(by_reference) < 2:
            return True
        spans = [
            (min(low for low, _ in bounds), max(high for _, high in bounds))
            for bounds in by_reference.values()
        ]
        return min(high for _, high in spans) >= max(low for low, _ in spans)

    @property
    def collapse_routes(self) -> tuple["KaoliniteResult", ...]:
        """The collapse tests that could be measured, 7.15 A and 3.58 A.

        Two independent measurements of the same thing: different reflections of
        both minerals, against chlorite orders that survive heating differently -
        32 to 72 % for the 002 under 7.15 A, 53 to 80 % for the 004 under 3.58 A.
        Reading only the first discarded half of what the heated mount measured.
        """
        return tuple(
            route for route in (self.collapse, self.second_collapse)
            if route is not None and route.air.is_present
        )

    @property
    def detected(self) -> bool:
        """Kaolinite is there whatever the chlorite under it is doing.

        The heated mount decides this where it exists, and the reason is that it
        is a *measurement* rather than an inference.  Heating to 550 C destroys
        kaolinite and leaves chlorite, so what disappears from the 7.15 A window
        is kaolinite, bounded only by how much of itself a chlorite 002 keeps -
        which was measured on two chlorite standards carrying no kaolinite.  The
        chlorite-ratio routes, by contrast, infer the chlorite 002 from the 001
        or the 003 through a ratio that varies from one chlorite to the next.

        Taking the weakest of all the routes, which this did, let one noisy
        estimate veto every other.  On a real separate the four ratio routes
        gave 2-35, 36-44, 40-51 and 52-66 per cent and the collapse test gave
        56-81, with 86 per cent of the peak gone after heating; the verdict was
        "not established" because one route's lower bound was 2.  A specimen
        whose 7.15 A reflection almost entirely disappears on heating has
        kaolinite in it, and no amount of disagreement between the ratio routes
        makes that untrue.
        """
        # The second order vetoes, and it has to: both chlorite standards have a
        # 7.15 A reflection that collapses on heating and neither contains any
        # kaolinite.  The collapse test alone passes them; the missing 3.58 A
        # order is what catches them.
        if self.second_order is not None and not self.second_order[2]:
            return False
        for route in self.collapse_routes:
            if route.kaolinite_detected:
                return True
        if self.collapse_routes:
            return False
        least, _ = self.bounds
        return bool(least == least and least > 0.1)

    @property
    def excluded(self) -> bool:
        """Whether the evidence positively rules kaolinite *out*.

        Not the negation of :attr:`detected`, and the distinction is the whole
        point: failing to establish a lower bound is not the same as measuring
        an upper one near zero.  Only the second is a reason to take kaolinite
        out of a fit, because taking it out asserts that there is none - which
        is one end of the range, not the middle of it.
        """
        if self.collapse_routes:
            highs = [route.kaolinite_bounds[1] for route in self.collapse_routes]
            highs = [h for h in highs if h == h]
            return bool(highs and max(highs) < 0.05)
        if not self.usable:
            return False
        _, most = self.bounds
        return bool(most == most and most < 0.05)

    def summary(self) -> str:
        if not self.usable:
            present = [name for name, peak in self.windows.items() if peak.is_present]
            if not present:
                return (
                    "Nothing above the background at 7.15 or 3.58 A, so there is no "
                    "kaolinite and no chlorite 002 to argue about."
                )
            return (
                f"Intensity at {' and '.join(present)} A, but no chlorite 001 or 003 "
                f"to measure it against, so none of it can be attributed. With no "
                f"chlorite in the specimen all of it is kaolinite."
            )
        least, most = self.bounds
        lines = [
            f"Kaolinite is {100.0 * least:.0f} to {100.0 * most:.0f}% of the "
            f"overlapped intensity, from {len(self.usable)} "
            f"{'estimate' if len(self.usable) == 1 else 'independent estimates'}:"
        ]
        lines.extend("  " + share.describe() + "." for share in self.usable)
        if not self.references_agree:
            lines.append(
                "  The chlorite 001 and 003 routes do not overlap, so one of them is "
                "being mismeasured - the 001 sits on the steepest part of the "
                "low-angle air scatter - and the range above is the wider of the two "
                "rather than a measurement."
            )
        for route in self.collapse_routes:
            lines.append("  " + route.summary())
        if self.second_order is not None:
            expected, found, ok = self.second_order
            if ok:
                lines.append(
                    f"  The 3.58 A second order holds {found:.4g} against the "
                    f"{expected:.4g} a kaolinite with this 001 would show, once the "
                    f"chlorite 004's tail is removed: consistent."
                )
            else:
                lines.append(
                    f"  The 3.58 A second order holds {found:.4g} against the "
                    f"{expected:.4g} a kaolinite with this 001 would show, once the "
                    f"chlorite 004's tail is removed. A 7.15 A reflection without a "
                    f"second order is not kaolinite - both chlorite standards collapse "
                    f"at 7.15 A too, and this is what tells them apart."
                )
        return "\n".join(lines)


def _detection_limit_area(metrics: PeakMetrics) -> float:
    """Area of the largest reflection this window could have hidden.

    A peak at the threshold has a matched height of ``minimum_sigmas`` standard
    errors of that estimate, and a Gaussian of that height and the instrumental
    width has an area of ``1.064 * height * FWHM``.  It is a bound, not a
    measurement, and it is what lets "no chlorite reflection here" be turned
    into a number instead of a shrug.
    """
    if metrics.noise <= 0.0 or metrics.averaged_points < 1:
        return 0.0
    height = (
        metrics.minimum_sigmas * metrics.noise / math.sqrt(metrics.averaged_points)
    )
    return float(1.064 * height * PEAK_WIDTH)


def kaolinite_evidence(
    air: Pattern,
    heated: Pattern | None = None,
    windows: dict[str, tuple[float, float]] | None = None,
    references: dict[str, tuple[float, float]] | None = None,
    ratios: dict[tuple[str, str], tuple[float, float]] | None = None,
    scale: float | None = None,
    wavelength: float = CU_KA1,
    minimum_sigmas: float = MINIMUM_SIGMAS,
) -> KaoliniteEvidence:
    """How much kaolinite there is, measured against the chlorite's own reflections.

    The two windows kaolinite occupies are both shared with chlorite: its 001 at
    12.36 deg against the chlorite 002 at 12.46, and its 002 at 24.85 against the
    chlorite 004 at 25.06.  Neither can be read alone.  What can be read alone is
    the chlorite's 001 at 6.22 deg and its 003 at 18.73 deg, which kaolinite does
    not have: each of those, times a ratio measured on chlorite standards
    (:data:`CHLORITE_RATIOS`), is the chlorite's share of an overlapped window,
    and the remainder is kaolinite.

    Four estimates follow, two windows times two reference reflections, and they
    are reported together.  Where they agree the answer is measured; where they
    disagree the disagreement is the uncertainty and is said so.

    ``heated`` adds the classical collapse test as a fifth line of evidence, but
    it is no longer what the answer rests on: heating enhances a chlorite 001 by
    two to three and a half times while collapsing its even orders, so what
    survives at 7.15 A after heating bounds nothing tightly.
    """
    windows = windows or {"7.15": KAOLINITE_001_WINDOW, "3.58": KAOLINITE_002_WINDOW}
    references = references or {
        "001": CHLORITE_001_WINDOW, "003": CHLORITE_003_WINDOW
    }
    ratios = CHLORITE_RATIOS if ratios is None else ratios

    measured = {
        name: measure_peak(air, window, wavelength=wavelength,
                           minimum_sigmas=minimum_sigmas)
        for name, window in windows.items()
    }
    reference_peaks = {
        name: measure_peak(air, window, wavelength=wavelength,
                           minimum_sigmas=minimum_sigmas)
        for name, window in references.items()
    }
    shares: list[ChloriteShare] = []
    for (window_name, reference_name), (low, high) in sorted(ratios.items()):
        window = measured.get(window_name)
        reference = reference_peaks.get(reference_name)
        if window is None or reference is None:
            continue
        if not window.is_present:
            continue
        window_area = window.area
        if window_area <= 0.0:
            continue
        # A chlorite reflection that did not clear the noise is not absent, it is
        # smaller than what this measurement could see - so it bounds the
        # chlorite rather than removing the estimate.  Without this a pure
        # kaolinite, which has no chlorite reflection at all, gets no answer.
        if reference.is_present:
            reference_area = reference.area
            limited = False
        else:
            reference_area = _detection_limit_area(reference)
            limited = True
        if reference_area <= 0.0:
            continue
        shares.append(
            ChloriteShare(
                window=window_name,
                reference=reference_name,
                reference_area=reference_area,
                window_area=window_area,
                chlorite=(
                    0.0 if limited else min(1.0, low * reference_area / window_area),
                    min(1.0, high * reference_area / window_area),
                ),
                limited=limited,
            )
        )
    collapse = second_collapse = measured_collapse = None
    survival_note = ""
    if heated is not None:
        # What the chlorite 002 under the 7.15 A window was expected to survive
        # is the whole of this test, and the standards' absolute range is the
        # wrong thing to assume: it spans 32 to 72 %, and a chlorite that
        # collapses harder than either standard then hands kaolinite a drop the
        # chlorite managed alone.  The specimen's own 003 says what its chlorite
        # did, in a window kaolinite cannot reach, and the standards supply only
        # the ratio between the orders.  See
        # :func:`chlorite_002_survival_from_003`.
        least, most, survival_note = chlorite_002_survival_from_003(
            air, heated, scale=scale, wavelength=wavelength,
            minimum_sigmas=minimum_sigmas,
        )
        collapse = kaolinite_collapse(
            air, heated, window=windows.get("7.15", KAOLINITE_001_WINDOW),
            scale=scale, wavelength=wavelength,
            survival=CHLORITE_002_SURVIVAL, label="7.15 A",
        )
        # The same window again, against what this specimen's own chlorite did
        # rather than what the standards' chlorites did.  It is carried beside
        # the standards' route and not in place of it, because the two can
        # disagree by a factor of three and which is right depends on how the
        # heated 003 is integrated - a peak that is often at the detection
        # limit.  Where they disagree that is the finding, and it belongs in
        # front of whoever is reading the report rather than resolved silently:
        # a specimen whose chlorite collapses harder than either standard gets
        # a kaolinite share from the standards' range that is far too high.
        measured_collapse = kaolinite_collapse(
            air, heated, window=windows.get("7.15", KAOLINITE_001_WINDOW),
            scale=scale, wavelength=wavelength,
            survival=(least, most), label="7.15 A, on this chlorite's own 003",
        )
        # The 3.58 A window is the same test on kaolinite's second order, and
        # it is independent of the first: a different reflection of each
        # mineral, against a chlorite order that survives heating differently.
        second_collapse = kaolinite_collapse(
            air, heated, window=windows.get("3.58", KAOLINITE_002_WINDOW),
            scale=scale, wavelength=wavelength,
            survival=CHLORITE_004_SURVIVAL, label="3.58 A",
        )
    return KaoliniteEvidence(
        windows=measured,
        references=reference_peaks,
        shares=tuple(shares),
        collapse=collapse,
        second_collapse=second_collapse,
        survival_note=survival_note,
        measured_collapse=measured_collapse,
        second_order=second_order_check(
            air, wavelength=wavelength,
            # the lowest kaolinite share any route allows, so the expectation is
            # the smallest second order consistent with the evidence and the
            # veto fires only when even that would have been visible
            kaolinite_share=_least_kaolinite_share(shares, collapse),
        ),
    )


MINIMUM_DOUBLET_SEPARATION = 0.22
"""How far apart the kaolinite 002 and chlorite 004 must be to be separated, in deg.

The split between the two windows cannot be a fixed angle, because the chlorite
004 is wherever that chlorite's 001 puts it: a 14.1 A chlorite has it at
25.25 deg, clear of the kaolinite 002 at 24.86, and a 14.4 A chlorite at 24.72 -
on the wrong side of it.  Deriving the window from the measured chlorite 001
handles the position; this handles the cases where no window would help, because
the two reflections are closer together than the width of either.

Set at 0.22 deg, which is above the 0.14 deg width of a clay reflection on this
instrument and below the 0.29 deg the specimens that motivated the split showed.
Where the separation is smaller the deconvolution is not attempted and the second
order is reported as untestable rather than as absent - an unresolved doublet is
missing information, not evidence.
"""


def chlorite_004_window(
    pattern: Pattern,
    wavelength: float = CU_KA1,
    default: tuple[float, float] = CHLORITE_004_WINDOW,
) -> tuple[float, float] | None:
    """Where this specimen's chlorite 004 is, from where its own 001 is.

    A chlorite's fourth order sits at a quarter of its 001 spacing, and that
    spacing varies enough between chlorites to move the reflection across the
    kaolinite 002.  Measuring it rather than assuming it is what lets the two be
    split at all.  ``None`` where there is no chlorite 001 to measure.
    """
    first = measure_peak(pattern, CHLORITE_001_WINDOW, wavelength=wavelength)
    if not first.is_present or first.d_spacing <= 0.0:
        return default
    d004 = first.d_spacing / 4.0
    argument = wavelength / (2.0 * d004)
    if not -1.0 < argument < 1.0:
        return default
    centre = math.degrees(2.0 * math.asin(argument))
    half = 0.5 * (default[1] - default[0])
    return (centre - half, centre + half)


def kaolinite_002_area(
    pattern: Pattern,
    kaolinite_window: tuple[float, float] = KAOLINITE_002_WINDOW,
    chlorite_window: tuple[float, float] | None = None,
    leakage: tuple[float, float] = CHLORITE_004_LEAKAGE,
    wavelength: float = CU_KA1,
) -> tuple[float, float]:
    """Kaolinite's 002 area with the chlorite 004's tail removed, as a range.

    The third window is what makes this possible.  A chlorite can be measured
    where no kaolinite reaches - its 001, its 003 and the high-angle side of the
    3.5 A doublet - and the last of those says how much of the doublet's
    low-angle side is the chlorite's tail rather than kaolinite's second order.

    Returns ``(least, most)`` from the two ends of :data:`CHLORITE_004_LEAKAGE`,
    clipped at zero.  A range that reaches zero means the 002 window holds
    nothing the chlorite does not account for.
    """
    if chlorite_window is None:
        chlorite_window = chlorite_004_window(pattern, wavelength=wavelength)
    kao = measure_peak(pattern, kaolinite_window, wavelength=wavelength)
    chl = measure_peak(pattern, chlorite_window, wavelength=wavelength)
    kao_area = kao.area if kao.is_present else 0.0
    chl_area = chl.area if chl.is_present else 0.0
    high_leak, low_leak = max(leakage), min(leakage)
    return (
        float(max(0.0, kao_area - high_leak * chl_area)),
        float(max(0.0, kao_area - low_leak * chl_area)),
    )


def second_order_check(
    pattern: Pattern,
    first_window: tuple[float, float] = KAOLINITE_001_WINDOW,
    ratio: tuple[float, float] = KAOLINITE_002_TO_001,
    wavelength: float = CU_KA1,
    kaolinite_share: float = 1.0,
    **kwargs,
) -> tuple[float, float, bool]:
    """Whether the 3.58 A second order is consistent with the 7.15 A first.

    Three windows make this possible, which two could not: the chlorite 004
    window says how much of the 3.5 A doublet is the chlorite's tail, leaving
    what is actually kaolinite's second order, and that can then be held against
    the first order in the overlapped 7.15 A window.

    ``kaolinite_share`` is the fraction of the 7.15 A window that is kaolinite
    rather than chlorite 002, and it must be passed or the test is wrong in a
    way that matters.  The 7.15 A window is *shared*: computing the expected
    second order from the whole of it assumes every count there is kaolinite,
    which inflates the expectation by however much chlorite is present and can
    turn an undetectably small second order into a refusal.  A specimen whose
    7.15 A window is a quarter kaolinite expects a quarter of the second order,
    and if that sits in the noise its absence says nothing at all.

    Returns ``(expected_least, measured_most, consistent)``.  ``consistent`` is
    False only where a second order would have had to be visible and is not.
    """
    # An unresolved doublet is missing information rather than evidence, so the
    # check stands down where the two reflections are closer than their width.
    window = chlorite_004_window(pattern, wavelength=wavelength)
    kaolinite_centre = 0.5 * sum(KAOLINITE_002_WINDOW)
    if abs(0.5 * sum(window) - kaolinite_centre) < MINIMUM_DOUBLET_SEPARATION:
        return 0.0, 0.0, True
    first = measure_peak(pattern, first_window, wavelength=wavelength)
    least, most = kaolinite_002_area(pattern, wavelength=wavelength, **kwargs)
    if not first.is_present or first.area <= 0.0:
        return 0.0, float(most), True
    expected_least = min(ratio) * first.area * max(0.0, float(kaolinite_share))
    # Below this the second order would sit in the noise whatever is there, so
    # its absence says nothing.
    if expected_least < 3.0 * first.noise:
        return float(expected_least), float(most), True
    return float(expected_least), float(most), bool(most >= 0.5 * expected_least)


def _least_kaolinite_share(shares, collapse) -> float:
    """The smallest kaolinite share of the 7.15 A window the evidence allows.

    Used to size the second order the specimen should show.  The *smallest* share
    rather than the best estimate, because this feeds a refusal: the test should
    only refuse kaolinite when even the least of it the evidence permits would
    have produced a visible 3.58 A reflection.
    """
    candidates = [
        share.kaolinite[0] for share in shares
        if share.window == "7.15" and share.kaolinite[0] == share.kaolinite[0]
    ]
    if collapse is not None and collapse.air.is_present:
        least, _ = collapse.kaolinite_bounds
        if least == least:
            candidates.append(least)
    if not candidates:
        return 1.0
    return float(max(0.0, min(candidates)))


def chlorite_survival_from_type(
    air: Pattern,
    calibration: tuple[tuple[float, float], tuple[float, float]] = CHLORITE_TYPE_CALIBRATION,
    fallback: tuple[float, float] = CHLORITE_002_SURVIVAL,
    wavelength: float = CU_KA1,
) -> tuple[float, float, str]:
    """Narrow the 7.15 A survival range using which chlorite this specimen has.

    The 003/001 ratio is measured on two reflections kaolinite does not have, so
    unlike every other way of characterising the chlorite in a kaolinite-bearing
    specimen it is not contaminated by the mineral it is being used to separate.
    That is the whole reason it can be used here: a specimen's own 002 or 004
    cannot, both being shared.

    Returns ``(least, most, note)``.  Inside the bracket the range is narrowed
    about the interpolated value; outside it the full range is kept and the note
    says so.
    """
    first = measure_peak(air, CHLORITE_001_WINDOW, wavelength=wavelength)
    third = measure_peak(air, CHLORITE_003_WINDOW, wavelength=wavelength)
    if not (first.is_present and third.is_present) or first.area <= 0.0:
        return (*fallback, "no chlorite 001 and 003 to type the chlorite by")
    ratio = third.area / first.area
    (r_low, s_low), (r_high, s_high) = calibration
    if ratio < r_low - 1e-6:
        return (*fallback,
                f"003/001 = {ratio:.3f} is below both standards ({r_low:.3f}, "
                f"{r_high:.3f}), so the chlorite is at or beyond the clinochloritic "
                f"end and the full range is kept rather than extrapolated")
    if ratio > r_high + 1e-6:
        return (*fallback,
                f"003/001 = {ratio:.3f} is above both standards, so the chlorite is "
                f"at or beyond the ferrous end and the full range is kept")
    fraction = (ratio - r_low) / (r_high - r_low)
    centre = s_low + fraction * (s_high - s_low)
    half = 0.5 * abs(s_high - s_low) * 0.35
    return (max(0.0, centre - half), min(1.0, centre + half),
            f"003/001 = {ratio:.3f} puts this chlorite {100 * fraction:.0f} % of the "
            f"way from the clinochlore to the prochlorite, on a two-point calibration")
