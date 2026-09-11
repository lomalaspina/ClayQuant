"""Diagnostic comparisons between the air-dried, heated and glycolated mounts.

Two classical tests are implemented, both of which compare two mounts of the
same clay separate:

Kaolinite, from air-dried against heated (500 C)
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
]

CU_KA1 = 1.540596

KAOLINITE_001_WINDOW = (11.6, 13.2)
"""2theta window around the kaolinite 001 / chlorite 002 reflection at ~7.15 A."""

EXPANDABLE_001_WINDOW = (3.0, 10.5)
"""2theta window in which the expandable 001 reflection is sought."""

ILLITE_002_WINDOW = (17.0, 18.4)
"""2theta window around the illite 002 reflection at 5.01 A."""


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

    @property
    def is_present(self) -> bool:
        return self.height > 0.0 and self.area > 0.0


def measure_peak(
    pattern: Pattern,
    window: tuple[float, float],
    baseline: str = "snip",
    wavelength: float = CU_KA1,
) -> PeakMetrics:
    """Measure the peak inside ``window``.

    The position is the centroid of the points above half the window maximum,
    which is robust against the asymmetry the Ka doublet introduces.

    Parameters
    ----------
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

    height = float(y.max())
    if height <= 0.0:
        return PeakMetrics(
            window=(low, high),
            position=float("nan"),
            d_spacing=float("nan"),
            height=0.0,
            area=0.0,
            n_points=int(inside.size),
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
    )


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
    def kaolinite_detected(self) -> bool:
        return self.collapse_fraction > 0.1

    @property
    def chlorite_indicated(self) -> bool:
        """Whether intensity survives at 7.15 A, indicating chlorite 002."""
        return self.residual_fraction > 0.1

    def summary(self) -> str:
        if not self.air.is_present:
            return "No reflection at 7.15 A in the air-dried mount: no kaolinite and no chlorite."
        return (
            f"7.15 A peak area {self.air.area:.4g} (air) vs {self.scale * self.heated.area:.4g} "
            f"(heated, scaled): {100.0 * self.collapse_fraction:.1f}% collapsed on heating "
            f"(kaolinite), {100.0 * self.residual_fraction:.1f}% survived (chlorite 002)."
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

    lost_area = air_peak.area - scale * heated_peak.area
    lost_height = air_peak.height - scale * heated_peak.height
    if air_peak.area > 0:
        collapse = float(np.clip(lost_area / air_peak.area, 0.0, 1.0))
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
            return "No low-angle reflection in the glycolated mount."
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
        if air_peak.area > 0 and glycol_peak.is_present
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
