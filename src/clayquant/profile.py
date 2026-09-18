"""Peak shapes and instrumental broadening.

Reflection widths follow the usual Caglioti form for the instrumental
contribution,

    FWHM_inst(theta)^2 = U tan^2(theta) + V tan(theta) + W    [deg^2]

optionally combined with a Scherrer size broadening.  Clay crystallites are
strongly anisotropic - thin along ``c*`` and wider in the ``ab`` plane - so the
effective coherent domain size is interpolated between the two directions as

    1 / L(alpha) = cos^2(alpha) / L_c + sin^2(alpha) / L_ab

with ``alpha`` the angle between the reflection vector and ``c*``.  For basal
reflections of a mixed-layer stack the thickness broadening is already contained
in the interference function of :mod:`clayquant.mixed_layer`, so ``size_c``
should be left at ``None`` there to avoid counting it twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "CU_KALPHA2_OVER_KALPHA1",
    "KALPHA2_INTENSITY_RATIO",
    "QUARTZ_SEARCH_WINDOW",
    "QUARTZ_UNSHIFTED_WINDOW",
    "QUARTZ_WIDTH_LIMITS",
    "LineWidth",
    "kalpha2_two_theta",
    "quartz_line_width",
    "shape_from_line_width",

    "MeasuredWidths",
    "PeakShape",
    "accumulate_peaks",
    "convolve_variable_fwhm",
    "fit_peak_shape",
    "measure_peak_widths",
    "pseudo_voigt",
]

_GAUSS_NORM = 2.0 * math.sqrt(math.log(2.0) / math.pi)
_LORENTZ_NORM = 2.0 / math.pi
_SCHERRER = 0.9


def pseudo_voigt(x: np.ndarray, fwhm: np.ndarray | float, eta: float) -> np.ndarray:
    """Normalised pseudo-Voigt profile evaluated at offsets ``x``.

    ``eta`` is the Lorentzian fraction: 0 gives a Gaussian, 1 a Lorentzian.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    fwhm = np.asarray(fwhm, dtype=float)
    gauss = (_GAUSS_NORM / fwhm) * np.exp(-4.0 * math.log(2.0) * (x / fwhm) ** 2)
    lorentz = (_LORENTZ_NORM / fwhm) / (1.0 + 4.0 * (x / fwhm) ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


@dataclass
class PeakShape:
    """Reflection width model.

    Attributes
    ----------
    u, v, w:
        Caglioti coefficients of the instrumental FWHM in deg^2.
    eta:
        Lorentzian fraction of the pseudo-Voigt profile.
    size_c, size_ab:
        Coherent domain sizes in A along ``c*`` and in the ``ab`` plane.
        ``None`` disables the corresponding size broadening.
    """

    u: float = 0.0
    v: float = 0.0
    w: float = 0.01
    eta: float = 0.5
    size_c: float | None = None
    size_ab: float | None = None

    def fwhm(
        self,
        two_theta: np.ndarray,
        wavelength: float,
        alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        """Total FWHM in degrees at the given ``two_theta`` values."""
        two_theta = np.asarray(two_theta, dtype=float)
        theta = np.radians(two_theta) / 2.0
        tan_theta = np.tan(theta)
        instrumental = self.u * tan_theta**2 + self.v * tan_theta + self.w
        variance = np.clip(instrumental, 1e-8, None)

        if self.size_c is not None or self.size_ab is not None:
            size_c = self.size_c if self.size_c is not None else np.inf
            size_ab = self.size_ab if self.size_ab is not None else np.inf
            if alpha is None:
                inverse_size = 1.0 / size_c
            else:
                alpha = np.asarray(alpha, dtype=float)
                inverse_size = np.cos(alpha) ** 2 / size_c + np.sin(alpha) ** 2 / size_ab
            with np.errstate(divide="ignore", invalid="ignore"):
                size_fwhm = np.degrees(
                    _SCHERRER * wavelength * inverse_size / np.maximum(np.cos(theta), 1e-6)
                )
            variance = variance + np.nan_to_num(size_fwhm) ** 2
        return np.sqrt(variance)


def accumulate_peaks(
    grid: np.ndarray,
    centers: np.ndarray,
    intensities: np.ndarray,
    fwhms: np.ndarray,
    eta: float,
    cutoff: float = 12.0,
) -> np.ndarray:
    """Sum peak profiles onto ``grid``.

    Each peak is evaluated only within ``cutoff`` times its FWHM, which keeps
    the cost proportional to the number of reflections rather than to their
    product with the grid size.
    """
    grid = np.asarray(grid, dtype=float)
    total = np.zeros_like(grid)
    if len(centers) == 0:
        return total
    step = float(np.mean(np.diff(grid))) if len(grid) > 1 else 1.0
    for center, intensity, fwhm in zip(centers, intensities, fwhms):
        if intensity == 0.0:
            continue
        half_window = cutoff * fwhm
        start = int(np.searchsorted(grid, center - half_window))
        stop = int(np.searchsorted(grid, center + half_window))
        if stop <= start:
            # Peak falls between grid points: put it on the nearest one.
            nearest = int(np.clip(np.searchsorted(grid, center), 0, len(grid) - 1))
            total[nearest] += intensity / step
            continue
        window = grid[start:stop]
        total[start:stop] += intensity * pseudo_voigt(window - center, fwhm, eta)
    return total


def convolve_variable_fwhm(
    grid: np.ndarray,
    y: np.ndarray,
    fwhm: np.ndarray,
    eta: float,
    cutoff: float = 12.0,
) -> np.ndarray:
    """Convolve ``y`` with a pseudo-Voigt kernel whose width varies along ``grid``.

    Used to apply instrumental broadening to the continuous intensity curve of a
    mixed-layer stack, where the width changes with angle.
    """
    grid = np.asarray(grid, dtype=float)
    y = np.asarray(y, dtype=float)
    fwhm = np.broadcast_to(np.asarray(fwhm, dtype=float), grid.shape)
    if len(grid) < 2:
        return y.copy()
    step = float(np.mean(np.diff(grid)))
    result = np.zeros_like(y)
    for index, (position, value, width) in enumerate(zip(grid, y, fwhm)):
        if value == 0.0:
            continue
        span = max(1, int(cutoff * width / step))
        start = max(0, index - span)
        stop = min(len(grid), index + span + 1)
        kernel = pseudo_voigt(grid[start:stop] - position, width, eta)
        total = kernel.sum()
        if total > 0:
            result[start:stop] += value * kernel / total
    return result


@dataclass(frozen=True)
class MeasuredWidths:
    """The isolated peaks a width model was fitted to, and how well it fits.

    Attributes
    ----------
    shape:
        The fitted :class:`PeakShape`.  Its ``eta`` is whichever of the
        candidates fitted the peak *shapes* best, and its ``u``, ``v``, ``w``
        come from the widths.
    two_theta, fwhm:
        The peaks used and their measured full widths at half maximum, in
        degrees.
    residual:
        Root mean square difference between the measured widths and the model,
        in degrees.  A value much above the step size means the peaks were not
        all from one specimen broadening - mixed phases with different
        crystallite sizes, say - and the model is a compromise.
    note:
        What happened, for the record: how many peaks were found and whether
        the default was kept.
    """

    shape: "PeakShape"
    two_theta: np.ndarray
    fwhm: np.ndarray
    residual: float
    note: str


def measure_peak_widths(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    minimum_height: float = 0.02,
    separation: float = 0.45,
    maximum_width: float = 0.6,
    tolerance: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Positions and full widths at half maximum of well-separated peaks.

    Deliberately conservative about which peaks count.  A width is only usable
    if the peak stands alone: the half-maximum crossing has to be found on both
    flanks inside ``separation`` degrees, the profile has to fall monotonically
    to each crossing, and the result has to be narrower than ``maximum_width``.
    An overlapped doublet read as one peak reports the width of the pair, and a
    width model fitted to that is broader than the instrument everywhere, which
    is precisely the error this exists to avoid.

    ``intensity`` should already have its background removed.  ``minimum_height``
    is a fraction of the strongest point, and ``tolerance`` is how much of the
    peak's own height a point on the flank may rise by before it is taken as the
    next peak beginning.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if two_theta.size < 5:
        return np.array([]), np.array([])
    step = float(np.median(np.diff(two_theta)))
    reach = max(2, int(round(separation / step)))
    threshold = minimum_height * float(np.max(intensity))

    positions: list[float] = []
    widths: list[float] = []
    for index in range(reach, intensity.size - reach):
        height = intensity[index]
        if height < threshold:
            continue
        window = intensity[index - reach:index + reach + 1]
        if height < window.max():
            continue
        half = height / 2.0
        # Walk out to the half-maximum crossing, refusing to cross a minimum on
        # the way: a rise on the flank means the neighbour peak has taken over.
        edges = []
        for direction in (-1, 1):
            previous = height
            crossing = None
            for step_count in range(1, reach + 1):
                value = intensity[index + direction * step_count]
                # A rise has to be a real one.  Counting every upward wiggle as
                # the next peak taking over loses almost every width on a sharp,
                # well counted pattern, where the flanks are noisy in relative
                # terms precisely because they are low.
                if value > previous + tolerance * height:
                    break
                if value <= half:
                    before = intensity[index + direction * (step_count - 1)]
                    span = before - value
                    fraction = (before - half) / span if span > 0 else 0.0
                    crossing = two_theta[index + direction * (step_count - 1)] + (
                        direction * fraction * step
                    )
                    break
                previous = value
            if crossing is None:
                break
            edges.append(crossing)
        if len(edges) != 2:
            continue
        width = abs(edges[1] - edges[0])
        # Three steps is the fewest that can describe a width rather than a
        # spike: two points above half maximum say only that the peak is
        # narrower than the step, and a single hot channel satisfies every other
        # test here.  On a 0.0167 deg step this rejects anything under 0.05 deg.
        if not 3.0 * step <= width <= maximum_width:
            continue
        positions.append(float(two_theta[index]))
        widths.append(float(width))
    return np.array(positions), np.array(widths)


def fit_peak_shape(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    wavelength: float,
    default: "PeakShape | None" = None,
    etas: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9, 1.0),
    minimum_peaks: int = 3,
    envelope: float = 0.4,
) -> MeasuredWidths:
    """Scale a width model to a measurement, and pick its mixing parameter.

    Why this is worth doing rather than carrying a default.  The calculated
    pattern's peak height is set by its width: a profile ten per cent too broad
    is ten per cent too short, and a fit of scale factors alone cannot recover
    that, so it leaves the fit short of intensity at every strong peak at once -
    which looks exactly like a missing phase and is not one.  The width belongs
    to the diffractometer and the specimen, not to the software.

    What is fitted is one number: the factor by which ``default``'s whole width
    curve is stretched, taken from the measured widths as
    ``k = median(measured / modelled)`` over the peaks used.  Fitting all three
    Caglioti coefficients instead was tried first and abandoned - a real pattern
    yields between two and a dozen usable isolated peaks, three coefficients
    from three peaks fit exactly and so say nothing about whether the form is
    right, and the extrapolations that came out of it reached ``U`` near 2, half
    a degree of width at the top of the scan.  The angular *trend* of the
    instrumental width is generic; what differs between machines and
    configurations is its size, and that is the one thing estimated here.

    The peaks used are the lower envelope, not all of them.  On a clay mount the
    measured widths belong to two things at once: the instrument, which every
    phase shares, and the crystallite size of each phase, which the library
    spans separately through its thickness and CSDS axes.  Averaging over all
    peaks lands between the sharp quartz and the broad clay, double-counting the
    clay broadening and leaving quartz too wide - both errors together.  So the
    narrowest ``envelope`` fraction of the peaks is used, which is the part of
    the width that is not specimen broadening.

    ``eta`` is chosen by how well each candidate reproduces the *shape* of the
    strongest peak used, at the width the scaled model gives, because a width
    alone cannot tell a Gaussian from a Lorentzian of the same half width.

    With fewer than ``minimum_peaks`` usable peaks the ``default`` comes back
    unchanged and ``note`` says so.
    """
    default = default or PeakShape()
    positions, widths = measure_peak_widths(two_theta, intensity)
    if positions.size < minimum_peaks:
        return MeasuredWidths(
            shape=default, two_theta=positions, fwhm=widths, residual=float("nan"),
            note=(f"{positions.size} isolated peaks found, fewer than the "
                  f"{minimum_peaks} needed; kept the default width model"),
        )

    modelled = default.fwhm(positions, wavelength)
    ratio = widths / np.clip(modelled, 1e-9, None)
    order = np.argsort(ratio)
    keep = order[:max(minimum_peaks, int(round(envelope * positions.size)))]
    factor = float(np.median(ratio[keep]))
    if not np.isfinite(factor) or factor <= 0.0:
        return MeasuredWidths(
            shape=default, two_theta=positions, fwhm=widths, residual=float("nan"),
            note="the measured widths gave no usable scale; kept the default width model",
        )

    scaled = replace(
        default,
        u=default.u * factor**2,
        v=default.v * factor**2,
        w=default.w * factor**2,
        size_c=None if default.size_c is None else default.size_c / factor,
        size_ab=None if default.size_ab is None else default.size_ab / factor,
    )

    strongest = positions[keep][int(np.argmax([
        intensity[int(np.argmin(np.abs(np.asarray(two_theta) - position)))]
        for position in positions[keep]
    ]))]
    best_eta, best_cost = default.eta, float("inf")
    window = np.abs(np.asarray(two_theta) - strongest) < 0.7
    x = np.asarray(two_theta)[window]
    y = np.asarray(intensity)[window]
    if y.size > 4 and y.max() > 0:
        y = y / y.max()
        width = float(scaled.fwhm(np.array([strongest]), wavelength)[0])
        for eta in etas:
            model = pseudo_voigt(x - strongest, width, eta)
            model = model / model.max() if model.max() > 0 else model
            cost = float(np.mean((y - model) ** 2))
            if cost < best_cost:
                best_eta, best_cost = eta, cost
    shape = replace(scaled, eta=best_eta)

    final = shape.fwhm(positions[keep], wavelength)
    return MeasuredWidths(
        shape=shape, two_theta=positions, fwhm=widths,
        residual=float(np.sqrt(np.mean((widths[keep] - final) ** 2))),
        note=(f"width model scaled by {factor:.3f} to the {keep.size} narrowest "
              f"of {positions.size} isolated peaks"),
    )


# --------------------------------------------------------------------------
# The instrumental width, measured where it is instrumental
# --------------------------------------------------------------------------

CU_KALPHA2_OVER_KALPHA1 = 1.544426 / 1.540596
"""Wavelength ratio of the Cu K-alpha doublet, for placing the second line."""

KALPHA2_INTENSITY_RATIO = 0.5
"""Intensity of K-alpha2 relative to K-alpha1.  Clayfit's ``CU_KALPHA2_TO_KALPHA1``."""

QUARTZ_WIDTH_LIMITS = (0.02, 0.40)
"""Range a laboratory instrument's own line width can credibly lie in, in degrees.

Wider than Clayfit's 0.035 to 0.16, which are the bounds of the instrument it
was written for.  A bound that the data would exceed is worse than no bound: the
fit then sits *on* it and returns it as though it had been measured, and a
library built from that number is wrong in the one way nothing downstream can
detect.  So the range covers any laboratory powder diffractometer, and a fit
that lands on either end is rejected rather than reported (see
:func:`quartz_line_width`).
"""

QUARTZ_SEARCH_WINDOW = 0.15
"""How far from its expected angle the quartz 101 is looked for, in degrees.

The line has to be located before it can be fitted, and centring the fit on the
nominal angle instead was a real error: on a scan whose quartz sat 0.04 deg
away, the centre pinned against its own bound and the width came back 15 per
cent narrow.

The window is tight on purpose, and takes a measured ``shift`` rather than
widening to cover an unknown one.  A clay separate's pattern is crowded around
26.6 deg - illite 003 is at 26.9 - so a window of half a degree finds the
strongest peak in the neighbourhood rather than the quartz 101: on one real
mount it settled on something 0.30 deg away and fitted its width instead.  The
load-time screen measures the displacement of each mount from the quartz pair
itself (:func:`clayquant.detection.quartz_zero_shift`), and that is what should
be passed in.
"""

QUARTZ_UNSHIFTED_WINDOW = 0.45
"""The window used when no displacement has been measured yet.

Wider, because with no shift the line really may be a few tenths away, and a
window that cannot reach it finds nothing.  What the width then rests on is
whichever peak is strongest in that stretch, so the note says how far from
nominal the fitted line sits and the caller can judge it.
"""


def kalpha2_two_theta(two_theta_kalpha1: float) -> float:
    """Where the K-alpha2 partner of a K-alpha1 reflection falls."""
    sine = CU_KALPHA2_OVER_KALPHA1 * math.sin(math.radians(float(two_theta_kalpha1) / 2.0))
    if not 0.0 <= sine <= 1.0:
        raise ValueError("the reflection is outside the K-alpha2 Bragg range")
    return 2.0 * math.degrees(math.asin(sine))


def _unit_pseudo_voigt(two_theta: np.ndarray, centre: float, fwhm: float,
                       eta: float) -> np.ndarray:
    scaled = (np.asarray(two_theta, dtype=float) - float(centre)) / float(fwhm)
    gaussian = np.exp(-4.0 * math.log(2.0) * scaled * scaled)
    lorentzian = 1.0 / (1.0 + 4.0 * scaled * scaled)
    return (1.0 - eta) * gaussian + eta * lorentzian


@dataclass
class LineWidth:
    """One reflection's own width, fitted as a K-alpha doublet."""

    fwhm: float
    eta: float
    centre: float
    note: str


def quartz_line_width(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    reference: float,
    shift: float = 0.0,
    limits: tuple[float, float] = QUARTZ_WIDTH_LIMITS,
) -> LineWidth | None:
    """Fit the instrumental width to the quartz K-alpha doublet.

    This exists because the alternative does not work on an oriented clay
    mount, and the failure is not subtle.  :func:`fit_peak_shape` scales the
    width model to the narrowest isolated peaks of the pattern, and on such a
    mount the isolated peaks *are* the clay basal reflections, which are broad
    by nature: crystallite thickness and interstratification both widen them,
    and neither belongs to the instrument.  Measured on three real glycol
    mounts, that route gave 0.077, 0.124 and 0.234 deg for the same
    diffractometer, and changed by a factor of two on one mount depending on
    whether a background had been subtracted first.  A library built with one of
    those numbers and checked against another is what makes the program warn
    about its own build.

    Quartz is the standard answer and the reason is that its lines are
    instrumental: a detrital quartz grain in a clay separate is large enough
    that its size broadening is immeasurable, so what its 101 shows is the
    machine.  The fit here is Clayfit's: a pseudo-Voigt doublet with the
    wavelength separation and the 2:1 intensity ratio held fixed, on a local
    linear background, returning the width of *one* component.  Holding the
    doublet fixed is the point - a half-height width read off an unresolved
    K-alpha1/K-alpha2 pair is the width of the envelope, which at 26.6 deg is
    about 0.04 deg too broad, and a library calculated from it suppresses its
    own K-alpha2 shoulder a second time.

    ``reference`` is the K-alpha1 position of the line and ``shift`` a known zero
    error; the line is then *located* within :data:`QUARTZ_SEARCH_WINDOW` of
    there before being fitted, so an uncorrected displacement does not bias the
    width.  Returns ``None`` when the line is outside the scan, too weak to fit,
    or comes back on either end of ``limits`` - in which case the caller should
    say the width was not measured rather than use a bound as a measurement.
    """
    from scipy.optimize import least_squares

    angles = np.asarray(two_theta, dtype=float)
    counts = np.asarray(intensity, dtype=float)
    if angles.shape != counts.shape or angles.size < 10:
        return None
    nominal = float(reference) - float(shift)
    window = QUARTZ_SEARCH_WINDOW if shift else QUARTZ_UNSHIFTED_WINDOW
    near = np.flatnonzero(np.abs(angles - nominal) <= window)
    if near.size < 5:
        return None
    expected = float(angles[near][int(np.argmax(counts[near]))])
    try:
        partner = kalpha2_two_theta(expected + float(shift)) - float(shift)
    except ValueError:
        return None
    inside = (angles >= expected - 0.12) & (angles <= partner + 0.08)
    if int(np.count_nonzero(inside)) < 10:
        return None
    local_angles = angles[inside]
    local_counts = counts[inside]
    span = float(np.ptp(local_counts))
    if not np.isfinite(span) or span <= 0.0:
        return None
    step = float(np.median(np.diff(angles)))
    middle = 0.5 * (expected + partner)
    half_window = max(0.5 * float(np.ptp(local_angles)), 1e-6)
    ramp = (local_angles - middle) / half_window

    def residual(parameters):
        centre, width, eta, amplitude, intercept, slope = parameters
        try:
            partner_centre = kalpha2_two_theta(centre + shift) - shift
        except ValueError:
            return np.full(local_counts.shape, 1e3)
        profile = _unit_pseudo_voigt(local_angles, centre, width, eta)
        profile += KALPHA2_INTENSITY_RATIO * _unit_pseudo_voigt(
            local_angles, partner_centre, width, eta)
        calculated = intercept + slope * ramp + amplitude * profile
        return (calculated - local_counts) / span

    floor = max(limits[0], 1.25 * step)
    start = [expected, float(np.clip(max(0.055, 2.5 * step), floor, limits[1])),
             0.35, span, float(np.min(local_counts)), 0.0]
    lower = [expected - 0.06, floor, 0.0, 0.0, -np.inf, -np.inf]
    upper = [expected + 0.06, limits[1], 1.0, np.inf, np.inf, np.inf]
    start = [float(np.clip(value, low, high))
             for value, low, high in zip(start, lower, upper)]
    try:
        solved = least_squares(residual, start, bounds=(lower, upper),
                               loss="soft_l1", f_scale=0.05, max_nfev=4000)
    except (ValueError, FloatingPointError):
        return None
    width = float(solved.x[1])
    if not solved.success or not np.isfinite(width):
        return None
    # A width sitting on either bound was decided by the bound, not by the
    # data, and returning it would put a number nobody measured into the
    # library.  Rejecting sends the caller to the fallback, which says so.
    if width <= floor * 1.02 or width >= limits[1] * 0.98:
        return None
    return LineWidth(
        fwhm=width,
        eta=float(np.clip(solved.x[2], 0.0, 1.0)),
        centre=float(solved.x[0]),
        note=(
            f"width {width:.3f} deg from the {reference:.2f} deg quartz K-alpha "
            f"doublet, fitted at {float(solved.x[0]):.3f} deg with the doublet "
            f"separation and 2:1 ratio held fixed"
            + (f" - {abs(float(solved.x[0]) - nominal):.3f} deg from where the "
               f"d-spacing puts it, which is a long way and worth checking: the "
               f"line fitted may not be the quartz 101"
               if abs(float(solved.x[0]) - nominal) > 0.15 else "")
        ),
    )


def shape_from_line_width(
    measured: LineWidth,
    reference: float,
    default: PeakShape | None = None,
    wavelength: float = 1.540596,
) -> PeakShape:
    """Stretch a width model so it gives ``measured`` at the reference angle.

    The same one-parameter scaling :func:`fit_peak_shape` uses - the angular
    trend of an instrumental width is generic and its size is what differs
    between machines - but anchored on one reflection whose width is the
    instrument's rather than on the lower envelope of a specimen's own peaks.
    The mixing parameter comes from the doublet fit, which sees the line shape
    directly.
    """
    default = default or PeakShape()
    modelled = float(default.fwhm(np.array([float(reference)]), wavelength)[0])
    if modelled <= 0.0:
        return default
    scale = measured.fwhm / modelled
    return replace(
        default,
        u=default.u * scale**2,
        v=default.v * scale**2,
        w=default.w * scale**2,
        eta=measured.eta,
        size_ab=None if default.size_ab is None else default.size_ab / scale,
    )
