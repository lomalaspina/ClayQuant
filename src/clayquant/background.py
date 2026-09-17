"""Background models.

Every model is linear in its amplitudes once a small number of shape parameters
is fixed, which makes fitting a single linear least-squares problem and makes
the shape parameters natural slider controls in the GUI.  Components are
additive and freely combinable, so the ``1/x`` term - which describes the
low-angle rise from air scatter and the beam knife - can be accumulated on top
of any of the others, as can the exponential term.

Available components
--------------------
``polynomial``
    ``sum_k c_k u^k`` with ``u`` the angle scaled to ``[-1, 1]``.
``chebyshev``
    ``sum_k c_k T_k(u)``, better conditioned than the plain polynomial at high
    degree and the usual choice in Rietveld software.
``exponential``
    ``c * exp(-decay * (two_theta - two_theta_min))``; ``decay`` is the shape
    parameter, in 1/degree.
``inverse``
    ``c / (two_theta + offset)``, the ``1/x`` term; ``offset`` in degrees keeps
    it finite as the angle approaches zero.

Automatic fitting and its bias
------------------------------
:meth:`BackgroundModel.fit` does not fit the raw counts.  It first strips the
peaks with :func:`snip_baseline` and fits the model to that estimate, then makes
a few asymmetric passes that drop the points where the model has risen above the
data.  A clay pattern has no peak-free points below about 8 deg, so fitting the
raw intensities there drags the background up into the basal reflections - the
single most damaging error in this workflow, since it removes the very signal
being quantified.

Peak stripping errs the other way: under a broad, strong low-angle reflection it
*underestimates* the background, by a third or more in the first few degrees.
That is the safe direction, because it leaves signal in rather than taking it
out, but it means the low-angle background should always be checked by eye and
adjusted, which is what the GUI is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ESTIMATORS",
    "ESTIMATOR_LABELS",
    "SONNEVELD_VISSER_BENDING",
    "SONNEVELD_VISSER_CURVATURE",
    "SONNEVELD_VISSER_GRANULARITY",
    "SONNEVELD_VISSER_ITERATIONS",
    "SONNEVELD_VISSER_REACH",
    "snip_edge_width",
    "snip_iterations",
    "sonneveld_visser_granularity",
    "sonneveld_visser_reach",
    "BackgroundModel",
    "BackgroundFit",
    "NoiseLevel",
    "StrippedBackground",
    "baseline_estimate",
    "noise_level",
    "snip_baseline",
    "sonneveld_visser_baseline",
    "select_background_points",
]


@dataclass
class BackgroundModel:
    """A combination of additive background components.

    Attributes
    ----------
    polynomial_degree, chebyshev_degree:
        Degree of the respective polynomial component, or ``None`` to omit it.
        Degree ``n`` contributes ``n + 1`` terms.
    exponential_decay:
        Decay constant in 1/degree of the exponential component, or ``None``.
    inverse:
        Whether to include the ``1/x`` term.
    inverse_offset:
        Offset in degrees added to the angle in the ``1/x`` term.
    clip_negative:
        Clip the evaluated background at zero, so that a fitted background
        cannot go below the physically possible count level.
    """

    polynomial_degree: int | None = 3
    chebyshev_degree: int | None = None
    exponential_decay: float | None = None
    inverse: bool = False
    inverse_offset: float = 1.0
    clip_negative: bool = True

    def __post_init__(self) -> None:
        for name in ("polynomial_degree", "chebyshev_degree"):
            degree = getattr(self, name)
            if degree is not None and degree < 0:
                raise ValueError(f"{name} must be >= 0 or None")
        if self.exponential_decay is not None and self.exponential_decay <= 0:
            raise ValueError("exponential_decay must be positive")
        if not self.components:
            raise ValueError("a background model needs at least one component")

    @property
    def components(self) -> list[str]:
        names = []
        if self.polynomial_degree is not None:
            names.append("polynomial")
        if self.chebyshev_degree is not None:
            names.append("chebyshev")
        if self.exponential_decay is not None:
            names.append("exponential")
        if self.inverse:
            names.append("inverse")
        return names

    @property
    def n_terms(self) -> int:
        total = 0
        if self.polynomial_degree is not None:
            total += self.polynomial_degree + 1
        if self.chebyshev_degree is not None:
            total += self.chebyshev_degree + 1
        if self.exponential_decay is not None:
            total += 1
        if self.inverse:
            total += 1
        return total

    def term_labels(self) -> list[str]:
        labels: list[str] = []
        if self.polynomial_degree is not None:
            labels += [f"poly u^{k}" for k in range(self.polynomial_degree + 1)]
        if self.chebyshev_degree is not None:
            labels += [f"cheb T{k}" for k in range(self.chebyshev_degree + 1)]
        if self.exponential_decay is not None:
            labels.append(f"exp(-{self.exponential_decay:g} dx)")
        if self.inverse:
            labels.append(f"1/(x+{self.inverse_offset:g})")
        return labels

    def basis(self, two_theta: np.ndarray, limits: tuple[float, float] | None = None) -> np.ndarray:
        """Design matrix of shape ``(n_terms, len(two_theta))``.

        ``limits`` fixes the angular range used to scale the polynomial
        variable, so that a background fitted on a subset of points can be
        evaluated on the full pattern.
        """
        two_theta = np.asarray(two_theta, dtype=float)
        low, high = limits if limits is not None else (two_theta.min(), two_theta.max())
        span = high - low
        if span <= 0:
            raise ValueError("the angular range must be positive")
        u = 2.0 * (two_theta - low) / span - 1.0

        rows: list[np.ndarray] = []
        if self.polynomial_degree is not None:
            rows.extend(u**k for k in range(self.polynomial_degree + 1))
        if self.chebyshev_degree is not None:
            for k in range(self.chebyshev_degree + 1):
                coefficients = np.zeros(k + 1)
                coefficients[k] = 1.0
                rows.append(np.polynomial.chebyshev.chebval(u, coefficients))
        if self.exponential_decay is not None:
            rows.append(np.exp(-self.exponential_decay * (two_theta - low)))
        if self.inverse:
            denominator = two_theta + self.inverse_offset
            with np.errstate(divide="ignore", invalid="ignore"):
                rows.append(np.where(denominator > 0, 1.0 / denominator, 0.0))
        return np.vstack(rows)

    def fit(
        self,
        two_theta: np.ndarray,
        intensity: np.ndarray,
        points: np.ndarray | None = None,
        strip_peaks: bool = True,
        snip_window: float = 4.0,
        refinements: int = 3,
        estimator: str = "snip",
    ) -> "BackgroundFit":
        """Fit the model to the background of a measured pattern.

        Parameters
        ----------
        points:
            Boolean mask or integer indices of the points to use.  When given,
            only those points are fitted, which is how manually picked anchor
            points are applied.
        strip_peaks:
            Fit the model to a :func:`snip_baseline` estimate of the background
            rather than to the raw intensities.  This is what makes the fit
            usable at low angle, where a clay pattern has no peak-free points
            for a quantile method to find and fitting the raw data drags the
            background up into the basal reflections.
        snip_window:
            Width in degrees of the widest peak the stripping should remove.
        refinements:
            Passes of asymmetric refinement, each excluding the points where the
            current background lies above the data.  This keeps the fitted
            background underneath the measurement.
        estimator:
            Which estimate of the background to fit: ``"snip"`` for
            :func:`snip_baseline`, or ``"sonneveld-visser"`` for
            :func:`sonneveld_visser_baseline`.  ``snip_window`` means the same
            thing to both.
        """
        two_theta = np.asarray(two_theta, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        if two_theta.shape != intensity.shape:
            raise ValueError("two_theta and intensity must have the same shape")

        if points is None:
            mask = np.ones(two_theta.shape, dtype=bool)
        else:
            mask = np.zeros(two_theta.shape, dtype=bool)
            mask[points] = True
        if mask.sum() < self.n_terms:
            raise ValueError(
                f"{mask.sum()} background points for {self.n_terms} free terms; "
                f"reduce the degree or select more points"
            )

        if strip_peaks:
            target = baseline_estimate(two_theta, intensity, window=snip_window,
                                       estimator=estimator)
        else:
            target = intensity

        limits = (float(two_theta.min()), float(two_theta.max()))
        coefficients = self._solve(two_theta[mask], target[mask], limits)

        for _ in range(max(0, refinements)):
            model_values = coefficients @ self.basis(two_theta, limits=limits)
            below = mask & (model_values <= intensity)
            if below.sum() < max(self.n_terms + 1, 8):
                break
            refined = self._solve(two_theta[below], target[below], limits)
            if np.allclose(refined, coefficients, rtol=1e-6, atol=1e-9):
                coefficients = refined
                mask = below
                break
            coefficients, mask = refined, below

        return BackgroundFit(
            model=self,
            coefficients=coefficients,
            limits=limits,
            points=mask,
            target=target,
        )

    def _solve(self, two_theta: np.ndarray, target: np.ndarray, limits) -> np.ndarray:
        """Least squares with the basis rows scaled to unit norm.

        The ``1/x`` and exponential components are nearly collinear with the
        low-order polynomial terms over a typical scan, so without scaling the
        solver returns large coefficients of opposite sign that cancel, and the
        result extrapolates wildly outside the fitted range.
        """
        design = self.basis(two_theta, limits=limits)
        norms = np.linalg.norm(design, axis=1)
        norms[norms == 0.0] = 1.0
        scaled, *_ = np.linalg.lstsq((design / norms[:, None]).T, target, rcond=None)
        return scaled / norms


@dataclass
class BackgroundFit:
    """A fitted background."""

    model: BackgroundModel
    coefficients: np.ndarray
    limits: tuple[float, float]
    points: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    target: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    """The background estimate that was fitted, i.e. the peak-stripped data."""

    def __call__(self, two_theta: np.ndarray) -> np.ndarray:
        design = self.model.basis(np.asarray(two_theta, dtype=float), limits=self.limits)
        values = self.coefficients @ design
        if self.model.clip_negative:
            values = np.clip(values, 0.0, None)
        return values

    def subtract(self, two_theta: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Background-subtracted intensity, clipped at zero."""
        return np.clip(np.asarray(intensity, dtype=float) - self(two_theta), 0.0, None)

    @property
    def components(self) -> list[str]:
        """Delegated, so a caller need not know which kind of background it holds."""
        return self.model.components

    @property
    def n_terms(self) -> int:
        return self.model.n_terms

    def r_squared(self, two_theta: np.ndarray, intensity: np.ndarray | None = None) -> float:
        """How well the model describes the background it was fitted to.

        The comparison is against the peak-stripped estimate stored in
        :attr:`target`, not against the raw counts: the model is meant to follow
        the background, so measuring it against data that still contains the
        peaks would reward a model that runs up into them.  Pass ``intensity``
        to compare against something else instead.
        """
        if not self.points.any():
            return float("nan")
        reference = self.target if intensity is None else np.asarray(intensity, dtype=float)
        if reference.shape != self.points.shape:
            return float("nan")
        observed = reference[self.points]
        predicted = self(np.asarray(two_theta, dtype=float)[self.points])
        residual = float(np.sum((observed - predicted) ** 2))
        total = float(np.sum((observed - observed.mean()) ** 2))
        return 1.0 - residual / total if total > 0 else float("nan")


@dataclass
class StrippedBackground:
    """A non-parametric background: the peak-stripped estimate itself.

    The parametric models cannot always follow a measured background.  On an
    oriented clay mount the direct-beam tail below about 6 deg falls steeply and
    is not a low-order polynomial; fitting one leaves oscillating residuals of
    hundreds of counts exactly where the 16.9 A glycol reflection sits, which
    then appear as spurious peaks.  Using the stripped estimate directly follows
    any shape, at the cost of having no parameters to report.

    It carries the same interface as :class:`BackgroundFit`, so it can be used
    anywhere one is expected.
    """

    two_theta: np.ndarray
    baseline: np.ndarray
    window: float = 4.0
    estimator: str = "snip"
    granularity: int | None = None
    bending: float | None = None
    """Sonneveld-Visser's own two parameters, when that is the estimator.

    Set instead of ``window``, which they replace: granularity says how wide a
    feature is removed, in points rather than in degrees
    (:func:`sonneveld_visser_baseline`).
    """

    def __post_init__(self) -> None:
        self.two_theta = np.asarray(self.two_theta, dtype=float)
        self.baseline = np.asarray(self.baseline, dtype=float)
        if self.two_theta.shape != self.baseline.shape:
            raise ValueError("two_theta and baseline must have the same shape")

    @classmethod
    def fit(
        cls,
        two_theta: np.ndarray,
        intensity: np.ndarray,
        window: float = 4.0,
        estimator: str = "snip",
        granularity: int | None = None,
        bending: float | None = None,
    ) -> "StrippedBackground":
        """The estimate itself, as the background.

        Pass ``granularity`` and ``bending`` to drive Sonneveld-Visser by its
        own parameters, as HighScore does, rather than by a width in degrees.
        """
        two_theta = np.asarray(two_theta, dtype=float)
        if granularity is not None or bending is not None:
            estimator = "sonneveld-visser"
            baseline = sonneveld_visser_baseline(
                two_theta,
                intensity,
                granularity=(SONNEVELD_VISSER_GRANULARITY if granularity is None
                             else granularity),
                bending=SONNEVELD_VISSER_BENDING if bending is None else bending,
            )
        else:
            baseline = baseline_estimate(two_theta, intensity, window=window,
                                         estimator=estimator)
        return cls(
            two_theta=two_theta,
            baseline=baseline,
            window=window,
            estimator=estimator,
            granularity=granularity,
            bending=bending,
        )

    @property
    def model(self) -> str:
        if self.estimator == "sonneveld-visser" and self.granularity is not None:
            bending = SONNEVELD_VISSER_BENDING if self.bending is None else self.bending
            return (f"{ESTIMATOR_LABELS[self.estimator]} "
                    f"(granularity {self.granularity:d}, bending {bending:g})")
        return f"{ESTIMATOR_LABELS[self.estimator]} ({self.window:g} deg)"

    @property
    def points(self) -> np.ndarray:
        return np.ones(self.two_theta.shape, dtype=bool)

    @property
    def target(self) -> np.ndarray:
        return self.baseline

    def __call__(self, two_theta: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(two_theta, dtype=float),
            self.two_theta,
            self.baseline,
            left=self.baseline[0],
            right=self.baseline[-1],
        )

    def subtract(self, two_theta: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(intensity, dtype=float) - self(two_theta), 0.0, None)

    def r_squared(self, two_theta: np.ndarray, intensity: np.ndarray | None = None) -> float:
        """Unity by construction: the background *is* the estimate it follows."""
        return 1.0

    @property
    def components(self) -> list[str]:
        return [ESTIMATOR_LABELS[self.estimator]]

    @property
    def n_terms(self) -> int:
        """Nothing is fitted, so there are no free terms to report."""
        return 0


def snip_iterations(two_theta: np.ndarray, window: float) -> int:
    """Number of stripping passes for ``window`` on this grid."""
    two_theta = np.asarray(two_theta, dtype=float)
    if len(two_theta) < 5:
        return 1
    step = float(np.mean(np.diff(two_theta)))
    if step <= 0:
        raise ValueError("two_theta must be increasing")
    return min(max(1, int(round(window / (2.0 * step)))), (len(two_theta) - 1) // 2)


def snip_edge_width(two_theta: np.ndarray, window: float) -> int:
    """Points at each end where the stripping had to narrow its comparison.

    The estimate there is not a fully stripped background - see
    :func:`snip_baseline` - so a model fitted to it should not be fitted there.
    """
    return snip_iterations(two_theta, window)


def snip_baseline(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    window: float = 4.0,
    iterations: int | None = None,
) -> np.ndarray:
    """Estimate the background by iterative peak stripping (SNIP).

    At each pass every point is replaced by the smaller of itself and the mean
    of the two points a distance ``p`` away, for shrinking ``p``; anything
    narrower than ``window`` degrees is clipped away and the smooth background
    survives.  The method is that of Ryan, Clayton, Griffin, Sie & Cousens
    (1988) Nucl. Instrum. Methods B34, 396-402, and is the standard background
    estimator for X-ray spectra and diffractograms.

    What makes it right for a clay mount is a property worth stating, because
    getting it wrong destroys the measurement rather than the peaks: where the
    background is convex, the mean of two symmetric neighbours is never below
    the point itself, so the operation leaves it *exactly* unchanged.  The
    low-angle tail of an oriented mount - air scatter and the shoulder of the
    direct beam - is such a background, and a correct implementation returns it
    untouched.  Tested against a known tail of 1820 counts falling to 339, this
    returns it to within a count with no peaks present, and to within 30 counts
    with ten peaks up to 7000 counts high standing on it.

    Two details decide whether that holds.  The comparison has to stay
    symmetric, so near the ends of the scan the reach is narrowed to what the
    data allows rather than padded, since any invented neighbour either lowers a
    falling edge or inverts a peak that sits near it.  And the stripping is done
    on the logarithm: a tail of the form ``a/x^n`` is log-convex, so log space
    preserves it exactly while still removing peaks a little more sharply than
    linear space does.

    The consequence of narrowing rather than inventing is that within one
    window of either end the estimate is not a fully stripped background, and at
    the very first and last point it is the measurement itself: with no data on
    one side there is no evidence that a point is a peak rather than background.
    On a clay mount the scan opens on the direct-beam tail, where keeping the
    measurement is the right answer and is why the fitted background now follows
    that tail instead of cutting under it.  Where a pattern is stripped over a
    sub-range that opens on a reflection, the estimate is too high within one
    window of that end; :func:`snip_edge_width` gives the width of the zone.
    The quantitative fit starts at 4 deg for this reason among others, so on a
    scan that begins near 3 deg most of that zone lies outside it.

    Parameters
    ----------
    window:
        Width in degrees of the widest feature to strip.
    iterations:
        Number of passes; by default derived from ``window`` and the step size.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    values = np.asarray(intensity, dtype=float).copy()
    if len(values) < 5:
        return values
    step = float(np.mean(np.diff(two_theta)))
    if step <= 0:
        raise ValueError("two_theta must be increasing")
    if iterations is None:
        iterations = snip_iterations(two_theta, window)
    iterations = min(iterations, (len(values) - 1) // 2)

    floor = max(float(np.min(values[values > 0.0])) if np.any(values > 0.0) else 1.0, 1e-6)
    transformed = np.log(np.clip(values, floor, None))

    index = np.arange(len(transformed))
    to_edge = np.minimum(index, len(transformed) - 1 - index)
    for p in range(iterations, 0, -1):
        reach = np.minimum(p, to_edge)
        neighbours = 0.5 * (transformed[index - reach] + transformed[index + reach])
        transformed = np.minimum(transformed, neighbours)

    restored = np.exp(transformed)
    # A background is never above the measurement it came from.
    return np.minimum(np.clip(restored, 0.0, None), values)


def select_background_points(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    n_windows: int = 40,
    quantile: float = 0.05,
    iterations: int = 3,
    tolerance: float = 1.5,
) -> np.ndarray:
    """Choose points that lie on the background.

    A low quantile of each of ``n_windows`` equal angular windows gives a first
    estimate of the lower envelope; the estimate is then refined by discarding
    points that sit more than ``tolerance`` times the local noise above a
    smooth fit through them.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if len(two_theta) < 4:
        return np.ones(two_theta.shape, dtype=bool)

    edges = np.linspace(two_theta[0], two_theta[-1], n_windows + 1)
    selected: list[int] = []
    for low, high in zip(edges[:-1], edges[1:]):
        window = np.flatnonzero((two_theta >= low) & (two_theta <= high))
        if window.size == 0:
            continue
        threshold = np.quantile(intensity[window], quantile)
        chosen = window[intensity[window] <= threshold]
        selected.extend((chosen if chosen.size else window[[np.argmin(intensity[window])]]).tolist())

    mask = np.zeros(two_theta.shape, dtype=bool)
    mask[np.unique(selected)] = True

    for _ in range(max(0, iterations)):
        if mask.sum() < 6:
            break
        coefficients = np.polyfit(two_theta[mask], intensity[mask], deg=3)
        trend = np.polyval(coefficients, two_theta)
        residual = intensity[mask] - trend[mask]
        spread = float(np.std(residual)) or 1.0
        refined = mask & (intensity - trend <= tolerance * spread)
        if refined.sum() < 6:
            break
        mask = refined
    return mask


# --------------------------------------------------------------------------
# Sonneveld & Visser (1975)
# --------------------------------------------------------------------------

SONNEVELD_VISSER_GRANULARITY = 20
"""Default granularity: points between the samples the erosion runs on.

Sonneveld & Visser used every twentieth point of their film scan - their 5 %
of the data - and this is that number.  It is also the parameter HighScore
exposes under this name, where the number of intervals is recommended between
15 and 30 and the literature value is 20, so a granularity set here means the
same thing as a granularity set there.

It is given in points rather than in degrees because that is what both the
paper and HighScore mean by it.  The consequence is worth knowing: the width of
feature the method removes is set by ``granularity * step``, so the same
granularity on a finer scan reaches less far in degrees.
"""

SONNEVELD_VISSER_BENDING = 1.0
"""Default bending factor: how much downward curvature the background may keep.

HighScore's slider of the same name, where 0 to 4 is the usual span and 0 to 2
the usual advice.  Here 1 is anchored on the paper's own value - ``c`` = 0.02 on
the eight-bit scale of their microdensitometer, so 7.8e-5 of the intensity
range - which makes 0 the strictly-linear first form of their algorithm and 4
four times their allowance.

That anchoring is a choice, not a conversion: HighScore's slider position is not
documented as a multiple of anything, so the same number here and there need not
give the same curve.  It matters less than it sounds, because the parameter is
weak: across 39 real patterns its whole span moves the baseline by at most
0.70 % of the intensity range against granularity's 30 %.  The measurement is
reported under :func:`sonneveld_visser_baseline`.
"""

SONNEVELD_VISSER_CURVATURE = 0.02 / 255.0
"""The paper's own curvature allowance, as a fraction of the intensity range.

``c ~ 0.02`` "on the intensity scale from 0 to 255", the eight bits of their
microdensitometer, so 7.8e-5 of full scale.  This is what a bending factor of 1
means; passing it as a fraction rather than in counts is what keeps the
parameter meaningful on a pattern of 10^5 counts, where the literal 0.02 would
be indistinguishable from zero.
"""

SONNEVELD_VISSER_REACH = 2.2
"""Coefficient in the reach law ``FWHM = REACH * spacing * sqrt(passes)``.

Measured, not derived: half of a Gaussian's height is removed at this width,
and the coefficient holds to 2 % over 5 to 120 passes and spacings from 0.1 to
0.4 deg (:func:`sonneveld_visser_reach`).
"""

SONNEVELD_VISSER_ITERATIONS = 30
"""Erosion passes.  The paper's "about 30 times", and not exposed in the
interface for the same reason HighScore does not expose it: with the passes
fixed, granularity is the single control of how wide a feature is removed.
"""

def sonneveld_visser_reach(granularity: int, step: float, iterations: int) -> float:
    """Width in degrees of the feature the erosion half-removes.

    The replacement ``p_i <- (p_(i+1) + p_(i-1))/2`` is one explicit step of the
    diffusion equation, so the passes do not march outwards one sample at a
    time: they spread as the square root of their number.  Measured on Gaussian
    peaks,

        FWHM = 2.2 * granularity * step * sqrt(passes)

    which holds to 2 % from 5 to 120 passes and over a fourfold range of
    spacing.  This is what granularity does, and it is the parameter that
    decides the baseline.

    Two readings worth having.  The paper's own settings - granularity 20 on a
    0.01 deg film scan, 30 passes - reach about 2.4 deg, not the 12 deg a linear
    reading of the iteration suggests: well matched to the sharp lines of a
    Guinier camera and deliberately short of the broad features of a clay mount.
    And on a 0.0167 deg clay scan the same granularity reaches 4 deg, because the
    granularity is counted in points and the points are wider apart.

    Unlike the ``window`` of :func:`snip_baseline` this is a soft cutoff: a
    feature of exactly this width keeps half its height, one of half the width
    keeps a few per cent.
    """
    return SONNEVELD_VISSER_REACH * granularity * step * math.sqrt(max(1, iterations))


def sonneveld_visser_granularity(window: float, step: float,
                                 iterations: int = SONNEVELD_VISSER_ITERATIONS) -> int:
    """Granularity that half-removes a feature ``window`` degrees wide.

    The inverse of :func:`sonneveld_visser_reach`, for asking this estimator and
    :func:`snip_baseline` for the same thing.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    spacing = window / (SONNEVELD_VISSER_REACH * math.sqrt(max(1, iterations)))
    return max(1, int(round(spacing / step)))


def sonneveld_visser_baseline(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    granularity: int = SONNEVELD_VISSER_GRANULARITY,
    bending: float = SONNEVELD_VISSER_BENDING,
    iterations: int = SONNEVELD_VISSER_ITERATIONS,
    window: float | None = None,
    sequential: bool = True,
) -> np.ndarray:
    """Estimate the background by the method of Sonneveld & Visser (1975).

    The method (their Sec. 3.1) is to take every ``granularity``-th point of the
    pattern as a first approximation of the background, then repeatedly replace
    each sample by the mean of its two neighbours wherever it stands more than
    ``c`` above that mean::

        m_i = (p_(i+1) + p_(i-1)) / 2
        if p_i > m_i + c:  p_i <- m_i

    and finally interpolate the eroded samples back onto the measured grid.
    Peaks, being local maxima, are pulled down pass by pass; a background is
    not.  It is the oldest of the automatic baseline estimators still in use,
    and the one HighScore determines its background with - which is why the two
    parameters carry HighScore's names.

    **Granularity** is the number of points between samples, HighScore's
    "number of intervals", recommended there between 15 and 30.  It is the
    parameter that decides the baseline, because it sets how wide a feature the
    erosion can remove: ``2.2 * granularity * step * sqrt(passes)``, which
    :func:`sonneveld_visser_reach` reports in degrees.

    **Bending factor** is how much *downward* curvature the background may
    keep.  At the fixed point of the rule the second difference of the retained
    background is ``-2c``, so with a sample spacing ``h``

        d2p/dx2 >= -2c/h^2

    which is to say that ``c`` - and so the bending factor - is exactly the
    largest downward curvature allowed.  At 0 only a straight or convex
    background survives, which is the paper's first form and why they had to
    introduce ``c`` at all; 1 is their own value; HighScore's usual advice is 0
    to 2.

    On real patterns the bending factor turns out to matter very little, and
    granularity to decide almost everything.  Measured over 39 clay and standard
    patterns, taking the bending factor across its whole useful span moves the
    baseline by at most 0.70 % of the intensity range - 0.16 % from 0 to 1 -
    while taking granularity from 10 to 40 moves it by 30 %.  The reason is the
    relation above: the bound scales with the range of the data, so a real
    background's curvature is either far above it, in which case the rule fires
    whatever the bending, or far below, where the erosion is negligible anyway.
    Documented here so that nobody spends an afternoon on the bending slider
    while leaving granularity at whatever it happened to be.

    **What survives.** A convex background - the direct-beam tail of an oriented
    mount, which falls as roughly ``a/x^n`` - has ``m_i >= p_i`` everywhere, so
    the rule never fires and the tail is returned untouched, exactly as with
    peak stripping.  What the method acts on is concave features: peaks, and
    also the broad hump of a poorly crystalline or interstratified phase, which
    is why granularity matters more here than bending.

    Parameters
    ----------
    granularity:
        Points between samples.  HighScore's parameter of the same name.
    bending:
        Curvature allowance, 1 being the value the paper used.
    iterations:
        Erosion passes; the paper's about 30, and left alone by the interface so
        that granularity is the single control of reach.
    window:
        Width in degrees of the widest feature to remove, converted to a
        granularity by :func:`sonneveld_visser_granularity` so that this
        estimator and :func:`snip_baseline` can be asked for the same thing.
        Overrides ``granularity``.
    sequential:
        Erode in place along the samples, as the paper's loop does, so that a
        sample already lowered in this pass is what its right-hand neighbour
        sees.  With ``False`` every ``m_i`` is formed from the previous pass,
        which makes a pass symmetric and independent of the direction of travel.
        The two differ while the erosion is still relaxing - up to 2.8 % of the
        intensity range after one pass - and agree exactly once it reaches its
        fixed point, which on a pattern with peaks takes about 21 passes.  At
        the paper's 30 and above the choice therefore does not matter, which is
        why the paper's own order is the default.

    Returns
    -------
    The background on the ``two_theta`` grid, never above the measurement.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    values = np.asarray(intensity, dtype=float)
    if two_theta.shape != values.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    if len(values) < 5:
        return values.copy()
    step = float(np.mean(np.diff(two_theta)))
    if step <= 0:
        raise ValueError("two_theta must be increasing")
    if bending < 0:
        raise ValueError("bending must not be negative")
    if window is not None:
        granularity = sonneveld_visser_granularity(window, step, iterations)
    granularity = int(granularity)
    if granularity < 1:
        raise ValueError("granularity must be at least 1 point")

    # The last point is sampled as well as the first: the samples are the only
    # evidence the interpolation has, and without the right-hand end it would
    # extrapolate the last interval across whatever remains of the scan.
    index = np.unique(np.append(np.arange(0, len(values), granularity), len(values) - 1))
    samples = values[index].copy()
    if len(samples) < 3:
        return values.copy()

    span = float(samples.max() - samples.min())
    c = bending * SONNEVELD_VISSER_CURVATURE * span

    for _ in range(max(0, iterations)):
        if sequential:
            previous = samples.copy()
            for i in range(1, len(samples) - 1):
                mean = 0.5 * (samples[i + 1] + samples[i - 1])
                if samples[i] > mean + c:
                    samples[i] = mean
            if np.allclose(samples, previous, rtol=0.0, atol=1e-12):
                break
        else:
            mean = 0.5 * (samples[2:] + samples[:-2])
            replace = samples[1:-1] > mean + c
            if not replace.any():
                break
            samples[1:-1] = np.where(replace, mean, samples[1:-1])

    baseline = np.interp(two_theta, two_theta[index], samples)
    # A background is never above the measurement it came from.  The samples
    # themselves satisfy this, but a straight line between two of them can cross
    # above a point that dips between them.
    return np.minimum(np.clip(baseline, 0.0, None), values)


@dataclass
class NoiseLevel:
    """The noise of a background-subtracted pattern, and what it took to get it.

    Attributes
    ----------
    sigma, mean:
        ``sigma_noise`` and ``mu_noise`` of Sonneveld & Visser's Sec. 3.2.
    threshold:
        ``mean + sigmas * sigma``, the level at which the signal is taken to
        differ significantly from the background.
    rejected:
        Points excluded as peak rather than noise.
    iterations:
        Clipping passes used.
    converged:
        Whether the passes ended because sigma stopped moving rather than
        because the cap was reached.
    """

    sigma: float
    mean: float
    threshold: float
    rejected: int
    iterations: int
    converged: bool

    @property
    def range(self) -> float:
        """The noise range, six sigma - the paper's own assumption."""
        return 6.0 * self.sigma


def noise_level(
    difference: np.ndarray,
    sigmas: float = 3.0,
    sample: int | None = None,
    tolerance: float = 0.01,
    max_iterations: int = 50,
) -> NoiseLevel:
    """Noise level of a difference signal, after Sonneveld & Visser (1975).

    Their Sec. 3.2: take the background-subtracted data, compute its mean and
    standard deviation, reject everything above ``mu + 3 sigma``, and repeat
    until sigma settles.  What is left is noise, and its spread is the level
    against which a peak has to be judged.

    The clipping is deliberately **one-sided**.  The contaminating population is
    the peaks, which lie above the background and nowhere below it, so rejecting
    symmetrically would throw away the lower half of the noise it is trying to
    measure and return a sigma too small by about a third.  The paper's step (4)
    rejects ``i > mu + 3 sigma`` and nothing else, and so does this.

    Parameters
    ----------
    sample:
        Use this many evenly spaced points rather than all of them.  The paper
        takes "N data ... N large enough, ~500", a concession to a 1975
        computer; all the data is strictly better and is the default.
    tolerance:
        Stop when sigma changes by less than this fraction between passes -
        the paper's "if the shift is large enough go to (3)".
    """
    values = np.asarray(difference, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("need at least two finite points to estimate the noise")
    if sample is not None:
        if sample < 2:
            raise ValueError("sample must be at least 2")
        if sample < values.size:
            values = values[np.linspace(0, values.size - 1, sample).round().astype(int)]

    total = values.size
    kept = values
    sigma = float(np.std(kept))
    mean = float(np.mean(kept))
    used = 0
    converged = False
    for used in range(1, max_iterations + 1):
        if sigma <= 0.0:
            converged = True
            break
        retained = kept[kept <= mean + sigmas * sigma]
        # Every point above the threshold: the distribution is all peak, and
        # clipping further would leave nothing to measure.
        if retained.size < 2:
            converged = True
            break
        new_sigma = float(np.std(retained))
        new_mean = float(np.mean(retained))
        shift = abs(new_sigma - sigma) / sigma if sigma > 0 else 0.0
        kept, sigma, mean = retained, new_sigma, new_mean
        if shift < tolerance:
            converged = True
            break

    return NoiseLevel(
        sigma=sigma,
        mean=mean,
        threshold=mean + sigmas * sigma,
        rejected=total - kept.size,
        iterations=used,
        converged=converged,
    )


ESTIMATORS = ("snip", "sonneveld-visser")
"""The non-parametric background estimators, by name."""

ESTIMATOR_LABELS = {
    "snip": "peak-stripped",
    "sonneveld-visser": "Sonneveld-Visser",
}


def baseline_estimate(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    window: float = 4.0,
    estimator: str = "snip",
) -> np.ndarray:
    """Estimate the background with the named estimator.

    The two are asked for the same thing - remove everything narrower than
    ``window`` degrees, keep the rest - so they can be exchanged, and the
    difference between their answers is a measure of how much of the background
    is a matter of method rather than of measurement (Sec. A.19 of the manual).
    """
    if estimator not in ESTIMATORS:
        raise ValueError(
            f"unknown background estimator {estimator!r}; expected one of {', '.join(ESTIMATORS)}"
        )
    if estimator == "snip":
        return snip_baseline(two_theta, intensity, window=window)
    return sonneveld_visser_baseline(two_theta, intensity, window=window)
