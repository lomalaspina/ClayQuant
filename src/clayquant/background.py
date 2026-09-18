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

import hashlib
import math
import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CLAYFIT_ALS_ASYMMETRY",
    "CLAYFIT_ALS_ITERATIONS",
    "CLAYFIT_ALS_SMOOTHNESS",
    "CLAYFIT_ANCHOR_RADIUS",
    "CLAYFIT_ANCHOR_STRIDE",
    "CLAYFIT_INVERSE_X_AMPLITUDE",
    "CLAYFIT_MODELS",
    "CLAYFIT_MODEL_LABELS",
    "CLAYFIT_ORDER",
    "CLAYFIT_ORDER_LIMITS",
    "QPA_BASELINE_ORDER",
    "QPA_BASELINE_SMOOTH",
    "QPA_FINAL_ORDER",
    "QPA_FINAL_SMOOTH",
    "QPA_LOWER_PERCENTILE",
    "QPA_MODEL_NAME",
    "QPA_PERCENTILE_WINDOW",
    "ClayfitBackground",
    "QpaBaseline",
    "als_baseline_2d",
    "background_anchors",
    "chebyshev_baseline",
    "clayfit_background",
    "exponential_baseline",
    "inverse_x_baseline",
    "polynomial_baseline",
    "qpa_percentile_baseline",
    "rolling_ball",
    "smooth_by_degrees",
    "ESTIMATORS",
    "ESTIMATOR_LABELS",
    "SONNEVELD_VISSER_BENDING",
    "SONNEVELD_VISSER_CURVATURE",
    "SONNEVELD_VISSER_GRANULARITY",
    "SONNEVELD_VISSER_ITERATIONS",
    "SONNEVELD_VISSER_CEILING",
    "SONNEVELD_VISSER_FLOOR",
    "SONNEVELD_VISSER_REACH",
    "SONNEVELD_VISSER_SAFETY",
    "snip_edge_width",
    "snip_iterations",
    "sonneveld_visser_granularity",
    "sonneveld_visser_reach",
    "suggest_sonneveld_visser",
    "BackgroundModel",
    "BackgroundFit",
    "NoiseLevel",
    "PeakGroup",
    "SonneveldVisserSuggestion",
    "StrippedBackground",
    "baseline_estimate",
    "noise_level",
    "peak_groups",
    "als_baseline",
    "percentile_baseline",
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
give the same curve.  HighScore's slider runs 0 to 100, and so does this one:
the bottom of that range does almost nothing - 0 to 4 moves a baseline by well
under 1 % of the intensity range - and it takes the upper part for the parameter
to bite at all.  The measurements are reported under
:func:`sonneveld_visser_baseline`.
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

    The bending factor is weak over the bottom of its range and only becomes
    useful towards the top, which is worth stating because the bottom is where
    a reader assumes the interesting values are.  Measured on real patterns, it
    moves the baseline by 0.08 to 0.16 % of the intensity range going from 0 to
    1, about 0.4 % by 4, and 2.3 to 12.4 % by 100.  Granularity from 10 to 40
    moves it by 30 %, so granularity is still the parameter to set first - but
    "inert", which an earlier version of this note said on the strength of a
    0-to-4 span, was wrong.

    Which value to prefer is a separate question and the fit answers it
    consistently: over eight glycol mounts, R_wp is lowest at bending 0 on every
    one, and rises monotonically to 4.6 to 11.9 points worse at 100.  The reason is
    the relation above - a real clay background's curvature far exceeds the
    bound, so raising the allowance cannot help it follow the background, and
    all it does is let the baseline sit higher and take more of the broad basal
    intensity out as background.

    **What survives.** A convex background - the direct-beam tail of an oriented
    mount, which falls as roughly ``a/x^n`` - has ``m_i >= p_i`` everywhere, so
    the rule never fires and the tail is returned untouched, exactly as with
    peak stripping.  What the method acts on is concave features: peaks, and
    also the broad hump of a poorly crystalline or interstratified phase, which
    is why granularity matters more here than bending.

    Parameters
    ----------
    granularity:
        Points between samples.  HighScore's parameter of the same name, whose
        slider runs 0 to 50; 0 and 1 both mean every point.
    bending:
        Curvature allowance, 1 being the value the paper used.  HighScore's
        slider runs 0 to 100, and the span matters: over 0 to 4 the baseline
        barely moves, and it takes the upper part of the range for the parameter
        to do anything at all.
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
    # HighScore's slider starts at 0 and 0 can only mean every point, which is
    # what 1 means here; accepting both keeps a setting transferable.
    granularity = max(1, int(granularity))

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

    # Seeded from the median and the median absolute deviation rather than from
    # the mean and the standard deviation, which is the one departure from the
    # paper's Sec. 3.2 and is there to make its own loop work.
    #
    # Their step (2) takes sigma over everything, peaks included, and step (4)
    # rejects above mu + 3 sigma.  On a film scan of sharp lines that is a small
    # contamination and the loop tightens from pass to pass.  On a pattern
    # carrying a broad feature - an amorphous hump, a smectite band, a
    # poorly crystalline phase - the feature is a fifth of the points, sigma
    # comes out inflated by two orders of magnitude, mu + 3 sigma lands above
    # every point in the pattern, and the first pass rejects nothing.  Sigma
    # then does not move, which their stopping rule reads as convergence, and
    # the returned noise level is the spread of the feature rather than of the
    # noise.  Measured on an 8 deg Gaussian: sigma 2924 against a true 22, and a
    # threshold above the feature's own height, so it was reported as noise.
    #
    # The median and the MAD are not moved by a fifth of the points, so the
    # first threshold engages and the paper's loop proceeds as intended.  It is
    # left a little conservative rather than made exact: the flanks of a broad
    # feature stay within three sigma of the median over a long stretch and
    # survive the clipping, so sigma comes back about a third high in that case.
    # A threshold slightly too high is slightly less sensitive, which is the
    # safe direction for deciding what counts as a reflection.  It is
    # left a little conservative rather than made exact: the flanks of a broad
    # feature stay within three sigma of the median over a long stretch and
    # survive the clipping, so sigma comes back about a third high in that case.
    # A threshold slightly too high is slightly less sensitive, which is the safe
    # direction for deciding what counts as a reflection.
    mean = float(np.median(kept))
    deviation = float(np.median(np.abs(kept - mean)))
    # 1.4826 makes the MAD an estimate of sigma for normally distributed noise.
    sigma = 1.4826 * deviation
    if sigma <= 0.0:
        sigma = float(np.std(kept))
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


def als_baseline(
    intensity: np.ndarray,
    smoothness: float = 1e6,
    asymmetry: float = 0.01,
    iterations: int = 20,
) -> np.ndarray:
    """Asymmetrically reweighted least squares baseline, after Eilers & Boelens.

    Minimises ``sum w_i (y_i - z_i)^2 + smoothness * sum (second difference of
    z)^2`` with ``w_i`` small where the data lies above the current baseline and
    one where it lies below, which is what makes it ignore peaks: a peak pulls
    on the baseline with weight ``asymmetry`` and a gap pulls with weight one.

    It answers a different question from peak stripping and that is the point of
    having it.  Stripping asks what is left when everything narrower than a
    window is removed, which on a clay pattern is a smooth curve - so smooth
    that a fourth-degree polynomial already describes it and higher degrees have
    almost nothing to fit, which is not a statement about the specimen but about
    the estimate.  This asks instead for the smoothest curve that stays under
    the data, and ``smoothness`` sets how smooth: lower follows the measurement
    more closely, higher approaches a straight line.

    Eilers, P.H.C. and Boelens, H.F.M. (2005) Baseline correction with
    asymmetric least squares smoothing, Leiden University Medical Centre report.
    """
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import spsolve

    y = np.asarray(intensity, dtype=float)
    size = y.size
    if size < 5:
        return y.copy()
    if not 0.0 < asymmetry < 1.0:
        raise ValueError("asymmetry must lie in (0, 1)")
    if smoothness <= 0.0:
        raise ValueError("smoothness must be positive")
    second = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(size - 2, size), format="csc")
    penalty = smoothness * (second.T @ second)
    weights = np.ones(size)
    baseline = y.copy()
    for _ in range(max(1, iterations)):
        system = diags(weights, 0, format="csc") + penalty
        baseline = spsolve(system, weights * y)
        updated = np.where(y > baseline, asymmetry, 1.0 - asymmetry)
        if np.allclose(updated, weights):
            break
        weights = updated
    return np.asarray(baseline, dtype=float)


def percentile_baseline(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    window: float = 4.0,
    percentile: float = 12.0,
    smooth: float = 2.0,
) -> np.ndarray:
    """A rolling low percentile of the pattern, then smoothed.

    The simplest of the estimators here and the one that follows a structured
    background most closely: in a window of ``window`` degrees the background is
    taken as the ``percentile``-th percentile of the counts, which a peak
    occupying less than that fraction of the window cannot raise, and the result
    is smoothed over ``smooth`` degrees to remove the steps the rolling window
    leaves behind.

    A percentile too high eats into the peaks and one too low follows the noise
    down; a twelfth is about where a clay pattern's peaks stop mattering, and it
    is a parameter rather than a constant because how much of a pattern is peak
    varies with the specimen.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if y.size < 5:
        return y.copy()
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must lie in (0, 100)")
    step = float(np.median(np.diff(two_theta))) if two_theta.size > 1 else 0.02
    half = max(1, int(round(0.5 * window / step)))
    padded = np.pad(y, half, mode="edge")
    rolling = np.array([
        np.percentile(padded[index:index + 2 * half + 1], percentile)
        for index in range(y.size)
    ])
    if smooth > 0.0:
        span = max(1, int(round(smooth / step)))
        kernel = np.ones(span) / span
        rolling = np.convolve(np.pad(rolling, span, mode="edge"), kernel, mode="same")
        rolling = rolling[span:span + y.size]
    return rolling


ESTIMATORS = ("snip", "sonneveld-visser", "als", "percentile")
"""The non-parametric background estimators, by name."""

ESTIMATOR_LABELS = {
    "snip": "peak-stripped",
    "sonneveld-visser": "Sonneveld-Visser",
    "als": "asymmetric least squares",
    "percentile": "rolling percentile",
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
    if estimator == "als":
        return als_baseline(intensity)
    if estimator == "percentile":
        return percentile_baseline(two_theta, intensity, window=window)
    return sonneveld_visser_baseline(two_theta, intensity, window=window)


# --------------------------------------------------------------------------
# Choosing the two parameters from the measurement
# --------------------------------------------------------------------------

SONNEVELD_VISSER_SAFETY = 2.0
"""How far the reach must exceed the widest reflection to be kept.

At a reach equal to a feature's width half its height stays in the baseline,
which is half the signal lost; at twice the width a few per cent stays.  The
factor follows from the reach law rather than from taste
(:func:`sonneveld_visser_reach`).
"""

SONNEVELD_VISSER_CEILING = 0.25
"""Largest reach the pre-screen will suggest, as a fraction of the fitted range.

Past this the background is being defined by a handful of samples spread across
the whole pattern, and a feature that wide cannot be told from background by any
local criterion anyway - a 12 deg feature on a 37 deg scan is a third of the
pattern.  Where the cap binds, the suggestion says so, because that is the case
in which the operator has to decide what is background rather than be told.
"""

SONNEVELD_VISSER_FLOOR = 4.0
"""Smallest reach in degrees the pre-screen will suggest.

A scan in which no broad reflection rises above the noise is not evidence that
none is present, and the interstratified phases whose intensity is broad by
nature are exactly the ones a weak pattern hides.  Four degrees is the width
eleven mounts supported for the peak stripping (Sec. A.11), and erring towards
it is the safe direction: it leaves background in the signal rather than taking
signal out.
"""


@dataclass
class PeakGroup:
    """A stretch of a difference signal that rises above the noise.

    Sonneveld & Visser's Sec. 3.3: "a peak group is considered to begin at a
    point where the signal surpasses this level and ends when the signal becomes
    lower".  Within one there is at least one reflection.
    """

    start: int
    stop: int
    """Half-open index range into the pattern."""

    centre: float
    """Angle of the group's maximum, in degrees."""

    height: float
    width: float
    """Height above the background, and full width at half that height."""

    @property
    def points(self) -> int:
        return self.stop - self.start


def peak_groups(
    two_theta: np.ndarray,
    difference: np.ndarray,
    threshold: float,
    minimum_points: int = 3,
) -> list[PeakGroup]:
    """Stretches of ``difference`` above ``threshold``, with their widths.

    The threshold is the one :func:`noise_level` computes, so what comes back is
    the set of features that differ significantly from the background rather
    than every wiggle.  Groups shorter than ``minimum_points`` are dropped: a
    single point over the line is noise that got through, not a reflection.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    difference = np.asarray(difference, dtype=float)
    step = float(np.mean(np.diff(two_theta))) if len(two_theta) > 1 else 0.0
    above = difference > threshold

    groups: list[PeakGroup] = []
    start: int | None = None
    for index, flag in enumerate(above):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            groups.append((start, index))
            start = None
    if start is not None:
        groups.append((start, len(above)))

    found: list[PeakGroup] = []
    for first, last in groups:
        if last - first < minimum_points:
            continue
        piece = difference[first:last]
        height = float(piece.max())
        # Width at half the group's own height, measured inside the group.  A
        # reflection standing on the shoulder of another is measured against its
        # own maximum, which is what its width means.
        over = np.flatnonzero(piece >= 0.5 * height)
        found.append(
            PeakGroup(
                start=first,
                stop=last,
                centre=float(two_theta[first + int(np.argmax(piece))]),
                height=height,
                width=float((over[-1] - over[0] + 1) * step),
            )
        )
    return found


@dataclass
class SonneveldVisserSuggestion:
    """Starting values for the two parameters, and what they were read from."""

    granularity: int
    bending: float
    reach: float
    """What the suggested granularity reaches, in degrees."""

    widest_reflection: float
    """Full width at half maximum of the broadest feature above the noise."""

    groups: int
    sigma: float
    step: float
    floored: bool
    """Whether the floor decided the answer rather than the measurement."""

    capped: bool = False
    """Whether the ceiling did, which means the specimen needs a judgement."""

    note: str = ""


def suggest_sonneveld_visser(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    range_two_theta: tuple[float, float] | None = (4.0, 39.0),
    safety: float = SONNEVELD_VISSER_SAFETY,
    floor: float = SONNEVELD_VISSER_FLOOR,
    ceiling: float = SONNEVELD_VISSER_CEILING,
    iterations: int = SONNEVELD_VISSER_ITERATIONS,
) -> SonneveldVisserSuggestion:
    """Read starting values for granularity and bending off a measurement.

    The two parameters are settled differently, because only one of them is
    determined by the data.

    **Granularity** follows from what has to survive.  The erosion removes a
    feature from the baseline once its reach exceeds the feature's width, so the
    reach has to clear the broadest reflection that is meant to stay in the
    signal.  That width is measured rather than assumed: the pattern is stripped
    generously, the noise of what is left is measured by the paper's own Sec. 3.2
    (:func:`noise_level`), the stretches standing above it are its Sec. 3.3 peak
    groups (:func:`peak_groups`), and the broadest of those is the width to
    clear - by the factor of two the reach law calls for, and never by less than
    the floor.

    **Bending** is not determined by the data in the same way, and asking the
    fit which value it prefers is the wrong question for granularity but the
    right one here.  Asked, over eight glycol mounts across the whole 0 to 100
    range, the answer is 0 on every mount, and the penalty for raising it grows
    monotonically to between 4.6 and 11.9 points of R_wp at 100.  The
    reason is (A.2) of the manual: a real clay background's curvature exceeds
    the bound that any bending factor in HighScore's range can grant, so the
    allowance cannot do the job the paper introduced it for, and all that is
    left of it is a baseline sitting slightly higher - which on an oriented
    mount means slightly more of the broad basal intensity taken out as
    background.  So 0 is suggested, with the paper's own 1 a defensible choice
    for a specimen whose background really is gently curved.

    Granularity is deliberately *not* chosen by fit quality, though it would be
    easy to: R_wp falls almost monotonically as the baseline drops, because a
    lower background always lets the fit explain more of the pattern whether or
    not what was removed was background.  It is the same bias that makes the
    residual of a whole-pattern fit useless as a measure of missing material.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if two_theta.shape != intensity.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    if len(two_theta) < 16:
        raise ValueError("too few points to read a suggestion from")

    if range_two_theta is not None:
        low, high = range_two_theta
        inside = (two_theta >= low) & (two_theta <= high)
        if inside.sum() >= 16:
            two_theta, intensity = two_theta[inside], intensity[inside]

    step = float(np.mean(np.diff(two_theta)))
    if step <= 0:
        raise ValueError("two_theta must be increasing")

    # Strip generously first: the widths are wanted against a background that is
    # under everything, not against one that has already climbed into the
    # reflections being measured.
    difference = intensity - snip_baseline(two_theta, intensity, window=12.0)
    level = noise_level(difference)
    groups = peak_groups(two_theta, difference, level.threshold)

    widest = max((group.width for group in groups), default=0.0)
    limit = ceiling * float(two_theta[-1] - two_theta[0])
    wanted = max(safety * widest, floor)
    floored = safety * widest <= floor
    capped = wanted > limit
    wanted = min(wanted, limit)
    granularity = max(1, int(round(
        wanted / (SONNEVELD_VISSER_REACH * step * math.sqrt(max(1, iterations)))
    )))
    reach = sonneveld_visser_reach(granularity, step, iterations)

    if not groups:
        note = (
            f"No reflection rose above the noise ({level.sigma:.0f} counts), so the "
            f"granularity is the floor: {granularity} points, reaching {reach:.1f} deg."
        )
    elif floored:
        note = (
            f"Widest reflection {widest:.2f} deg, at the noise level of "
            f"{level.sigma:.0f} counts over {len(groups)} groups. Twice that is under the "
            f"{floor:.0f} deg floor, so granularity {granularity} ({reach:.1f} deg) is the "
            f"floor rather than the measurement."
        )
    else:
        note = (
            f"Widest reflection {widest:.2f} deg of {len(groups)} above the noise "
            f"({level.sigma:.0f} counts), so granularity {granularity} to reach "
            f"{reach:.1f} deg - twice the widest, so it keeps all of it."
        )
    if capped:
        note += (
            f" The widest feature is so broad that twice it would reach past a quarter "
            f"of the scan, so the reach is capped at {limit:.1f} deg; a feature that wide "
            f"cannot be told from background by any local test, and the choice is yours."
        )
    note += (
        " Bending 0: every one of eight real mounts fitted best there, and worse "
        "by up to 12 points of R_wp at 100."
    )

    return SonneveldVisserSuggestion(
        granularity=granularity,
        bending=0.0,
        reach=reach,
        widest_reflection=widest,
        groups=len(groups),
        sigma=level.sigma,
        step=step,
        floored=floored,
        capped=capped,
        note=note,
    )


# --------------------------------------------------------------------------
# The Clayfit background models
# --------------------------------------------------------------------------
#
# Everything from here down is Clayfit's background step, reimplemented so that
# the same choice made here and there gives the same curve.  It replaces an
# earlier design in which one parametric model (above) was fitted to a
# peak-stripped estimate of the background, and it is a different thing in two
# ways that matter.
#
# First, Clayfit's menu is a menu of *algorithms*, not of bases.  "Polynomial"
# is a least-squares polynomial through a set of anchor points, in raw 2theta.
# "Chebyshev" is a Chebyshev series fitted to ``I/2theta`` over the *whole*
# pattern - peaks included - which is then rescaled by a single amplitude fitted
# to the anchor points.  The two therefore do not agree, and cannot: only the
# first is constrained to pass near the anchor points, and only the second sees
# the peaks.  An earlier version of this module offered "Polynomial" and
# "Chebyshev" as two bases for the same unconstrained least-squares fit, where
# they do span the same space and do give the same curve, and the note here said
# so.  That was a statement about that implementation, not about the two
# methods, and it does not describe Clayfit - where the order slider visibly
# moves both curves and the two sit in different places.
#
# Second, the anchor points are found by a rolling ball rather than by peak
# stripping, and the order runs from 4 to 12 with 6 the default.
#
# The one deliberate difference, besides Sonneveld-Visser being offered as a
# sixth model, is that :meth:`ClayfitBackground.subtract` clips the corrected
# pattern at zero for every model rather than for the percentile model alone.
# That is a requirement of what comes after it here - the weights of the
# non-negative least squares are 1/counts - and it changes the corrected
# pattern, never the background curve, which is what the two programs are being
# compared on.

CLAYFIT_ANCHOR_RADIUS = 0.5
"""Rolling-ball radius, as a fraction of the pattern's own extent.

The ball rolls under the pattern scaled into the unit square, so the radius is
dimensionless and means the same thing on any scan: 0.5 is Clayfit's value and
a ball of half the width of the whole pattern.
"""

CLAYFIT_ANCHOR_STRIDE = 5
"""Points between the samples the ball rolls on - Clayfit's ``array[::5]``.

The rolling ball compares every pair of samples against every sample, so the
work grows as the cube of their number; on a 2200-point scan every fifth point
is 440 samples and about a second.  It also sets the finest feature the ball can
enter, which on a 0.017 deg scan is 0.08 deg - narrower than any reflection.
"""

CLAYFIT_ORDER = 6
"""Default polynomial/Chebyshev order.  Clayfit's ``polydegree``."""

CLAYFIT_ORDER_LIMITS = (4, 12)
"""Order range Clayfit accepts, and the range of its slider."""

CLAYFIT_INVERSE_X_AMPLITUDE = 500.0
"""Default amplitude ``A`` of the additive ``A/2theta`` term.

Set rather than fitted, which is the point of it: it is there to be turned up
until the low-angle rise is accounted for, and a fitted ``A`` would instead be
free to eat the 001 reflections that sit on that rise.
"""

CLAYFIT_ALS_SMOOTHNESS = 1.0
CLAYFIT_ALS_ASYMMETRY = 0.001
CLAYFIT_ALS_ITERATIONS = 100
"""Clayfit's ``baseline_als_2d(lam=1, p=0.001, niter=100)``.

``lam`` looks small only because the second-difference operator is divided by
the square of the step: at 0.0167 deg that is a factor of 3600, so the
smoothing is comparable to ``lam`` of 10^7 on an unscaled grid.
"""

QPA_PERCENTILE_WINDOW = 1.50
QPA_LOWER_PERCENTILE = 15.0
QPA_BASELINE_SMOOTH = 0.30
QPA_BASELINE_ORDER = 2
QPA_FINAL_SMOOTH = 0.08
QPA_FINAL_ORDER = 2
"""Clayfit's percentile/Savitzky-Golay defaults, in degrees and orders."""

QPA_MODEL_NAME = "qpa"
"""Clayfit calls it "QPA percentile + Savitzky-Golay"."""

CLAYFIT_MODELS = (
    "exponential",
    "polynomial",
    "chebyshev",
    "als",
    QPA_MODEL_NAME,
    "sonneveld-visser",
)
"""The models the background step offers, in Clayfit's own order.

The first five are Clayfit's; Sonneveld-Visser is the addition.
"""

CLAYFIT_MODEL_LABELS = {
    "exponential": "Exponential",
    "polynomial": "Polynomial",
    "chebyshev": "Chebyshev",
    "als": "ALS",
    QPA_MODEL_NAME: "QPA percentile + Savitzky-Golay",
    "sonneveld-visser": "Sonneveld-Visser",
}

_ANCHOR_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def rolling_ball(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    radius: float = CLAYFIT_ANCHOR_RADIUS,
) -> np.ndarray:
    """Indices of the points a ball of ``radius`` touches rolling underneath.

    The pattern is scaled into the unit square, so ``radius`` is a fraction of
    its own extent rather than a number of degrees or counts.  For every pair of
    points a circle of that radius is passed through both; of its two possible
    centres the lower one that lies under the pattern is taken, and the pair is
    a pair of contact points when no other point lies inside that circle.  What
    comes back is the set of all such points: the lower envelope of the pattern
    at the scale the ball can follow.

    This is Clayfit's ``rolling_ball``, and the two agree point for point.  It
    is written over whole arrays rather than in a compiled loop because the
    quadratic part - a circle through each of the 10^5 pairs - is the cheap
    part, and the cubic part is done in chunks.
    """
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    if radius <= 0.0:
        raise ValueError("the rolling-ball radius must be positive")
    count = x.size
    if count < 3:
        return np.arange(count)

    span_x = float(x.max() - x.min())
    span_y = float(y.max() - y.min())
    if span_x <= 0.0 or span_y <= 0.0:
        # A flat pattern has no envelope to find; every point is on it.
        return np.arange(count)
    u = (x - x.min()) / span_x
    v = (y - y.min()) / span_y

    first, second = np.triu_indices(count, k=1)
    separation = np.hypot(u[second] - u[first], v[second] - v[first])
    # Points further apart than the diameter admit no such circle, and
    # coincident points give no direction to offset the centre along.
    reachable = (separation < 2.0 * radius - 1e-9) & (separation >= 1e-9)
    first, second, separation = first[reachable], second[reachable], separation[reachable]
    if first.size == 0:
        return np.arange(count)

    step_u = u[second] - u[first]
    step_v = v[second] - v[first]
    offset = np.sqrt(np.maximum(radius**2 - (0.5 * separation) ** 2, 0.0))
    normal_u = -step_v / separation
    normal_v = step_u / separation
    middle_u = 0.5 * (u[first] + u[second])
    middle_v = 0.5 * (v[first] + v[second])
    centre_u = np.stack((middle_u + offset * normal_u, middle_u - offset * normal_u))
    centre_v = np.stack((middle_v + offset * normal_v, middle_v - offset * normal_v))

    # Clayfit's own validity test: the centre must not sit above the pattern,
    # judged at the sample nearest in x.
    right = np.clip(np.searchsorted(u, centre_u), 1, count - 1)
    left = right - 1
    nearer_left = np.abs(centre_u - u[left]) <= np.abs(u[right] - centre_u)
    beneath = centre_v <= v[np.where(nearer_left, left, right)]

    both = beneath[0] & beneath[1]
    lower = centre_v[0] < centre_v[1]
    take_first = (both & lower) | (beneath[0] & ~beneath[1])
    usable = take_first | (both & ~lower) | (beneath[1] & ~beneath[0])
    chosen_u = np.where(take_first, centre_u[0], centre_u[1])[usable]
    chosen_v = np.where(take_first, centre_v[0], centre_v[1])[usable]
    first, second = first[usable], second[usable]
    if first.size == 0:
        return np.arange(count)

    empty = np.zeros(first.size, dtype=bool)
    # Four million distances at a time, which is a few tens of megabytes.
    block = max(1, 4_000_000 // count)
    for start in range(0, first.size, block):
        stop = min(start + block, first.size)
        distance = np.hypot(
            u[None, :] - chosen_u[start:stop, None],
            v[None, :] - chosen_v[start:stop, None],
        )
        inside = distance < radius - 1e-6
        rows = np.arange(stop - start)
        # The two contact points lie on the circle, not inside it, but floating
        # point puts them either side of that; Clayfit excludes them explicitly.
        inside[rows, first[start:stop]] = False
        inside[rows, second[start:stop]] = False
        empty[start:stop] = ~inside.any(axis=1)

    return np.unique(np.concatenate((first[empty], second[empty])))


def background_anchors(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    radius: float = CLAYFIT_ANCHOR_RADIUS,
    stride: int = CLAYFIT_ANCHOR_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """The anchor points the polynomial and exponential models are fitted to.

    Clayfit's ``find_bkg_points(array[::5], r=0.5)``: every ``stride``-th point
    of the pattern is offered to the rolling ball, and the points it touches are
    the anchors.  On a clay mount there are typically twenty to thirty of them,
    which is what makes the order of the polynomial fitted through them
    meaningful and why an order above 12 is refused.

    Results are cached on the contents of the pattern, so moving the order
    slider does not roll the ball again.
    """
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    stride = max(1, int(stride))
    key = (
        hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=16).digest(),
        hashlib.blake2b(np.ascontiguousarray(y).tobytes(), digest_size=16).digest(),
        float(radius),
        stride,
    )
    cached = _ANCHOR_CACHE.get(key)
    if cached is not None:
        return cached

    sampled_x, sampled_y = x[::stride], y[::stride]
    contacts = rolling_ball(sampled_x, sampled_y, radius=radius)
    anchors = (sampled_x[contacts].copy(), sampled_y[contacts].copy())
    if len(_ANCHOR_CACHE) > 32:
        _ANCHOR_CACHE.clear()
    _ANCHOR_CACHE[key] = anchors
    return anchors


def polynomial_baseline(
    two_theta: np.ndarray,
    anchor_two_theta: np.ndarray,
    anchor_intensity: np.ndarray,
    order: int = CLAYFIT_ORDER,
) -> np.ndarray:
    """Least-squares polynomial of ``order`` through the anchor points.

    Clayfit's ``poly_bkg``: ``np.polyfit`` on the anchors, evaluated on the
    pattern, in raw 2theta rather than in a scaled variable.  Raw 2theta is
    worth naming because it is what makes the higher orders behave as they do -
    at order 12 the monomials of an argument running from 3 to 40 span 18 orders
    of magnitude, so the fit is ill-conditioned and the curve is free to move a
    long way between anchors.  That is the same arithmetic Clayfit does.
    """
    anchor_x = np.asarray(anchor_two_theta, dtype=float)
    anchor_y = np.asarray(anchor_intensity, dtype=float)
    order = int(order)
    if order < 0:
        raise ValueError("the order must not be negative")
    if anchor_x.size <= order:
        raise ValueError(
            f"{anchor_x.size} anchor points for an order-{order} polynomial; "
            f"lower the order or widen the rolling ball"
        )
    with warnings.catch_warnings():
        # np.polyfit warns that a high-order fit in raw 2theta is badly
        # conditioned, which is true and is the method being reproduced.
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        coefficients = np.polyfit(anchor_x, anchor_y, order)
    return np.polyval(coefficients, np.asarray(two_theta, dtype=float))


def chebyshev_baseline(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    anchor_two_theta: np.ndarray,
    anchor_intensity: np.ndarray,
    order: int = CLAYFIT_ORDER,
) -> np.ndarray:
    """A Chebyshev shape fitted to ``I/2theta``, rescaled to the anchor points.

    Clayfit's ``cheby_bkg``, which is TOPAS's way of writing a background with a
    ``1/x`` term built in:

    1. a Chebyshev series of ``order`` is least-squares fitted to
       ``intensity / two_theta`` over the *whole* pattern, peaks included;
    2. that shape is multiplied by ``2theta`` implicitly - the series is the
       background *divided* by the angle - and one amplitude is fitted to the
       anchor points.

    Because the shape is fitted to the whole pattern it follows the peaks, and
    because only an amplitude is then free the curve is not obliged to pass
    through the anchors.  That is why it sits well below the measurement on a
    pattern with strong basal reflections, and why it moves when the order
    changes: the shape is refitted at every order.

    The amplitude is solved in closed form rather than by ``curve_fit``, which
    is the same answer for a model linear in its one parameter.
    """
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    if np.any(x <= 0.0):
        raise ValueError("the Chebyshev background needs positive 2theta values")
    order = int(order)
    if order < 0:
        raise ValueError("the order must not be negative")
    shape = np.polynomial.chebyshev.Chebyshev.fit(x, y / x, order)
    at_anchors = shape(np.asarray(anchor_two_theta, dtype=float))
    anchor_y = np.asarray(anchor_intensity, dtype=float)
    denominator = float(np.dot(at_anchors, at_anchors))
    if denominator <= 0.0:
        raise ValueError("the Chebyshev shape vanishes at every anchor point")
    amplitude = float(np.dot(at_anchors, anchor_y)) / denominator
    return amplitude * shape(x)


def exponential_baseline(
    two_theta: np.ndarray,
    anchor_two_theta: np.ndarray,
    anchor_intensity: np.ndarray,
) -> np.ndarray:
    """``a exp(-b 2theta) + c`` fitted to the anchor points.

    Clayfit's ``exp_bkg``, with its starting values: ``c`` the lowest anchor,
    ``a`` the range of the anchors and ``b`` the reciprocal of the highest
    angle.  It is Clayfit's default model and the one that describes an oriented
    mount's background best over the first few degrees, where the direct-beam
    tail falls faster than any low-order polynomial through the same points.

    Raises :class:`RuntimeError` when the fit does not converge, which is what
    Clayfit catches to fall back on the polynomial.
    """
    from scipy.optimize import curve_fit

    anchor_x = np.asarray(anchor_two_theta, dtype=float)
    anchor_y = np.asarray(anchor_intensity, dtype=float)
    if anchor_x.size < 3:
        raise RuntimeError("an exponential background needs at least three anchor points")

    def model(angle, amplitude, decay, floor):
        return amplitude * np.exp(-decay * angle) + floor

    floor = float(anchor_y.min())
    start = [float(anchor_y.max()) - floor, 1.0 / (float(anchor_x.max()) or 1.0), floor]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parameters, _ = curve_fit(model, anchor_x, anchor_y, p0=start, maxfev=5000)
    return model(np.asarray(two_theta, dtype=float), *parameters)


def als_baseline_2d(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    smoothness: float = CLAYFIT_ALS_SMOOTHNESS,
    asymmetry: float = CLAYFIT_ALS_ASYMMETRY,
    iterations: int = CLAYFIT_ALS_ITERATIONS,
) -> np.ndarray:
    """Asymmetric least squares with Clayfit's scaling and parameters.

    The same method as :func:`als_baseline` with two differences, both
    Clayfit's: the second-difference operator is divided by the square of the
    mean step, so ``smoothness`` is in physical units and 1 is a strong
    smoothing rather than none; and a point exactly on the baseline is given
    weight zero rather than being counted on either side.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    if smoothness <= 0.0:
        raise ValueError("smoothness must be positive")
    if asymmetry <= 0.0:
        raise ValueError("asymmetry must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    size = y.size
    if size < 5:
        return y.copy()
    step = float(np.mean(np.diff(x)))
    if step <= 0.0:
        raise ValueError("two_theta must be increasing")

    second = sparse.diags(
        [1.0, -2.0, 1.0], [0, -1, -2], shape=(size, size - 2), format="csr"
    ) / step**2
    penalty = smoothness * (second @ second.T)
    weights = np.ones(size)
    baseline = y.copy()
    for _ in range(iterations):
        system = sparse.diags(weights, 0, format="csc") + penalty
        baseline = spsolve(system.tocsc(), weights * y)
        updated = asymmetry * (y > baseline) + (1.0 - asymmetry) * (y < baseline)
        if np.array_equal(updated, weights):
            break
        weights = updated
    return np.asarray(baseline, dtype=float)


def _odd_window(points: int, minimum: int) -> int:
    value = max(int(minimum), int(points))
    return value if value % 2 else value + 1


def _savgol_window(
    length: int,
    step: float,
    window_degrees: float,
    order: int,
    minimum_points: int,
) -> int:
    """Points in a Savitzky-Golay window of ``window_degrees``, Clayfit's rule."""
    if window_degrees == 0.0:
        return 0
    largest_odd = length if length % 2 else length - 1
    least_for_order = order + 1
    if least_for_order % 2 == 0:
        least_for_order += 1
    requested = _odd_window(round(window_degrees / step), max(minimum_points, least_for_order))
    window = min(largest_odd, requested)
    if window <= order:
        raise ValueError("the Savitzky-Golay order is too high for this pattern")
    return window


def smooth_by_degrees(
    values: np.ndarray,
    step: float,
    window_degrees: float,
    order: int,
    minimum_points: int = 5,
) -> tuple[np.ndarray, int]:
    """Savitzky-Golay smoothing over a window given in degrees, not in points.

    Clayfit's ``smooth_by_degrees``.  A window of zero degrees means no
    smoothing and returns the signal unchanged, which is how the percentile
    model's two smoothing stages are switched off.
    """
    from scipy.signal import savgol_filter

    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or signal.size < 3 or not np.all(np.isfinite(signal)):
        raise ValueError("the signal must be a finite one-dimensional array")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("the angular step must be finite and positive")
    if not np.isfinite(window_degrees) or window_degrees < 0.0:
        raise ValueError("the smoothing window must be finite and non-negative")
    order = int(order)
    if order < 0:
        raise ValueError("the Savitzky-Golay order must not be negative")
    points = _savgol_window(signal.size, float(step), float(window_degrees), order,
                            minimum_points)
    if points == 0:
        return signal.copy(), 0
    return np.asarray(savgol_filter(signal, points, order), dtype=float), points


@dataclass
class QpaBaseline:
    """The percentile background, what it left, and the windows it used."""

    baseline: np.ndarray
    corrected: np.ndarray
    smoothed: np.ndarray
    step: float
    percentile_points: int
    baseline_smooth_points: int
    final_smooth_points: int


def qpa_percentile_baseline(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    window: float = QPA_PERCENTILE_WINDOW,
    baseline_smooth: float = QPA_BASELINE_SMOOTH,
    baseline_order: int = QPA_BASELINE_ORDER,
    final_smooth: float = QPA_FINAL_SMOOTH,
    final_order: int = QPA_FINAL_ORDER,
    percentile: float = QPA_LOWER_PERCENTILE,
) -> QpaBaseline:
    """Clayfit's percentile/Savitzky-Golay background, for quantification.

    A running ``percentile``-th percentile over ``window`` degrees, smoothed by
    a Savitzky-Golay filter over ``baseline_smooth`` degrees; the corrected
    pattern is clipped at zero and smoothed again over ``final_smooth``
    degrees.  It is the one model here whose corrected pattern is smoothed,
    which is what Clayfit's "QPA" means: the pattern is being prepared for a
    quantitative fit rather than displayed.

    A fifteenth percentile over 1.5 deg cannot be raised by a reflection that
    occupies less than that fraction of the window, and follows a structured
    background far more closely than a polynomial through twenty anchor points
    can.  The price is that it follows a broad reflection too, which is why the
    window is a parameter: on an oriented mount with a smectite band it has to
    be wide enough that the band is not taken for background.
    """
    from scipy.ndimage import percentile_filter

    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    if y.size < 3:
        raise ValueError("at least three points are needed")
    if not np.isfinite(window) or window <= 0.0:
        raise ValueError("the percentile window must be finite and positive")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("the percentile must lie between 0 and 100")

    step = float(np.median(np.diff(x)))
    if step <= 0.0:
        raise ValueError("two_theta must be increasing")
    percentile_points = _odd_window(round(float(window) / step), 21)
    baseline = np.asarray(
        percentile_filter(y, percentile=float(percentile), size=percentile_points,
                          mode="nearest"),
        dtype=float,
    )
    baseline, baseline_points = smooth_by_degrees(
        baseline, step, baseline_smooth, baseline_order, minimum_points=7
    )
    corrected = np.clip(y - baseline, 0.0, None)
    smoothed, final_points = smooth_by_degrees(
        corrected, step, final_smooth, final_order, minimum_points=5
    )
    return QpaBaseline(
        baseline=baseline,
        corrected=corrected,
        smoothed=smoothed,
        step=step,
        percentile_points=percentile_points,
        baseline_smooth_points=baseline_points,
        final_smooth_points=final_points,
    )


def inverse_x_baseline(two_theta: np.ndarray, amplitude: float) -> np.ndarray:
    """The additive ``A / 2theta`` term, with ``A`` set rather than fitted."""
    x = np.asarray(two_theta, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("the 2theta values must all be finite")
    if np.any(x <= 0.0):
        raise ValueError("the 1/x background needs positive 2theta values")
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("the 1/x amplitude must be finite and non-negative")
    return float(amplitude) / x


@dataclass
class ClayfitBackground:
    """A background calculated by one of Clayfit's models.

    It carries the interface of :class:`BackgroundFit`, so it can be stored and
    subtracted anywhere one is expected, and additionally the anchor points and
    the ``A/x`` amplitude, which are what the operator is adjusting.
    """

    two_theta: np.ndarray
    baseline: np.ndarray
    kind: str
    order: int | None = None
    anchor_two_theta: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    anchor_intensity: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    inverse_x_amplitude: float = 0.0
    smooth_degrees: float = 0.0
    smooth_order: int = QPA_FINAL_ORDER
    smooth_points: int = 0
    note: str = ""
    """Set when a model fell back on another, as the exponential does."""

    def __post_init__(self) -> None:
        self.two_theta = np.asarray(self.two_theta, dtype=float)
        self.baseline = np.asarray(self.baseline, dtype=float)
        if self.two_theta.shape != self.baseline.shape:
            raise ValueError("two_theta and baseline must have the same shape")

    @property
    def model(self) -> str:
        parts = [CLAYFIT_MODEL_LABELS.get(self.kind, self.kind)]
        if self.order is not None:
            parts.append(f"order {self.order:d}")
        if self.inverse_x_amplitude > 0.0:
            parts.append(f"+ {self.inverse_x_amplitude:g}/2θ")
        if self.anchor_two_theta.size:
            parts.append(f"{self.anchor_two_theta.size:d} anchor points")
        return f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else parts[0]

    @property
    def components(self) -> list[str]:
        names = [CLAYFIT_MODEL_LABELS.get(self.kind, self.kind)]
        if self.inverse_x_amplitude > 0.0:
            names.append("A/2theta")
        return names

    @property
    def n_terms(self) -> int:
        """Free parameters fitted to the anchor points.

        The polynomial has ``order + 1``; the Chebyshev model has one, its
        amplitude, however high the order, because the shape is fitted to the
        pattern and not to the anchors; the exponential has three; the
        percentile and erosion models fit nothing.  The ``A/x`` amplitude is
        never fitted and never counted.
        """
        if self.kind == "polynomial":
            return int(self.order or 0) + 1
        if self.kind == "chebyshev":
            return 1
        if self.kind == "exponential":
            return 3
        return 0

    @property
    def points(self) -> np.ndarray:
        """Which points of the pattern the background was judged against."""
        if self.anchor_two_theta.size == 0:
            return np.ones(self.two_theta.shape, dtype=bool)
        mask = np.zeros(self.two_theta.shape, dtype=bool)
        mask[np.searchsorted(self.two_theta, self.anchor_two_theta).clip(
            0, self.two_theta.size - 1)] = True
        return mask

    @property
    def target(self) -> np.ndarray:
        """The curve the model was fitted to, for drawing.

        For the models fitted to anchor points there is no such curve - the
        anchors are the data - so the background itself is returned and the view
        has nothing extra to draw.
        """
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
        """The corrected pattern, clipped at zero and smoothed where asked.

        Clayfit clips only the percentile model's corrected pattern; here every
        model is clipped, because the non-negative least squares that follows
        weights each point by 1/counts and a negative count has no meaning as a
        variance.  The background curve is unaffected.
        """
        angles = np.asarray(two_theta, dtype=float)
        corrected = np.clip(np.asarray(intensity, dtype=float) - self(angles), 0.0, None)
        if self.smooth_degrees > 0.0 and angles.size > 2:
            step = float(np.median(np.diff(angles)))
            if step > 0.0:
                corrected, _ = smooth_by_degrees(
                    corrected, step, self.smooth_degrees, self.smooth_order
                )
                corrected = np.clip(corrected, 0.0, None)
        return corrected

    def r_squared(self, two_theta: np.ndarray, intensity: np.ndarray | None = None) -> float:
        """How well the model describes the anchor points it was fitted to.

        Against the anchors, not against the pattern: the background is meant to
        pass through the lower envelope, and measuring it against data that
        still carries the peaks would reward a model that ran up into them.
        For the models that fit nothing there is nothing to report.
        """
        if self.anchor_two_theta.size < 3 or self.n_terms == 0:
            return float("nan")
        observed = self.anchor_intensity
        predicted = self(self.anchor_two_theta)
        total = float(np.sum((observed - observed.mean()) ** 2))
        if total <= 0.0:
            return float("nan")
        return 1.0 - float(np.sum((observed - predicted) ** 2)) / total


def clayfit_background(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    kind: str = "exponential",
    order: int = CLAYFIT_ORDER,
    inverse_x_amplitude: float = 0.0,
    radius: float = CLAYFIT_ANCHOR_RADIUS,
    stride: int = CLAYFIT_ANCHOR_STRIDE,
    qpa_window: float = QPA_PERCENTILE_WINDOW,
    qpa_baseline_smooth: float = QPA_BASELINE_SMOOTH,
    qpa_baseline_order: int = QPA_BASELINE_ORDER,
    qpa_final_smooth: float = QPA_FINAL_SMOOTH,
    qpa_final_order: int = QPA_FINAL_ORDER,
    granularity: int = SONNEVELD_VISSER_GRANULARITY,
    bending: float = SONNEVELD_VISSER_BENDING,
) -> ClayfitBackground:
    """Calculate one of the background models, as Clayfit's background step does.

    ``kind`` is one of :data:`CLAYFIT_MODELS`.  ``order`` drives the polynomial
    and the Chebyshev model and is ignored by the rest; the ``qpa_*`` arguments
    drive the percentile model, ``granularity`` and ``bending`` the erosion.
    ``inverse_x_amplitude`` is added to whichever model was chosen, and is a
    setting rather than a fitted coefficient.
    """
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.shape != y.shape:
        raise ValueError("two_theta and intensity must have the same shape")
    if kind not in CLAYFIT_MODELS:
        raise ValueError(
            f"unknown background model {kind!r}; expected one of {', '.join(CLAYFIT_MODELS)}"
        )
    low, high = CLAYFIT_ORDER_LIMITS
    if kind in ("polynomial", "chebyshev") and not low <= int(order) <= high:
        raise ValueError(f"the polynomial/Chebyshev order must be from {low} to {high}")

    anchor_x = np.array([], dtype=float)
    anchor_y = np.array([], dtype=float)
    used_order: int | None = None
    smooth_degrees = 0.0
    note = ""

    if kind in ("exponential", "polynomial", "chebyshev"):
        anchor_x, anchor_y = background_anchors(x, y, radius=radius, stride=stride)

    if kind == "polynomial":
        baseline = polynomial_baseline(x, anchor_x, anchor_y, order=int(order))
        used_order = int(order)
    elif kind == "chebyshev":
        baseline = chebyshev_baseline(x, y, anchor_x, anchor_y, order=int(order))
        used_order = int(order)
    elif kind == "exponential":
        try:
            baseline = exponential_baseline(x, anchor_x, anchor_y)
        except (RuntimeError, ValueError) as exc:
            # Clayfit's own fallback, and worth reporting rather than hiding:
            # the two curves are not interchangeable.
            baseline = polynomial_baseline(x, anchor_x, anchor_y, order=int(order))
            used_order = int(order)
            note = (f"The exponential fit did not converge ({exc}); "
                    f"the order-{int(order)} polynomial is shown instead.")
    elif kind == "als":
        baseline = als_baseline_2d(x, y)
    elif kind == QPA_MODEL_NAME:
        result = qpa_percentile_baseline(
            x, y,
            window=qpa_window,
            baseline_smooth=qpa_baseline_smooth,
            baseline_order=qpa_baseline_order,
            final_smooth=qpa_final_smooth,
            final_order=qpa_final_order,
        )
        baseline = result.baseline
        smooth_degrees = float(qpa_final_smooth)
    else:
        baseline = sonneveld_visser_baseline(
            x, y, granularity=int(granularity), bending=float(bending)
        )

    amplitude = float(inverse_x_amplitude)
    if amplitude > 0.0:
        baseline = baseline + inverse_x_baseline(x, amplitude)

    return ClayfitBackground(
        two_theta=x,
        baseline=baseline,
        kind=kind,
        order=used_order,
        anchor_two_theta=anchor_x,
        anchor_intensity=anchor_y,
        inverse_x_amplitude=amplitude,
        smooth_degrees=smooth_degrees,
        smooth_order=int(qpa_final_order),
        note=note,
    )
