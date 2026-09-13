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

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BackgroundModel",
    "BackgroundFit",
    "StrippedBackground",
    "snip_baseline",
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
            target = snip_baseline(two_theta, intensity, window=snip_window)
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

    def __post_init__(self) -> None:
        self.two_theta = np.asarray(self.two_theta, dtype=float)
        self.baseline = np.asarray(self.baseline, dtype=float)
        if self.two_theta.shape != self.baseline.shape:
            raise ValueError("two_theta and baseline must have the same shape")

    @classmethod
    def fit(
        cls, two_theta: np.ndarray, intensity: np.ndarray, window: float = 4.0
    ) -> "StrippedBackground":
        two_theta = np.asarray(two_theta, dtype=float)
        return cls(
            two_theta=two_theta,
            baseline=snip_baseline(two_theta, intensity, window=window),
            window=window,
        )

    @property
    def model(self) -> str:
        return f"peak-stripped ({self.window:g} deg)"

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


def snip_baseline(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    window: float = 4.0,
    iterations: int | None = None,
) -> np.ndarray:
    """Estimate the background by iterative peak stripping (SNIP).

    At each pass every point is replaced by the smaller of itself and the mean
    of the two points a distance ``p`` away, for growing ``p``; anything
    narrower than ``window`` degrees is clipped away and the smooth background
    survives.  The method is that of Ryan, Clayton, Griffin, Sie & Cousens
    (1988) Nucl. Instrum. Methods B34, 396-402, and is the standard background
    estimator for X-ray spectra and diffractograms.

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
        iterations = max(1, int(round(window / (2.0 * step))))

    # Work on a log-log-ish scale so that strong peaks do not dominate.
    offset = float(np.min(values))
    transformed = np.log(np.log(np.sqrt(np.clip(values - offset, 0.0, None) + 1.0) + 1.0) + 1.0)

    # Every comparison must be genuinely two-sided, which at the ends of the scan
    # means inventing the missing neighbour. How it is invented decides what
    # happens to a sloping edge, and both of the easy answers are wrong:
    #
    #   * substituting the point itself makes the average sit below the point
    #     whenever the edge falls, so every pass drags it further down. The
    #     direct-beam tail of an oriented clay mount is exactly such an edge, and
    #     it was stripped away as though it were a peak - the background came out
    #     near 500 counts where the measurement was 1900, and the difference was
    #     left behind as false signal;
    #   * narrowing the window to fit pins the first point to the measurement,
    #     which is right on a tail but wrong when the scan opens on a reflection,
    #     and the error then propagates into the fitted model.
    #
    # Extending the trend instead is neutral: a falling edge is continued upwards
    # and survives, while an edge that rises into a peak is continued downwards
    # and is stripped.
    pad = min(iterations, max(len(transformed) // 4, 1))
    if pad > 0:
        span = min(len(transformed) - 1, max(3, pad))
        steps = np.arange(1, pad + 1, dtype=float)
        left_slope = (transformed[span] - transformed[0]) / span
        right_slope = (transformed[-1] - transformed[-1 - span]) / span
        transformed = np.concatenate(
            [
                transformed[0] - left_slope * steps[::-1],
                transformed,
                transformed[-1] + right_slope * steps,
            ]
        )

    for p in range(iterations, 0, -1):
        shifted_left = np.roll(transformed, p)
        shifted_right = np.roll(transformed, -p)
        shifted_left[:p] = transformed[0]
        shifted_right[-p:] = transformed[-1]
        transformed = np.minimum(transformed, 0.5 * (shifted_left + shifted_right))

    if pad > 0:
        transformed = transformed[pad:-pad]
    restored = (np.exp(np.exp(transformed) - 1.0) - 1.0) ** 2 - 1.0
    # A background is never above the measurement it came from.
    return np.minimum(np.clip(restored, 0.0, None) + offset, values)


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
