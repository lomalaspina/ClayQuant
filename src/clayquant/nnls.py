"""Non-negative least-squares fitting of a measured pattern against the library.

The measured (background-subtracted) pattern is modelled as a non-negative
combination of calculated patterns,

    y_obs(2theta) ~ sum_p c_p * y_p(2theta),   c_p >= 0

which is solved with :func:`scipy.optimize.nnls`.  Non-negativity is what makes
the fit usable with a large, highly correlated library: it is the constraint
that stops the solver from cancelling one clay pattern against another.

What the coefficients mean
--------------------------
The ``c_p`` are scale factors, not weight fractions.  Each entry's share of the
*calculated scattering* is reported as ``scattering_fraction``, which is the
directly measured quantity.  Converting to weight fractions needs a reference
intensity ratio (RIR) per phase, i.e. the calculated intensity per unit mass in
the same geometry, and for oriented mounts also depends on the mount
preparation; pass ``reference_intensity_ratios`` if you have calibrated them,
otherwise read the output as relative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from .background import BackgroundFit
from .pattern import Pattern

__all__ = ["FitResult", "nnls_fit"]


@dataclass
class FitResult:
    """Outcome of a non-negative least-squares fit."""

    two_theta: np.ndarray
    observed: np.ndarray
    calculated: np.ndarray
    coefficients: np.ndarray
    names: list[str]
    phases: list[str]
    scattering_fraction: np.ndarray
    amplitude_fraction: np.ndarray
    r_wp: float
    r_p: float
    mask: np.ndarray
    components: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    march_dollase: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    """The texture parameter each entry was calculated with."""

    relative_mass: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    """Mass of each entry in the specimen, in arbitrary but common units.

    ``coefficient / normalization`` is proportional to the number of scattering
    units of that entry in the beam, whatever height its stored pattern was
    normalised to; multiplying by the mass of one unit gives a mass.  The
    constant of proportionality is the same for every entry - the instrument,
    the irradiated volume, the counting time, all of which the phases of one
    measurement share - so ratios of these are ratios of mass.  Zero where the
    library does not record the mass of its units.
    """

    @property
    def residual(self) -> np.ndarray:
        return self.observed - self.calculated

    def pattern(self, name: str = "fit") -> Pattern:
        return Pattern(self.two_theta, self.calculated, name=name, metadata=self.metadata)

    @property
    def metrics_note(self) -> str:
        return (
            "amplitude_fraction is each pattern's share of the fitted scale factors; "
            "scattering_fraction is its share of the calculated integrated intensity. "
            "Broad interstratified patterns carry more area per unit peak height, so the "
            "two differ; neither is a weight fraction without a reference intensity ratio."
        )

    def phase_components(self) -> dict[str, np.ndarray]:
        """Fitted curve of each phase, summed over that phase's library entries.

        Only phases with a non-zero coefficient appear.  The curves are on
        :attr:`two_theta` and already carry their fitted scale, so they sum to
        :attr:`calculated`.
        """
        by_phase: dict[str, np.ndarray] = {}
        for name, phase in zip(self.names, self.phases):
            curve = self.components.get(name)
            if curve is None:
                continue
            if phase in by_phase:
                by_phase[phase] = by_phase[phase] + curve
            else:
                by_phase[phase] = curve.copy()
        return by_phase

    def active(self, threshold: float = 1e-4) -> list[tuple[str, float, float]]:
        """Entries with a non-negligible share, as ``(name, coefficient, share)``."""
        rows = [
            (name, float(coefficient), float(share))
            for name, coefficient, share in zip(
                self.names, self.coefficients, self.scattering_fraction
            )
            if share > threshold
        ]
        return sorted(rows, key=lambda row: row[2], reverse=True)

    def by_phase(self, metric: str = "scattering") -> dict[str, float]:
        """Share summed over the entries of each phase.

        ``metric`` selects ``"scattering"`` (share of calculated integrated
        intensity) or ``"amplitude"`` (share of the fitted scale factors).
        """
        if metric == "scattering":
            values = self.scattering_fraction
        elif metric == "amplitude":
            values = self.amplitude_fraction
        else:
            raise ValueError("metric must be 'scattering' or 'amplitude'")
        totals: dict[str, float] = {}
        for phase, share in zip(self.phases, values):
            totals[phase] = totals.get(phase, 0.0) + float(share)
        return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))

    def weight_fractions(self, reference_intensity_ratios: dict[str, float]) -> dict[str, float]:
        """Weight fractions from per-phase reference intensity ratios.

        Each phase's scattering share is divided by its RIR and the results are
        renormalised.  Phases absent from the mapping are omitted, and the
        result is only as good as the calibration supplied.
        """
        shares = self.by_phase()
        missing = set(shares) - set(reference_intensity_ratios)
        if missing:
            raise KeyError(f"no reference intensity ratio given for {sorted(missing)}")
        scaled = {
            phase: share / reference_intensity_ratios[phase]
            for phase, share in shares.items()
            if share > 0.0
        }
        total = sum(scaled.values())
        if total <= 0:
            return {phase: 0.0 for phase in scaled}
        return {phase: value / total for phase, value in scaled.items()}

    def summary(self) -> str:
        amplitude = self.by_phase("amplitude")
        lines = [
            f"Rwp = {100.0 * self.r_wp:.2f}%   Rp = {100.0 * self.r_p:.2f}%",
            f"{'phase':<16s} {'scattering':>12s} {'amplitude':>11s}",
        ]
        for phase, share in self.by_phase().items():
            if share > 1e-4 or amplitude.get(phase, 0.0) > 1e-4:
                lines.append(
                    f"  {phase:<14s} {100.0 * share:11.2f}% {100.0 * amplitude.get(phase, 0.0):10.2f}%"
                )
        lines.append("Entries:")
        for name, _, share in self.active(threshold=1e-3):
            lines.append(f"  {name:<34s} {100.0 * share:6.2f}%")
        return "\n".join(lines)


def nnls_fit(
    measured: Pattern,
    library,
    mask: np.ndarray | None = None,
    background: BackgroundFit | None = None,
    range_two_theta: tuple[float, float] | None = None,
) -> FitResult:
    """Fit ``measured`` as a non-negative combination of library patterns.

    Parameters
    ----------
    library:
        A :class:`clayquant.library.PatternLibrary`.
    mask:
        Boolean mask selecting the points to fit; combined with
        ``range_two_theta`` when both are given.
    background:
        A fitted background to subtract from the measurement first.  When
        omitted the measurement is used as given, so it should already be
        background-corrected.
    range_two_theta:
        Angular range to fit, as ``(low, high)`` in degrees.
    """
    if len(library.entries) == 0:
        raise ValueError("the library is empty")

    two_theta = measured.two_theta
    raw = measured.intensity.astype(float)
    observed = background.subtract(two_theta, raw) if background is not None else raw

    selection = np.ones(two_theta.shape, dtype=bool)
    if range_two_theta is not None:
        low, high = range_two_theta
        selection &= (two_theta >= low) & (two_theta <= high)
    if mask is not None:
        selection &= np.asarray(mask, dtype=bool)
    # Library patterns are only defined on their own grid.
    selection &= (two_theta >= library.two_theta[0]) & (two_theta <= library.two_theta[-1])
    if selection.sum() < len(library.entries):
        raise ValueError(
            f"{selection.sum()} points selected for {len(library.entries)} library entries; "
            f"widen the fitting range or reduce the library"
        )

    design = library.matrix(two_theta[selection]).T  # (n_points, n_entries)
    target = observed[selection]

    # Counting statistics come from the *raw* counts: the variance of a point is
    # the number of photons recorded there, which background subtraction does not
    # reduce.  Weighting by the subtracted intensity instead would put the
    # largest weight on the flat regions between peaks, where the subtracted
    # value is near zero and only noise remains, and almost none on the peaks
    # that carry the information - which inflates Rwp and biases the solution
    # towards the background.
    variance = np.clip(raw[selection], 1.0, None)
    weights = 1.0 / variance
    root = np.sqrt(weights)
    coefficients, _ = nnls(design * root[:, None], target * root)

    calculated_selected = design @ coefficients
    calculated = np.zeros_like(observed)
    calculated[selection] = calculated_selected

    components: dict[str, np.ndarray] = {}
    for column, (name, coefficient) in enumerate(zip(library.names, coefficients)):
        if coefficient <= 0.0:
            continue
        curve = np.zeros_like(observed)
        curve[selection] = coefficient * design[:, column]
        components[name] = curve

    areas = np.array(
        [np.trapezoid(column, two_theta[selection]) for column in design.T]
    )
    contributions = coefficients * areas
    # What each entry weighs, up to one constant shared by all of them.
    normalizations = np.array([
        entry.normalization if entry.normalization else 1.0 for entry in library.entries
    ])
    unit_masses = np.array([
        0.0 if entry.unit_mass is None else entry.unit_mass for entry in library.entries
    ])
    # W proportional to S(ZMV): the volume is as necessary as the mass, because a
    # powder pattern carries 1/V**2 - one factor from how many cells fit in a
    # given volume of specimen, one from the density of reciprocal lattice
    # points.  An entry that does not know its volume contributes no mass rather
    # than a mass computed without it.
    unit_volumes = np.array([
        0.0 if entry.unit_volume is None else entry.unit_volume for entry in library.entries
    ])
    relative_mass = coefficients / normalizations * unit_masses * unit_volumes
    # A phase that cannot be weighed must not be quietly counted as weighing
    # nothing: the others would then be renormalised to 100% between them and
    # the answer would look complete while a mineral was missing from it.
    unweighable = bool(
        np.any((coefficients > 0.0) & ((unit_masses <= 0.0) | (unit_volumes <= 0.0)))
    )

    total = contributions.sum()
    shares = contributions / total if total > 0 else np.zeros_like(contributions)
    amplitude_total = coefficients.sum()
    amplitudes = (
        coefficients / amplitude_total if amplitude_total > 0 else np.zeros_like(coefficients)
    )

    denominator = float(np.sum(weights * target**2))
    residual = target - calculated_selected
    r_wp = (
        float(np.sqrt(np.sum(weights * residual**2) / denominator))
        if denominator > 0
        else float("nan")
    )
    total_observed = float(np.sum(np.abs(target)))
    r_p = float(np.sum(np.abs(residual)) / total_observed) if total_observed > 0 else float("nan")

    return FitResult(
        two_theta=two_theta,
        observed=observed,
        calculated=calculated,
        coefficients=coefficients,
        names=[entry.name for entry in library.entries],
        phases=[entry.phase for entry in library.entries],
        scattering_fraction=shares,
        amplitude_fraction=amplitudes,
        relative_mass=np.zeros_like(relative_mass) if unweighable else relative_mass,
        march_dollase=np.array([entry.march_dollase for entry in library.entries]),
        r_wp=r_wp,
        r_p=r_p,
        mask=selection,
        components=components,
        metadata={
            "measurement": measured.name,
            "n_points": int(selection.sum()),
            "n_entries": len(library.entries),
            "background_subtracted": background is not None,
            "unweighable_entries": [
                entry.name for entry, coefficient in zip(library.entries, coefficients)
                if coefficient > 0.0
                and (not entry.unit_mass or not entry.unit_volume)
            ],
        },
    )
