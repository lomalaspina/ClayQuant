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

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.optimize import nnls

from .background import BackgroundFit
from .pattern import Pattern

__all__ = [
    "FAMILY_SWEEPS",
    "MAX_EXHAUSTIVE_COMBINATIONS",
    "FamilySelection",
    "FitResult",
    "clay_families",
    "nnls_fit",
    "select_one_per_family",
]


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
    fraction: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    """Host-layer fraction of each entry, for the interstratified ones.

    1 for a discrete phase and for a pure end member, 0.8 for an 80/20 stack.
    Worth carrying through to the report: an entry at 0.99 is a stack of
    essentially pure host layers, and a result that calls it "I/S" without
    saying so reads as if smectite had been found.
    """
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


PHYSICAL_BOUNDS: dict[str, dict[float, str]] = {
    "fraction": {
        1.0: "the pure host end member",
        0.0: "the pure second component",
    },
    "march_dollase": {
        1.0: "a random powder, the least oriented a platy clay can be",
    },
}
"""Ends of a range that are physical limits rather than where the library stops.

The distinction changes what a caller can do about it.  A crystallite thickness
piled up on the largest value calculated means the data wanted a larger one and
the library should be extended.  A composition piled up on 1.00 means the data
wants the pure phase, and there is nothing beyond it to extend to - the fit has
given an answer, not run out of room.  Telling the user to span it further there
is advice they cannot take.
"""


def parameters_at_an_edge(
    result: "FitResult", library, threshold: float = 0.8
) -> list[str]:
    """Parameters whose fitted weight piles up on the lowest or highest value spanned.

    The fit chooses among pre-computed patterns, so it cannot report a value the
    library does not hold.  When a phase's coefficients collect on the end of a
    spanned range, the honest reading is not that the parameter equals that
    value but that the data wanted to go further and the library stopped it -
    the answer is a bound, not a measurement.  It is the same signal that marks
    a refined parameter sitting on its limit, and it is easy to miss in a table
    of numbers that all look like results.

    ``threshold`` is the share of a phase's coefficient that has to sit on the
    end value before it is reported, so a fit that merely includes the end value
    among others is not flagged.
    """
    axes = library.spanned()
    if not axes:
        return []
    weight: dict[tuple[str, float], float] = {}
    total: dict[str, float] = {}
    for entry, coefficient in zip(library.entries, result.coefficients):
        if coefficient <= 0.0:
            continue
        for name, value in (("march_dollase", entry.march_dollase),
                            ("thickness", entry.thickness),
                            ("csds_mean", entry.csds_mean),
                            ("fraction", entry.fraction)):
            if value is None:
                continue
            key = f"{entry.phase}/{name}"
            if key not in axes:
                continue
            weight[(key, float(value))] = weight.get((key, float(value)), 0.0) + float(coefficient)
            total[key] = total.get(key, 0.0) + float(coefficient)

    notes = []
    for key, values in axes.items():
        if total.get(key, 0.0) <= 0.0:
            continue
        for end, where in ((values[0], "lowest"), (values[-1], "highest")):
            share = weight.get((key, end), 0.0) / total[key]
            if share < threshold:
                continue
            # A phase name may itself carry a slash - "I/S", "C/S" - so the
            # parameter is what follows the *last* one.
            phase, name = key.rsplit("/", 1)
            meaning = next(
                (text for value, text in PHYSICAL_BOUNDS.get(name, {}).items()
                 if math.isclose(end, value)),
                None,
            )
            if meaning is not None:
                notes.append(
                    f"{phase}: {share * 100:.0f} % of the fit sits on {name} = {end:g}, which "
                    f"is {meaning} - the end of what can exist rather than the end of the "
                    f"library, so read it as the answer"
                )
            else:
                notes.append(
                    f"{phase}: {share * 100:.0f} % of the fit sits on {name} = {end:g}, the "
                    f"{where} value the library holds, so {name} is a bound here and not a "
                    f"measurement - span it further to find out where it wanted to go"
                )
    return notes


def nnls_fit(
    measured: Pattern,
    library,
    mask: np.ndarray | None = None,
    background: BackgroundFit | None = None,
    range_two_theta: tuple[float, float] | None = None,
    constraints: "list | tuple" = (),
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
    constraints:
        :class:`~clayquant.treatment.ExtraObservation` blocks - further things
        the same coefficients have to account for, appended to the
        least-squares system as extra rows.  This is how the air-dried mount
        restrains the expandable clays without being fitted itself
        (:mod:`clayquant.treatment`): a quantity measured on the *pair* of
        mounts cannot enter a fit of one of them any other way.

        Only the coefficients are affected.  R_wp, the shares and the
        residual are all computed from the pattern block alone, so they remain
        what they would be for an unconstrained fit of the same coefficients
        and stay comparable with one.
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
    # More entries than points is not an error, and refusing it was wrong.  This
    # library is meant to be over-complete: it spans layer spacing, orientation,
    # crystallite thickness and composition so that the fit can choose among
    # them, and spanning one more layer spacing per host was enough to put the
    # entry count above the number of points in a thirty degree window and stop
    # the fit running at all.  A non-negative least squares is well posed either
    # way, because the non-negativity is itself a constraint and the solution
    # sits on a face of the positive cone; what an under-determined one loses is
    # not the fit but the uniqueness of *which* entries carry a phase.  So it is
    # said out loud in the result rather than refused or passed over in silence.
    crowded = (
        f"{int(selection.sum())} points for {len(library.entries)} library entries, so which "
        f"entries carry a phase is not determined by the data. The fit and the phase totals "
        f"are still meaningful; read a fitted spacing or orientation as a range rather than a "
        f"value, or narrow the library."
        if selection.sum() < len(library.entries) else ""
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
    blocks = [design * root[:, None]]
    targets = [target * root]
    for constraint in constraints or ():
        if constraint.weight == 0.0:
            continue
        if constraint.n_entries != len(library.entries):
            raise ValueError(
                f"the {constraint.name!r} constraint describes "
                f"{constraint.n_entries} entries, not the library's "
                f"{len(library.entries)}"
            )
        rows, values = constraint.rows()
        blocks.append(rows)
        targets.append(values)
    coefficients, _ = nnls(np.vstack(blocks), np.concatenate(targets))

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

    result = FitResult(
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
        fraction=np.array([
            1.0 if entry.fraction is None else float(entry.fraction)
            for entry in library.entries
        ]),
        r_wp=r_wp,
        r_p=r_p,
        mask=selection,
        components=components,
        metadata={
            "measurement": measured.name,
            "n_points": int(selection.sum()),
            "n_entries": len(library.entries),
            "crowded": crowded,
            "background_subtracted": background is not None,
            "constraints": [
                {"name": item.name, "weight": float(item.weight),
                 "points": int(np.count_nonzero(
                     np.ones(item.target.shape) if item.point_weights is None
                     else item.point_weights))}
                for item in (constraints or ()) if item.weight != 0.0
            ],
            "unweighable_entries": [
                entry.name for entry, coefficient in zip(library.entries, coefficients)
                if coefficient > 0.0
                and (not entry.unit_mass or not entry.unit_volume)
            ],
        },
    )
    # Worked out after the result exists, since it needs the coefficients.
    result.metadata["parameters_at_an_edge"] = parameters_at_an_edge(result, library)
    return result


# --------------------------------------------------------------------------
# One pattern per clay family, as Clayfit chooses them
# --------------------------------------------------------------------------
#
# :func:`nnls_fit` lets every library entry take a coefficient.  That is right
# when the library is a set of candidates to choose among, and wrong when it is
# larger than the measurement: with 1751 entries over 1795 points the fit is
# very nearly square, and what an under-determined non-negative fit does with
# the extra freedom is not neutral.  Measured on one real glycol mount, thinning
# the clay library from 1755 columns to 145 raised quartz from 1.50 % of the
# pattern to 11.24 % - seven and a half times - and dropped the interstratified
# series from 50.1 % to 9.7 %.  The interstratified patterns are broad and there
# are 1440 of them, so between them they can absorb a sharp reflection that
# belongs to quartz or rutile, and the residual rewards them for it:
# R_wp was 14 points *better* in the fit that gives the less defensible answer
# (Sec. A.20 of the manual).
#
# Clayfit does not have the problem, and not because its library is small by
# accident: it fits *one member per family*.  Each clay family contributes
# exactly one pattern to the design matrix, its composition and orientation
# chosen by which choice fits best, and the accompanying minerals are present in
# every fit.  A clay family then has one coefficient and cannot be spent on a
# peak that is not the family's.
#
# This is that, transcribed.  The search, the objective, the stopping rule and
# the margin it reports are Clayfit's ``select_exclusive_family_fit``.

MAX_EXHAUSTIVE_COMBINATIONS = 1000
"""Combinations below which every one is tried rather than searched.

Clayfit's ``MAX_EXHAUSTIVE_FAMILY_COMBINATIONS``.  Above it the Cartesian
product is hopeless - ClayQuant's own library gives 2x10^10 - and the
coordinate search below is used instead.
"""

FAMILY_SWEEPS = 4
"""Passes of the coordinate search over the families, per starting point."""


@dataclass
class FamilySelection:
    """The fit that results from keeping one pattern per family."""

    result: "FitResult"
    """The fit itself, over the fixed entries and the chosen one per family.

    A :class:`FitResult` like any other, so :func:`clayquant.quantification.quantify`
    and the plots take it unchanged; its ``names`` are the entries that were kept.
    """

    chosen: dict[str, str]
    """Which entry each family contributed, by name."""

    objective: float
    """Weighted sum of squared residuals of the winning combination."""

    margins: dict[str, float]
    """How much worse the best *alternative* is for each family, relatively.

    ``(next best objective - best objective) / |best objective|``, over the
    combinations that differ from the winner in that family alone.  It is the
    number that says whether the choice was determined by the data: a margin of
    0.3 means the runner-up fitted 30 % worse and the family's composition is
    settled, and one of 0.001 means the data does not distinguish them and the
    reported spacing or orientation should be read as a range.  ``inf`` where a
    family had only one member to offer.
    """

    evaluations: int
    """Combinations actually fitted, which the search keeps far below the total."""

    exhaustive: bool
    """Whether every combination was tried rather than searched."""

    def summary(self) -> str:
        lines = [
            f"One pattern per family, chosen from {len(self.chosen)} families in "
            f"{self.evaluations} fits"
            f"{' (exhaustive)' if self.exhaustive else ''}: "
            f"Rwp = {100.0 * self.result.r_wp:.2f}%."
        ]
        for family in sorted(self.chosen):
            margin = self.margins.get(family, float("nan"))
            confidence = (
                "no alternative" if math.isinf(margin)
                else f"next best {100.0 * margin:.1f}% worse" if margin >= 0.02
                else f"next best only {100.0 * margin:.2f}% worse, so not determined"
            )
            lines.append(f"  {family:<16s} {self.chosen[family]}  ({confidence})")
        return "\n".join(lines)


def clay_families(library, fixed: set[str] | None = None) -> list[str]:
    """One family label per library entry, the way Clayfit labels them.

    A clay entry's family is its phase, so each clay phase contributes exactly
    one pattern; anything else - the accompanying minerals appended to the
    library - gets the empty label and is present in every fit, because a
    quartz is not an alternative to a rutile.

    ``fixed`` names phases to keep in every fit regardless.
    """
    from .quantification import _is_clay

    fixed = fixed or set()
    return [
        "" if entry.phase in fixed or not _is_clay(entry.phase) else entry.phase
        for entry in library.entries
    ]


def _family_starts(groups: list[np.ndarray]) -> list[tuple[int, ...]]:
    """Clayfit's four deterministic starting combinations.

    All-first, all-middle, all-last, and first/last alternating by position.
    Deterministic on purpose: a search that starts somewhere different each run
    gives a different composition each run, which is not a property an analysis
    may have.
    """
    return sorted({
        tuple(int(indices[0]) for indices in groups),
        tuple(int(indices[indices.size // 2]) for indices in groups),
        tuple(int(indices[-1]) for indices in groups),
        tuple(int(indices[0 if position % 2 == 0 else -1])
              for position, indices in enumerate(groups)),
    })


def select_one_per_family(
    measured: Pattern,
    library,
    families: list[str] | None = None,
    mask: np.ndarray | None = None,
    background: BackgroundFit | None = None,
    range_two_theta: tuple[float, float] | None = None,
    max_exhaustive: int = MAX_EXHAUSTIVE_COMBINATIONS,
    sweeps: int = FAMILY_SWEEPS,
    constraints: "list | tuple" = (),
) -> FamilySelection:
    """Fit with exactly one pattern from each family, as Clayfit does.

    ``families`` gives one label per library entry; an empty label means the
    entry is in every fit and a non-empty one names a mutually exclusive family.
    The default is :func:`clay_families`: one pattern per clay phase, every
    accompanying mineral always present.

    Every combination is tried when there are at most ``max_exhaustive`` of
    them.  Otherwise the search is Clayfit's: from each of four deterministic
    starting combinations, sweep the families in turn and move each to its best
    member with the others held, up to ``sweeps`` times or until nothing moves;
    then scan each family once more around the best combination found, so the
    reported margins rest on the winner's own neighbours rather than on whatever
    the sweeps happened to visit.

    The objective is the same weighted sum of squares :func:`nnls_fit`
    minimises, so the two are comparable and the choice is made on the quantity
    the fit is judged by.

    What this does *not* do is refine a coherent d-spacing scale per family, as
    Clayfit's ``select_exclusive_family_fit_with_lattice_scaling`` does.
    Clayfit needs it because its calculated profiles carry one fixed spacing
    each; ClayQuant's library spans the layer spacing explicitly - four
    thicknesses per host - so the same freedom is already in the family, chosen
    by the same search, and reported as the spacing of the entry it picked.
    """
    if len(library.entries) == 0:
        raise ValueError("the library is empty")

    labels = list(clay_families(library) if families is None else families)
    if len(labels) != len(library.entries):
        raise ValueError("families must give one label per library entry")

    order = list(dict.fromkeys(label for label in labels if label))
    if not order:
        # Nothing to choose between, so this is an ordinary fit.
        result = nnls_fit(measured, library, mask=mask, background=background,
                          range_two_theta=range_two_theta, constraints=constraints)
        return FamilySelection(result=result, chosen={}, objective=_objective_of(result),
                               margins={}, evaluations=1, exhaustive=True)

    as_array = np.asarray(labels, dtype=object)
    fixed = np.flatnonzero(as_array == "")
    groups = [np.flatnonzero(as_array == family) for family in order]

    # The design is built once and its columns sliced per trial: interpolating
    # 1751 patterns onto the grid for every one of thousands of trials is what
    # would make this unusable, and it is the same matrix every time.
    two_theta = measured.two_theta
    raw = measured.intensity.astype(float)
    observed = background.subtract(two_theta, raw) if background is not None else raw
    points = np.ones(two_theta.shape, dtype=bool)
    if range_two_theta is not None:
        low, high = range_two_theta
        points &= (two_theta >= low) & (two_theta <= high)
    if mask is not None:
        points &= np.asarray(mask, dtype=bool)
    points &= (two_theta >= library.two_theta[0]) & (two_theta <= library.two_theta[-1])
    if points.sum() < 2:
        raise ValueError("fewer than two points lie in both the range and the library")

    design = library.matrix(two_theta[points]).T
    root = np.sqrt(1.0 / np.clip(raw[points], 1.0, None))
    weighted = design * root[:, None]
    target = observed[points] * root

    # The constraint blocks are part of the objective the choice is made on -
    # a family chosen without them would be chosen on the glycol mount alone,
    # which is the freedom they exist to remove.
    extra = [item for item in (constraints or ()) if item.weight != 0.0]
    tried: dict[tuple[int, ...], float] = {}

    def evaluate(choice: tuple[int, ...]) -> float:
        key = tuple(int(index) for index in choice)
        cached = tried.get(key)
        if cached is not None:
            return cached
        columns = np.sort(np.concatenate((fixed, np.asarray(key, dtype=int))))
        block = weighted[:, columns]
        if extra:
            rows = [block]
            values = [target]
            for item in extra:
                item_rows, item_values = item.subset(columns).rows()
                rows.append(item_rows)
                values.append(item_values)
            block = np.vstack(rows)
            observed = np.concatenate(values)
        else:
            observed = target
        coefficients, _ = nnls(block, observed)
        residual = observed - block @ coefficients
        objective = float(residual @ residual)
        tried[key] = objective
        return objective

    total = 1
    for indices in groups:
        total *= int(indices.size)
    exhaustive = total <= max_exhaustive
    if exhaustive:
        for choice in product(*groups):
            evaluate(choice)
        best = min(tried, key=lambda key: (tried[key], key))
    else:
        optima: list[tuple[int, ...]] = []
        for start in _family_starts(groups):
            choice = list(start)
            for _sweep in range(max(1, sweeps)):
                moved = False
                for position, indices in enumerate(groups):
                    scores = []
                    for index in indices:
                        trial = list(choice)
                        trial[position] = int(index)
                        scores.append((evaluate(tuple(trial)), int(index)))
                    _, winner = min(scores)
                    if winner != choice[position]:
                        choice[position] = winner
                        moved = True
                if not moved:
                    break
            optima.append(tuple(choice))
        best = min(optima, key=lambda key: (tried[key], key))
        # One more scan around the winner, so every family's margin below is
        # measured against the winner's own alternatives.  Without it a margin
        # could be reported against a combination the sweeps never reached.
        for position, indices in enumerate(groups):
            for index in indices:
                trial = list(best)
                trial[position] = int(index)
                evaluate(tuple(trial))
        best = min(tried, key=lambda key: (tried[key], key))

    best_objective = tried[best]
    denominator = max(abs(best_objective), float(np.finfo(float).tiny))
    margins: dict[str, float] = {}
    for position, family in enumerate(order):
        alternatives = [
            objective for choice, objective in tried.items()
            if choice[position] != best[position]
        ]
        margins[family] = (
            max(0.0, (min(alternatives) - best_objective) / denominator)
            if alternatives else float("inf")
        )

    kept = np.sort(np.concatenate((fixed, np.asarray(best, dtype=int))))
    result = nnls_fit(measured, _library_subset(library, kept), mask=mask,
                      background=background, range_two_theta=range_two_theta,
                      constraints=[item.subset(kept) for item in extra])
    result.metadata["family_selection"] = {
        "families": len(order),
        "evaluations": len(tried),
        "exhaustive": exhaustive,
        "combinations": total,
        "margins": dict(margins),
    }
    chosen = {
        family: library.entries[int(index)].name
        for family, index in zip(order, best)
    }
    return FamilySelection(
        result=result,
        chosen=chosen,
        objective=best_objective,
        margins=margins,
        evaluations=len(tried),
        exhaustive=exhaustive,
    )


def _library_subset(library, indices: np.ndarray):
    """The library restricted to the given entries, metadata carried over."""
    kind = type(library)
    return kind(
        two_theta=library.two_theta,
        entries=[library.entries[int(index)] for index in indices],
        metadata=dict(library.metadata),
    )


def _objective_of(result: "FitResult") -> float:
    """The weighted sum of squares a :class:`FitResult` reached."""
    inside = result.mask
    observed = result.observed[inside]
    calculated = result.calculated[inside]
    weights = 1.0 / np.clip(np.abs(observed), 1.0, None)
    residual = observed - calculated
    return float(np.sum(weights * residual**2))
