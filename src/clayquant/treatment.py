"""What the three treatments say about the expandable clays, as a constraint.

The quantitative fit is made on the glycolated mount (Sec. 2.1), and on that one
mount a smectite-rich interstratified pattern and an illite-rich one are not far
apart: both put intensity near 8.8 deg, the difference between them is a broad
shoulder, and a fit free to choose can take either.  What distinguishes them is
not in that mount at all.  It is the *difference* between the air-dried and the
glycolated scan: an expandable interlayer takes glycol in and moves from about
15 to 17 A, so intensity appears near 5.2 deg that was not there before, and a
partly expandable stack moves its 001 within the 9 to 10 deg region.  A
non-expandable clay does neither.

So the air-dried mount is used here as a second observation of the same
coefficients, restrained rather than fitted: the expandable-bearing patterns,
with whatever scale the glycol fit gives them, must also account for the
intensity glycolation *added*, in the two windows where it would appear.

This follows Clayfit's ``treatment_evidence.build_air_eg_smectite_constraint``,
with its windows, its weight and its two acceptance tests, and departs from it
in one way that is the point of having it.  Clayfit adds the term only when the
evidence for expansion passes both tests, and leaves the fit unrestrained
otherwise.  Here it is added either way, because the two cases are the same
statement: the target is the measured expansion, and when there is no expansion
the target is zero and the term says the expandable component must be near
zero.  A specimen whose air-dried and glycolated scans are the same is evidence
*against* expandable clay, not an absence of evidence, and leaving the fit free
in that case is how a mount with no shift at all came back with 72 % of its clay
reported as illite/smectite.  Pass ``when="supported"`` for Clayfit's own
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .diagnostics import scale_to_reference
from .pattern import Pattern

__all__ = [
    "AIR_DRIED_STATES",
    "ExpandableBound",
    "LEAST_MOVING_HYDRATION",
    "PeakShift",
    "ShiftEvidence",
    "expandable_bound",
    "shift_evidence",
    "AIR_OBSERVATION_WEIGHT",
    "AirDriedObservation",
    "MINIMUM_AIR_OVERLAP",
    "air_dried_observation",
    "EXPANSION_CONSTRAINT_WEIGHT",
    "INTERSTRATIFIED_SHIFT_WINDOW",
    "MINIMUM_EXPANSION_GAIN",
    "MINIMUM_PROFILE_AGREEMENT",
    "SMECTITE_GLYCOL_001_WINDOW",
    "ExpansionEvidence",
    "ExtraObservation",
    "expandable_entries",
    "expansion_evidence",
    "BASAL_AGREEMENT_WEIGHT",
    "BASAL_AGREEMENT_WINDOW",
    "IMPLAUSIBLE_SCALE",
]

SMECTITE_GLYCOL_001_WINDOW = (4.75, 5.55)
"""Where a fully expandable 001 lands once it has taken up two glycol layers.

Clayfit's ``SMECTITE_2EG_001_WINDOW``.  A 17 A spacing is 5.20 deg at Cu
K-alpha1, and the window is wide enough for the 16.9 to 17.7 A range a
glycolated smectite actually shows.
"""

INTERSTRATIFIED_SHIFT_WINDOW = (9.10, 9.90)
"""Where a partly expandable stack's 001 moves to, which is the subtler signal.

Clayfit's ``INTERSTRATIFIED_SHIFT_WINDOW``.  An illite-rich interstratified
stack does not grow a reflection at 5 deg; its 001 moves by a few tenths of a
degree, and 9.1 to 9.9 deg is where it moves *from* and *to*.  Including it is
what lets the term say something about a 90/10 illite/smectite, which the 5 deg
window alone cannot.
"""

EXPANSION_CONSTRAINT_WEIGHT = 0.25
"""How strongly the expansion is weighted against the glycol pattern itself.

Clayfit's ``SMECTITE_SHIFT_CONSTRAINT_WEIGHT``.  A quarter, not one: the two
mounts are two preparations with their own textures and their own irradiated
volumes, so their intensities are not the same measurement even after the
quartz scaling, and a term that insisted they were would distort the phase that
carries the strongest lines.  The glycol scan stays the primary target.
"""

MINIMUM_EXPANSION_GAIN = 0.08
MINIMUM_PROFILE_AGREEMENT = 0.10
"""Clayfit's two tests for calling the expansion real.

The gain is the area glycolation added in the 5 deg window as a fraction of what
is there; the agreement is the cosine between that addition and the envelope of
the expandable patterns over both windows.  Failing either means the difference
between the mounts does not look like expansion - noise, a scaling error, a
displacement - and what is reported then is that the evidence is weak, not that
the specimen has no expandable clay.
"""


@dataclass
class ExtraObservation:
    """A second observation described by the same coefficients.

    Clayfit's ``LinearConstraint``.  ``target`` is what is observed,
    ``design`` how each fitted entry contributes to it, ``point_weights`` which
    points of it count and by how much, and ``weight`` how much the whole block
    counts against the pattern being fitted.  Appended to the least-squares
    system as extra rows, which is the only thing a linear constraint of this
    kind is.
    """

    target: np.ndarray
    design: np.ndarray
    weight: float = 1.0
    point_weights: np.ndarray | None = None
    name: str = "extra observation"

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=float)
        self.design = np.asarray(self.design, dtype=float)
        if self.target.ndim != 1:
            raise ValueError("the target must be one-dimensional")
        if self.design.shape[0] != self.target.size:
            raise ValueError("the design must have one row per target point")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("the weight must be finite and non-negative")
        if self.point_weights is not None:
            self.point_weights = np.asarray(self.point_weights, dtype=float)
            if self.point_weights.shape != self.target.shape:
                raise ValueError("the point weights must match the target")
            if not np.all(np.isfinite(self.point_weights)) or np.any(self.point_weights < 0.0):
                raise ValueError("the point weights must be finite and non-negative")

    @property
    def n_entries(self) -> int:
        return int(self.design.shape[1])

    def rows(self) -> tuple[np.ndarray, np.ndarray]:
        """The block as weighted rows to append to a least-squares system."""
        scale = self.weight * (
            np.ones(self.target.shape) if self.point_weights is None else self.point_weights
        )
        return self.design * scale[:, None], self.target * scale

    def subset(self, columns: np.ndarray) -> "ExtraObservation":
        """The same observation restricted to selected entries."""
        return ExtraObservation(
            target=self.target,
            design=self.design[:, np.asarray(columns, dtype=int)],
            weight=self.weight,
            point_weights=self.point_weights,
            name=self.name,
        )


@dataclass
class ExpansionEvidence:
    """What glycolation added, and the constraint built from it."""

    two_theta: np.ndarray
    expansion: np.ndarray
    """Intensity present in the glycol mount and not in the scaled air-dried one."""

    scale: float
    """Factor that put the air-dried mount on the glycol mount's scale."""

    gain: float
    """Area glycolation added in the 5 deg window, as a fraction of what is there."""

    agreement: float
    """Cosine between that addition and the expandable patterns' own envelope."""

    peak_two_theta: float | None
    """Where in the 5 deg window the addition is largest."""

    supported: bool
    """Whether both of Clayfit's tests passed."""

    constraint: "ExtraObservation | None"
    status: str
    entries: list[int] = field(default_factory=list)
    """Library entries the constraint acts on, by index."""

    @property
    def window(self) -> np.ndarray:
        """The diagnostic points, as a mask over :attr:`two_theta`."""
        return _windows(self.two_theta)


def _windows(two_theta: np.ndarray) -> np.ndarray:
    low, high = SMECTITE_GLYCOL_001_WINDOW
    primary = (two_theta >= low) & (two_theta <= high)
    low, high = INTERSTRATIFIED_SHIFT_WINDOW
    return primary | ((two_theta >= low) & (two_theta <= high))


def expandable_entries(library) -> list[int]:
    """Indices of the library entries that carry expandable interlayers.

    An entry's host fraction is the proportion of its layers that are *not*
    expandable, so anything below one carries some; the pure glycolated smectite
    is stored at zero.  A discrete illite, chlorite or kaolinite is one and is
    left out, which is what makes the constraint act on the expandable component
    rather than on the clays generally.
    """
    return [
        index for index, entry in enumerate(library.entries)
        if entry.fraction is not None and float(entry.fraction) < 1.0
    ]


def expansion_evidence(
    glycol: Pattern,
    air: Pattern,
    library,
    entries: list[int] | None = None,
    scale: float | None = None,
    weight: float = EXPANSION_CONSTRAINT_WEIGHT,
    when: str = "always",
    range_two_theta: tuple[float, float] | None = None,
) -> ExpansionEvidence:
    """Measure what glycolation added, and make a constraint of it.

    Both patterns must already be zero-corrected and background-subtracted.  The
    air-dried mount is put on the glycol mount's intensity scale on quartz 100,
    which the treatments do not touch, unless ``scale`` is given.

    ``when`` is ``"always"`` - add the term whichever way the evidence went - or
    ``"supported"``, which is Clayfit's behaviour and adds it only when both
    tests pass.  The module docstring says why the default is the other one.
    """
    if when not in ("always", "supported"):
        raise ValueError('when must be "always" or "supported"')
    indices = expandable_entries(library) if entries is None else list(entries)
    grid = np.asarray(library.two_theta, dtype=float)
    if range_two_theta is not None:
        low, high = range_two_theta
        grid = grid[(grid >= low) & (grid <= high)]
    diagnostic = _windows(grid)
    primary = (grid >= SMECTITE_GLYCOL_001_WINDOW[0]) & (grid <= SMECTITE_GLYCOL_001_WINDOW[1])

    if not indices:
        return ExpansionEvidence(
            grid, np.zeros_like(grid), 1.0, 0.0, 0.0, None, False, None,
            "No expandable pattern is in the library, so there is nothing to restrain.",
        )
    if int(np.count_nonzero(primary)) < 3 or int(np.count_nonzero(diagnostic)) < 6:
        return ExpansionEvidence(
            grid, np.zeros_like(grid), 1.0, 0.0, 0.0, None, False, None,
            f"The fitted range excludes the diagnostic windows "
            f"({SMECTITE_GLYCOL_001_WINDOW[0]:.2f}-{SMECTITE_GLYCOL_001_WINDOW[1]:.2f} and "
            f"{INTERSTRATIFIED_SHIFT_WINDOW[0]:.2f}-{INTERSTRATIFIED_SHIFT_WINDOW[1]:.2f} deg), "
            f"so the air-dried mount cannot restrain the expandable clays. Widen it to "
            f"{SMECTITE_GLYCOL_001_WINDOW[0]:.1f} deg.",
            indices,
        )

    if scale is None:
        try:
            scale = scale_to_reference(glycol, air)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return ExpansionEvidence(
                grid, np.zeros_like(grid), 1.0, 0.0, 0.0, None, False, None,
                f"The two mounts could not be put on a common intensity scale "
                f"({exc}), so their difference is not interpretable and the "
                f"expandable clays are unrestrained.",
                indices,
            )
    scale = float(scale)

    glycol_on_grid = np.interp(grid, glycol.two_theta, glycol.intensity)
    air_on_grid = scale * np.interp(grid, air.two_theta, air.intensity)
    # Only what glycolation *added*.  Intensity it removed is a different
    # statement - about the air-dried mount's own texture, or about the
    # scaling - and clipping keeps the target a statement about expansion.
    expansion = np.clip(glycol_on_grid - air_on_grid, 0.0, None)

    here = float(np.trapezoid(np.clip(glycol_on_grid[primary], 0.0, None), grid[primary]))
    added = float(np.trapezoid(expansion[primary], grid[primary]))
    gain = added / here if here > 0.0 else 0.0

    columns = np.clip(library.matrix(grid).T[:, indices], 0.0, None)
    peaks = np.max(columns, axis=0)
    normalised = np.divide(columns, peaks[None, :], out=np.zeros_like(columns),
                           where=peaks[None, :] > 0.0)
    envelope = np.max(normalised, axis=1)
    denominator = float(np.linalg.norm(expansion[diagnostic])
                        * np.linalg.norm(envelope[diagnostic]))
    agreement = (float(np.dot(expansion[diagnostic], envelope[diagnostic]) / denominator)
                 if denominator > 0.0 else 0.0)
    peak_two_theta = (float(grid[primary][int(np.argmax(expansion[primary]))])
                      if np.any(expansion[primary] > 0.0) else None)
    supported = gain >= MINIMUM_EXPANSION_GAIN and agreement >= MINIMUM_PROFILE_AGREEMENT

    design = np.zeros((grid.size, len(library.entries)), dtype=float)
    design[:, indices] = library.matrix(grid).T[:, indices]
    constraint = None
    if supported or when == "always":
        constraint = ExtraObservation(
            target=expansion,
            design=design,
            weight=float(weight),
            point_weights=diagnostic.astype(float),
            name="air-to-glycol expansion",
        )

    if supported:
        status = (
            f"Glycolation added {100.0 * gain:.1f}% of the intensity in the "
            f"{SMECTITE_GLYCOL_001_WINDOW[0]:.2f}-{SMECTITE_GLYCOL_001_WINDOW[1]:.2f} deg "
            f"window, peaking at {peak_two_theta:.2f} deg, and what it added agrees with "
            f"the expandable patterns' own shape to {agreement:.2f}. The "
            f"{len(indices)} expandable entries are restrained to account for it."
        )
    elif constraint is not None:
        status = (
            f"The two mounts show no expansion worth the name: glycolation added "
            f"{100.0 * gain:.1f}% of the intensity in the "
            f"{SMECTITE_GLYCOL_001_WINDOW[0]:.2f}-{SMECTITE_GLYCOL_001_WINDOW[1]:.2f} deg "
            f"window (it takes {100.0 * MINIMUM_EXPANSION_GAIN:.0f}%) and what little it "
            f"added agrees with the expandable patterns' shape to only {agreement:.2f} "
            f"(it takes {MINIMUM_PROFILE_AGREEMENT:.2f}). That is evidence against "
            f"expandable clay, not an absence of it, so the {len(indices)} expandable "
            f"entries are restrained to the little that is there rather than left free."
        )
    else:
        status = (
            f"The expansion evidence was weak (gain {100.0 * gain:.1f}%, agreement "
            f"{agreement:.2f}) and no constraint was added, so the expandable clays are "
            f"free. This is Clayfit's behaviour; pass when=\"always\" to restrain them "
            f"to the measured expansion instead."
        )

    return ExpansionEvidence(
        two_theta=grid,
        expansion=expansion,
        scale=scale,
        gain=gain,
        agreement=agreement,
        peak_two_theta=peak_two_theta,
        supported=supported,
        constraint=constraint,
        status=status,
        entries=indices,
    )


# --------------------------------------------------------------------------
# The 001 region, where an illite-rich stack is actually distinguished
# --------------------------------------------------------------------------

BASAL_AGREEMENT_WINDOW = (7.8, 9.4)
"""Where the clay 001 sits, and where the two mounts have to agree, in degrees.

Not Clayfit's, and added because its two windows cannot do this job on
ClayQuant's library.  Measured over the interstratified series, the area each
composition puts in Clayfit's 4.75-5.55 deg window runs 0.709 at 80 % smectite
down to 0.196 at 10 % - and back *up* to 0.208 at none at all, so between an
illite-rich stack and pure illite that window has no signal to give.  Its
9.10-9.90 deg window carries 0.005 to 0.025 for every composition, because
ClayQuant's 001s are at 8.2 to 9.1 deg and not there.

What does vary monotonically is the 001 itself: 0.001 at 80 % smectite rising
without a break to 0.622 at none.  So this is the window that separates a 90/10
illite/smectite from an illite, and the one an operator looks at when they say
the air-dried and glycolated scans show no shift.
"""

BASAL_AGREEMENT_WEIGHT = 0.25
"""As Clayfit weights its own extra observation: a restraint, not a constraint."""

IMPLAUSIBLE_SCALE = 2.5
"""Beyond this factor between the mounts, their difference is not interpretable.

The two mounts are separate preparations and their quartz areas differ, but a
factor of this size means one of the two quartz measurements failed rather than
that the specimen changed.  Measured: eight of nine real mounts scale within
1.2, and the ninth came out at 3.3.
"""


def basal_agreement(
    glycol: Pattern,
    air: Pattern,
    library,
    scale: float | None = None,
    weight: float = BASAL_AGREEMENT_WEIGHT,
    window: tuple[float, float] = BASAL_AGREEMENT_WINDOW,
    range_two_theta: tuple[float, float] | None = None,
) -> tuple["ExtraObservation | None", str]:
    """Require the fitted coefficients to explain the air-dried 001 as well.

    **Measured, and it does not work.**  On a real glycol mount this restraint
    moved the interstratified phase from 50.1 % of the pattern to 69.8 % - the
    wrong way - and R_wp from 58.0 to 80.9 %, while dropping quartz out of the
    fit altogether.  The reason is that the two mounts differ over the 001
    window for reasons that are not expansion: they are separate preparations
    with their own textures, and 96 points of restraint at a quarter weight is
    enough to make the fit satisfy the air-dried mount by inflating whichever
    entry is broad enough to cover both.  It is kept, unused and not wired into
    the interface, because the negative result is worth having where someone
    can find it: the idea is the obvious one and it is wrong.

    What follows is what it was meant to do.

    This is the statement an operator makes when they say there is no shift
    between the air-dried and the glycolated scan, written so that a fit can be
    held to it.  Over the 001 window the same coefficients, with the same
    scales, must account for both mounts.  A clay whose interlayer does not take
    up glycol satisfies that exactly - its 001 is in the same place in both -
    and one whose interlayer does cannot, because the library entry that
    describes it in the glycol mount has its 001 somewhere the air-dried mount
    has none.  So an expandable composition is admitted only to the extent the
    glycol mount needs it *and* the air-dried mount tolerates it.

    Every entry is in the design, not only the expandable ones: quartz, illite
    and chlorite all contribute in that window and have to be allowed to, and it
    is only against them that the expandable ones are being judged.

    Returns the observation and a sentence about it, or ``(None, why not)``.
    """
    grid = np.asarray(library.two_theta, dtype=float)
    if range_two_theta is not None:
        low, high = range_two_theta
        grid = grid[(grid >= low) & (grid <= high)]
    inside = (grid >= float(window[0])) & (grid <= float(window[1]))
    if int(np.count_nonzero(inside)) < 8:
        return None, (
            f"The fitted range excludes the 001 window "
            f"({window[0]:.1f}-{window[1]:.1f} deg), so the air-dried mount cannot "
            f"say anything about the interstratified composition."
        )
    if scale is None:
        try:
            scale = scale_to_reference(glycol, air)
        except Exception as exc:  # noqa: BLE001
            return None, (f"The mounts could not be put on a common intensity scale "
                          f"({exc}), so their 001 regions are not comparable.")
    scale = float(scale)
    if not 1.0 / IMPLAUSIBLE_SCALE <= scale <= IMPLAUSIBLE_SCALE:
        return None, (
            f"The air-dried mount scales to the glycolated one by {scale:.2f} on "
            f"quartz, which is past the factor of {IMPLAUSIBLE_SCALE:g} that two "
            f"preparations of one specimen plausibly differ by. One of the two quartz "
            f"measurements has failed, so the 001 comparison is not made - it would "
            f"otherwise restrain the composition on a scaling error."
        )

    target = scale * np.interp(grid, air.two_theta, air.intensity)
    design = library.matrix(grid).T
    observation = ExtraObservation(
        target=target,
        design=design,
        weight=float(weight),
        point_weights=inside.astype(float),
        name="air-dried 001 agreement",
    )
    return observation, (
        f"The fitted coefficients are also held to the air-dried mount over "
        f"{window[0]:.1f}-{window[1]:.1f} deg, at {int(np.count_nonzero(inside))} points, "
        f"with the air-dried intensities scaled by {scale:.2f} on quartz. An expandable "
        f"composition is admitted only so far as both mounts allow it, which is what "
        f"'no shift between air and glycol' means as a restraint on the fit."
    )


# --------------------------------------------------------------------------
# The movement between the mounts, measured, and the bound it puts on
# expandable clay
# --------------------------------------------------------------------------
#
# The air-dried mount is not fitted, and the reason is the reason glycol
# solvation exists.  An air-dried smectite interlayer holds zero, one, two or
# three layers of water - roughly 9.6, 12.4, 15 and 18 A - depending on the
# exchangeable cation and on the humidity of the room, and a real specimen
# carries several of those states at once, interstratified within single
# crystallites.  There is no single air-dried structure to calculate.
# Glycolation exists to remove exactly that variability: it drives the
# interlayer to a reproducible two-layer complex near 16.9 A whatever the
# cation, and that is what makes the glycolated mount the one a pattern can be
# calculated for.
#
# What the air-dried mount gives instead is a measurement: whether the basal
# reflections *moved*.  That is read off the two scans without calculating
# anything, and it is the classical test - a clay whose 001 does not move on
# glycolation is not expandable.  Below, the movement is measured, and then
# turned into a bound on how much expandable layer the specimen can contain
# without having shown it.

AIR_DRIED_STATES = {
    "0 water layers": 9.6,
    "1 water layer": 12.4,
    "2 water layers": 15.0,
    "3 water layers": 18.0,
}
"""Basal spacing in A of a smectite at each hydration state.

Not a set of models to fit - a list of the reasons an air-dried mount cannot be
fitted.  Which of these a specimen shows depends on the exchangeable cation and
the humidity, and it usually shows more than one at once.
"""

SHIFT_SEARCH_WINDOW = 0.30
"""How far from a glycol peak to look for its air-dried partner, in degrees."""

SHIFT_RANGE = (4.0, 16.0)
"""Where expansion shows, and where the evidence is therefore read.

A glycolated smectite puts its 001 at 5.2 degrees, an illite-rich I/S moves its
10 A 001 and 5 A 002, and a chlorite/smectite moves its 14 A 001 - all of it
below about 16 degrees.  Above that the clay basal orders are weak and sit among
the accompanying minerals: on one real specimen the strongest apparent
"movement" in a 4-30 degree window was 0.036 degrees at 3.24 A, in the middle of
albite and sekaninaite, on a specimen whose clay basal reflections had not moved
by more than 0.006.  Reading the evidence where the evidence is avoids that.
"""

MINIMUM_SHIFT_SHARE = 0.05
"""A peak below this share of the strongest in the window is not read."""

MINIMUM_SIGNAL_TO_NOISE = 8.0
"""Prominence-to-noise a peak needs before its position counts as evidence.

A peak this test cannot place is not evidence about anything, and the failure
it guards against is specific: two maxima that are only noise, one in each
mount, land a couple of tenths of a degree apart and are matched to each other,
and a specimen with no expandable clay is reported as having some.
"""

MINIMUM_PEAK_SHARE = 0.05
"""A peak below this share of the strongest is too weak to place reliably."""


@dataclass
class PeakShift:
    """One reflection, and how far glycolation moved it."""

    glycol_two_theta: float
    air_two_theta: float
    displacement: float
    """Glycol minus air, in degrees, before the mount offset is removed."""

    height: float
    uncertainty: float
    """How well this pair of peaks can be placed, in degrees.

    A strong sharp line is located to a small fraction of its width; a weak
    broad one is barely located at all.  Comparing every displacement against
    one number instead treats the difference between two noise maxima as
    evidence, which it is not.
    """

    d_glycol: float
    d_air: float

    @property
    def moved(self) -> bool:
        return abs(self.displacement) > 2.0 * self.uncertainty


@dataclass
class ShiftEvidence:
    """What the two mounts say about movement, measured rather than calculated."""

    shifts: list[PeakShift]
    offset: float
    """Systematic displacement between the mounts, from reflections that cannot move."""

    largest: float
    """Largest displacement that exceeded the peak's own placement, in degrees.

    Zero when nothing moved, which is the point - but it is not the largest
    displacement *measured*, and reporting it as though it were reads as if no
    measurement had been made.  :attr:`measured` is that number.
    """

    uncertainty: float
    """How large a displacement the measurement could have hidden, in degrees."""

    significant: bool

    measured: float = 0.0
    """Largest displacement measured, whether or not it means anything.

    This is what the bound is drawn from: the specimen is allowed whatever
    expandable content could hide behind the movement the scans actually show,
    which is this rather than zero.
    """

    appeared: list[float] = field(default_factory=list)
    """Reflections the glycolated mount has and the air-dried one does not.

    A peak that appears on glycolation is the plainest evidence there is, and
    it is not a displacement - it has no air-dried partner to be displaced
    from.  Counting it as one is how a shoulder becomes a tenth of a degree of
    movement that never happened.
    """

    status: str = ""

    @property
    def moved(self) -> bool:
        """Whether anything moved by more than the measurement can account for."""
        return self.significant


def _peaks(pattern: Pattern, low: float, high: float, minimum_share: float,
           separation: float = 0.35,
           floor: float | None = None) -> list[tuple[float, float, float]]:
    """Local maxima of one scan, as ``(two_theta, height)``, strongest first.

    The peaks come from :func:`clayquant.detection.treatment_peaks`, which finds
    them Clayfit's way - a prominence of four times the *local* noise or a
    quarter of a per cent of the pattern.  A flat threshold does not work here
    and the failure is instructive: at 5.2 degrees, where a glycolated smectite
    would put its 17 A 001, one real specimen's background-subtracted counts
    wander between 150 and 440 with no structure at all, and taking 5 % of the
    pattern maximum as a floor picked the largest of them as a peak - reporting
    a 17 A reflection that is not there, on the one specimen whose whole point
    was that it has no expandable clay.

    Positions are then refined by a parabola through the maximum and its
    neighbours, which places a peak to a fraction of the step.  That matters
    because the displacements being measured here are themselves smaller than
    one step.
    """
    from .detection import treatment_peaks

    angles = np.asarray(pattern.two_theta, dtype=float)
    values = np.asarray(pattern.intensity, dtype=float)
    inside = (angles >= low) & (angles <= high)
    if int(np.count_nonzero(inside)) < 3:
        return []
    if floor is None:
        floor = minimum_share * (float(np.max(values[inside])) or 1.0)

    peaks = treatment_peaks(pattern)
    widths = np.asarray(peaks.width, dtype=float)
    prominences = np.asarray(peaks.prominence, dtype=float)
    found: list[tuple[float, float, float]] = []
    for order, centre in enumerate(np.asarray(peaks.two_theta, dtype=float)):
        if not (low <= centre <= high):
            continue
        position = int(np.argmin(np.abs(angles - centre)))
        if position == 0 or position == angles.size - 1:
            continue
        y0, y1, y2 = values[position - 1], values[position], values[position + 1]
        if y1 < floor:
            continue
        refined = float(angles[position])
        denominator = y0 - 2.0 * y1 + y2
        if denominator != 0.0:
            delta = 0.5 * (y0 - y2) / denominator
            if abs(delta) <= 1.0:
                refined += float(delta) * float(angles[position + 1] - angles[position])
        # How well this peak can be placed.  A strong sharp line is located to
        # a small fraction of its width; a weak broad one is not located at
        # all, and treating the two alike is how the difference between two
        # noise maxima becomes a tenth of a degree of "movement".
        _, noise = _height_at(pattern, refined, 0.5 * separation)
        width = float(widths[order]) if order < widths.size else separation
        # Prominence, not height: a bump on a 300-count background has a fine
        # ratio to the noise if its own height is used, and none at all once
        # the background under it is taken off.  The difference decides whether
        # two noise maxima become a tenth of a degree of "movement".
        prominence = float(prominences[order]) if order < prominences.size else float(y1)
        ratio = prominence / max(noise, 1.0)
        if ratio < MINIMUM_SIGNAL_TO_NOISE:
            continue
        placed = max(width / (2.0 * max(ratio, 1.0)), 0.0)
        found.append((refined, float(y1), placed))
    found.sort(key=lambda item: -item[1])
    kept: list[tuple[float, float, float]] = []
    for item in found:
        if all(abs(item[0] - taken[0]) > separation for taken in kept):
            kept.append(item)
    return kept


def _match(glycol_peaks, air_peaks, window):
    """Pair each glycol peak with at most one air peak, strongest first.

    One-to-one matters: without it a shoulder that glycolation *created* - with
    no counterpart in the air-dried mount at all - is matched to whichever
    neighbouring peak is nearest, and reported as a displacement of a tenth of a
    degree that never happened.  That is the difference between reading
    movement off these scans and inventing it.
    """
    taken: set[int] = set()
    pairs: list[tuple[float, float, float, float]] = []
    unmatched: list[tuple[float, float, float]] = []
    for centre, height, placed in glycol_peaks:
        best, distance = None, window
        for index, other in enumerate(air_peaks):
            if index in taken:
                continue
            if abs(other[0] - centre) <= distance:
                best, distance = index, abs(other[0] - centre)
        if best is None:
            unmatched.append((centre, height, placed))
            continue
        taken.add(best)
        # The pair can be separated no better than the worse-placed of the two.
        together = float(np.hypot(placed, air_peaks[best][2]))
        pairs.append((centre, air_peaks[best][0], height, together))
    return pairs, unmatched


def _intensity_scale_rough(glycol: Pattern, air: Pattern, reference, window) -> float:
    """How much brighter the glycol mount is, from the reflections that cannot move.

    Needed before any peak is chosen, to put one detection threshold on both
    scans; the pairs give a better estimate afterwards.
    """
    ratios = []
    for line in reference:
        here, _ = _height_at(glycol, line, window)
        there, _ = _height_at(air, line, window)
        if here > 0.0 and there > 0.0:
            ratios.append(here / there)
    return float(np.median(ratios)) if ratios else 1.0


def _intensity_scale(glycol: Pattern, air: Pattern, pairs) -> float:
    """How much brighter the glycol mount is, from reflections present in both.

    Two mounts of one specimen are separate preparations and rarely carry the
    same amount of material, so a peak is not "missing" from the air-dried
    mount merely because it is smaller there.
    """
    ratios = []
    for here, there, height, _ in pairs:
        other, _ = _height_at(air, there, 0.05)
        if other > 0.0 and height > 0.0:
            ratios.append(height / other)
    return float(np.median(ratios)) if ratios else 1.0


def _height_at(pattern: Pattern, centre: float, window: float) -> tuple[float, float]:
    """The largest value near ``centre``, and the local scatter around it."""
    angles = np.asarray(pattern.two_theta, dtype=float)
    values = np.asarray(pattern.intensity, dtype=float)
    near = (angles >= centre - window) & (angles <= centre + window)
    if not np.any(near):
        return 0.0, 1.0
    wide = (angles >= centre - 10.0 * window) & (angles <= centre + 10.0 * window)
    local = values[wide] if int(np.count_nonzero(wide)) > 8 else values[near]
    noise = float(np.median(np.abs(np.diff(local)))) / 0.9539 if local.size > 2 else 1.0
    return float(np.max(values[near])), max(noise, 1.0)


def _is_new(glycol: Pattern, air: Pattern, centre: float, height: float,
            scale: float, window: float, sigmas: float = 4.0) -> bool:
    """Whether the air-dried mount really lacks the intensity, not just the peak."""
    there, noise = _height_at(air, centre, 0.5 * window)
    expected = height / max(scale, 1e-9)
    # It appeared only if the air-dried mount falls short of what this peak
    # would be there by more than the scatter of the air-dried scan itself.
    return (expected - there) > sigmas * noise


def shift_evidence(
    air: Pattern,
    glycol: Pattern,
    range_two_theta: tuple[float, float] = SHIFT_RANGE,
    window: float = SHIFT_SEARCH_WINDOW,
    minimum_share: float = MINIMUM_SHIFT_SHARE,
    reference: tuple[float, ...] = (20.859, 26.640),
    wavelength: float = 1.540596,
) -> ShiftEvidence:
    """Measure how far glycolation moved each basal reflection.

    Nothing is calculated and no structure is assumed.  The strong peaks of the
    glycolated mount in ``range_two_theta`` are located, each is matched to the
    nearest maximum of the air-dried mount, and the difference is the
    displacement.  The quartz lines, which no treatment moves, give the
    systematic offset between the two mounts and the scatter of that offset
    gives the uncertainty - so the question the result answers is not "did
    anything move" but "did anything move by more than these two scans can
    disagree by anyway".

    This is the classical test, and it is the one the specimen actually
    supports: a clay whose 001 does not move on glycolation is not expandable,
    whatever hydration state its interlayer would have been in.
    """
    angles = np.asarray(glycol.two_theta, dtype=float)
    low, high = range_two_theta
    if int(np.count_nonzero((angles >= low) & (angles <= high))) < 8:
        return ShiftEvidence([], 0.0, 0.0, float("inf"), False, 0.0, [],
                             "The two mounts do not overlap over the basal region, so "
                             "no movement could be measured.")
    step = float(np.median(np.diff(angles))) if angles.size > 1 else 0.01

    # Reflections that cannot move give the offset between the mounts and, from
    # their spread, how far apart the two scans place the same line anyway.
    offsets: list[float] = []
    for line in reference:
        if not (angles[0] <= line <= angles[-1]):
            continue
        here = _peaks(glycol, line - window, line + window, 0.0, separation=window)
        there = _peaks(air, line - window, line + window, 0.0, separation=window)
        if here and there:
            offsets.append(here[0][0] - there[0][0])
    offset = float(np.median(offsets)) if offsets else 0.0
    uncertainty = max(
        float(np.max(np.abs(np.asarray(offsets) - offset))) if len(offsets) > 1 else step,
        step,
    )

    # One threshold for both mounts, scaled onto each.  Applied separately, a
    # reflection that sits just above it in one scan and just below it in the
    # other is reported as having appeared on glycolation, which is a statement
    # about the threshold and not about the specimen.
    inside = (angles >= low) & (angles <= high)
    reference_floor = minimum_share * (
        float(np.max(np.asarray(glycol.intensity, dtype=float)[inside])) or 1.0)
    glycol_peaks = [item for item in _peaks(glycol, low, high, minimum_share,
                                            floor=reference_floor)
                    if all(abs(item[0] - line) > window for line in reference)]
    rough = _intensity_scale_rough(glycol, air, reference, window)
    air_peaks = [item for item in _peaks(air, low, high, minimum_share,
                                         floor=reference_floor / max(rough, 1e-9))
                 if all(abs(item[0] - line) > window for line in reference)]
    pairs, candidates = _match(glycol_peaks, air_peaks, window)

    # A reflection that *appeared* on glycolation is the strongest claim this
    # can make, so it takes more than the air-dried mount failing to show a
    # local maximum there: the air-dried mount must be missing the intensity.
    # Without this the test reports a 17 A smectite 001 on a specimen that has
    # none, because at 5.2 degrees the counts wander by more than the feature
    # being called a peak and one of them is always the largest.
    scale = _intensity_scale(glycol, air, pairs)
    appeared = [
        (centre, height) for centre, height, _placed in candidates
        if _is_new(glycol, air, centre, height, scale, window)
    ]

    shifts = [
        PeakShift(
            glycol_two_theta=here,
            air_two_theta=there,
            displacement=float(here - there - offset),
            height=float(height),
            uncertainty=max(float(together), uncertainty),
            d_glycol=float(wavelength / (2.0 * np.sin(np.radians(here / 2.0)))),
            d_air=float(wavelength / (2.0 * np.sin(np.radians(there / 2.0)))),
        )
        for here, there, height, together in pairs
    ]
    moved = [item for item in shifts
             if abs(item.displacement) > 2.0 * item.uncertainty]
    # The largest movement that is actually a movement; a displacement smaller
    # than its own peak can be placed is not one.
    largest = max((abs(item.displacement) for item in moved), default=0.0)
    measured = max((abs(item.displacement) for item in shifts), default=0.0)
    new_peaks = [centre for centre, _ in appeared]
    significant = bool(moved) or bool(new_peaks)

    if not shifts and not new_peaks:
        status = ("No basal reflection was strong enough to place in both mounts, so "
                  "the air-dried mount says nothing about expandable clay here.")
    elif significant:
        parts = []
        if moved:
            worst = max(moved, key=lambda item: abs(item.displacement))
            parts.append(
                f"glycolation moved {len(moved)} of {len(shifts)} basal reflections, the "
                f"largest at {worst.glycol_two_theta:.2f} deg ({worst.d_air:.2f} A air-dried "
                f"to {worst.d_glycol:.2f} A glycolated, {worst.displacement:+.3f} deg)"
            )
        if new_peaks:
            parts.append(
                f"{len(new_peaks)} reflection{'s' if len(new_peaks) != 1 else ''} appeared "
                f"on glycolation that the air-dried mount does not have "
                f"({', '.join(f'{x:.2f}' for x in new_peaks[:3])} deg)"
            )
        status = ("There is expandable clay: " + "; ".join(parts)
                  + f". The two scans place an unmoving quartz line to {uncertainty:.3f} deg.")
    else:
        status = (
            f"Nothing moved. Across {len(shifts)} basal reflections the largest displacement "
            f"is {measured:.3f} deg, within what those peaks can be placed to, and against "
            f"{uncertainty:.3f} deg of disagreement between the "
            f"mounts on the quartz lines, which no treatment moves, and no reflection "
            f"appeared on glycolation. A clay whose basal series does not move on "
            f"glycolation is not expandable, whatever hydration state its interlayer "
            f"would have been in."
        )
    return ShiftEvidence(shifts, offset, float(largest), float(uncertainty),
                         bool(significant), float(measured), new_peaks, status)


# --------------------------------------------------------------------------
# The air-dried mount as a second observation of the same coefficients
# --------------------------------------------------------------------------

AIR_OBSERVATION_WEIGHT = 1.0
"""How much the air-dried mount counts against the mount being fitted.

One, because it is a measurement of the same specimen of the same quality, and
down-weighting it would be a statement that it is less trustworthy rather than
a statement about clay.  Lower it to let the glycol mount win where the two
disagree; zero is the same as not passing the constraint at all.
"""

MINIMUM_AIR_OVERLAP = 0.5
"""Fraction of the fitted range the air-dried mount must cover to be usable."""


@dataclass
class AirDriedObservation:
    """The air-dried mount, and the constraint that carries it into the fit."""

    constraint: "ExtraObservation | None"
    scale: float
    """Factor that put the air-dried mount on the fitted mount's scale."""

    shift: float
    """Zero shift of the air-dried mount against the fitted one, in degrees."""

    expandable: int
    """Library entries that predict a change between the two treatments."""

    status: str


def air_dried_observation(
    air: Pattern,
    fitted: Pattern,
    library,
    range_two_theta: tuple[float, float] | None = None,
    scale: float | None = None,
    shift: float | None = 0.0,
    weight: float = AIR_OBSERVATION_WEIGHT,
    counts: np.ndarray | None = None,
) -> AirDriedObservation:
    """Fit the glycol mount and the air-dried mount with one set of coefficients.

    This is the measurement that decides whether an expandable clay is there,
    and it is not the one Clayfit makes.  Clayfit - and ClayQuant before this -
    looks for intensity glycolation *added* near 5.2 deg, where a fully
    expandable smectite puts its 17 A 001.  That works for a smectite-rich
    phase and is powerless against an illite-rich one: an I/S at 3% expandable
    layers has no peak there to add, so the window sees nothing whichever way
    the truth goes, and the fit is free to use as much of it as it likes.  On a
    real specimen that freedom produced 72% I/S from a pair of mounts that a
    reader could see were the same pattern.

    What an illite-rich I/S does do on glycolation is change the *whole* 00l
    series, because interstratification is not additive: putting even a few
    layers of a different spacing into the stack breaks the coherence of every
    order (Mering's principle).  Measured on ClayQuant's own library, going from
    the air-dried to the glycol state changes an I/S at 2% expandable layers by
    44% of peak height at the 10 A 001 and by 33% at the 3.32 A 003, while it
    changes Clayfit's 5.2 deg window by 0.4%.  The evidence is enormous, and it
    is nowhere near where anyone was looking.

    So this uses it directly.  Each library entry contributes its glycol pattern
    to the mount being fitted and its stored air-dried counterpart - the same
    composition with the smectite interlayer collapsed - to this block, with the
    *same* coefficient.  An entry that claims a share of the glycol mount
    thereby predicts a definite, different air-dried mount, and if the specimen
    does not show it the fit pays for it here.  A phase that does not expand
    contributes the same pattern to both blocks and is unaffected, so this
    restrains the expandable clays and nothing else - which is what
    "the evidence of peak shift should restrict the smectite family" asks for.

    The two mounts are separate preparations, so they are put on a common
    intensity scale on quartz 100, which no treatment touches, and on a common
    angular scale by the quartz zero shift.  Both are measured unless given.

    Parameters
    ----------
    air:
        The air-dried mount, zero-corrected and background-subtracted the same
        way as the mount being fitted.
    fitted:
        The mount being fitted - the glycol one - which the air-dried mount is
        scaled against on quartz 100.
    shift:
        Zero shift of the air-dried mount against the fitted one, in degrees,
        signed as :func:`clayquant.calibration.apply_zero_error` signs it - the
        error to be *subtracted* from the measured angles, not the correction to
        be added.  Zero by default, because ClayQuant zero-corrects each mount
        as it is loaded and correcting it twice would misalign the two blocks;
        ``None`` measures it from the quartz lines instead, for a mount that has
        not been corrected.

        Measuring is not the safe default it looks.  On a specimen with no
        quartz in it, :func:`~clayquant.detection.quartz_zero_shift` will match
        a clay reflection to a quartz line and read a shift of several tenths of
        a degree off it - on a test case here, -0.40 deg from a single peak -
        and a misaligned air-dried block does not restrain the expandable clays,
        it wrecks the fit: 2.24% became 37.91%.  So only a shift that function
        confirms, meaning two quartz lines were found and agreed on it, is taken
        here; a one-line reading is reported in the note and the mount is left
        on the fitted mount's scale.
    counts:
        Raw counts of the air-dried mount, for the 1/counts weighting that the
        pattern block uses.  Without them every point of the air-dried mount
        counts alike, which over-weights its strong peaks.

    Raises
    ------
    ValueError
        If the library carries no air-dried counterparts.  Falling back to the
        glycol patterns would assert that nothing expands, which is the question
        being asked, so this refuses instead and says to rebuild.
    """
    from .detection import ZeroShift, quartz_zero_shift, treatment_peaks

    if not library.has_air_dried():
        raise ValueError(
            "this library carries no air-dried patterns, so the air-dried mount cannot "
            "restrain the expandable clays; rebuild it with the current version "
            "(build_library(air_dried_thickness=...)) or leave the air-dried mount out"
        )

    grid = np.asarray(library.two_theta, dtype=float)
    if range_two_theta is not None:
        low, high = range_two_theta
        grid = grid[(grid >= low) & (grid <= high)]
    if grid.size < 2:
        raise ValueError("fewer than two library points lie in the fitted range")

    expandable = sum(entry.air_intensity is not None for entry in library.entries)
    if expandable == 0:
        return AirDriedObservation(
            None, 1.0, 0.0, 0,
            "No library entry changes on glycolation, so the air-dried mount has "
            "nothing to restrain.",
        )

    covered = float(np.mean((grid >= air.two_theta.min()) & (grid <= air.two_theta.max())))
    if covered < MINIMUM_AIR_OVERLAP:
        return AirDriedObservation(
            None, 1.0, 0.0, expandable,
            f"The air-dried mount covers only {100.0 * covered:.0f}% of the fitted range "
            f"(it takes {100.0 * MINIMUM_AIR_OVERLAP:.0f}%), so it is not used. Scan it "
            f"over the same range as the mount being fitted.",
        )

    measured_shift = None
    declined = ""
    measuring = shift is None
    if shift is None:
        try:
            measured = quartz_zero_shift(treatment_peaks(air))
        except Exception:  # noqa: BLE001 - reported below, not raised
            measured = ZeroShift(None, (), False, "The air-dried mount could not be "
                                                  "searched for quartz.")
        if measured.confirmed and measured.shift is not None:
            # `quartz_zero_shift` returns the correction to add to the measured
            # angles; this function's `shift` is the error to subtract from
            # them, the way `apply_zero_error` signs it.  Same quantity, opposite
            # sign, and taking it across unnegated doubles the displacement
            # instead of removing it.
            measured_shift = -float(measured.shift)
        elif measured.shift is not None:
            declined = measured.note
        shift = 0.0 if measured_shift is None else measured_shift
    shift = float(shift)
    corrected = replace(air, two_theta=air.two_theta - shift)

    note = ""
    if scale is None:
        try:
            scale = scale_to_reference(fitted, corrected)
        except Exception as exc:  # noqa: BLE001 - reported below, not raised
            return AirDriedObservation(
                None, 1.0, shift, expandable,
                f"The two mounts could not be put on a common intensity scale on quartz "
                f"100 ({exc}), so one coefficient cannot describe both and the air-dried "
                f"mount is not used. Give the scale explicitly to use it anyway.",
            )
    if declined:
        note = (" The air-dried mount's own zero shift was measured but not applied, so "
                "it was taken to share the fitted mount's. " + declined)
    elif measuring and measured_shift is None:
        note = (" No quartz line gave the air-dried mount a zero shift, so it was taken "
                "to share the fitted mount's.")
    scale = float(scale)

    # The air-dried mount is left in its own measured units and the design is
    # divided by the scale instead of the target multiplied by it.  Both express
    # the same relation, but this one keeps the block's counting statistics
    # native: multiplying the target by `scale` multiplies its standard
    # deviation by `scale` too, and weighting it as though it had not is how the
    # air block silently comes to outweigh the pattern it is restraining.
    if scale <= 0.0:
        raise ValueError("the intensity scale must be positive")
    target = np.interp(grid, corrected.two_theta, corrected.intensity, left=0.0, right=0.0)
    design = library.air_matrix(grid).T / scale
    if counts is None:
        # Without counts, match the pattern block's own convention as closely as
        # possible: weight by the observation itself, which is Poisson.
        point_weights = np.sqrt(1.0 / np.clip(target, 1.0, None))
    else:
        on_grid = np.interp(grid, corrected.two_theta, np.asarray(counts, dtype=float),
                            left=1.0, right=1.0)
        point_weights = np.sqrt(1.0 / np.clip(on_grid, 1.0, None))

    constraint = ExtraObservation(
        target=target,
        design=design,
        weight=float(weight),
        point_weights=point_weights,
        name="air-dried mount",
    )
    return AirDriedObservation(
        constraint=constraint,
        scale=scale,
        shift=shift,
        expandable=expandable,
        status=(
            f"The air-dried mount is fitted alongside, with one set of coefficients, over "
            f"{grid.size} points at a zero shift of {shift:+.3f} deg and an intensity scale "
            f"of {scale:.3f}.{note} {expandable} library entries predict a different "
            f"air-dried pattern, and the mount decides whether they are there."
        ),
    )


# --------------------------------------------------------------------------
# The bound that evidence puts on the fit
# --------------------------------------------------------------------------

LEAST_MOVING_HYDRATION = 15.0
"""Air-dried spacing in A that moves least on glycolation, of the common states.

Of the states in :data:`AIR_DRIED_STATES` the two-water-layer one at 15 A is
nearest the 16.86 A glycol complex, so it is the state a smectite would show
*least* movement from.  Using it makes the bound below conservative: a phase it
excludes would have moved detectably whichever of these states its interlayer
had been in, so excluding it does not depend on knowing which.

A three-water-layer smectite near 18 A would move less still.  It needs a
divalent cation and a humid room, it is rare in a routine air-dried mount, and
nothing here would detect it - which is a limit of the evidence, not of the
arithmetic, and is why the bound is reported rather than applied silently.
"""


@dataclass
class ExpandableBound:
    """How much expandable layer the measured movement allows."""

    maximum: float
    """Largest expandable layer content an entry may carry, as a fraction."""

    allowed: list[int]
    excluded: list[int]
    unrestricted: bool
    """Whether movement was seen, in which case nothing is excluded."""

    status: str


def expandable_bound(
    evidence: ShiftEvidence,
    library,
    hydrated: float = LEAST_MOVING_HYDRATION,
    glycol: float = 16.86,
    margin: float = 2.0,
) -> ExpandableBound:
    """Turn the measured movement into a ceiling on expandable layer content.

    The argument is entirely about *spacings*, and uses no air-dried structure.
    An interstratified stack's basal reflections lie between those of its two
    layer types - Mering's rule - so a stack of host spacing ``h`` carrying a
    fraction ``x`` of expandable layers puts its 001 near ``h + x(s - h)``,
    where ``s`` is the expandable layer's own spacing.  Glycolation changes
    ``s`` from its air-dried value to 16.86 A, so it moves the 001 by about
    ``x(16.86 - s_air)``.  The least that can be, over the hydration states an
    air-dried smectite actually takes, is at ``s_air`` = 15 A.

    So a specimen whose basal reflections did not move by more than ``m`` can
    carry at most ``x = m / (16.86 - 15)`` of expandable layers *per angstrom of
    unmoved spacing*, and entries above that are excluded from the fit.  When
    movement was seen the bound is not applied: the evidence then says
    expandable clay is present and the fit is left to say how much.

    This is the constraint the measurement supports.  It is not a fit of the
    air-dried mount, which cannot be done - see
    :meth:`clayquant.library.PatternLibrary.for_treatment`.
    """
    indices = expandable_entries(library)
    if evidence.significant:
        return ExpandableBound(
            1.0, list(range(len(library.entries))), [], True,
            "Movement was seen between the mounts, so the expandable clays are not "
            "restrained: the evidence says they are there and the fit says how much. "
            + evidence.status,
        )
    if not indices:
        return ExpandableBound(
            1.0, list(range(len(library.entries))), [], True,
            "No entry in the library carries expandable layers, so there is nothing "
            "to restrain.",
        )

    reach = float(glycol) - float(hydrated)
    if reach <= 0.0:
        raise ValueError("the glycol spacing must exceed the air-dried one")
    # How far the 001 of a fully expandable stack would move; a stack that is a
    # fraction x expandable moves x as far.
    allowance = float(evidence.measured) + float(margin) * float(evidence.uncertainty)
    # Movement in degrees at the 001 of a 10 A series, converted to a fraction:
    # d(2theta)/d(d) at 10 A is about 0.88 deg per angstrom for Cu K-alpha.
    degrees_per_angstrom = 0.88
    maximum = min(1.0, max(0.0, allowance / (degrees_per_angstrom * reach)))

    allowed, excluded = [], []
    for position, entry in enumerate(library.entries):
        expandable = (0.0 if entry.fraction is None
                      else max(0.0, 1.0 - float(entry.fraction)))
        (allowed if expandable <= maximum + 1e-9 else excluded).append(position)

    return ExpandableBound(
        maximum=maximum,
        allowed=allowed,
        excluded=excluded,
        unrestricted=False,
        status=(
            f"Nothing moved between the mounts: the largest displacement of a basal "
            f"reflection is {evidence.measured:.3f} deg, within what those peaks can be "
            f"placed to, against {evidence.uncertainty:.3f} deg "
            f"of disagreement on the quartz lines. Glycolation moves a fully expandable "
            f"001 by about {degrees_per_angstrom * reach:.2f} deg even from the "
            f"least-moving hydration state ({hydrated:g} A), so a stack that did not move "
            f"by more than {allowance:.3f} deg can be at most {100.0 * maximum:.1f}% "
            f"expandable. {len(excluded)} of the library's {len(indices)} expandable "
            f"entries are above that and are left out of the fit."
        ),
    )
