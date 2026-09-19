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
        Zero shift of the air-dried mount against the fitted one, in degrees.
        Zero by default, because ClayQuant zero-corrects each mount as it is
        loaded and correcting it twice would misalign the two blocks; ``None``
        measures it from the quartz lines instead, for a mount that has not been
        corrected.

        Measuring is not the safe default it looks.  On a specimen with no
        quartz in it, :func:`~clayquant.detection.quartz_zero_shift` will match
        a clay reflection to a quartz line and return a shift of several
        hundredths of a degree - on a test case here, -0.40 deg from a single
        peak - and a misaligned air-dried block does not restrain the expandable
        clays, it wrecks the fit: 2.24% became 37.91%.
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
    from .detection import quartz_zero_shift, treatment_peaks

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
    if shift is None:
        try:
            measured_shift, _ = quartz_zero_shift(treatment_peaks(air))
        except Exception:  # noqa: BLE001 - reported below, not raised
            measured_shift = None
        shift = 0.0 if measured_shift is None else float(measured_shift)
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
    if measured_shift is None and shift == 0.0:
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
