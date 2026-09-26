"""The reference pattern library and its builder.

The library holds the calculated patterns that the non-negative least-squares
fit draws on.  As specified for ClayQuant it contains:

* the discrete phases - illite, chlorite, kaolinite 1M and kaolinite 2M - each
  at ten March-Dollase orientation parameters from 0.1 to 1.0;
* the ethylene glycol smectite layer of Reynolds (1965) as a pure basal pattern;
* randomly interstratified illite/smectite over fifteen compositions from
  0.20/0.80 to 0.99/0.01, at each orientation parameter;
* randomly interstratified chlorite/smectite at 0.95/0.05, 0.90/0.10 and
  0.85/0.15, at each orientation parameter.

Patterns are stored in a single ``.npz`` file together with their metadata, so a
library is built once and reused.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .background import snip_baseline
from .crystal import Crystal
from .emission import CU_KA_5LINE
from .mixed_layer import MixedLayerStack, lognormal_csds
from .optics import Divergence
from .models import (
    AIR_DRIED_SMECTITE_D001,
    CIF_SOURCES,
    air_dried_smectite_layer,
    available_phases,
    chlorite_crystal,
    eg_smectite_layer,
    illite_crystal,
    load_crystal,
    load_layer,
    use_refined_structures,
)
from .pattern import (
    Instrument,
    Pattern,
    basal_pattern,
    basal_scale_factor,
    mixed_layer_pattern,
    powder_pattern,
    reflections,
    two_theta_grid,
)
from .profile import PeakShape

__all__ = [
    "LibraryEntry",
    "PatternLibrary",
    "build_library",
    "PREFERRED_ORIENTATIONS",
    "ILLITE_SMECTITE_FRACTIONS",
    "CHLORITE_SMECTITE_FRACTIONS",
    "CSDS_MEANS",
    "DEFAULT_PEAK_SHAPE",
    "DISCRETE_STRAINS",
    "HOST_THICKNESSES",
    "ILLITE_COMPOSITION",
    "ILLITE_SMECTITE_HOST",
    "describe_instrument_mismatch",
    "instrument_from_measurement",
    "CONTINUUM_WINDOW",
    "scaled_to_d001",
    "NORMALIZATION_FLOOR",
    "main",
]

CONTINUUM_WINDOW = 6.0
"""Peak-stripping width in degrees used to take the continuum off a calculated pattern."""

CONTINUUM_MIN_ANGLE = 1.0
"""Lowest angle a pattern is calculated at; below it the Lorentz factor is useless."""

PREFERRED_ORIENTATIONS: tuple[float, ...] = tuple(round(0.1 * k, 1) for k in range(1, 11))
"""March-Dollase parameters 0.1 to 1.0 in steps of 0.1."""

ILLITE_SMECTITE_FRACTIONS: tuple[float, ...] = (
    0.20,
    0.40,
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    1.00,
)
"""Illite fraction of the illite/smectite series.

The series reaches 1.00, the pure illite end member, and that matters more than
it looks.  Without it the closest a pure illite can be described as is 0.99, and
a fit of one piles its whole coefficient onto that entry - which reports 1 % of
smectite the specimen does not have, and, worse, reports a *bound* as though it
were a measurement.  A measured pure illite standard is what showed this: the
fit put 94 % of its illite on the 0.99 entry, the end of the range.

The end member is not redundant with the discrete ``illite`` entries either.
Those are three-dimensional powder patterns whose basal widths come from the
instrumental profile; the interstratified entries carry the crystallite
thickness distribution of Sec. 2.6, which is what actually sets a basal width.
Only through this series can a pure illite be fitted with its thickness spanned.
"""

CHLORITE_SMECTITE_FRACTIONS: tuple[float, ...] = (1.00, 0.95, 0.90, 0.85)
"""Chlorite fraction of the chlorite/smectite series.

Reaching 1.00 for the same reason as the illite series: a chlorite with no
expandable component has to be describable without one.
"""

CHLORITE_IRON: tuple[tuple[float, float], ...] = (
    (0.0, 0.000),
    (0.0, 0.006),
    (0.0, 0.015),
    (0.0, 0.025),
    (0.0, 0.035),
    (0.0, 0.050),
)
"""Octahedral iron of the chlorite entries, as (2:1 sheet, hydroxide sheet).

In the **hydroxide** sheet, and finely, and both of those are measurements
rather than choices.

A chlorite's octahedral iron varies from one deposit to the next and acts
directly on its basal intensities, so one published clinochlore cannot describe
two chlorites of different composition.  This axis used to span the 2:1 sheet,
from 0 to 0.6 of an atom, and it ran the wrong way.  Chlorite's 001 comes from
the *contrast* between the 2:1 layer and the hydroxide sheet and its even orders
from their *sum*, so iron in the 2:1 sheet raises the contrast and lowers the
even orders relative to the 001: the old series took 002/001 from 1.80 down to
0.74, while two chlorite standards measured on this instrument want 1.87 and
2.30.  Every value of it was further from both standards than the published
structure.

Iron in the hydroxide sheet raises the even orders, and steeply, because the 001
passes through zero where the two sheets scatter alike - which is why so little
of it does so much, and why the axis has to be fine near zero to be useful at
all.  Calculated on this instrument's geometry:

======  =======  =======  =======
 OH Fe  002/001  003/001  004/001
======  =======  =======  =======
 0.000    1.805    1.236    1.204
 0.006    1.877    1.283    1.246
 0.015    1.989    1.355    1.313
 0.025    2.118    1.437    1.386
 0.035    2.259    1.527    1.473
 0.050    2.519    1.692    1.616
======  =======  =======  =======

against 1.874 / 0.791 / 1.142 measured on a clinochloritic chlorite and 2.298 /
1.357 / 1.701 on an iron-rich prochlorite.  The prochlorite's three higher orders
are matched together to within 15 % at 0.035; the chlorite's 002 and 004 at
0.006, its 003 by nothing here, which is left standing rather than tuned away.

Whether the parameter is literally iron, or the hydroxide sheet's height, or its
occupancy, is not settled by this: it is one number that moves the sum of the two
sheets against their contrast, and with it two real chlorites are describable and
with the published structure alone neither is.
"""

ILLITE_COMPOSITION: tuple[tuple[float, float], ...] = (
    (1.0, 0.000),
    (0.9, 0.075),
    (0.9, 0.150),
    (0.8, 0.150),
)
"""Illite compositions to calculate, as ``(interlayer K, octahedral Fe)`` pairs.

Both substitutions are real in an illite and neither is known in advance, any
more than a chlorite's iron is, so they are spanned and the fit chooses; the
entry it picks reports the pair.

They are here because the published structure does not reproduce a measured
illite.  ICSD 90144 carries K at full occupancy - that is a muscovite - and no
octahedral iron, and it calculates a basal series of 1 : 0.507 : 0.868 where the
standard measures 1 : 0.164 : 0.477: the 5 A order three times too strong and
the 3.33 A order nearly twice.  The potassium sits exactly between the layers,
so its phase factor alternates and removing it lowers the higher orders; it
corrects most of the 3.33 A error and almost none of the 5 A one.  The
octahedral iron sits at the middle of the 2:1 layer and does the rest: 0.15 of
it gives 1 : 0.170 : 0.493, the measurement to within 4 per cent on both orders.

The potassium stays inside the range that defines the mineral, 0.75 to 0.9 per
O10(OH)2, and that is a constraint on the axis rather than an outcome of it.
Fitted freely, the measured series is matched marginally better by full
potassium with 0.15 of iron - but that is a phengite, and a phengite is
identifiable in hand specimen by its glitter, so a separate prepared from a
sample with no visible mica is not one.  Inside the illite range the best
member, K = 0.9 with 0.15 of iron, calculates 1 : 0.173 : 0.449 against
1 : 0.164 : 0.477 measured, which is within 6 per cent on both orders: nothing
is given up by respecting the mineralogy.

The four pairs are the published structure, kept under its plain name so that
nothing already written down stops matching, and three illites spanning both
substitutions.  Sec. A.35 of the manual records the scan and what else was
ruled out.
"""

ILLITE_SMECTITE_HOST: tuple[float, float] = (0.9, 0.150)
"""The illite composition used for the illite/smectite host layer.

One pair, not an axis. The discrete illite is calculated at each of
:data:`ILLITE_COMPOSITION` and the fit chooses among them, but the
interstratified series already spans fifteen compositions, ten orientations,
three crystallite sizes and four layer spacings, so giving it a fourth axis
would multiply eighteen hundred entries by four for a distinction the fit
cannot make anyway: an illite/smectite is identified by where its
interstratification orders fall, not by the relative heights of a host's basal
series.

What it must not be is the published structure, which is an iron-free muscovite
and calculates a 5 A order three times too strong (Sec. A.35). The host would
then be systematically brighter at 5 A than the discrete illite beside it, and
the split between the two - the one number an illite/smectite analysis is really
for - would absorb the difference.  The default is the member of
:data:`ILLITE_COMPOSITION` that best reproduces a measured illite on this
instrument, so that the host and the discrete mineral beside it are the same
mineral and both are illites rather than micas.  Pass ``None`` for the published
structure, or another pair measured on your own illite.
"""

DISCRETE_STRAINS: dict[str, tuple[float, ...]] = {
    "kaolinite_1M": (0.0, 0.25, 0.5, 0.75, 1.0),
    "kaolinite_2M": (0.0, 0.25, 0.5, 0.75, 1.0),
}
"""Microstrain values to span per discrete phase, and which phases get the axis.

Each value is the broadening FWHM in degrees at ``tan(theta) = 1``
(:class:`clayquant.profile.PeakShape`), so zero is the instrumental width alone
and one adds about 0.11 deg at the kaolinite 001.  A phase not named here is
calculated at zero strain only.

The axis exists because the instrument does not account for the width of a clay
peak.  On one real oriented mount the instrumental FWHM near 12 deg is 0.051 deg
while the measured basal reflections are 0.093 to 0.131 deg wide - a factor of
two to two and a half.  A reference peak half as wide as the measured one cannot
be scaled to it: the fit can match its height or its area but not both, and it
is the area that carries the weight percent.

Only the kaolinites carry the axis by default, and the reason is that they are
the only clays without one already.  Illite and chlorite have their basal widths
spanned through the crystallite thickness distribution of the interstratified
series, which reaches the discrete end member at a host fraction of 1.00; the
kaolinites have no such series, so without this they are fitted at essentially
instrumental width.  Giving the discrete illite and chlorite a *second* width
axis on top of that one was measured on a mount with an independent TOPAS
refinement to answer against: it lowered ``R_wp`` by a further half point and
moved the clay assemblage away from the refinement, the discrete illite
outcompeting the interstratified entries and taking the illite share from 27 to
43 % of the clay where the refinement says 5 %.  Span them here if your own
standards call for it - the cost is a larger library, not a wrong one.
"""

NORMALIZATION_FLOOR = 4.0
"""Patterns are normalised on their maximum above this 2theta, in degrees."""

HOST_THICKNESSES: dict[str, tuple[float, ...]] = {
    "illite": (9.90, 9.95, 10.02, 10.10),
    "chlorite": (14.05, 14.15, 14.25, 14.35),
}
"""Layer repeat distances in A spanned for each interstratification host.

Neither spacing is a constant.  Illite runs from about 9.90 to 10.10 A with
interlayer potassium content and hydration, and chlorite from about 14.0 to
14.4 A with its octahedral composition; the value that comes out of any one
refined structure is a single point in that range.  Measured here, ICSD 90144
gives 10.022 A while a real sample's 001 and 002 both put it at 9.95 A, and
ICSD 164234 gives 14.278 A where the same sample's 002 and 004 put its chlorite
at 14.15 A.

What that costs when the library does not span it is out of proportion to the
numbers.  A chlorite 0.13 A away is 0.11 deg off at the 002 reflection and
0.24 deg at the 004 - one and two peak widths - so the calculated peaks stand
beside the measured ones rather than on them, and a fit of scale factors cannot
move them.  On M_26_1041 the fit responded by using a chlorite/smectite entry at
14.2 A as a stand-in for chlorite, which is worse than a bad fit: it reports an
expandable component that is not there.  Widening this table to reach 14.05 A
recovered the chlorite as itself and took Rwp from 36.9 to 33.3 per cent.
"""

CSDS_MEANS: tuple[float, ...] = (5.0, 15.0, 50.0)
"""Default mean crystallite thicknesses, in layers, spanned by the library.

Basal peak width is set by how many layers a crystallite stacks, and that varies
far too much between samples to fix in advance: 10 layers give a 001 width near
0.8 deg, while a well crystallised illite measures nearer 0.12 deg and needs
about 70 layers.  The library therefore carries each interstratified composition
at several thicknesses and lets the fit choose.
"""


def scaled_to_d001(crystal: Crystal, layers_per_cell: int, thickness: float) -> Crystal:
    """A copy of ``crystal`` whose single-layer basal spacing is ``thickness`` [A].

    Only the ``c`` axis is scaled, since the basal spacing of a phyllosilicate
    changes with what sits between the layers and not with the layer's own
    in-plane dimensions.
    """
    current = crystal.d001 / layers_per_cell
    if current <= 0:
        raise ValueError(f"{crystal.name}: basal spacing is not positive")
    return replace(crystal, c=crystal.c * thickness / current)


@dataclass
class LibraryEntry:
    """One calculated pattern in the library."""

    name: str
    phase: str
    intensity: np.ndarray
    march_dollase: float = 1.0
    fraction: float | None = None
    csds_mean: float | None = None
    thickness: float | None = None
    strain: float = 0.0
    """Microstrain the pattern was calculated with, in deg at ``tan(theta) = 1``.

    Zero for a pattern carrying the instrumental width alone, which is every
    interstratified entry and every discrete one in a library built before the
    strain axis existed.
    """

    normalization: float = 1.0
    """What the calculated pattern was divided by before storage.

    Patterns are stored at unit maximum so that the fit is well conditioned,
    which throws the absolute scale away.  Keeping the divisor puts it back:
    ``coefficient / normalization`` is proportional to how many scattering units
    of the phase are in the beam, whatever the stored pattern's height.
    """

    unit_mass: float | None = None
    """Mass in g/mol of the unit the pattern was calculated for.

    One unit cell for a discrete phase, one layer for an interstratified stack -
    whichever the structure factor was computed from.
    """

    air_intensity: np.ndarray | None = None
    """The same pattern with the smectite interlayer collapsed, or ``None``.

    Only an entry that carries expandable layers has one: everything else - a
    discrete illite, kaolinite, chlorite or an accompanying mineral - diffracts
    the same before and after glycolation, and its own :attr:`intensity` serves
    for both mounts.

    It is stored divided by the *same* :attr:`normalization` as
    :attr:`intensity`, not by its own maximum.  That is what makes one fitted
    coefficient describe the phase in both treatments, which is the whole
    purpose: an entry claiming a share of the glycol mount is thereby claiming a
    definite, different share of the air-dried one, and the air-dried mount can
    say whether that is there.  Normalising it separately would throw exactly
    that information away.
    """

    unit_volume: float | None = None
    """Volume in A^3 of that same unit.

    Mass and volume are both needed, and the volume is the one that is easy to
    forget: a powder pattern carries ``1/V**2``, one factor from the number of
    cells in a given volume of specimen and one from the density of reciprocal
    lattice points.  Together with :attr:`unit_mass` and the normalisation this
    gives ``W`` proportional to ``S(ZMV)``, the Hill-Howard relation.  Leaving
    the volume out over-weights phases with small cells: quartz, at 113 A^3, is
    six times over-weighted against albite at 664.
    """

    metadata: dict = field(default_factory=dict)


@dataclass
class PatternLibrary:
    """A set of calculated patterns on a common 2theta grid."""

    two_theta: np.ndarray
    entries: list[LibraryEntry] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.two_theta = np.asarray(self.two_theta, dtype=float)
        for entry in self.entries:
            if entry.intensity.shape != self.two_theta.shape:
                raise ValueError(f"entry {entry.name!r} does not match the library grid")

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> list[str]:
        return [entry.name for entry in self.entries]

    @property
    def phases(self) -> list[str]:
        return sorted({entry.phase for entry in self.entries})

    def add(
        self,
        pattern: Pattern,
        phase: str,
        march_dollase: float = 1.0,
        fraction: float | None = None,
        csds_mean: float | None = None,
        thickness: float | None = None,
        strain: float = 0.0,
        unit_mass: float | None = None,
        unit_volume: float | None = None,
        normalization_floor: float = NORMALIZATION_FLOOR,
        strip_continuum: bool = True,
        air_pattern: "Pattern | None" = None,
    ) -> None:
        """Append a pattern, scaled to unit maximum above ``normalization_floor``.

        The scale is taken from the maximum above ``normalization_floor`` rather
        than from the global maximum, because intensity rises steeply towards
        zero angle - the Lorentz factor still diverges as ``1/sin(theta)`` after
        the beam overflow correction - and no clay basal reflection of interest
        lies below about 4 deg.  Normalising on that rise would otherwise shrink
        the diffraction peaks themselves and leave the fit poorly conditioned.

        With ``strip_continuum`` the pattern's own smooth continuum is removed,
        leaving the diffraction features alone.  An interstratified stack of
        finite thickness scatters a genuine diffuse continuum that rises towards
        zero angle, but in a measurement it is inseparable from air scatter and
        the direct beam and is always treated as background.  Leaving it in the
        basis patterns would let a phase soak up the measured background and
        would inflate that phase's apparent share.
        """
        def prepare(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
            if strip_continuum:
                values = np.clip(
                    values - snip_baseline(grid, values, window=CONTINUUM_WINDOW), 0.0, None
                )
            if grid.shape != self.two_theta.shape or not np.allclose(grid, self.two_theta):
                # Patterns are calculated on a grid reaching beyond the stored
                # one so that the continuum can be stripped without the end
                # effect; the margin is dropped here.  The grids share a step,
                # so this is a trim rather than a resampling.
                values = np.interp(self.two_theta, grid, values, left=0.0, right=0.0)
            return values

        intensity = prepare(pattern.intensity, pattern.two_theta)
        usable = self.two_theta >= normalization_floor
        scale = float(np.max(intensity[usable])) if usable.any() else 0.0
        if scale <= 0.0:
            scale = float(np.max(intensity)) or 1.0
        air = None
        if air_pattern is not None:
            # Its own continuum, because the diffuse scatter genuinely differs
            # between the treatments, but the glycol pattern's normalisation,
            # because the two must stay on one scale.
            air = prepare(air_pattern.intensity, air_pattern.two_theta) / scale
        self.entries.append(
            LibraryEntry(
                name=pattern.name,
                phase=phase,
                intensity=intensity / scale,
                air_intensity=air,
                march_dollase=march_dollase,
                fraction=fraction,
                csds_mean=csds_mean,
                thickness=thickness,
                strain=strain,
                normalization=scale,
                unit_mass=unit_mass,
                unit_volume=unit_volume,
                metadata=pattern.metadata,
            )
        )

    def matrix(self, two_theta: np.ndarray | None = None) -> np.ndarray:
        """Design matrix of shape ``(n_entries, n_points)``.

        With ``two_theta`` the patterns are resampled onto that grid.
        """
        if two_theta is None:
            return np.vstack([entry.intensity for entry in self.entries])
        target = np.asarray(two_theta, dtype=float)
        return np.vstack(
            [
                np.interp(target, self.two_theta, entry.intensity, left=0.0, right=0.0)
                for entry in self.entries
            ]
        )

    def has_air_dried(self) -> bool:
        """Whether the library carries the air-dried counterparts."""
        return any(entry.air_intensity is not None for entry in self.entries)

    def air_matrix(self, two_theta: np.ndarray | None = None) -> np.ndarray:
        """Design matrix for the air-dried mount, of shape ``(n_entries, n_points)``.

        Row ``i`` is what entry ``i`` contributes to an air-dried mount at the
        same scale at which :meth:`matrix` gives what it contributes to a glycol
        one.  For an entry with expandable layers that is its stored
        :attr:`~LibraryEntry.air_intensity`; for everything else it is the same
        pattern, because glycolation does not touch it.

        An expandable entry in a library built without the air-dried
        counterparts falls back to its glycol pattern, which asserts that
        glycolation changes nothing - the one thing the air-dried mount is there
        to test.  :meth:`has_air_dried` is how to find out, and
        :func:`clayquant.treatment.air_dried_observation` refuses rather than
        fall back silently.
        """
        rows = [
            entry.intensity if entry.air_intensity is None else entry.air_intensity
            for entry in self.entries
        ]
        if two_theta is None:
            return np.vstack(rows)
        target = np.asarray(two_theta, dtype=float)
        return np.vstack(
            [np.interp(target, self.two_theta, row, left=0.0, right=0.0) for row in rows]
        )

    def for_treatment(self, treatment: str = "glycol") -> "PatternLibrary":
        """The library as the specimen diffracts it - which is the glycolated state.

        Only ``"glycol"`` is a state this library can be used in, and the reason
        is the reason ethylene glycol solvation exists.  An air-dried smectite
        interlayer is not one thing: it holds zero, one, two or three layers of
        water depending on the exchangeable cation and on the humidity of the
        room, at roughly 9.6, 12.4, 15 and 18 A, and a real specimen carries a
        mixture of them, interstratified within single crystallites.  There is
        no single air-dried structure to calculate, and choosing one - this code
        once chose a 12.4 A one-water-layer smectite - is choosing arbitrarily
        among states the specimen may be in any combination of.  Glycolation
        exists to remove that: it drives the interlayer to a reproducible
        two-layer complex near 16.9 A whatever the cation, which is what makes
        the glycolated mount the one a pattern can be calculated for at all.

        So the air-dried mount is not fitted.  It is still evidence, and the
        strongest there is about expandable clay, but the evidence is the
        *movement* between the two mounts, which is measured rather than
        calculated: see :func:`clayquant.treatment.shift_evidence`.

        The heated mount is refused for a related reason and a simpler one:
        heating to 500 C takes the interlayer to 10 A, dehydroxylates the
        kaolinite altogether and alters the chlorite, and none of that is
        calculated either.  It is used for the kaolinite diagnostic, which needs
        no calculated pattern.
        """
        if treatment == "glycol":
            return self
        if treatment == "air":
            raise ValueError(
                "an air-dried mount cannot be fitted: its smectite interlayer may hold "
                "zero, one, two or three layers of water - about 9.6, 12.4, 15 or 18 A - "
                "depending on the exchangeable cation and the humidity, and a specimen "
                "usually carries several of them at once, so there is no single structure "
                "to calculate. That is what glycol solvation is for. Fit the glycolated "
                "mount, and use the air-dried one as evidence of movement instead "
                "(clayquant.treatment.shift_evidence)."
            )
        if treatment == "heated":
            raise ValueError(
                "ClayQuant has no model for a mount heated to 500 C - the smectite "
                "interlayer collapses to 10 A, the kaolinite is destroyed and the "
                "chlorite is altered, and none of it is calculated. Fit the glycolated "
                "mount and use the heated one for the kaolinite diagnostic in step 5."
            )
        raise ValueError(
            f"unknown treatment {treatment!r}; ClayQuant fits the glycolated mount"
        )

    def spanned(self) -> dict[str, list[float]]:
        """The distinct values of each parameter this library samples, per phase.

        The fit chooses among pre-computed patterns, so a parameter is only as
        well determined as the library samples it - and a result that lands on
        the lowest or highest value sampled has not been determined at all, it
        has been stopped there.  This is what makes that detectable; see
        :func:`clayquant.nnls.parameters_at_an_edge`.
        """
        axes: dict[str, set[float]] = {}
        for entry in self.entries:
            for name, value in (("march_dollase", entry.march_dollase),
                                ("thickness", entry.thickness),
                                ("csds_mean", entry.csds_mean),
                                ("fraction", entry.fraction)):
                if value is None:
                    continue
                axes.setdefault(f"{entry.phase}/{name}", set()).add(float(value))
        return {
            key: sorted(values) for key, values in axes.items() if len(values) > 1
        }

    def select(
        self,
        phase: str | list[str] | None = None,
        march_dollase: float | list[float] | None = None,
    ) -> "PatternLibrary":
        """A sub-library filtered by phase and/or orientation parameter."""
        phases = {phase} if isinstance(phase, str) else (set(phase) if phase else None)
        if march_dollase is None:
            orientations = None
        elif isinstance(march_dollase, (int, float)):
            orientations = {float(march_dollase)}
        else:
            orientations = {float(value) for value in march_dollase}

        kept = [
            entry
            for entry in self.entries
            if (phases is None or entry.phase in phases)
            and (orientations is None or any(
                math.isclose(entry.march_dollase, value) for value in orientations
            ))
        ]
        return PatternLibrary(self.two_theta, kept, dict(self.metadata))

    def pattern(self, index: int) -> Pattern:
        entry = self.entries[index]
        return Pattern(self.two_theta, entry.intensity, name=entry.name, metadata=entry.metadata)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            two_theta=self.two_theta,
            intensity=self.matrix(),
            # NaN marks an entry with no air-dried counterpart, which is not the
            # same as one whose counterpart happens to be zero everywhere.
            air_intensity=np.vstack([
                np.full_like(entry.intensity, np.nan) if entry.air_intensity is None
                else entry.air_intensity
                for entry in self.entries
            ]),
            names=np.array(self.names, dtype=object),
            phases=np.array([entry.phase for entry in self.entries], dtype=object),
            march_dollase=np.array([entry.march_dollase for entry in self.entries]),
            fraction=np.array(
                [np.nan if entry.fraction is None else entry.fraction for entry in self.entries]
            ),
            csds_mean=np.array(
                [np.nan if entry.csds_mean is None else entry.csds_mean for entry in self.entries]
            ),
            thickness=np.array(
                [np.nan if entry.thickness is None else entry.thickness for entry in self.entries]
            ),
            strain=np.array([entry.strain for entry in self.entries]),
            normalization=np.array([entry.normalization for entry in self.entries]),
            unit_mass=np.array(
                [np.nan if entry.unit_mass is None else entry.unit_mass for entry in self.entries]
            ),
            unit_volume=np.array(
                [np.nan if entry.unit_volume is None else entry.unit_volume
                 for entry in self.entries]
            ),
            entry_metadata=json.dumps([entry.metadata for entry in self.entries], default=str),
            library_metadata=json.dumps(self.metadata, default=str),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PatternLibrary":
        with np.load(Path(path), allow_pickle=True) as data:
            entry_metadata = json.loads(str(data["entry_metadata"]))
            fractions = data["fraction"]
            # Libraries written before crystallite thickness became a library
            # dimension carry no csds_mean column.
            sizes = data["csds_mean"] if "csds_mean" in data.files else np.full(len(fractions), np.nan)
            spacings = (
                data["thickness"] if "thickness" in data.files else np.full(len(fractions), np.nan)
            )
            # Libraries written before microstrain became a library dimension
            # carry the instrumental width alone.
            strains = (
                data["strain"] if "strain" in data.files else np.zeros(len(fractions))
            )
            # Libraries written before weight percent was possible carry neither
            # column; without them a fit still runs and reports shares, and the
            # weight percent says it cannot be computed rather than guessing.
            scales = (
                data["normalization"] if "normalization" in data.files
                else np.ones(len(fractions))
            )
            masses = (
                data["unit_mass"] if "unit_mass" in data.files
                else np.full(len(fractions), np.nan)
            )
            volumes = (
                data["unit_volume"] if "unit_volume" in data.files
                else np.full(len(fractions), np.nan)
            )
            # Libraries written before the air-dried mount could restrain the
            # expandable clays carry no counterparts at all.
            air = (
                data["air_intensity"] if "air_intensity" in data.files
                else np.full_like(data["intensity"], np.nan)
            )
            entries = [
                LibraryEntry(
                    name=str(name),
                    phase=str(phase),
                    intensity=np.asarray(row, dtype=float),
                    air_intensity=(None if not np.isfinite(air_row).all()
                                   else np.asarray(air_row, dtype=float)),
                    march_dollase=float(orientation),
                    fraction=None if np.isnan(fraction) else float(fraction),
                    csds_mean=None if np.isnan(size) else float(size),
                    thickness=None if np.isnan(spacing) else float(spacing),
                    strain=0.0 if np.isnan(strain) else float(strain),
                    normalization=float(scale),
                    unit_mass=None if np.isnan(mass) else float(mass),
                    unit_volume=None if np.isnan(volume) else float(volume),
                    metadata=metadata,
                )
                for (name, phase, row, air_row, orientation, fraction, size, spacing,
                     strain, scale, mass, volume, metadata) in zip(
                    data["names"],
                    data["phases"],
                    data["intensity"],
                    air,
                    data["march_dollase"],
                    fractions,
                    sizes,
                    spacings,
                    strains,
                    scales,
                    masses,
                    volumes,
                    entry_metadata,
                )
            ]
            return cls(
                two_theta=np.asarray(data["two_theta"], dtype=float),
                entries=entries,
                metadata=json.loads(str(data["library_metadata"])),
            )

    def describe(self) -> str:
        lines = [
            f"{len(self.entries)} patterns on {self.two_theta[0]:.2f}-{self.two_theta[-1]:.2f} "
            f"deg 2theta, step {np.mean(np.diff(self.two_theta)):.3f} deg"
        ]
        for phase in self.phases:
            members = self.select(phase=phase)
            lines.append(f"  {phase:<16s} {len(members):4d} patterns")
        return "\n".join(lines)


def _layer_footprint(crystal) -> float:
    """Area of the (001) face of a cell, in A^2.

    A cell is that area times its (001) interplanar spacing, so a layer of
    thickness ``d`` cut from the same structure has volume ``footprint * d``.
    This keeps a layer-based calculation and a cell-based one on one scale: for
    illite, whose cell holds two layers, the layer has half the mass and half
    the volume of the cell, and its structure factor is half the amplitude, so
    the two routes give the same mass for the same specimen.
    """
    return crystal.volume / crystal.d001


def build_library(
    grid: np.ndarray | None = None,
    instrument: Instrument | None = None,
    orientations: tuple[float, ...] = PREFERRED_ORIENTATIONS,
    illite_smectite: tuple[float, ...] = ILLITE_SMECTITE_FRACTIONS,
    chlorite_smectite: tuple[float, ...] = CHLORITE_SMECTITE_FRACTIONS,
    chlorite_iron: tuple[tuple[float, float], ...] = CHLORITE_IRON,
    illite_composition: tuple[tuple[float, float], ...] = ILLITE_COMPOSITION,
    illite_smectite_host: tuple[float, float] | None = ILLITE_SMECTITE_HOST,
    csds_means: tuple[float, ...] = CSDS_MEANS,
    strains: dict[str, tuple[float, ...]] | None = None,
    csds_beta: float = 0.35,
    host_thicknesses: dict[str, tuple[float, ...]] | None = None,
    smectite_thickness: float | None = None,
    smectite_orientation: float = 0.1,
    air_dried_thickness: float | None = AIR_DRIED_SMECTITE_D001,
    progress: bool = False,
) -> PatternLibrary:
    """Calculate the full reference library.

    Parameters
    ----------
    csds_means:
        Mean numbers of layers per crystallite for the interstratified stacks,
        one set of patterns per value.  Crystallite thickness along ``c*`` sets
        the basal peak width, and it varies far too much between samples to fix
        in advance: a mean of 10 layers gives a 001 width near 0.8 deg, while a
        well crystallised illite measures nearer 0.12 deg, which needs about 70
        layers.  Spanning the range lets the fit choose, instead of forcing
        every interstratified pattern to one sharpness and leaving the peak tops
        unexplained.
    csds_beta:
        Width of each lognormal distribution in ``ln(N)``.
    host_thicknesses:
        Layer repeat distances in A to span per phase; see
        :data:`HOST_THICKNESSES`, which is the default.  Pass ``{}`` to use each
        structure's own refined spacing only.
    smectite_thickness:
        Layer repeat of the glycolated smectite in A; defaults to the 16.86 A
        measured by Reynolds (1965).
    air_dried_thickness:
        Layer repeat in A of the smectite interlayer *before* glycolation.  Each
        interstratified entry is then calculated a second time with the smectite
        collapsed to it, and that pattern stored beside the glycol one, so that
        a fit of the glycol mount can be asked whether the air-dried mount
        agrees - which is the only measurement that distinguishes an illite-rich
        illite/smectite from an illite.  ``None`` skips it, which halves the
        build and leaves the expandable clays unrestrained.
    smectite_orientation:
        March-Dollase parameter the pure smectite pattern is calculated at.  The
        default is 0.1, the orientation a fit typically gives the other platy
        clays on an oriented mount, rather than the 1 of a random powder that it
        used to be: the two differ by a factor of 1000 in the mass behind one
        fitted coefficient, and 1 put a smectite carrying 0.95 % of the
        scattering at 22 % of the clay weight on a real mount.  The quantifier
        rebases it onto the texture the fit actually measured
        (:func:`clayquant.quantification.rebase_fixed_orientation`), so this
        value decides the basis only when that is switched off.  It
        does not change the pattern's shape - every reflection is basal, so the
        factor is one constant - and it does set the basis its weight percent is
        on, because a basal series scales as ``r ** -3``.  The default of 1 is a
        random powder, which is not what an oriented mount of a glycolated
        smectite is; set it to the orientation the fit gives the other platy
        clays if the clay percentages are to be compared with each other.
    strains:
        Microstrain values to span per discrete phase, as the broadening FWHM in
        degrees at ``tan(theta) = 1``; see :data:`DISCRETE_STRAINS`, which is
        the default and gives the axis to the kaolinites alone.  A phase not
        named is calculated at zero strain.  The interstratified series is left
        out of this axis altogether because its basal widths already come from a
        crystallite thickness distribution that the fit chooses among.  Pass
        ``{}`` for the instrumental width throughout.
    illite_composition:
        Illite compositions to calculate, as ``(interlayer potassium, octahedral
        iron)`` pairs; see :data:`ILLITE_COMPOSITION`.  Each pair produces its
        own set of illite entries, all grouped under the phase ``illite``, so
        the fit chooses among compositions as it chooses among layer spacings.
        Empty calculates the published structure alone, which is a muscovite
        with no iron and does not reproduce a measured illite.
    illite_smectite_host:
        The single ``(potassium, iron)`` pair the illite/smectite host layer is
        built from; see :data:`ILLITE_SMECTITE_HOST`.  ``None`` uses the
        published structure.
    chlorite_iron:
        Octahedral iron fractions to calculate chlorite for, as
        ``(2:1 sheet, hydroxide sheet)`` pairs; see :data:`CHLORITE_IRON`.
        Each pair produces its own set of chlorite entries, all grouped under
        the phase ``chlorite``, so the fit chooses among compositions as it
        chooses among layer spacings.  Empty calculates the published structure
        alone.
    """
    grid = two_theta_grid(2.0, 40.0, 0.02) if grid is None else np.asarray(grid, dtype=float)
    # Every pattern is calculated on a grid reaching below the one it is stored
    # on.  The continuum is stripped before storage, and peak stripping needs
    # data on both sides of a point, so its estimate is unreliable within one
    # window of either end (see clayquant.background.snip_baseline) - which for
    # a calculated clay pattern is exactly where the diffuse continuum is
    # largest.  Calculating a margin beyond both ends and trimming afterwards
    # puts that zone outside the stored range: on an I/S 0.80/0.20 pattern it
    # takes the continuum left behind at 3.5 deg from 21% of the strongest peak
    # down to 5%, with the peaks themselves unchanged.
    step = float(np.mean(np.diff(grid))) if len(grid) > 1 else 0.02
    margin = np.arange(1, int(round(CONTINUUM_WINDOW / step)) + 1) * step
    extended = np.concatenate([
        grid[0] - margin[::-1],
        grid,
        grid[-1] + margin,
    ])
    extended = extended[extended >= CONTINUUM_MIN_ANGLE]
    instrument = instrument or Instrument(
        emission=CU_KA_5LINE,
        peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0),
        lp_mode="powder",
        divergence=Divergence(),
    )
    host_thicknesses = HOST_THICKNESSES if host_thicknesses is None else host_thicknesses
    strains = DISCRETE_STRAINS if strains is None else strains
    distributions = [lognormal_csds(float(mean), csds_beta) for mean in csds_means]
    if not distributions:
        raise ValueError("at least one CSDS mean is needed")
    csds = distributions[0]

    missing = [key for key, present in available_phases().items() if not present]
    if missing:
        raise FileNotFoundError(
            f"cannot build the library: CIF files missing for {missing}. "
            f"See clayquant.models.load_crystal for where to put them."
        )

    library = PatternLibrary(
        two_theta=grid,
        metadata={
            "emission": instrument.emission.name,
            "lp_mode": instrument.lp_mode,
            # The width model and the geometry the patterns were calculated with.
            # Recorded because a library built with one instrument and fitted
            # against a measurement calculated with another is wrong in a way
            # that has no symptom of its own: the peaks are the wrong width and
            # the wrong height, the fit absorbs it into the scale factors, and
            # the only trace is intensity missing at every strong peak.  With
            # this here, describe_instrument can say so.
            "peak_shape": {
                "u": instrument.peak_shape.u,
                "v": instrument.peak_shape.v,
                "w": instrument.peak_shape.w,
                "eta": instrument.peak_shape.eta,
                "size_c": instrument.peak_shape.size_c,
                "size_ab": instrument.peak_shape.size_ab,
                "strain": instrument.peak_shape.strain,
            },
            "geometry": None if instrument.divergence is None else {
                "specimen_length": instrument.divergence.specimen_length,
                "goniometer_radius": instrument.divergence.goniometer_radius,
                "divergence": instrument.divergence.divergence,
            },
            "orientations": list(orientations),
            "illite_smectite": list(illite_smectite),
            "chlorite_smectite": list(chlorite_smectite),
            "chlorite_iron": [list(pair) for pair in chlorite_iron],
            "illite_composition": [list(pair) for pair in illite_composition],
            "illite_smectite_host": (None if illite_smectite_host is None
                                     else list(illite_smectite_host)),
            "strains": {key: list(value) for key, value in strains.items()},
            "csds_means": [distribution.mean for distribution in distributions],
            "csds_beta": csds_beta,
            "host_thicknesses": {key: list(value) for key, value in host_thicknesses.items()},
            "smectite_source": "Reynolds (1965) Am. Mineral. 50, 990-1001",
            "smectite_orientation": float(smectite_orientation),
            "air_dried_thickness": (None if air_dried_thickness is None
                                    else float(air_dried_thickness)),
            "cif_sources": {
                key: f"ICSD {source.icsd}: {source.description}"
                for key, source in CIF_SOURCES.items()
            },
        },
    )

    def announce(message: str) -> None:
        if progress:
            print(message, flush=True)

    # Discrete phases at each orientation parameter and layer spacing, and for
    # chlorite at each octahedral iron content asked for.
    for key, source in CIF_SOURCES.items():
        variants: list[tuple[Crystal, str]] = [(load_crystal(key), "")]
        if key == "illite" and illite_composition:
            # The published structure keeps its plain name, as the chlorite does.
            variants = [
                (illite_crystal(k, fe),
                 "" if (k, fe) == (1.0, 0.0) else f" K={k:g}/Fe={fe:g}")
                for k, fe in illite_composition
            ]
        if key == "chlorite" and chlorite_iron:
            # The published structure keeps its plain name; only a substituted
            # one is labelled.  Otherwise adding the iron series renames an
            # entry that was already there, and every reference to it - a saved
            # result, a test, a note in a lab book - stops matching.
            variants = [
                (chlorite_crystal(a, b),
                 "" if (a, b) == (0.0, 0.0) else f" Fe={a:g}/{b:g}")
                for a, b in chlorite_iron
            ]
        for base, iron_tag in variants:
            spacings = host_thicknesses.get(key) or (base.d001 / source.layers_per_cell,)
            values = strains.get(key) or (0.0,)
            announce(f"{key}{iron_tag}: {len(orientations)} orientations "
                     f"x {len(spacings)} layer spacings x {len(values)} strains")
            for thickness in spacings:
                crystal = scaled_to_d001(base, source.layers_per_cell, thickness)
                spacing_tag = f" d={thickness:g}" if len(spacings) > 1 else ""
                for strain in values:
                    strained = replace(
                        instrument,
                        peak_shape=replace(instrument.peak_shape, strain=float(strain)),
                    )
                    # Zero strain keeps the plain name, as the published
                    # chlorite does in the iron series above: adding an axis
                    # must not rename an entry that was already there, or every
                    # reference to it - a saved result, a test, a note in a lab
                    # book - stops matching.
                    strain_tag = "" if not strain else f" e={strain:g}"
                    for r in orientations:
                        pattern = powder_pattern(
                            crystal,
                            extended,
                            strained,
                            r_march_dollase=r,
                            name=f"{key}{iron_tag} PO={r:g}{spacing_tag}{strain_tag}",
                        )
                        library.add(pattern, phase=key, march_dollase=r, thickness=thickness,
                                    strain=float(strain),
                                    unit_mass=crystal.cell_mass, unit_volume=crystal.volume)

    # Pure glycolated smectite.  One entry, and the reason is worth setting out
    # because the consequence is not obvious.  Every reflection of this phase is
    # basal, so the March-Dollase factor is one constant for the whole series and
    # the orientation parameter scales the pattern without changing its shape.
    # Ten entries would therefore be ten identical columns once stored at unit
    # maximum, and a non-negative fit would divide the phase among them
    # arbitrarily - which is worse than one entry, not better.
    #
    # What the single entry does decide is the basis its weight percent is on.
    # A basal series is enhanced by r to the power -3, so the mass behind a
    # given fitted coefficient depends entirely on the orientation assumed, and
    # ``smectite_orientation`` is that assumption.  It used to be fixed at 1, a
    # random powder, while illite, chlorite and the kaolinites had their
    # orientation spanned and chosen by the fit from their non-basal
    # reflections.  A smectite reported as a random powder beside an illite
    # fitted at r = 0.2 is not on the same basis as that illite - the two differ
    # by a factor of 125 per gram - so the clay percentages could not be
    # compared with each other.  Set it to the orientation the fit gives the
    # other platy clays in the same mount, which is the assumption that makes
    # them comparable; the value used is recorded in the library's metadata.
    smectite = eg_smectite_layer(
        smectite_thickness if smectite_thickness is not None else eg_smectite_layer().thickness
    )
    # The same smectite before glycolation.  Every interstratified entry is
    # calculated with both, and the pair is what lets the air-dried mount say
    # whether a candidate expandable phase is really there.
    air_smectite = (
        None if air_dried_thickness is None
        else air_dried_smectite_layer(float(air_dried_thickness))
    )
    announce(f"smectite_EG: 1 pattern at r = {smectite_orientation:g}")
    library.add(
        basal_pattern(
            MixedLayerStack(smectite, smectite, 1.0, csds=csds, name="smectite_EG"),
            extended,
            instrument,
            r_march_dollase=smectite_orientation,
            name="smectite_EG",
        ),
        air_pattern=None if air_smectite is None else basal_pattern(
            MixedLayerStack(air_smectite, air_smectite, 1.0, csds=csds, name="smectite_air"),
            extended,
            instrument,
            r_march_dollase=smectite_orientation,
            name="smectite_EG air",
        ),
        phase="smectite_EG",
        march_dollase=smectite_orientation,
        fraction=0.0,
        unit_mass=smectite.mass,
        # A layer occupies the area of the (001) face of the host cell times its
        # own thickness; the smectite layer is modelled on the same footprint.
        unit_volume=_layer_footprint(load_crystal("illite")) * smectite.thickness,
    )

    # Interstratified series, at each layer spacing and crystallite thickness.
    for host_key, fractions, label in (
        ("illite", illite_smectite, "I/S"),
        ("chlorite", chlorite_smectite, "C/S"),
    ):
        layers_per_cell = CIF_SOURCES[host_key].layers_per_cell
        if host_key == "illite" and illite_smectite_host is not None:
            base_host = illite_crystal(*illite_smectite_host)
            base_layer = base_host.layer_model(layers_per_cell=layers_per_cell,
                                               name=host_key)
        else:
            base_host = load_crystal(host_key)
            base_layer = load_layer(host_key)
        wavelengths, _ = instrument.sample_emission()
        d_min = float(np.max(wavelengths)) / (2.0 * math.sin(math.radians(extended[-1] / 2.0)))
        spacings = host_thicknesses.get(host_key) or (base_layer.thickness,)
        announce(
            f"{label}: {len(fractions)} compositions x {len(orientations)} orientations "
            f"x {len(distributions)} crystallite sizes x {len(spacings)} layer spacings"
        )

        for thickness in spacings:
            host = scaled_to_d001(base_host, layers_per_cell, thickness)
            host_layer = base_layer.with_thickness(thickness, scale_z=True)
            host_reflections = reflections(host, d_min)
            spacing_tag = f" d={thickness:g}" if len(spacings) > 1 else ""
            for csds in distributions:
                scale = basal_scale_factor(host, layers_per_cell, extended, instrument, csds)
                for fraction in fractions:
                    composition = f"{fraction:.2f}/{1.0 - fraction:.2f}"
                    stack = MixedLayerStack(
                        host_layer,
                        smectite,
                        fraction_a=fraction,
                        csds=csds,
                        name=f"{label} {composition}",
                    )
                    air_stack = None if air_smectite is None else MixedLayerStack(
                        host_layer,
                        air_smectite,
                        fraction_a=fraction,
                        csds=csds,
                        name=f"{label} {composition} air",
                    )
                    for r in orientations:
                        entry_name = (
                            f"{label} {composition} PO={r:g}"
                            + (f" N={csds.mean:g}" if len(distributions) > 1 else "")
                            + spacing_tag
                        )
                        pattern = mixed_layer_pattern(
                            stack,
                            host,
                            layers_per_cell,
                            extended,
                            instrument,
                            r_march_dollase=r,
                            name=entry_name,
                            basal_scale=scale,
                            host_reflections=host_reflections,
                        )
                        # A fully collapsed stack - fraction 1.0 - is the host
                        # alone and does not change on glycolation, so it needs
                        # no counterpart and the fit should not be told it has
                        # one.  Everything else does.
                        air_pattern = None
                        if air_stack is not None and fraction < 1.0:
                            air_pattern = mixed_layer_pattern(
                                air_stack,
                                host,
                                layers_per_cell,
                                extended,
                                instrument,
                                r_march_dollase=r,
                                name=entry_name + " air",
                                basal_scale=scale,
                                host_reflections=host_reflections,
                            )
                        library.add(
                            pattern,
                            air_pattern=air_pattern,
                            phase=label,
                            march_dollase=r,
                            fraction=fraction,
                            csds_mean=csds.mean,
                            thickness=thickness,
                            # The interstratification model computes intensity
                            # per layer, and a layer of this stack is a host
                            # layer with probability `fraction` and a smectite
                            # layer otherwise, so the mass of the average layer
                            # is what a scale factor here counts.
                            unit_mass=(fraction * host_layer.mass
                                       + (1.0 - fraction) * smectite.mass),
                            unit_volume=_layer_footprint(host) * (
                                fraction * host_layer.thickness
                                + (1.0 - fraction) * smectite.thickness
                            ),
                        )

    return library


def _parse_strains(arguments: list[str] | None) -> dict[str, tuple[float, ...]] | None:
    """Read ``--strains kaolinite_2M=0,0.5,1`` into the mapping build_library takes."""
    if arguments is None:
        return None
    if len(arguments) == 1 and arguments[0].lower() == "none":
        return {}
    parsed: dict[str, tuple[float, ...]] = {}
    for argument in arguments:
        phase, separator, values = argument.partition("=")
        if not separator:
            raise SystemExit(
                f"--strains takes PHASE=VALUES, as in kaolinite_2M=0,0.5,1; got {argument!r}"
            )
        if phase not in CIF_SOURCES:
            raise SystemExit(
                f"--strains: {phase!r} is not a discrete phase. "
                f"The phases are {', '.join(CIF_SOURCES)}."
            )
        try:
            parsed[phase] = tuple(float(value) for value in values.split(",") if value)
        except ValueError:
            raise SystemExit(f"--strains: {values!r} is not a list of numbers") from None
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Command line entry point for ``clayquant-build-library``."""
    parser = argparse.ArgumentParser(
        prog="clayquant-build-library",
        description="Calculate the ClayQuant reference pattern library.",
    )
    parser.add_argument("-o", "--out", default="library/clayquant_library.npz", type=Path)
    parser.add_argument("--start", type=float, default=2.0, help="first 2theta in degrees")
    parser.add_argument("--stop", type=float, default=40.0, help="last 2theta in degrees")
    parser.add_argument("--step", type=float, default=0.02, help="2theta step in degrees")
    parser.add_argument(
        "--csds-means",
        type=float,
        nargs="+",
        default=list(CSDS_MEANS),
        help="mean numbers of layers per crystallite, one pattern set each",
    )
    parser.add_argument(
        "--csds-beta", type=float, default=0.35, help="width of the lognormal CSDS in ln(N)"
    )
    parser.add_argument(
        "--strains",
        nargs="+",
        default=None,
        metavar="PHASE=VALUES",
        help="microstrain to span for a discrete phase, as the broadening FWHM in "
             "degrees at tan(theta) = 1, e.g. kaolinite_2M=0,0.5,1. Repeat per "
             "phase; a phase not named is calculated at zero strain. A clay peak "
             "on an oriented mount is about twice as wide as the instrument alone "
             "gives, and the kaolinites, which have no interstratified series to "
             "take a width from, carry this axis by default. Pass 'none' for the "
             "instrumental width throughout.",
    )
    parser.add_argument(
        "--measurement",
        default=None,
        help="a scan from the instrument the samples will be measured on; its "
             "goniometer radius and divergence slit are read from the file and its "
             "peaks are measured for the width. No data from it enters the library.",
    )
    parser.add_argument(
        "--air-dried-thickness",
        type=float,
        default=AIR_DRIED_SMECTITE_D001,
        help="layer repeat in A of the smectite before glycolation. Every "
             "interstratified pattern is calculated a second time with the "
             "interlayer collapsed to it and stored beside the glycol one, which is "
             "what lets the air-dried mount say whether an expandable clay is really "
             f"there (default: {AIR_DRIED_SMECTITE_D001:g}, a one-water-layer "
             "smectite; 15 for a Ca-saturated two-layer one)",
    )
    parser.add_argument(
        "--no-air-dried",
        action="store_true",
        help="skip the air-dried patterns. Halves the build and leaves the "
             "expandable clays unrestrained by the air-dried mount.",
    )
    parser.add_argument(
        "--smectite-orientation",
        type=float,
        default=0.1,
        help="March-Dollase parameter for the pure smectite pattern; it sets the "
             "basis its weight percent is on, not its shape (default: 0.1, the "
             "texture a fit usually gives the other platy clays)",
    )
    parser.add_argument(
        "--smectite-thickness",
        type=float,
        default=None,
        help="glycolated smectite layer repeat in A (default: 16.86, after Reynolds 1965)",
    )
    parser.add_argument(
        "--emission-subsampling",
        type=int,
        default=1,
        help="sample points per emission line; >1 reproduces the natural line widths",
    )
    parser.add_argument(
        "--specimen-length", type=float, default=25.0,
        help="size of the mount along the beam in mm, or the diameter of a round one. "
             "It sets where the beam stops overflowing the specimen, which for a 0.5 deg "
             "slit on a 240 mm goniometer is 6.9 deg at 35 mm and 12.0 deg at 20 mm - the "
             "chlorite 001 lies between them, so getting this wrong costs tens of per cent "
             "in the reflection the chlorite is measured by (default: 25, the glass disc "
             "an oriented separate is dried on)"
    )
    parser.add_argument(
        "--specimen-shape", choices=("round", "rectangular"), default="round",
        help="a round mount loses the beam strip's corners before its middle, so it "
             "intercepts a few per cent less than a rectangle of the same length "
             "(default: round)"
    )
    parser.add_argument(
        "--beam-width", type=float, default=10.0,
        help="axial width of the beam at the specimen in mm, the mask setting. Only a "
             "round mount uses it, and barely (default: 10)"
    )
    parser.add_argument(
        "--goniometer-radius", type=float, default=280.0, help="goniometer radius in mm"
    )
    parser.add_argument(
        "--divergence-slit", type=float, default=0.5, help="equatorial divergence in degrees"
    )
    parser.add_argument(
        "--no-divergence-correction",
        action="store_true",
        help="omit the beam overflow correction (leaves a 1/sin^2 ramp at low angle)",
    )
    parser.add_argument(
        "--refined-structures",
        type=Path,
        default=None,
        metavar="REFINEMENT",
        help=(
            "a TOPAS refinement (.out or .inp) whose clay structures to use in place of "
            "the published CIFs; its refined cell and occupancies describe the specimen "
            "it was refined on rather than somebody else's"
        ),
    )
    parser.add_argument(
        "--refined-only",
        default=None,
        metavar="PHASES",
        help=(
            "take only these phases from the refinement, comma separated "
            "(illite, chlorite, kaolinite_1M, kaolinite_2M)"
        ),
    )
    parser.add_argument(
        "--refined-except",
        default=None,
        metavar="PHASES",
        help=(
            "take every phase the refinement supplies except these, comma separated; "
            "use it to keep a published structure you would rather cite, such as the "
            "chlorite, while taking the refined illite and kaolinite"
        ),
    )
    parser.add_argument(
        "--chlorite-iron",
        default=None,
        metavar="PAIRS",
        help=(
            "octahedral iron contents to calculate chlorite for, as "
            "'a/b,a/b' pairs of (2:1 sheet)/(hydroxide sheet) fractions, "
            "for example 0.35/0.20,0.55/0.30; measure them from patterns of your "
            "own pure chlorites with clayquant.composition.fit_chlorite_iron"
        ),
    )
    parser.add_argument(
        "--illite-composition",
        default=None,
        metavar="PAIRS",
        help=(
            "illite compositions to calculate, as 'k/fe,k/fe' pairs of "
            "(interlayer potassium)/(octahedral iron) occupancies, for example "
            "1/0,0.9/0.15; the published structure is 1/0, which is an iron-free "
            "muscovite and calculates a 5 A order three times too strong"
        ),
    )
    parser.add_argument(
        "--illite-smectite-host",
        default=None,
        metavar="K/FE",
        help=(
            "the single illite composition the illite/smectite host layer is built "
            "from, as (interlayer potassium)/(octahedral iron); 'published' uses the "
            "CIF as it stands, which is an iron-free muscovite"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    def occupancy_pairs(text: str, shape: str) -> tuple[tuple[float, float], ...]:
        """Parse ``a/b,a/b`` into pairs, reporting the caller's own wording."""
        pairs = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            halves = part.split("/")
            if len(halves) != 2:
                parser.error(f"{part!r} is not a pair; write each as {shape}")
            try:
                a, b = (float(half) for half in halves)
            except ValueError:
                parser.error(f"{part!r} is not a pair of numbers")
            if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
                parser.error(f"{part!r}: an occupancy lies between 0 and 1")
            pairs.append((a, b))
        return tuple(pairs)

    iron: tuple[tuple[float, float], ...] = CHLORITE_IRON
    if arguments.chlorite_iron:
        iron = occupancy_pairs(
            arguments.chlorite_iron,
            "(2:1 sheet)/(hydroxide sheet), for example 0.35/0.20",
        )

    composition: tuple[tuple[float, float], ...] = ILLITE_COMPOSITION
    if arguments.illite_composition:
        composition = occupancy_pairs(
            arguments.illite_composition,
            "(interlayer K)/(octahedral Fe), for example 0.9/0.15",
        )

    host_composition: tuple[float, float] | None = ILLITE_SMECTITE_HOST
    if arguments.illite_smectite_host:
        if arguments.illite_smectite_host.strip().lower() == "published":
            host_composition = None
        else:
            pair = occupancy_pairs(
                arguments.illite_smectite_host,
                "(interlayer K)/(octahedral Fe), for example 0.9/0.15",
            )
            if len(pair) != 1:
                parser.error("--illite-smectite-host takes one pair, not a list")
            host_composition = pair[0]

    def phase_list(value):
        return [part.strip() for part in value.split(",") if part.strip()] if value else None

    if arguments.refined_structures is None and (arguments.refined_only or arguments.refined_except):
        parser.error("--refined-only and --refined-except need --refined-structures")
    if arguments.refined_only and arguments.refined_except:
        parser.error("give one of --refined-only and --refined-except, not both")

    if arguments.refined_structures is not None:
        from .bern import refined_clay_structures

        skipped: list[str] = []
        wanted = phase_list(arguments.refined_only)
        unwanted = phase_list(arguments.refined_except)
        for name in (wanted or []) + (unwanted or []):
            if name not in CIF_SOURCES:
                parser.error(
                    f"{name!r} is not a phase of the clay library; expected among "
                    f"{', '.join(sorted(CIF_SOURCES))}"
                )
        structures = refined_clay_structures(
            arguments.refined_structures, skipped=skipped, only=wanted, exclude=unwanted
        )
        if not structures:
            selection = (
                f" among {arguments.refined_only}" if arguments.refined_only
                else f" once {arguments.refined_except} is left out" if arguments.refined_except
                else ""
            )
            parser.error(
                f"{arguments.refined_structures} holds no clay structure the library uses"
                f"{selection}. A refinement that models its clays as hkl_Is peaks phases has "
                "no structure to take."
                + ("\n" + "\n".join(f"  {note}" for note in skipped) if skipped else "")
            )
        use_refined_structures(structures)
        if not arguments.quiet:
            print(f"structures taken from {arguments.refined_structures}:")
            for key, crystal in sorted(structures.items()):
                print(f"  {key:<14} {crystal.name}, d(001) = {crystal.d001:.3f} A, "
                      f"cell mass {crystal.cell_mass:.1f}")
            for key in sorted(set(CIF_SOURCES) - set(structures)):
                print(f"  {key:<14} published structure kept")
            for note in skipped:
                print(f"  note: {note}")

    divergence = (
        None
        if arguments.no_divergence_correction
        else Divergence(
            specimen_length=arguments.specimen_length,
            goniometer_radius=arguments.goniometer_radius,
            divergence=arguments.divergence_slit,
            shape=arguments.specimen_shape,
            beam_width=arguments.beam_width,
        )
    )
    instrument = Instrument(
        emission=CU_KA_5LINE,
        peak_shape=DEFAULT_PEAK_SHAPE,
        lp_mode="powder",
        lines_per_emission=arguments.emission_subsampling,
        divergence=divergence,
    )
    if arguments.measurement:
        # A scan from the instrument settles the two things the library cannot
        # guess and must not get wrong: the geometry, which the file records,
        # and the peak width, which its peaks measure.  No data from it enters
        # the library.
        from .background import clayfit_background
        from .io import read_pattern, resolve_user_path

        pattern = read_pattern(resolve_user_path(arguments.measurement))
        background = clayfit_background(pattern.two_theta, pattern.intensity)
        instrument, note = instrument_from_measurement(
            pattern, background=background,
            specimen_length=arguments.specimen_length,
            specimen_shape=arguments.specimen_shape,
        )
        if not arguments.quiet:
            print(f"instrument from {Path(arguments.measurement).name}: {note}")
    elif not arguments.quiet:
        print(
            f"no --measurement given: calculating for a "
            f"{arguments.goniometer_radius:g} mm goniometer radius, a "
            f"{arguments.divergence_slit:g} deg divergence slit and a generic peak "
            f"width. Pass a scan from your instrument if that is not it."
        )
    library = build_library(
        grid=two_theta_grid(arguments.start, arguments.stop, arguments.step),
        instrument=instrument,
        chlorite_iron=iron,
        illite_composition=composition,
        illite_smectite_host=host_composition,
        csds_means=tuple(arguments.csds_means),
        csds_beta=arguments.csds_beta,
        strains=_parse_strains(arguments.strains),
        smectite_thickness=arguments.smectite_thickness,
        smectite_orientation=arguments.smectite_orientation,
        air_dried_thickness=(None if arguments.no_air_dried
                             else arguments.air_dried_thickness),
        progress=not arguments.quiet,
    )
    path = library.save(arguments.out)
    if not arguments.quiet:
        print(library.describe())
        print(f"written to {path}")
    return 0




DEFAULT_PEAK_SHAPE = PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0)
"""Width model used when there is no measurement to take one from."""


def instrument_from_measurement(
    pattern,
    background=None,
    specimen_length: float = 25.0,
    shift: float = 0.0,
    specimen_shape: str = "round",
) -> tuple[Instrument, str]:
    """An instrument taken from a measurement: its geometry and its peak widths.

    The geometry comes from the file, which records the goniometer radius and
    the divergence slit; only the specimen length has to be supplied, because
    nothing in a data file knows how large the mount was.  It defaults to a
    round mount 25 mm across, which is the glass disc these oriented separates
    are dried on, and it is worth setting to the real one:
    the beam-overflow correction (:class:`clayquant.optics.Divergence`) acts
    only where the irradiated length exceeds it, which for a 0.5 deg slit on a
    240 mm goniometer is below 6.9 deg at 35 mm and below 12.0 deg at 20 mm.
    Between those two the difference is the whole of the chlorite 001, and on a
    real mount it is what took the calculated chlorite 001/002 from 0.26 to 0.47
    against 0.43 measured; above 12 deg neither value acts at all, which is why
    the pure-mineral ratios of Sec. A.17 are flat across this range.  Getting it
    wrong is not a small error: a 25 mm mount described as 35 mm keeps 100 per
    cent of the beam at the illite 001 where it really keeps 90, and 91 per cent
    at the chlorite 001 where it really keeps 63 (Sec. A.40).

    ``specimen_shape`` is ``"round"`` or ``"rectangular"``.  A disc loses the
    strip's corners before its middle, so it intercepts a few per cent less than
    a rectangle of the same length.

    The width model is anchored on the quartz 101 K-alpha doublet
    (:func:`~clayquant.profile.quartz_line_width`), whose width in a clay
    separate is the instrument's; only where no quartz can be fitted does it
    fall back on the pattern's own isolated peaks, and it says so when it does,
    because on an oriented mount those peaks are the clay basal reflections and
    are broader than the instrument by a factor that varies with the specimen.
    ``shift`` is the measurement's zero error, so the line is looked for where
    it actually is.

    Returns the instrument and a sentence saying what was taken from where, for
    the status line: a value silently guessed is a value nobody checks.
    """
    from .calibration import QUARTZ_101_D, reference_two_theta
    from .profile import fit_peak_shape, quartz_line_width, shape_from_line_width

    metadata = getattr(pattern, "metadata", {}) or {}
    radius = float(metadata.get("goniometer_radius", 280.0))
    slit = float(metadata.get("divergence_slit", 0.5))
    wavelength = float(metadata.get("wavelength", 1.540596))
    intensity = pattern.intensity
    if background is not None:
        intensity = background.subtract(pattern.two_theta, pattern.intensity)

    # Quartz first, and by a long way.  Its lines in a clay separate are
    # instrumental - the grains are large enough that their size broadening is
    # immeasurable - whereas the isolated peaks the fallback measures are the
    # clay basal reflections, which are broad by nature.  That fallback gave
    # 0.077, 0.124 and 0.234 deg for one diffractometer on three real mounts,
    # and moved by a factor of two on one of them depending on whether a
    # background had been subtracted first; a library built with one of those
    # numbers and checked against another is what made the program warn about
    # its own build.  See :func:`clayquant.profile.quartz_line_width`.
    line = quartz_line_width(
        pattern.two_theta, intensity,
        reference=reference_two_theta(QUARTZ_101_D, wavelength),
        shift=float(shift),
    )
    if line is not None:
        shape = shape_from_line_width(
            line, reference_two_theta(QUARTZ_101_D, wavelength),
            default=DEFAULT_PEAK_SHAPE, wavelength=wavelength,
        )
        width_note = line.note
        width_source = "quartz"
    else:
        widths = fit_peak_shape(pattern.two_theta, intensity, wavelength,
                                default=DEFAULT_PEAK_SHAPE)
        shape = widths.shape
        width_note = (
            f"No quartz 101 could be fitted, so the width is scaled to the pattern's "
            f"own isolated peaks instead, which on an oriented mount are the clay "
            f"basal reflections and are broader than the instrument: {widths.note}"
        )
        width_source = "isolated peaks"
    instrument = Instrument(
        emission=CU_KA_5LINE,
        peak_shape=shape,
        lp_mode="powder",
        divergence=Divergence(specimen_length=specimen_length,
                              shape=specimen_shape,
                              goniometer_radius=radius, divergence=slit),
        width_source=width_source,
    )
    return instrument, (
        f"Geometry from the file: {radius:.0f} mm goniometer radius, "
        f"{slit:g} deg divergence slit, specimen taken as {specimen_length:g} mm. "
        f"{width_note}"
    )


def describe_instrument_mismatch(library, instrument) -> str:
    """Say whether a library was calculated with the instrument now in use.

    A library holds finished patterns, so the width model and the geometry that
    made them are fixed at build time.  Fitting those patterns against a
    measurement whose own instrument differs is not a small inconsistency: the
    reference peaks are then the wrong width and so the wrong height, the fit
    absorbs the difference into its scale factors, and what comes out is
    intensity missing at every strong peak and a weight percent quietly wrong.
    It has no symptom that distinguishes it from a missing phase, which is why
    it is worth a sentence in the interface rather than a comment in the code.

    Returns an empty string when they agree, or when the library is too old to
    record what it was built with - in which case nothing can be said, and
    saying nothing is better than guessing agreement.
    """
    stored_shape = library.metadata.get("peak_shape")
    stored_geometry = library.metadata.get("geometry")
    if not stored_shape:
        return (
            "This library does not record the instrument it was calculated with, "
            "so ClayQuant cannot check it against this measurement. Rebuild it to "
            "be sure the reference peaks have the width of your diffractometer."
        )

    complaints: list[str] = []
    wavelength = np.array([26.0])
    built = PeakShape(
        u=float(stored_shape.get("u", 0.0)),
        v=float(stored_shape.get("v", 0.0)),
        w=float(stored_shape.get("w", 0.01)),
        eta=float(stored_shape.get("eta", 0.5)),
        size_c=stored_shape.get("size_c"),
        size_ab=stored_shape.get("size_ab"),
        strain=float(stored_shape.get("strain", 0.0) or 0.0),
    )
    built_width = float(built.fwhm(wavelength, 1.540596)[0])
    now_width = float(instrument.peak_shape.fwhm(wavelength, 1.540596)[0])
    # A quarter, not two per cent.  The width ClayQuant measures is the width
    # of a quartz line in that particular mount, and that is the instrument's
    # width *plus* whatever the specimen adds: across ten mounts from one
    # diffractometer, at one radius and one slit, the fitted width ranged from
    # 0.032 to 0.203 deg.  At two per cent every one of those disagrees with
    # every other and the interface asks for a rebuild each time, which is
    # wrong - the geometry is what must match, and one library serves every
    # scan from the same instrument.  A gross difference is still worth saying:
    # a library built at 0.100 deg against a 0.052 deg measurement cost 12
    # points of Rwp on a real mount.
    if abs(built_width - now_width) > 0.25 * max(built_width, now_width):
        complaints.append(
            f"peak width at 26 deg is {built_width:.3f} deg in the library and "
            f"{now_width:.3f} deg for this measurement, which is more than the "
            f"width varies between mounts on one instrument"
        )
    if stored_geometry and instrument.divergence is not None:
        for key, label in (("goniometer_radius", "goniometer radius"),
                           ("divergence", "divergence slit"),
                           ("specimen_length", "specimen length")):
            built_value = float(stored_geometry[key])
            now_value = float(getattr(instrument.divergence, key))
            if abs(built_value - now_value) > 1e-6 * max(1.0, abs(built_value)):
                complaints.append(f"{label} {built_value:g} in the library, {now_value:g} here")
    if not complaints:
        return ""
    return (
        "This library was calculated with a different instrument than this "
        "measurement: " + "; ".join(complaints) + ". Rebuild the library so the "
        "reference peaks have the right width, or the fit will be short of "
        "intensity at every strong peak. One library serves every scan from the "
        "same diffractometer - it does not need rebuilding per sample."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
