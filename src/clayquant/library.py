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
    CIF_SOURCES,
    available_phases,
    chlorite_crystal,
    eg_smectite_layer,
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
    "HOST_THICKNESSES",
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

CHLORITE_IRON: tuple[tuple[float, float], ...] = ()
"""Octahedral iron of the chlorite entries, as (2:1 sheet, hydroxide sheet).

Empty by default, which calculates the published structure alone.  A chlorite's
octahedral iron varies from one deposit to the next and acts directly on its
basal intensities, so one published clinochlore cannot describe two chlorites of
different composition - but neither is there a sensible generic set to span, the
way there is for the illite/smectite ratio.  What there is instead is a way to
measure it: fit the composition to a pattern of the pure mineral
(:func:`clayquant.composition.fit_chlorite_iron`) and span the values that come
out of your own standards.
"""

NORMALIZATION_FLOOR = 4.0
"""Patterns are normalised on their maximum above this 2theta, in degrees."""

HOST_THICKNESSES: dict[str, tuple[float, ...]] = {
    "illite": (9.95, 10.02, 10.10),
    "chlorite": (14.20, 14.35),
}
"""Layer repeat distances in A spanned for each interstratification host.

The basal spacing of illite is not a constant: it runs from about 9.95 to
10.10 A with interlayer potassium content and hydration, and the value that
comes out of any single refined structure is only one point in that range.
Measured here, ICSD 90144 gives 10.022 A while a real sample's 001 and 002 both
put it at 9.95 A - a difference of 0.066 deg at the 001 reflection, half a peak
width, which for a fixed library is the difference between fitting that peak and
missing it almost entirely.  Spanning the range lets the fit pick the spacing
instead of being told it.
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
        unit_mass: float | None = None,
        unit_volume: float | None = None,
        normalization_floor: float = NORMALIZATION_FLOOR,
        strip_continuum: bool = True,
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
        intensity = pattern.intensity
        if strip_continuum:
            intensity = np.clip(
                intensity
                - snip_baseline(pattern.two_theta, intensity, window=CONTINUUM_WINDOW),
                0.0,
                None,
            )
        if pattern.two_theta.shape != self.two_theta.shape or not np.allclose(
            pattern.two_theta, self.two_theta
        ):
            # Patterns are calculated on a grid reaching beyond the stored one so
            # that the continuum can be stripped without the end effect; the
            # margin is dropped here.  The grids share a step, so this is a
            # trim rather than a resampling.
            intensity = np.interp(self.two_theta, pattern.two_theta, intensity,
                                  left=0.0, right=0.0)
        usable = self.two_theta >= normalization_floor
        scale = float(np.max(intensity[usable])) if usable.any() else 0.0
        if scale <= 0.0:
            scale = float(np.max(intensity)) or 1.0
        self.entries.append(
            LibraryEntry(
                name=pattern.name,
                phase=phase,
                intensity=intensity / scale,
                march_dollase=march_dollase,
                fraction=fraction,
                csds_mean=csds_mean,
                thickness=thickness,
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
            entries = [
                LibraryEntry(
                    name=str(name),
                    phase=str(phase),
                    intensity=np.asarray(row, dtype=float),
                    march_dollase=float(orientation),
                    fraction=None if np.isnan(fraction) else float(fraction),
                    csds_mean=None if np.isnan(size) else float(size),
                    thickness=None if np.isnan(spacing) else float(spacing),
                    normalization=float(scale),
                    unit_mass=None if np.isnan(mass) else float(mass),
                    unit_volume=None if np.isnan(volume) else float(volume),
                    metadata=metadata,
                )
                for (name, phase, row, orientation, fraction, size, spacing, scale, mass,
                     volume, metadata) in zip(
                    data["names"],
                    data["phases"],
                    data["intensity"],
                    data["march_dollase"],
                    fractions,
                    sizes,
                    spacings,
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
    csds_means: tuple[float, ...] = CSDS_MEANS,
    csds_beta: float = 0.35,
    host_thicknesses: dict[str, tuple[float, ...]] | None = None,
    smectite_thickness: float | None = None,
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
            "orientations": list(orientations),
            "illite_smectite": list(illite_smectite),
            "chlorite_smectite": list(chlorite_smectite),
            "chlorite_iron": [list(pair) for pair in chlorite_iron],
            "csds_means": [distribution.mean for distribution in distributions],
            "csds_beta": csds_beta,
            "host_thicknesses": {key: list(value) for key, value in host_thicknesses.items()},
            "smectite_source": "Reynolds (1965) Am. Mineral. 50, 990-1001",
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
        if key == "chlorite" and chlorite_iron:
            variants = [
                (chlorite_crystal(a, b), f" Fe={a:g}/{b:g}") for a, b in chlorite_iron
            ]
        for base, iron_tag in variants:
            spacings = host_thicknesses.get(key) or (base.d001 / source.layers_per_cell,)
            announce(f"{key}{iron_tag}: {len(orientations)} orientations "
                     f"x {len(spacings)} layer spacings")
            for thickness in spacings:
                crystal = scaled_to_d001(base, source.layers_per_cell, thickness)
                spacing_tag = f" d={thickness:g}" if len(spacings) > 1 else ""
                for r in orientations:
                    pattern = powder_pattern(
                        crystal,
                        extended,
                        instrument,
                        r_march_dollase=r,
                        name=f"{key}{iron_tag} PO={r:g}{spacing_tag}",
                    )
                    library.add(pattern, phase=key, march_dollase=r, thickness=thickness,
                                unit_mass=crystal.cell_mass, unit_volume=crystal.volume)

    # Pure glycolated smectite.  All its reflections are basal, so the
    # orientation parameter only scales the pattern and one entry suffices.
    smectite = eg_smectite_layer(
        smectite_thickness if smectite_thickness is not None else eg_smectite_layer().thickness
    )
    announce("smectite_EG: 1 pattern")
    library.add(
        basal_pattern(
            MixedLayerStack(smectite, smectite, 1.0, csds=csds, name="smectite_EG"),
            extended,
            instrument,
            r_march_dollase=1.0,
            name="smectite_EG",
        ),
        phase="smectite_EG",
        march_dollase=1.0,
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
        base_host = load_crystal(host_key)
        base_layer = load_layer(host_key)
        layers_per_cell = CIF_SOURCES[host_key].layers_per_cell
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
                    for r in orientations:
                        pattern = mixed_layer_pattern(
                            stack,
                            host,
                            layers_per_cell,
                            extended,
                            instrument,
                            r_march_dollase=r,
                            name=(
                                f"{label} {composition} PO={r:g}"
                                + (f" N={csds.mean:g}" if len(distributions) > 1 else "")
                                + spacing_tag
                            ),
                            basal_scale=scale,
                            host_reflections=host_reflections,
                        )
                        library.add(
                            pattern,
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
        "--specimen-length", type=float, default=20.0, help="specimen length along the beam in mm"
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
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    iron: tuple[tuple[float, float], ...] = CHLORITE_IRON
    if arguments.chlorite_iron:
        pairs = []
        for part in arguments.chlorite_iron.split(","):
            part = part.strip()
            if not part:
                continue
            halves = part.split("/")
            if len(halves) != 2:
                parser.error(
                    f"{part!r} is not an iron pair; write each as (2:1 sheet)/(hydroxide "
                    "sheet), for example 0.35/0.20"
                )
            try:
                a, b = (float(half) for half in halves)
            except ValueError:
                parser.error(f"{part!r} is not a pair of numbers")
            if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
                parser.error(f"{part!r}: an occupancy lies between 0 and 1")
            pairs.append((a, b))
        iron = tuple(pairs)

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
        )
    )
    instrument = Instrument(
        emission=CU_KA_5LINE,
        peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0),
        lp_mode="powder",
        lines_per_emission=arguments.emission_subsampling,
        divergence=divergence,
    )
    library = build_library(
        grid=two_theta_grid(arguments.start, arguments.stop, arguments.step),
        instrument=instrument,
        chlorite_iron=iron,
        csds_means=tuple(arguments.csds_means),
        csds_beta=arguments.csds_beta,
        smectite_thickness=arguments.smectite_thickness,
        progress=not arguments.quiet,
    )
    path = library.save(arguments.out)
    if not arguments.quiet:
        print(library.describe())
        print(f"written to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
