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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .background import snip_baseline
from .emission import CU_KA_5LINE
from .mixed_layer import CSDS, MixedLayerStack, lognormal_csds
from .optics import Divergence
from .models import CIF_SOURCES, available_phases, eg_smectite_layer, load_crystal, load_layer
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
    "NORMALIZATION_FLOOR",
    "main",
]

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
)
"""Illite fraction of the illite/smectite series."""

CHLORITE_SMECTITE_FRACTIONS: tuple[float, ...] = (0.95, 0.90, 0.85)
"""Chlorite fraction of the chlorite/smectite series."""

NORMALIZATION_FLOOR = 4.0
"""Patterns are normalised on their maximum above this 2theta, in degrees."""


@dataclass
class LibraryEntry:
    """One calculated pattern in the library."""

    name: str
    phase: str
    intensity: np.ndarray
    march_dollase: float = 1.0
    fraction: float | None = None
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
                intensity - snip_baseline(pattern.two_theta, intensity, window=6.0), 0.0, None
            )
        usable = pattern.two_theta >= normalization_floor
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
            entry_metadata=json.dumps([entry.metadata for entry in self.entries], default=str),
            library_metadata=json.dumps(self.metadata, default=str),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PatternLibrary":
        with np.load(Path(path), allow_pickle=True) as data:
            entry_metadata = json.loads(str(data["entry_metadata"]))
            fractions = data["fraction"]
            entries = [
                LibraryEntry(
                    name=str(name),
                    phase=str(phase),
                    intensity=np.asarray(row, dtype=float),
                    march_dollase=float(orientation),
                    fraction=None if np.isnan(fraction) else float(fraction),
                    metadata=metadata,
                )
                for name, phase, row, orientation, fraction, metadata in zip(
                    data["names"],
                    data["phases"],
                    data["intensity"],
                    data["march_dollase"],
                    fractions,
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


def build_library(
    grid: np.ndarray | None = None,
    instrument: Instrument | None = None,
    orientations: tuple[float, ...] = PREFERRED_ORIENTATIONS,
    illite_smectite: tuple[float, ...] = ILLITE_SMECTITE_FRACTIONS,
    chlorite_smectite: tuple[float, ...] = CHLORITE_SMECTITE_FRACTIONS,
    csds: CSDS | None = None,
    smectite_thickness: float | None = None,
    progress: bool = False,
) -> PatternLibrary:
    """Calculate the full reference library.

    Parameters
    ----------
    csds:
        Crystallite size distribution along ``c*`` used for the interstratified
        stacks; defaults to a lognormal distribution with a mean of 10 layers.
    smectite_thickness:
        Layer repeat of the glycolated smectite in A; defaults to the 16.86 A
        measured by Reynolds (1965).
    """
    grid = two_theta_grid(2.0, 40.0, 0.02) if grid is None else np.asarray(grid, dtype=float)
    instrument = instrument or Instrument(
        emission=CU_KA_5LINE,
        peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0),
        lp_mode="powder",
        divergence=Divergence(),
    )
    csds = csds or lognormal_csds(10.0)

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
            "csds_mean": csds.mean,
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

    # Discrete phases at each orientation parameter.
    for key in CIF_SOURCES:
        crystal = load_crystal(key)
        announce(f"{key}: {len(orientations)} orientations")
        for r in orientations:
            pattern = powder_pattern(
                crystal, grid, instrument, r_march_dollase=r, name=f"{key} PO={r:g}"
            )
            library.add(pattern, phase=key, march_dollase=r)

    # Pure glycolated smectite.  All its reflections are basal, so the
    # orientation parameter only scales the pattern and one entry suffices.
    smectite = eg_smectite_layer(
        smectite_thickness if smectite_thickness is not None else eg_smectite_layer().thickness
    )
    announce("smectite_EG: 1 pattern")
    library.add(
        basal_pattern(
            MixedLayerStack(smectite, smectite, 1.0, csds=csds, name="smectite_EG"),
            grid,
            instrument,
            r_march_dollase=1.0,
            name="smectite_EG",
        ),
        phase="smectite_EG",
        march_dollase=1.0,
        fraction=0.0,
    )

    # Interstratified series.
    for host_key, fractions, label in (
        ("illite", illite_smectite, "I/S"),
        ("chlorite", chlorite_smectite, "C/S"),
    ):
        host = load_crystal(host_key)
        host_layer = load_layer(host_key)
        layers_per_cell = CIF_SOURCES[host_key].layers_per_cell
        scale = basal_scale_factor(host, layers_per_cell, grid, instrument, csds)
        wavelengths, _ = instrument.sample_emission()
        d_min = float(np.max(wavelengths)) / (2.0 * math.sin(math.radians(grid[-1] / 2.0)))
        host_reflections = reflections(host, d_min)
        announce(f"{label}: {len(fractions)} compositions x {len(orientations)} orientations")

        for fraction in fractions:
            stack = MixedLayerStack(
                host_layer,
                smectite,
                fraction_a=fraction,
                csds=csds,
                name=f"{label} {fraction:.2f}/{1.0 - fraction:.2f}",
            )
            for r in orientations:
                pattern = mixed_layer_pattern(
                    stack,
                    host,
                    layers_per_cell,
                    grid,
                    instrument,
                    r_march_dollase=r,
                    name=f"{label} {fraction:.2f}/{1.0 - fraction:.2f} PO={r:g}",
                    basal_scale=scale,
                    host_reflections=host_reflections,
                )
                library.add(pattern, phase=label, march_dollase=r, fraction=fraction)

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
        "--csds-mean", type=float, default=10.0, help="mean number of layers per crystallite"
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
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

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
        csds=lognormal_csds(arguments.csds_mean, arguments.csds_beta),
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
