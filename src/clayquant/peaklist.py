"""Patterns and a library built from measured peak lists.

Everything else in ClayQuant calculates a pattern from atomic positions.  This
module does the other thing: it takes a list of *d*-spacings and relative
intensities - the form a powder diffraction file entry is in - and builds a
pattern from that, so a library can be made of entries for which no structure
is available.

Why this exists and what it cannot do
-------------------------------------
ICDD's PDF-2 - the set a local HighScore installation usually searches, and the
one whose entry numbers look like ``00-012-0219`` - holds, per entry, a table of
interplanar spacings, relative intensities and often the diffraction indices.
It holds no atomic coordinates; those are in PDF-4+.  So an entry from it can
give a pattern, and cannot give a structure.

That difference decides what a peak-list library is good for:

* **Accompanying minerals: yes, and possibly better.**  Quartz, the feldspars,
  calcite and the rest enter the fit as discrete phases at a fixed orientation,
  and a file entry's intensities are *measured*, on a real specimen with its
  real substitutions.  A calculated pattern from one published structure need
  not describe the specimen in hand any better.
* **The interstratified clays: no, and not by any amount of work.**  An
  illite/smectite pattern is not a sum of reflections at all.  It comes from the
  matrix method (:mod:`clayquant.mixed_layer`), which needs the layer structure
  factor as a *continuous* function of the scattering vector, and that needs
  scatterers at heights within the layer.  A table of spacings cannot supply it.
  Nor can it supply the cell content that turns a fitted scale factor into a
  mass (:mod:`clayquant.masses`), so a peak-list phase has no weight percent of
  its own.

So this is an addition and not a replacement: the clay library is still built
from structures by :mod:`clayquant.library`, and a peak-list library is a
separate file, built separately, and useful beside it.

Licensing
---------
A powder diffraction file is licensed data.  A library built from one is a
derived work of it and belongs with the user's own files, not in a repository
and not in anything shared - the same position the ICSD structures are in
(&sect;4.2 of the manual).  Nothing here ships with ClayQuant; the input is
supplied by whoever holds the licence.

The input format
----------------
One block per phase, in a plain text file::

    PHASE   Montmorillonite-18A
    SOURCE  ICDD PDF 00-012-0219
    CELL    5.17 8.94 15.0 90 90 90
    #   d(A)     I    h  k  l
      15.000   100    0  0  1
       4.490    80    1  1  0
       2.570    60

``PHASE`` starts a block and is the only required line.  ``SOURCE`` is free text
and is carried through to the pattern's metadata, so a fit can be traced back to
the entry it came from.  ``CELL`` takes the six cell parameters and is needed
only for preferred orientation, which cannot be applied without it.  Then two
or more whitespace-separated columns: the spacing in angstrom, the relative
intensity, and optionally the indices, either as three columns or as one token
like ``001``.  Blank lines and ``#`` comments are ignored, and a block ends at
the next ``PHASE`` or at the end of the file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .crystal import Crystal
from .pattern import Instrument, Reflections, _build_from_reflections
from .pattern import two_theta_grid

__all__ = [
    "PeakList",
    "build_peaklist_library",
    "peaklist_pattern",
    "peaklist_reflections",
    "read_peak_lists",
]


@dataclass
class PeakList:
    """A measured peak list: spacings, relative intensities, and what they are.

    Attributes
    ----------
    d:
        Interplanar spacings in angstrom.
    intensity:
        Relative intensities as the file gives them - observed, so already
        carrying the Lorentz-polarization factor and the multiplicity.
    hkl:
        Diffraction indices, or ``None`` where the entry is unindexed.  Without
        them preferred orientation cannot be applied, because the angle between
        a reflection and ``c*`` is not known.
    cell:
        The six cell parameters, or ``None``.  Also needed only for orientation.
    """

    name: str
    d: np.ndarray
    intensity: np.ndarray
    hkl: np.ndarray | None = None
    cell: tuple[float, float, float, float, float, float] | None = None
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.d = np.atleast_1d(np.asarray(self.d, dtype=float))
        self.intensity = np.atleast_1d(np.asarray(self.intensity, dtype=float))
        if self.d.shape != self.intensity.shape:
            raise ValueError(
                f"{self.name}: {self.d.size} spacings but {self.intensity.size} intensities"
            )
        if self.d.size == 0:
            raise ValueError(f"{self.name}: no peaks")
        if np.any(self.d <= 0.0):
            raise ValueError(f"{self.name}: a spacing is zero or negative")
        if np.any(self.intensity < 0.0):
            raise ValueError(f"{self.name}: a negative intensity")
        if self.hkl is not None:
            self.hkl = np.atleast_2d(np.asarray(self.hkl, dtype=float))
            if self.hkl.shape != (self.d.size, 3):
                raise ValueError(
                    f"{self.name}: {self.hkl.shape[0]} index triples for {self.d.size} peaks"
                )
        if self.cell is not None:
            if len(self.cell) != 6:
                raise ValueError(f"{self.name}: a cell needs six parameters")
            self.cell = tuple(float(value) for value in self.cell)

    @property
    def can_orient(self) -> bool:
        """Whether preferred orientation can be applied to this entry."""
        return self.hkl is not None and self.cell is not None

    def two_theta(self, wavelength: float) -> np.ndarray:
        """Where these spacings fall at this wavelength, in degrees.

        A spacing below half the wavelength diffracts at no angle at all, and
        comes back as ``nan`` rather than being clipped to 180 degrees.  Clipping
        it was a real defect: the unreachable reflection survived every finite
        test and was placed at the end of the pattern, where nothing looks.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            argument = wavelength / (2.0 * self.d)
            return np.where(
                argument < 1.0,
                np.degrees(2.0 * np.arcsin(np.where(argument < 1.0, argument, 0.0))),
                np.nan,
            )


def _crystal_for(peaks: PeakList) -> Crystal:
    """A cell with no contents, for the geometry alone.

    Only the metric is wanted - the angle each reflection makes with ``c*`` -
    and that is a property of the cell, so the sites can be empty.  Nothing
    downstream asks this object for a structure factor or a mass.
    """
    if peaks.cell is None:
        raise ValueError(f"{peaks.name}: no cell, so no orientation can be computed")
    a, b, c, alpha, beta, gamma = peaks.cell
    return Crystal(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, sites=[],
                   name=peaks.name, source=peaks.source)


def peaklist_reflections(
    peaks: PeakList,
    instrument: Instrument | None = None,
) -> Reflections:
    """Turn a measured peak list into a reflection list the assembler can use.

    One correction is applied and it is not cosmetic.  A file entry's
    intensities are *observed*, so they already contain the
    Lorentz-polarization factor at the angle each peak was measured at, while
    :func:`~clayquant.pattern._build_from_reflections` applies that factor
    itself.  Handing the observed numbers over unchanged would apply it twice -
    a factor of thirty across a 4 to 40 degree scan, which would tilt the whole
    pattern.  So the observed intensity is divided by the factor here, leaving a
    structure-factor-like quantity for the assembler to put back.  The pattern
    that comes out therefore matches the entry at the entry's own geometry, and
    changes correctly if the instrument description changes.
    """
    instrument = instrument or Instrument()
    wavelength = instrument.emission.principal_wavelength
    two_theta = peaks.two_theta(wavelength)
    reachable = np.isfinite(two_theta) & (two_theta > 0.0)

    lp = instrument.lp(two_theta[reachable])
    lp = np.where(lp > 0.0, lp, 1.0)
    f_squared = peaks.intensity[reachable] / lp

    if peaks.hkl is not None:
        hkl = peaks.hkl[reachable]
    else:
        # Unindexed: the indices are only used for the angle to c*, which is
        # then refused rather than guessed (see peaklist_pattern).
        hkl = np.zeros((int(reachable.sum()), 3), dtype=float)

    if peaks.can_orient:
        alpha = _crystal_for(peaks).angle_to_cstar(hkl)
    else:
        # Zero means "along c*", which is wrong for most reflections - and
        # harmless, because this value is only reached at r = 1 where the
        # March-Dollase factor is 1 for every angle.
        alpha = np.zeros(hkl.shape[0], dtype=float)

    return Reflections(hkl=hkl, d=peaks.d[reachable], f_squared=f_squared, alpha=alpha)


def peaklist_pattern(
    peaks: PeakList,
    grid: np.ndarray | None = None,
    instrument: Instrument | None = None,
    r_march_dollase: float = 1.0,
    name: str = "",
):
    """Build a pattern from a measured peak list.

    ``r_march_dollase`` other than 1 needs the indices and the cell, and raises
    without them: an orientation correction applied to reflections whose angle
    to ``c*`` is unknown would be a number with no meaning, and silently
    treating them as basal would enhance every peak in the entry alike.
    """
    from .pattern import Pattern

    instrument = instrument or Instrument()
    grid = two_theta_grid() if grid is None else np.asarray(grid, dtype=float)
    if not math.isclose(r_march_dollase, 1.0) and not peaks.can_orient:
        missing = "indices" if peaks.hkl is None else "cell"
        raise ValueError(
            f"{peaks.name}: preferred orientation needs the {missing}, which this "
            f"entry does not carry; fit it as a random powder (r = 1) instead"
        )

    found = peaklist_reflections(peaks, instrument)
    intensity = _build_from_reflections(found, grid, instrument, r_march_dollase)
    return Pattern(
        two_theta=grid,
        intensity=intensity,
        name=name or f"{peaks.name} (peak list, PO r={r_march_dollase:g})",
        metadata={
            "kind": "peak list",
            "phase": peaks.name,
            "source": peaks.source,
            "march_dollase": r_march_dollase,
            "n_reflections": int(found.d.size),
            "indexed": peaks.hkl is not None,
            "emission": instrument.emission.name,
            "lp_mode": instrument.lp_mode,
        },
    )


def read_peak_lists(path: str | Path) -> list[PeakList]:
    """Read the peak-list format described in this module's documentation.

    Deliberately forgiving about spacing and column count and strict about
    everything else: a line that looks like data but cannot be read raises with
    its number, because a peak list silently one row short is a pattern that is
    wrong in a way nothing downstream can notice.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks: list[PeakList] = []
    name = source = ""
    cell: tuple | None = None
    rows: list[tuple[float, float, tuple[float, float, float] | None]] = []

    def finish() -> None:
        if not name:
            if rows:
                raise ValueError(f"{path}: peaks before any PHASE line")
            return
        if not rows:
            raise ValueError(f"{path}: PHASE {name} has no peaks")
        indexed = [row[2] for row in rows]
        hkl = None if any(item is None for item in indexed) else np.array(indexed, dtype=float)
        blocks.append(
            PeakList(
                name=name,
                d=np.array([row[0] for row in rows], dtype=float),
                intensity=np.array([row[1] for row in rows], dtype=float),
                hkl=hkl,
                cell=cell,
                source=source,
            )
        )

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        keyword, _, rest = line.partition(" ")
        upper = keyword.upper()
        if upper == "PHASE":
            finish()
            name, source, cell, rows = rest.strip(), "", None, []
            if not name:
                raise ValueError(f"{path}:{number}: PHASE with no name")
            continue
        if upper == "SOURCE":
            source = rest.strip()
            continue
        if upper == "CELL":
            parts = rest.split()
            if len(parts) != 6:
                raise ValueError(
                    f"{path}:{number}: CELL needs six parameters, found {len(parts)}"
                )
            try:
                cell = tuple(float(part) for part in parts)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            continue
        if upper == "END":
            continue

        parts = line.split()
        try:
            spacing = float(parts[0])
            relative = float(parts[1]) if len(parts) > 1 else 100.0
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{path}:{number}: cannot read a peak from {line!r}") from exc
        indices: tuple[float, float, float] | None = None
        if len(parts) >= 5:
            try:
                indices = (float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: cannot read indices from {line!r}") from exc
        elif len(parts) == 3:
            token = parts[2]
            # A single token like 001, or -1-11 with signs attached.
            digits = _split_indices(token)
            if digits is None:
                raise ValueError(f"{path}:{number}: cannot read indices from {token!r}")
            indices = digits
        rows.append((spacing, relative, indices))

    finish()
    if not blocks:
        raise ValueError(f"{path}: no PHASE blocks found")
    return blocks


def _split_indices(token: str) -> tuple[float, float, float] | None:
    """Read ``001``, ``1 1 0`` joined up, or ``-1-11`` into three indices."""
    values: list[float] = []
    sign = 1.0
    for character in token:
        if character == "-":
            sign = -1.0
            continue
        if not character.isdigit():
            return None
        values.append(sign * float(character))
        sign = 1.0
    return (values[0], values[1], values[2]) if len(values) == 3 else None


def build_peaklist_library(
    peak_lists: list[PeakList],
    grid: np.ndarray | None = None,
    instrument: Instrument | None = None,
    orientations: tuple[float, ...] = (1.0,),
):
    """A separate :class:`~clayquant.library.PatternLibrary` of peak-list phases.

    Separate on purpose.  The clay library is built from structures and carries
    the mass of what each pattern was calculated from, which is what makes a
    weight percent possible; these entries carry no mass, so mixing them into
    that file would put patterns of two different kinds behind one interface and
    make the weight percent quietly wrong for some of them.  Kept apart, a
    peak-list library is exactly what it looks like: a set of reference patterns
    for identification and for fitting the accompanying minerals.

    ``orientations`` spans the March-Dollase parameter as the clay library does.
    An unindexed entry can only be a random powder, so it contributes one
    pattern at ``r = 1`` whatever is asked for, and the library records which
    entries those were.
    """
    from .library import PatternLibrary

    instrument = instrument or Instrument()
    grid = two_theta_grid() if grid is None else np.asarray(grid, dtype=float)
    library = PatternLibrary(two_theta=grid, entries=[], metadata={
        "kind": "peak list",
        "note": "Reference patterns from measured peak lists; no masses, so no "
                "weight percent. Derived from licensed data - keep it with your "
                "own files.",
    })

    unindexed: list[str] = []
    for peaks in peak_lists:
        wanted = orientations if peaks.can_orient else (1.0,)
        if not peaks.can_orient and orientations != (1.0,):
            unindexed.append(peaks.name)
        for r in wanted:
            pattern = peaklist_pattern(peaks, grid, instrument, r_march_dollase=r)
            if float(np.max(pattern.intensity)) <= 0.0:
                continue
            library.add(pattern, phase=peaks.name, march_dollase=r)
    library.metadata["unindexed"] = unindexed
    library.metadata["n_phases"] = len({entry.phase for entry in library.entries})
    return library
