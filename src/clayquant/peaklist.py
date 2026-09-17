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
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .crystal import Crystal
from .pattern import Instrument, Reflections, _build_from_reflections
from .pattern import two_theta_grid

__all__ = [
    "PeakList",
    "PeakListComparison",
    "compare_with_calculated",
    "build_peaklist_library",
    "peaklist_pattern",
    "peaklist_reflections",
    "read_highscore_card",
    "read_peak_lists",
    "write_peak_lists",
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


@dataclass
class PeakListComparison:
    """How a peak list and a calculated pattern agree, line by line."""

    name: str
    source: str
    matched: int
    total: int
    worst_deviation: float
    mean_absolute_deviation: float
    intensity_agreement: float
    lines: list[tuple[float, float, float | None, float | None, float]]
    """``(d_obs, I_obs, d_calc, I_calc, deviation in degrees)`` per line."""

    unexplained_calculated: list[tuple[float, float]]
    """Calculated lines above 10 % that no line of the entry accounts for."""

    def summary(self) -> str:
        return (
            f"{self.name}: {self.matched} of {self.total} lines matched, "
            f"worst {self.worst_deviation:.3f} deg, mean |dev| "
            f"{self.mean_absolute_deviation:.3f} deg, intensity agreement "
            f"{self.intensity_agreement:.2f}"
        )


def compare_with_calculated(
    peaks: PeakList,
    calculated: tuple[np.ndarray, np.ndarray],
    instrument: Instrument | None = None,
    tolerance: float = 0.20,
    minimum_intensity: float = 0.10,
) -> PeakListComparison:
    """Score a measured peak list against a calculated peak list.

    ``calculated`` is ``(two_theta, relative intensity)`` as
    :func:`~clayquant.pattern.peak_list` returns it, so the comparison is
    against the same merged, Lorentz-corrected quantity a file entry reports
    rather than against bare structure factors.

    What the numbers mean.  The positions are compared in degrees rather than in
    angstrom, because that is where a disagreement matters and because an
    angstrom at 5 degrees and an angstrom at 35 are not the same thing at all.
    ``intensity_agreement`` is one minus the mean absolute difference of the two
    intensity sets, each normalised to its own strongest matched line, so 1 is
    perfect and 0 is no relation; it is computed over matched lines only, and
    ``unexplained_calculated`` carries what the entry has no line for, which is
    the half of the comparison an agreement number hides.
    """
    instrument = instrument or Instrument()
    wavelength = instrument.emission.principal_wavelength
    observed_angle = peaks.two_theta(wavelength)
    calculated_angle, calculated_intensity = (
        np.asarray(calculated[0], dtype=float),
        np.asarray(calculated[1], dtype=float),
    )

    lines: list[tuple[float, float, float | None, float | None, float]] = []
    deviations: list[float] = []
    pairs: list[tuple[float, float]] = []
    used: set[int] = set()
    for spacing, height, angle in zip(peaks.d, peaks.intensity, observed_angle):
        if not np.isfinite(angle) or calculated_angle.size == 0:
            lines.append((float(spacing), float(height), None, None, float("nan")))
            continue
        index = int(np.argmin(np.abs(calculated_angle - angle)))
        deviation = float(calculated_angle[index] - angle)
        if abs(deviation) > tolerance:
            lines.append((float(spacing), float(height), None, None, float("nan")))
            continue
        used.add(index)
        matched_d = float(wavelength / (2.0 * np.sin(np.radians(calculated_angle[index] / 2.0))))
        lines.append((float(spacing), float(height), matched_d,
                      float(calculated_intensity[index]), deviation))
        deviations.append(deviation)
        pairs.append((float(height), float(calculated_intensity[index])))

    if pairs:
        observed = np.array([pair[0] for pair in pairs], dtype=float)
        expected = np.array([pair[1] for pair in pairs], dtype=float)
        observed = observed / (observed.max() or 1.0)
        expected = expected / (expected.max() or 1.0)
        agreement = float(max(0.0, 1.0 - np.mean(np.abs(observed - expected))))
    else:
        agreement = 0.0

    unexplained = [
        (float(wavelength / (2.0 * np.sin(np.radians(angle / 2.0)))), float(height))
        for index, (angle, height) in enumerate(zip(calculated_angle, calculated_intensity))
        if index not in used and height >= minimum_intensity
    ]

    return PeakListComparison(
        name=peaks.name,
        source=peaks.source,
        matched=len(pairs),
        total=int(peaks.d.size),
        worst_deviation=float(np.max(np.abs(deviations))) if deviations else float("nan"),
        mean_absolute_deviation=float(np.mean(np.abs(deviations))) if deviations
        else float("nan"),
        intensity_agreement=agreement,
        lines=lines,
        unexplained_calculated=sorted(unexplained, key=lambda item: -item[1]),
    )


def _rtf_to_text(raw: str) -> str:
    """Flatten an RTF document to the text a reader sees.

    Not an RTF implementation and not trying to be one.  A reference card is a
    single-level document of paragraphs and tabs, so the four control words that
    carry its layout are translated, the escaped characters are decoded, and
    every other control word is dropped along with the group braces.  Anything
    more elaborate in the file is ignored, which for a card is the right
    outcome: the text is what matters and the fonts are not.
    """
    text = re.sub(
        r"\\u(-?\d+)\\?'?[0-9a-fA-F]{0,2}",
        lambda match: chr(int(match.group(1)) % 65536),
        raw,
    )
    for control, replacement in (
        (r"\pard", ""),
        (r"\tab", "\t"),
        (r"\par", "\n"),
        (r"\line", "\n"),
    ):
        text = text.replace(control, replacement)
    text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda match: chr(int(match.group(1), 16)), text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]+\n", "\n", text)


def _lines(text: str) -> list[str]:
    """Split on the three line endings a file actually uses.

    Not ``str.splitlines``, which also breaks at the vertical tab, the form
    feed and - the one that matters here - ``U+0085``.  A card written in UTF-8
    and decoded as a single-byte code page turns every angstrom sign into
    ``\xc3\x85``, and that second byte made ``str.splitlines`` cut the peak
    table's header in half, so the table could not be found at all.  Being
    explicit about the three endings costs nothing and removes the whole class.
    """
    return re.split(r"\r\n|\r|\n", text)


_FIELD = re.compile(r"^\s*([A-Za-z][^:]*?)\s*:+\s*(.*)$")
_CELL_LENGTH = re.compile(r"^([abc])\s*\(")
_CELL_ANGLE = re.compile(r"^(alpha|beta|gamma)\s*\(")


def _card_fields(text: str) -> dict[str, str]:
    """Every ``label: value`` line of a card, keyed by its lowercased label.

    A card repeats some labels - ``Color`` and the calculated-pattern remarks
    appear more than once - and the repeats are joined with ``"; "`` rather than
    overwritten, because the second remark is as much of the provenance as the
    first.
    """
    fields: dict[str, str] = {}
    for line in _lines(text):
        found = _FIELD.match(line)
        if found is None:
            continue
        label, value = found.group(1).strip().lower(), found.group(2).strip()
        if not value:
            continue
        if label in fields:
            if value not in fields[label]:
                fields[label] = f"{fields[label]}; {value}"
        else:
            fields[label] = value
    return fields


def _card_cell(fields: dict[str, str]) -> tuple[float, float, float, float, float, float] | None:
    """The six cell parameters of a card, or ``None`` if any is missing.

    A card writes an unknown quantity as ``-1.00`` rather than leaving it out,
    so a non-positive parameter is treated as absent; that is what keeps a cell
    that cannot be a cell from reaching :class:`PeakList` and being used for
    orientation.
    """
    lengths: dict[str, float] = {}
    angles: dict[str, float] = {}
    for label, value in fields.items():
        head = value.split()
        try:
            magnitude = float(head[0]) if head else float("nan")
        except ValueError:
            continue
        if _CELL_LENGTH.match(label):
            lengths[label[0]] = magnitude
        elif _CELL_ANGLE.match(label):
            angles[label.split()[0]] = magnitude
    if len(lengths) != 3 or len(angles) != 3:
        return None
    if any(value <= 0.0 for value in (*lengths.values(), *angles.values())):
        return None
    return (
        lengths["a"], lengths["b"], lengths["c"],
        angles["alpha"], angles["beta"], angles["gamma"],
    )


def _table_blob(text: str) -> str:
    """The run-together peak table of a card, header and trailer removed.

    A card's table arrives as one unbroken line - ``No.hkld [A]2th [deg]I
    [%]10017.0201412.599100.0`` and on - because the export writes the columns
    with tabs that the RTF's paragraph structure does not keep.  The header ends
    at its last bracket and no number in the table holds one, so that is where
    the digits start.
    """
    for line in _lines(text):
        stripped = line.strip()
        if stripped.startswith("No.") and "]" in stripped:
            return stripped[stripped.rindex("]") + 1:].strip()
    raise ValueError("no peak table: expected a line beginning 'No.'")


def _written_plainly(integer_part: str) -> bool:
    """Whether this is how a formatter writes the whole part of a number.

    Digits, at least one, and no redundant leading zero: ``4``, ``17`` and ``0``
    but not ``04``.  It reads like pedantry and it is the whole of what keeps
    the index column and the spacing column apart, because ``4.45298`` and
    ``04.45298`` are the same number, so without this rule the reading that
    takes a zero out of the indices and gives it to the spacing is admissible
    and indistinguishable - and it turns ``020`` into ``02``, which is no longer
    three indices.
    """
    return integer_part.isdigit() and (len(integer_part) == 1 or integer_part[0] != "0")


def _row_splits(
    blob: str, start: int, number: int, wavelength: float, tolerance: float
) -> list[tuple[tuple[int, int, int], int, str, float, float]]:
    """Every reading of one table line that Bragg's law and the next index allow.

    A candidate is ``(decimal signature, where it ends, the index token, d,
    intensity)``.  The signature is the number of decimal places the reading
    implies for the spacing, the angle and the intensity, and it is what
    :func:`_card_rows` uses to tie the whole table to one reading.
    """
    dots = [index for index, character in enumerate(blob[start:], start=start)
            if character == "."][:3]
    if len(dots) < 3:
        return []
    first, second, third = dots
    found: list[tuple[tuple[int, int, int], int, str, float, float]] = []
    for lead in range(start, first):
        if not _written_plainly(blob[lead:first]):
            continue
        token = blob[start:lead]
        if token and not all(character.isdigit() or character == "-" for character in token):
            continue
        for d_places in range(1, second - first):
            spacing_text = blob[lead:first + 1 + d_places]
            if not _written_plainly(blob[first + 1 + d_places:second]):
                continue
            for angle_places in range(1, third - second):
                angle_text = blob[first + 1 + d_places:second + 1 + angle_places]
                if not _written_plainly(blob[second + 1 + angle_places:third]):
                    continue
                for height_places in range(1, len(blob) - third):
                    end = third + 1 + height_places
                    if not blob[third + 1:end].isdigit():
                        break
                    if end != len(blob) and not blob.startswith(str(number + 1), end):
                        continue
                    spacing = float(spacing_text)
                    height = float(blob[second + 1 + angle_places:end])
                    if spacing <= 0.0 or not 0.0 <= height <= 100.0:
                        continue
                    argument = wavelength / (2.0 * spacing)
                    if argument >= 1.0:
                        continue
                    if abs(math.degrees(2.0 * math.asin(argument)) - float(angle_text)) > tolerance:
                        continue
                    found.append((
                        (d_places, angle_places, height_places),
                        end, token, spacing, height,
                    ))
    return found


def _card_rows(
    blob: str, wavelength: float, tolerance: float = 0.03
) -> list[tuple[str, float, float]]:
    """Split the run-together table into ``(index token, d, intensity)`` a line.

    The table has no separators at all, so on the characters alone the split is
    genuinely ambiguous: ``10017.0201412.599100.0`` reads as line 1, ``001``,
    *d* = 7.02014 A, 12.599 deg, 100 %, and reads equally well, as far as digits
    go, as *d* = 17.02014 at the same angle, or as 12.5991 deg at 0.0 %.  Three
    things together settle it, and each is needed:

    *Bragg's law.*  The card prints the spacing and the angle both, and at a
    known wavelength they determine each other, which disposes of every reading
    that moves a digit across the first decimal point.

    *One signature for the whole table.*  A card is written by one formatter and
    does not change its column widths part way down, so the reading is required
    to use the same number of decimal places on every line and to consume the
    table exactly.  This is what disposes of readings that borrow the
    intensity's leading digit for the angle, which Bragg's law cannot see: an
    angle of 12.5991 agrees with *d* = 7.02014 quite as well as 12.599 does,
    and better, because the borrowed digit adds precision the card never
    printed.

    *A strongest line of 100 %.*  Intensities in a powder diffraction file are
    relative to the strongest reflection of the entry, so exactly that reading
    which puts a 100 in the table is the reading in which the intensity column
    is the intensity column.  On the cards tested this is the constraint that
    decides it, the other two having each left a handful of candidates.

    A table no reading satisfies raises rather than being guessed at, naming the
    line where the reading broke down.  Index tokens come back as text because
    ``0012`` is three indices in more than one way and it takes the cell to say
    which; :func:`_indices_from_token` does that afterwards.

    ``tolerance`` is loose on purpose.  A card rounds its angle to the decimals
    it prints but computed it from its own wavelength, which for an entry
    indexed decades ago need not be the one in use here to the last digit.
    """
    signatures: list[tuple[int, int, int]] = []
    for candidate in _row_splits(blob, len("1"), 1, wavelength, tolerance):
        if candidate[0] not in signatures:
            signatures.append(candidate[0])
    if not blob.startswith("1"):
        raise ValueError(f"peak table line 1: expected the index 1 at {blob[:24]!r}")
    if not signatures:
        raise ValueError(f"peak table line 1: cannot read a line from {blob[:32]!r}")

    complete: list[list[tuple[str, float, float]]] = []
    furthest, reason, furthest_reason = -1, "", ""
    for signature in signatures:
        rows: list[tuple[str, float, float]] = []
        position, number = 0, 1
        while position < len(blob):
            label = str(number)
            if not blob.startswith(label, position):
                reason = (f"peak table line {number}: expected the index {label} at "
                          f"{blob[position:position + 24]!r}")
                break
            matching = [
                candidate
                for candidate in _row_splits(
                    blob, position + len(label), number, wavelength, tolerance
                )
                if candidate[0] == signature
            ]
            if not matching:
                reason = (f"peak table line {number}: cannot read a line from "
                          f"{blob[position:position + 32]!r}")
                break
            _, end, token, spacing, height = matching[0]
            rows.append((token, spacing, height))
            position, number = end, number + 1
        else:
            complete.append(rows)
            continue
        # Keep the complaint from the reading that got furthest down the table.
        # Any other is an early reading of a line that was never the right one,
        # and pointing at it sends the reader to a line that is not the problem.
        if number > furthest:
            furthest, furthest_reason = number, reason
    if not complete:
        raise ValueError(furthest_reason or f"peak table: cannot read {blob[:32]!r}")

    scaled = [rows for rows in complete
              if any(abs(row[2] - 100.0) < 0.05 for row in rows)]
    if not scaled:
        raise ValueError(
            "peak table: no reading of it has a strongest line of 100 %, so the "
            "intensity column cannot be identified"
        )
    if len(scaled) > 1:
        lengths = {len(rows) for rows in scaled}
        if len(lengths) > 1 or any(rows != scaled[0] for rows in scaled[1:]):
            raise ValueError(
                f"peak table: {len(scaled)} readings of it are equally consistent"
            )
    return scaled[0]


def _indices_from_token(token: str, spacing: float, crystal: Crystal | None) -> tuple[
    tuple[float, float, float] | None, float
]:
    """Read a card's index token into three indices, with the cell to decide.

    ``0012`` is ``0 0 12`` and is also ``0 1 2`` with a leading zero and is also
    ``00 1 2``; the card writes the indices with no separator and leaves it at
    that.  What decides is the spacing printed beside them: every way of
    cutting the token into three signed integers is enumerated, the spacing each
    would have in this cell is computed, and the one that matches the printed
    spacing is the indexing meant.  That is a check as much as a choice - a
    token whose best split is off by more than a few per mille is reported, not
    accepted - and it needs no convention about how the card pads its columns.

    Returns the indices and the relative spacing error of the split chosen, so
    the caller can say how well the card's own indexing holds together.  With no
    cell the single-digit reading is taken where there is one, which is what a
    three-character token always is, and otherwise nothing.
    """
    splits = _token_splits(token)
    if not splits:
        return None, float("nan")
    if crystal is None:
        single = [item for item in splits
                  if all(abs(value) < 10 for value in item)]
        return (single[0] if len(single) == 1 else None), float("nan")
    candidates = np.array(splits, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        computed = crystal.d_spacing(candidates)
    error = np.abs(computed - spacing) / spacing
    best = int(np.nanargmin(np.where(np.isfinite(error), error, np.inf)))
    return tuple(float(value) for value in candidates[best]), float(error[best])


def _token_splits(token: str) -> list[tuple[float, float, float]]:
    """Every way of cutting an index token into three signed integers.

    A part is an optional minus sign and then one or more digits, and a lone
    minus sign is not a part, so ``-1-11`` cuts one way and ``0012`` four.
    """
    if not token:
        return []
    parts: list[list[str]] = []

    def walk(rest: str, taken: list[str]) -> None:
        if len(taken) == 3:
            if not rest:
                parts.append(list(taken))
            return
        for length in range(1, len(rest) + 1):
            piece = rest[:length]
            if piece.lstrip("-").isdigit() and piece.count("-") <= 1 and (
                "-" not in piece[1:]
            ):
                walk(rest[length:], [*taken, piece])

    walk(token, [])
    return [(float(one), float(two), float(three)) for one, two, three in parts]


def _card_text(path: Path) -> str:
    """Decode a card, which may be UTF-8, may be a Windows code page, or ASCII.

    RTF is seven-bit and carries its accented characters as escapes, so it
    decodes either way; a plain-text export does not, and getting its encoding
    wrong is not merely cosmetic - see :func:`_lines`.  UTF-8 is tried first
    because it is what a modern export writes and because it fails loudly on
    anything that is not UTF-8, and CP1252 is the fallback because it decodes
    every byte and is what a Windows tool writes when it is not writing UTF-8.
    """
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def read_highscore_card(path: str | Path, name: str = "") -> PeakList:
    """Read one reference card as a powder-file front end exports it.

    This is the format that comes out of a local installation when a single
    entry is printed or saved from the reference-pattern view, in RTF or as
    plain text: a block of ``label: value`` header lines, then the peak table.
    It is read here rather than converted by hand because the alternative -
    retyping a hundred and forty-five lines - is where the errors would come
    from.

    The entry's name is its mineral name where it has one, falling back to the
    compound name and then to the file's stem, and ``source`` records the
    reference code, which is what makes a fit traceable to the entry.  Every
    header field is kept in ``metadata`` under its lowercased label, including
    the ones nothing here reads: the quality mark and the sample-preparation
    line decide whether an entry means anything for oriented work, and whoever
    reads the fit has to be able to see them.

    Unindexed entries are common in this format - a card marked *Indexed (I)*
    may print no indices at all - and come back with ``hkl`` as ``None``, so
    :attr:`PeakList.can_orient` is false and the entry can enter a fit only as a
    random powder.
    """
    path = Path(path)
    raw = _card_text(path)
    text = _rtf_to_text(raw) if raw.lstrip().startswith("{\\rtf") else raw
    fields = _card_fields(text)
    try:
        blob = _table_blob(text)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not blob:
        raise ValueError(f"{path}: the peak table is empty")
    try:
        rows = _card_rows(blob, Instrument().emission.principal_wavelength)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    label = name or fields.get("mineral name") or fields.get("compound name") or path.stem
    code = fields.get("reference code", "")
    cell = _card_cell(fields)
    crystal = (
        Crystal(a=cell[0], b=cell[1], c=cell[2], alpha=cell[3], beta=cell[4],
                gamma=cell[5], sites=[], name=label)
        if cell is not None
        else None
    )
    indexed = [_indices_from_token(row[0], row[1], crystal) for row in rows]
    hkl = (
        None if any(item[0] is None for item in indexed)
        else np.array([item[0] for item in indexed], dtype=float)
    )
    errors = np.array([item[1] for item in indexed], dtype=float)
    return PeakList(
        name=label,
        d=np.array([row[1] for row in rows], dtype=float),
        intensity=np.array([row[2] for row in rows], dtype=float),
        hkl=hkl,
        cell=cell,
        source=f"ICDD PDF {code}" if code else path.name,
        metadata={
            **fields,
            "indexing_worst_error": float(np.nanmax(errors)) if np.any(np.isfinite(errors))
            else float("nan"),
        },
    )


def write_peak_lists(peak_lists: list[PeakList], path: str | Path) -> Path:
    """Write entries out in the plain-text format :func:`read_peak_lists` reads.

    The bridge from cards to a library: export the entries wanted, read them
    here, write one file, and that file is the input to
    :func:`build_peaklist_library` and is editable by hand, which a card is not.
    It carries the reference code and the cell and drops the rest of the header,
    so the written file is the data and the card stays the record.
    """
    path = Path(path)
    lines = [
        "# Peak lists for ClayQuant, written by clayquant.peaklist.",
        "# Derived from licensed powder diffraction data - keep it with your own",
        "# files; it belongs neither in a repository nor in anything shared.",
    ]
    for peaks in peak_lists:
        lines.append("")
        lines.append(f"PHASE   {peaks.name}")
        if peaks.source:
            lines.append(f"SOURCE  {peaks.source}")
        if peaks.cell is not None:
            lines.append("CELL    " + " ".join(f"{value:g}" for value in peaks.cell))
        lines.append("#       d(A)      I    hkl")
        for index in range(peaks.d.size):
            row = f"  {peaks.d[index]:10.5f} {peaks.intensity[index]:6.1f}"
            if peaks.hkl is not None:
                row += "    " + "".join(f"{int(value):d}" for value in peaks.hkl[index])
            lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
