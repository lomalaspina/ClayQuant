"""Import of TOPAS structure libraries held as jEdit macro menus.

The structure collection this was written for is a jEdit macro menu whose items
each insert a TOPAS ``str`` block: phase name, cell parameters, space group in
TOPAS notation, and ``site`` lines with fractional coordinates, occupancies and
``beq`` displacement parameters.  Those are exactly the ingredients ClayQuant's
:class:`~clayquant.crystal.Crystal` needs, so the library becomes a phase
database for the non-clay ("main") minerals that accompany a clay separate:
quartz, the feldspars, carbonates, sulphates, oxides and so on.

Space group symbols are expanded to symmetry operations with ``gemmi``, which is
needed only for the import step: :func:`write_phase_database` stores the
operations explicitly, and :func:`load_phase_database` then reads the result
without any extra dependency.

The atomic coordinates in such a library normally originate from ICSD, which is
licensed data, so the generated database is not redistributed with ClayQuant -
import your own copy.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .crystal import AtomSite, Crystal, _unquote, read_cif
from .models import MINERAL_HABIT

__all__ = [
    "PhaseDefinition",
    "decode_macro_item",
    "parse_macro_library",
    "parse_topas_structures",
    "write_phase_database",
    "load_phase_database",
    "CLAY_PHASE_NAMES",
    "is_clay_phase",
    "main",
]

CLAY_PHASE_NAMES = frozenset(
    {
        "illite",
        "chlorite",
        "chamosite",
        "kaolinite",
        "kaolinite 1m",
        "kaolinite 2m",
        "dickite",
        "montmorillonite",
        "smectite",
        "vermiculite",
        "talc",
        "muscovite",
        "biotite",
        "chrysotile",
        "lizardite",
        "pyrophyllite",
        "glauconite",
        "nontronite",
        "saponite",
        "beidellite",
        "palygorskite",
        "sepiolite",
        "halloysite",
        "berthierine",
        "corrensite",
    }
)
"""Phyllosilicates treated as clay minerals rather than as accompanying phases.

Micas are included: in a clay separate mounted on a glass slide they orient and
diffract like the other phyllosilicates.  The GUI lets the classification be
overridden per phase, since whether a coarse mica counts as "clay" depends on
what the quantification is for.
"""


def is_clay_phase(name: str) -> bool:
    """Whether ``name`` is one of the phyllosilicates in :data:`CLAY_PHASE_NAMES`."""
    lowered = name.strip().lower()
    if lowered in CLAY_PHASE_NAMES:
        return True
    return any(clay in lowered for clay in CLAY_PHASE_NAMES)


# --------------------------------------------------------------------------- #
# Decoding the macro wrapper
# --------------------------------------------------------------------------- #

_INSERT_RE = re.compile(r'buffer\.insert\(textArea\.getCaretPosition\(\),\s*"(.*?)"\);', re.S)
_ITEM_SPLIT_RE = re.compile(r'(?=<item name=")')
_ITEM_NAME_RE = re.compile(r'<item name="([^"]+)"')


def decode_macro_item(block: str) -> str:
    """Recover the TOPAS text a macro item would insert."""
    pieces = [html.unescape(html.unescape(piece)) for piece in _INSERT_RE.findall(block)]
    text = "".join(pieces)
    return text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')


# --------------------------------------------------------------------------- #
# Parsing a TOPAS str block
# --------------------------------------------------------------------------- #

_FRACTION_RE = re.compile(r"^([-+]?\d+)\s*/\s*(\d+)$")
_NUMBER_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
_CELL_KEYS = {"a": "a", "b": "b", "c": "c", "al": "alpha", "be": "beta", "ga": "gamma"}
_SITE_RE = re.compile(
    r"""^\s*site\s+(?P<label>\S+)\s+
        # A site may state its own multiplicity instead of leaving it to the
        # space group.  TOPAS writes that as "num_posns 4", and a structure
        # written out by a refinement of a low-symmetry phase usually does,
        # since the refinement counted the positions once and recorded the
        # count.  ClayQuant expands the site by the symmetry operations and
        # removes duplicates, so it arrives at the same multiplicity itself and
        # only needs to not be confused by the statement.
        (?:num_posns\s+\S+\s+)?
        x\s+(?P<x>.+?)\s+
        y\s+(?P<y>.+?)\s+
        z\s+(?P<z>.+?)\s+
        occ\s+(?P<species>\S+)\s+(?P<occ>.+?)\s+
        beq\s+(?P<beq>.+?)\s*$""",
    re.VERBOSE,
)
_PO_RE = re.compile(r"PO\([^)]*?,,\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)")
# A peaks phase and the name it goes under, for a refinement read as plain TOPAS.
_PEAKS_PHASE_RE = re.compile(
    r"^[^\S\n]*(hkl_Is|xo_Is)\b(?:(?!^[^\S\n]*(?:str|hkl_Is|xo_Is)\b).)*?"
    r'phase_name\s+"?([^"\n]+?)"?\s*$',
    re.S | re.M,
)
# TOPAS writes back what the structure it refined weighs and how large its cell
# is, either bare or under a parameter name.  They are not needed to calculate
# anything - ClayQuant works both out from the sites and the cell - which is
# exactly what makes them worth reading: they are an independent statement of
# the same two numbers, by the program the library was written for.
_CELL_MASS_RE = re.compile(r"^cell_mass\s+(?:[A-Za-z_]\w*\s+)?([-+0-9.eE]+)")
_CELL_VOLUME_RE = re.compile(r"^cell_volume\s+(?:[A-Za-z_]\w*\s+)?([-+0-9.eE]+)")
_ICSD_RE = re.compile(r"No:\s*(\d+)")


def _strip_marks(token: str) -> str:
    """Remove the marks TOPAS puts on a refined value.

    A parameter that has been refined comes back from TOPAS carrying a backtick,
    and sometimes an error or a limit welded onto it with no space:
    ``16.875164```, ``9.663417`_LIMIT_MIN_9.5``, ``6.5e-08```.  A library written
    by hand has none of these; one saved out of a completed refinement has them
    on every value, which is the difference between a phase that imports and a
    phase that vanishes.  ``!`` and ``@`` mark a fixed or free parameter and are
    equally uninteresting here.
    """
    return token.lstrip("!@").split("`", 1)[0].strip().rstrip(",;")


def _value(text: str) -> float | None:
    """Read a TOPAS parameter value, which may be a number, a fraction or an expression.

    Refined parameters are written as ``name value min ... max ...``; expressions
    as ``= something;`` optionally followed by ``: value``, where that trailing
    value is the one TOPAS last evaluated.  Anything else yields ``None``.
    """
    token = text.strip().rstrip(",")
    if not token:
        return None

    evaluated = re.search(r":\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)`?\s*$", token)
    if evaluated:
        return float(evaluated.group(1))

    if token.startswith("="):
        body = token[1:].split(";")[0].strip()
        fraction = _FRACTION_RE.match(body)
        if fraction:
            return float(fraction.group(1)) / float(fraction.group(2))
        if _NUMBER_RE.match(body):
            return float(body)
        return None

    token = token.lstrip("!@")
    parts = token.split()
    for index, part in enumerate(parts):
        cleaned = _strip_marks(part)
        if cleaned in {"min", "max"}:
            break
        if _NUMBER_RE.match(cleaned):
            # A leading number is the value; otherwise the value follows the name.
            if index <= 1:
                return float(cleaned)
        fraction = _FRACTION_RE.match(cleaned)
        if fraction and index <= 1:
            return float(fraction.group(1)) / float(fraction.group(2))
    return None


def _explain_missing_mass(crystal, deficit: float) -> str:
    """Name the element the missing mass belongs to, when one accounts for it.

    "Sites are probably missing" is true and nearly useless: it does not say
    what to add, and the item reads as complete because the deficit is
    invisible site by site.  The deficit almost always belongs to one element -
    a hand-edited item drops a run of like sites - so each element already
    present is tried in turn, and the one whose atomic mass divides the deficit
    into a whole number of atoms is named, with the count.

    Sekaninaite is the case this was written for, and it is worth the detail.
    Its item carries every cation - aluminium 16 of 16, silicon 20 of 20, iron
    8 of 8 - and one oxygen site of the six that cordierite has.  So reading the
    item finds nothing wrong with it, and reading the arithmetic finds 16 oxygen
    atoms where 72 are needed: 56 x 15.999 = 895.97, which is the whole
    shortfall to the last decimal.  Saying "56 oxygen atoms" instead of "sites
    are probably missing" is the difference between a warning and an
    instruction.
    """
    from collections import Counter

    from .masses import atomic_weight

    present = Counter()
    for site in crystal.expanded_sites():
        element = site.species.rstrip("+-0123456789")
        present[element] += site.occupancy
    best: tuple[float, str, int] | None = None
    for element in present:
        try:
            mass = atomic_weight(element)
        except Exception:  # noqa: BLE001 - an unknown species simply cannot be blamed
            continue
        if mass <= 0.0:
            continue
        count = deficit / mass
        error = abs(count - round(count)) * mass / deficit if deficit else 1.0
        if round(count) >= 1 and (best is None or error < best[0]):
            best = (error, element, int(round(count)))
    if best is None or best[0] > 0.02:
        return (
            f"sites are probably missing, accounting for {deficit:.3f} of mass, "
            f"and no single element divides it evenly"
        )
    _, element, count = best
    have = present.get(element, 0.0)
    return (
        f"the shortfall is {deficit:.3f}, which is exactly {count} more {element} "
        f"atoms: the cell holds {have:.0f} and should hold {have + count:.0f}, so "
        f"{element} sites are missing from the item"
    )


@dataclass
class PhaseDefinition:
    """One phase read from a TOPAS structure library."""

    name: str
    cell: dict[str, float]
    space_group: str
    sites: list[AtomSite]
    icsd: int | None = None
    po_hkl: tuple[int, int, int] | None = None
    stated_mass: float | None = None
    stated_volume: float | None = None
    source_item: str = ""
    symops: list[str] | None = None
    """Symmetry operations, where the source states them rather than a symbol.

    A CIF lists them; a TOPAS block gives a space group symbol to expand.  One
    is as good as the other to everything downstream, and stating them is in
    fact the better of the two, because expanding a symbol needs ``gemmi`` and
    a symbol spelled the way gemmi expects.
    """

    warnings: list[str] = field(default_factory=list)

    @property
    def is_clay(self) -> bool:
        return is_clay_phase(self.name)

    def to_crystal(self, symops: list[str]) -> Crystal:
        return Crystal(
            a=self.cell["a"],
            b=self.cell["b"],
            c=self.cell["c"],
            alpha=self.cell["alpha"],
            beta=self.cell["beta"],
            gamma=self.cell["gamma"],
            sites=list(self.sites),
            symops=symops,
            name=self.name,
            source=(
                f"{self.name}"
                + (f" (ICSD {self.icsd})" if self.icsd else "")
                + f", space group {self.space_group}"
            ),
        )

    def check_stated_values(self, symops: list[str], tolerance: float = 0.005) -> list[str]:
        """Compare the cell against the mass and volume TOPAS itself wrote down.

        Neither number is used for anything - both are worked out from the sites
        and the cell - so a disagreement means the block does not describe the
        structure that was refined, and the commonest way for that to happen is
        a hand-edited item with sites left out.  The mass is the sensitive one:
        it is the only statement in the block that depends on every site being
        present, so a missing atom shows up here and nowhere else.  Five oxygens
        absent from a cordierite item cost it a third of its mass, its calculated
        pattern, and the weight percent that follows from both, and the item
        otherwise imported without complaint.
        """
        notes: list[str] = []
        crystal = self.to_crystal(symops)
        for label, stated, computed in (
            ("mass", self.stated_mass, crystal.cell_mass),
            ("volume", self.stated_volume, crystal.volume),
        ):
            if stated is None or stated <= 0.0:
                continue
            if abs(computed - stated) / stated > tolerance:
                detail = ""
                if label == "mass" and computed < stated:
                    detail = "; " + _explain_missing_mass(crystal, stated - computed)
                notes.append(
                    f"{self.name}: cell {label} works out to {computed:.3f} from the sites and "
                    f"cell given, but the block states {stated:.3f} "
                    f"({100 * computed / stated:.0f} % of it)" + detail
                )
        return notes

    def to_dict(self, symops: list[str]) -> dict:
        return {
            "name": self.name,
            "icsd": self.icsd,
            "space_group": self.space_group,
            "symops": symops,
            "cell": self.cell,
            "po_hkl": list(self.po_hkl) if self.po_hkl else None,
            "is_clay": self.is_clay,
            "sites": [
                {
                    "species": site.species,
                    "x": site.x,
                    "y": site.y,
                    "z": site.z,
                    "occupancy": site.occupancy,
                    "b_iso": site.b_iso,
                    "label": site.label,
                }
                for site in self.sites
            ],
        }


def _note_skipped(skipped: list[str] | None, name: str, reason: str) -> None:
    if skipped is not None:
        skipped.append(f"{name}: {reason}")


def parse_topas_structures(
    text: str, source_item: str = "", skipped: list[str] | None = None
) -> list[PhaseDefinition]:
    """Parse every ``str`` block in a piece of TOPAS input.

    A block that cannot be read is left out, and the reason is appended to
    ``skipped`` when one is given.  It is worth passing: an item that is dropped
    here never becomes a phase at all, so it is invisible to every count taken
    afterwards, and a structure library that silently imports the same number of
    phases as before is a hard thing to argue with.
    """
    lines = text.splitlines()
    # A structure block opens with "str", sometimes with a comment after it:
    # TOPAS takes everything from an apostrophe to the end of the line as one,
    # and the NIST LaB6 entry uses that to note that its cell is certified.
    starts = [index for index, line in enumerate(lines)
              if re.match(r"^str\s*(?:'.*)?$", line.strip())]
    if not starts:
        return []
    bounds = list(zip(starts, starts[1:] + [len(lines)]))

    icsd_match = _ICSD_RE.search(text)
    icsd = int(icsd_match.group(1)) if icsd_match else None

    # Parameter values seen so far, so that a block can refer to one defined in
    # an earlier block of the same item.  A two-component phase - the same
    # mineral at two crystallinities, which TOPAS models as two "str" blocks
    # sharing one cell - writes the second cell as "a =a_Calcite;" and cannot be
    # read without it.
    parameters: dict[str, float] = {}

    phases: list[PhaseDefinition] = []
    for start, stop in bounds:
        block = lines[start:stop]
        name = source_item
        cell: dict[str, float] = {}
        space_group = ""
        sites: list[AtomSite] = []
        po_hkl = None
        stated_mass: float | None = None
        stated_volume: float | None = None
        warnings: list[str] = []

        for line in block:
            stripped = line.strip()
            if not stripped or stripped.startswith("'"):
                continue

            named = re.match(r'^phase_name\s+"?([^"\']+)"?', stripped)
            if named:
                name = named.group(1).strip()
                continue
            group = re.match(r"""^space_group\s+(?:"([^"]+)"|'([^']+)'|(\S+))""", stripped)
            if group:
                # Quoted symbols may contain spaces - "C c c m" and "Cccm" are
                # the same group, and TOPAS writes either - so the quotes decide
                # where the symbol ends, not the first space inside them.
                space_group = next(part for part in group.groups() if part).strip()
                continue

            site = _SITE_RE.match(stripped)
            if site:
                coordinates = {}
                for axis in ("x", "y", "z"):
                    value = _value(site.group(axis))
                    if value is None:
                        warnings.append(
                            f"{name}: could not read {axis} of site {site.group('label')}"
                        )
                    coordinates[axis] = value
                occupancy = _value(site.group("occ"))
                b_iso = _value(site.group("beq"))
                if None in coordinates.values():
                    continue
                sites.append(
                    AtomSite(
                        species=site.group("species"),
                        x=coordinates["x"],
                        y=coordinates["y"],
                        z=coordinates["z"],
                        occupancy=1.0 if occupancy is None else occupancy,
                        b_iso=1.0 if b_iso is None else b_iso,
                        label=site.group("label"),
                    )
                )
                continue

            mass_line = _CELL_MASS_RE.match(stripped)
            if mass_line:
                stated_mass = _value(mass_line.group(1))
                continue
            volume_line = _CELL_VOLUME_RE.match(stripped)
            if volume_line:
                stated_volume = _value(volume_line.group(1))
                continue

            orientation = _PO_RE.search(stripped)
            if orientation:
                po_hkl = tuple(int(orientation.group(index)) for index in (1, 2, 3))
                continue

            cell_line = re.match(r"^(a|b|c|al|be|ga)\b(.*)$", stripped)
            if cell_line:
                key = _CELL_KEYS[cell_line.group(1)]
                rest = cell_line.group(2).strip()
                reference = re.match(r"^=\s*Get\(\s*(a|b|c|al|be|ga)\s*\)", rest)
                if reference:
                    source_key = _CELL_KEYS[reference.group(1)]
                    if source_key in cell:
                        cell[key] = cell[source_key]
                    else:
                        warnings.append(f"{name}: {key} refers to {source_key}, which is not set")
                    continue
                named_reference = re.match(r"^=\s*([A-Za-z_]\w*)\s*;", rest)
                if named_reference:
                    source = named_reference.group(1)
                    if source in parameters:
                        cell[key] = parameters[source]
                    else:
                        warnings.append(
                            f"{name}: {key} refers to the parameter {source}, which is not set"
                        )
                    continue
                value = _value(rest)
                if value is None:
                    warnings.append(f"{name}: could not read cell parameter {key} from {rest!r}")
                else:
                    cell[key] = value
                    # Remember it under its TOPAS name, for a later block.
                    first = _strip_marks(rest.split()[0]) if rest.split() else ""
                    if first and not _NUMBER_RE.match(first):
                        parameters[first] = value

        cell.setdefault("alpha", 90.0)
        cell.setdefault("beta", 90.0)
        cell.setdefault("gamma", 90.0)
        missing = [key for key in ("a", "b", "c") if key not in cell]
        if missing:
            # A trigonal or cubic phase may leave b and c to equal a.
            if "a" in cell:
                for key in missing:
                    cell[key] = cell["a"]
                warnings.append(f"{name}: {missing} not given, assumed equal to a")
            else:
                _note_skipped(skipped, name, "no cell parameters could be read")
                continue
        if not space_group or not sites:
            _note_skipped(
                skipped, name,
                "no space group" if not space_group else "no atom sites could be read",
            )
            continue

        phases.append(
            PhaseDefinition(
                name=name,
                cell=cell,
                space_group=space_group,
                sites=sites,
                icsd=icsd,
                po_hkl=po_hkl,
                stated_mass=stated_mass,
                stated_volume=stated_volume,
                source_item=source_item,
                warnings=warnings,
            )
        )

    if len(phases) > 1:
        seen: dict[str, int] = {}
        for phase in phases:
            seen[phase.name] = seen.get(phase.name, 0) + 1
            if seen[phase.name] > 1:
                phase.name = f"{phase.name} ({seen[phase.name]})"
    return phases


def looks_like_a_macro_library(text: str) -> bool:
    """Is this a jEdit macro menu, or plain TOPAS input?

    The two sources are read the same way once the macro escaping is undone, so
    the only thing that has to be decided is whether to undo it.
    """
    return "<item name=" in text


def parse_cif_structure(
    path: str | Path, skipped: list[str] | None = None
) -> list[PhaseDefinition]:
    """Read one CIF as a phase definition, for a mineral TOPAS has no item for.

    The accompanying minerals normally come from the TOPAS structure library,
    which is the right source when the library has them.  It does not have
    everything: a specimen can contain a mineral nobody in the lab has refined,
    and then a published CIF - from COD, AMCSD or a subscription - is what
    there is.  This reads one, so that such a mineral can be quantified beside
    the rest rather than left as unexplained intensity.

    The name is taken from ``_chemical_name_mineral`` where the file states one
    and from the file's own name otherwise, because that name is the phase's
    identity everywhere afterwards: in the fit, in the table, in the export.
    Symmetry is taken from the file's operation list, which is why this needs no
    space group symbol and no ``gemmi``.
    """
    path = Path(path)
    try:
        crystal = read_cif(path)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        _note_skipped(skipped, path.name, f"could not be read as a CIF: {exc}")
        return []

    text = path.read_text(errors="replace")
    name = ""
    for line in text.splitlines():
        if line.strip().startswith("_chemical_name_mineral"):
            name = _unquote(line.split(None, 1)[1].strip()) if len(line.split()) > 1 else ""
            break
    if not name:
        # "sepiolite_COD_9014723" -> "Sepiolite".  A file named after its
        # mineral is the ordinary case, and a name that is nothing but a
        # database code would be useless in a results table.
        stem = re.split(r"[_-]", path.stem)[0]
        name = stem[:1].upper() + stem[1:]

    if not crystal.sites:
        _note_skipped(skipped, name, f"{path.name} holds no atom sites")
        return []

    return [PhaseDefinition(
        name=name,
        po_hkl=MINERAL_HABIT.get(name.lower()),
        cell={"a": crystal.a, "b": crystal.b, "c": crystal.c,
              "alpha": crystal.alpha, "beta": crystal.beta, "gamma": crystal.gamma},
        space_group=_cif_value(text, "_space_group_name_H-M_alt")
        or _cif_value(text, "_symmetry_space_group_name_H-M")
        or "unstated",
        sites=list(crystal.sites),
        source_item=path.name,
        symops=list(crystal.symops),
    )]


def _cif_value(text: str, tag: str) -> str:
    """The value of a non-looped CIF tag, or an empty string."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(tag) and len(stripped) > len(tag):
            return _unquote(stripped[len(tag):].strip())
    return ""


def parse_refinement(
    path: str | Path, skipped: list[str] | None = None
) -> list[PhaseDefinition]:
    """Read the structures out of a TOPAS refinement's input or output file.

    A refinement is the best source of a structure there is for the specimen it
    was refined on: its cell, and any occupancy that was refined, are the values
    that actually fitted the measured intensities, rather than a published
    structure of a different specimen of the same mineral.  A ``.out`` file is
    plain TOPAS input - TOPAS writes the refined values back into the same
    syntax it read - so the same parser serves, with the macro unescaping
    skipped.

    The file holds much besides structures (the instrument, the background, the
    agreement factors), all of which is passed over: only ``str`` blocks become
    phases.  A ``hkl_Is`` block is reported through ``skipped`` as usual, which
    matters here, because a refinement that models a clay as a peaks phase
    contributes no structure for it.
    """
    if Path(path).suffix.lower() == ".cif":
        return parse_cif_structure(path, skipped=skipped)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if looks_like_a_macro_library(text):
        return parse_macro_library(path, skipped=skipped)
    phases = parse_topas_structures(text, source_item=Path(path).stem, skipped=skipped)
    # parse_topas_structures looks for "str" and passes over everything else in
    # silence, which is right for a file that is mostly instrument settings but
    # wrong for the one thing a reader of a refinement most needs to be told:
    # that a phase is in it as a peak set and so brought no structure with it.
    for match in _PEAKS_PHASE_RE.finditer(text):
        kind, name = match.group(1), match.group(2) or "an unnamed phase"
        _note_skipped(
            skipped, name,
            f"a peaks phase ({kind}) in this refinement, which fits its intensity without a "
            "structure; no cell or occupancy can be taken from it",
        )
    return phases


def parse_macro_library(
    path: str | Path, skipped: list[str] | None = None
) -> list[PhaseDefinition]:
    """Parse every phase in a jEdit macro menu of TOPAS structures.

    Items that hold no ``str`` block at all - a heading, a comment, a macro that
    does something else - are passed over in silence, since a menu is full of
    them.  An item that looks like a structure and cannot be read is reported
    through ``skipped``.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    phases: list[PhaseDefinition] = []
    for block in _ITEM_SPLIT_RE.split(text)[1:]:
        name_match = _ITEM_NAME_RE.match(block)
        if name_match is None:
            continue
        decoded = decode_macro_item(block)
        before = len(skipped) if skipped is not None else 0
        found = parse_topas_structures(decoded, source_item=name_match.group(1), skipped=skipped)
        if (not found and skipped is not None and len(skipped) == before
                and _looks_like_a_structure(decoded)):
            # Only when nothing inside the item has already explained itself:
            # a block that was read and rejected has said why, and saying it
            # again in other words helps no one.
            skipped.append(f"{name_match.group(1)}: {_why_no_structure(decoded)}")
        phases.extend(found)
    return phases


def _looks_like_a_structure(text: str) -> bool:
    """Does this item try to define a phase, whether or not it succeeds?"""
    return any(marker in text for marker in ("phase_name", "space_group", "\tsite", " site "))


def _why_no_structure(text: str) -> str:
    """Why an item that mentions a phase yields no structure."""
    if re.search(r"^\s*hkl_Is\b", text, re.M):
        return ("a peaks phase (hkl_Is), which lists reflections instead of atoms; "
                "ClayQuant calculates from structures and cannot use it")
    if re.search(r"^\s*xo_Is\b", text, re.M):
        return ("a peaks phase (xo_Is), which lists peak positions instead of atoms; "
                "ClayQuant calculates from structures and cannot use it")
    return "no 'str' block was found in the item"


# --------------------------------------------------------------------------- #
# Symmetry expansion and storage
# --------------------------------------------------------------------------- #


def _symbol_alternatives(space_group: str) -> list[str]:
    """Spellings to try when a TOPAS space group symbol is not recognised directly.

    TOPAS marks the two origin choices of a centrosymmetric cubic or tetragonal
    group with a trailing letter instead of the ``:1`` / ``:2`` of the
    International Tables: ``Fd-3mS`` against ``Fd-3mZ``, ``I41/amdS`` against
    ``I41/amdZ``.  Which is which is fixed by the structures themselves: in this
    library diamond, silicon and franklinite are written ``S`` with their 8a site
    at the origin, which is origin choice 1, while magnetite, spinel and chromite
    are written ``Z`` with that site at (1/8, 1/8, 1/8), which is choice 2.
    """
    candidates: list[str] = []
    if space_group and space_group[-1] in "SZ" and not space_group[-2].isdigit():
        candidates.append(f"{space_group[:-1]}:{'1' if space_group[-1] == 'S' else '2'}")
    candidates.extend(
        [
            space_group.rstrip("H"),
            space_group.replace(":", ""),
        ]
    )
    return candidates


def symmetry_operations(space_group: str) -> list[str]:
    """Symmetry operations of a TOPAS space group symbol, as ``x, y, z`` strings.

    Requires ``gemmi`` (``pip install clayquant[import]``); it is used at import
    time only, because :func:`write_phase_database` stores the result.
    """
    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "importing a TOPAS structure library needs gemmi to expand space group "
            "symbols: pip install gemmi"
        ) from exc

    group = gemmi.find_spacegroup_by_name(space_group)
    if group is None:
        for candidate in _symbol_alternatives(space_group):
            group = gemmi.find_spacegroup_by_name(candidate)
            if group is not None:
                break
    if group is None:
        raise ValueError(f"unknown space group symbol {space_group!r}")
    return [operation.triplet() for operation in group.operations()]


CLAY_KEY_ALIASES: dict[str, str] = {
    "illite": "illite",
    "chlorite": "chlorite",
    "clinochlore": "chlorite",
    "kaolinite": "kaolinite_1M",
    "kaolinite1m": "kaolinite_1M",
    "kaolinite2m": "kaolinite_2M",
}
"""Phase names a refinement may use, against the keys of :data:`models.CIF_SOURCES`.

A refinement names a phase whatever the person refining it typed, and the clay
library asks for it by a fixed key.  Only the clays need this: an accompanying
mineral is looked up in the phase database by its own name.
"""


def clay_key_for(name: str) -> str | None:
    """The clay library key a refinement's phase name corresponds to, if any."""
    squashed = re.sub(r"[^a-z0-9]", "", name.lower())
    if squashed in CLAY_KEY_ALIASES:
        return CLAY_KEY_ALIASES[squashed]
    # "Kaolinite 2M (2)" from a duplicate name, "Illite-HiCryst", and so on.
    for alias, key in sorted(CLAY_KEY_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if squashed.startswith(alias):
            return key
    return None


def refined_crystals(
    path: str | Path, skipped: list[str] | None = None
) -> dict[str, Crystal]:
    """Structures from a refinement, by the name the refinement gives them."""
    crystals: dict[str, Crystal] = {}
    for phase in parse_refinement(path, skipped=skipped):
        try:
            symops = symmetry_operations(phase.space_group)
        except ValueError as exc:
            _note_skipped(skipped, phase.name, str(exc))
            continue
        crystals[phase.name] = phase.to_crystal(symops)
    return crystals


def refined_clay_structures(
    path: str | Path,
    skipped: list[str] | None = None,
    only: "Iterable[str] | None" = None,
    exclude: "Iterable[str] | None" = None,
) -> dict[str, Crystal]:
    """The clay structures from a refinement, keyed for the clay library.

    Pass the result to :func:`clayquant.models.use_refined_structures` to have
    the library built from the structures as refined rather than as published.

    ``only`` and ``exclude`` select which phases to take, because the choice is
    made per mineral rather than per file.  A refined occupancy is worth having
    where it describes the specimen better than the published structure does,
    and worth leaving out where the published one is the more defensible
    reference - an Fe-rich chlorite refined on one deposit is not a chlorite
    anywhere else, and keeping the published clinochlore is a reasonable
    decision rather than a missed opportunity.  Names are the library's keys, so
    ``exclude=["chlorite"]`` keeps the published chlorite and takes the rest.
    """
    found: dict[str, Crystal] = {}
    wanted = {str(key) for key in only} if only is not None else None
    unwanted = {str(key) for key in exclude} if exclude is not None else set()
    for name, crystal in refined_crystals(path, skipped=skipped).items():
        key = clay_key_for(name)
        if key is None or key in unwanted or (wanted is not None and key not in wanted):
            continue
        found[key] = crystal
    return found


def write_phase_database(
    xml_path: str | Path | Sequence[str | Path], output: str | Path, verbose: bool = False
) -> dict[str, int]:
    """Import structure libraries and refinements and write a phase database.

    Each source may be a jEdit macro menu or a TOPAS refinement's ``.inp`` or
    ``.out``; which it is, is decided by looking at it.  Given several, a phase
    from a later source replaces one of the same name from an earlier, so a
    refinement can be laid over a library: the library supplies the breadth and
    the refinement supplies, for the few phases it contains, the cell and the
    occupancies that actually fitted a measurement.

    Returns counts of what happened, and prints per-phase problems when
    ``verbose``.
    """
    sources = (
        [xml_path]
        if isinstance(xml_path, (str, Path))
        else [Path(source) for source in xml_path]
    )
    skipped: list[str] = []
    phases: list[PhaseDefinition] = []
    replaced: list[str] = []
    for source in sources:
        found = parse_refinement(source, skipped=skipped)
        incoming = {phase.name for phase in found}
        kept = [phase for phase in phases if phase.name not in incoming]
        replaced.extend(
            f"{phase.name}: replaced by the one in {Path(source).name}"
            for phase in phases
            if phase.name in incoming
        )
        phases = kept + found
    records = []
    failures: list[str] = list(skipped)
    disagreements: list[str] = []
    for phase in phases:
        try:
            symops = phase.symops or symmetry_operations(phase.space_group)
        except ValueError as exc:
            failures.append(f"{phase.name}: {exc}")
            continue
        records.append(phase.to_dict(symops))
        # An import that reads a phase is not the same as an import that reads
        # it correctly, and the phase is written either way: this is the only
        # place the difference shows.
        disagreements.extend(phase.check_stated_values(symops))
        if verbose:
            for warning in phase.warnings:
                print(f"  note: {warning}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(
            {
                "source": ", ".join(Path(source).name for source in sources),
                "n_phases": len(records),
                "phases": records,
            },
            handle,
            indent=1,
        )
    if verbose and failures:
        print("\nphases that could not be imported:")
        for failure in failures:
            print(f"  {failure}")
    if verbose and replaced:
        print("\nphases replaced by a later source:")
        for note in replaced:
            print(f"  {note}")
    if verbose and disagreements:
        print("\nphases that were imported but do not match what TOPAS says they weigh:")
        for note in disagreements:
            print(f"  {note}")
    return {
        "parsed": len(phases) + len(skipped),
        "written": len(records),
        "failed": len(failures),
        "clay": sum(1 for record in records if record["is_clay"]),
        "disagreeing": len(disagreements),
        "disagreements": disagreements,
        # Which of them the clay library could be built from, which is not
        # obvious from a phase list: a refinement naming a phase "Clinochlore"
        # supplies ClayQuant's "chlorite", and one that models its clays as
        # peaks phases supplies none of them.
        "replaced": len(replaced),
        "clay_structures": sorted(
            {key for record in records
             if (key := clay_key_for(record["name"])) is not None}
        ),
    }


def load_phase_database(path: str | Path) -> dict[str, Crystal]:
    """Load a phase database written by :func:`write_phase_database`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Import a TOPAS structure library first:\n"
            f"  clayquant-import-structures your_structures.xml -o {path}"
        )
    with path.open() as handle:
        data = json.load(handle)

    crystals: dict[str, Crystal] = {}
    for record in data["phases"]:
        cell = record["cell"]
        crystals[record["name"]] = Crystal(
            a=cell["a"],
            b=cell["b"],
            c=cell["c"],
            alpha=cell["alpha"],
            beta=cell["beta"],
            gamma=cell["gamma"],
            sites=[
                AtomSite(
                    species=site["species"],
                    x=site["x"],
                    y=site["y"],
                    z=site["z"],
                    occupancy=site["occupancy"],
                    b_iso=site["b_iso"],
                    label=site.get("label", ""),
                )
                for site in record["sites"]
            ],
            symops=record["symops"],
            name=record["name"],
            po_axis=(tuple(record["po_hkl"]) if record.get("po_hkl") else None),
            source=(
                f"{record['name']}"
                + (f" (ICSD {record['icsd']})" if record.get("icsd") else "")
                + f", space group {record['space_group']}"
            ),
        )
    return crystals


def main(argv: list[str] | None = None) -> int:
    """Command line entry point for ``clayquant-import-structures``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="clayquant-import-structures",
        description="Import a TOPAS/jEdit structure library into a ClayQuant phase database.",
    )
    parser.add_argument(
        "xml", type=Path, nargs="+", metavar="SOURCE",
        help=(
            "jEdit macro menu XML files and TOPAS refinements (.inp or .out), in "
            "increasing order of authority: where two name the same phase, the later "
            "one is kept, so a refinement given last overrides the library"
        ),
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("structures/phases.json"))
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    counts = write_phase_database(arguments.xml, arguments.out, verbose=not arguments.quiet)
    if not arguments.quiet:
        print(
            f"\n{counts['written']} phases written to {arguments.out} "
            f"({counts['clay']} classified as clay minerals, {counts['failed']} failed)"
        )
        if counts["clay_structures"]:
            print(
                "clay structures these sources can supply to the pattern library: "
                + ", ".join(sorted(counts["clay_structures"]))
                + "\n  clayquant-build-library --refined-structures "
                + str(arguments.xml[-1])
            )
        if counts["disagreeing"]:
            n = counts["disagreeing"]
            print(
                f"{n} of them {'disagrees' if n == 1 else 'disagree'} with the cell mass or "
                "volume the library states; check "
                f"{'that item' if n == 1 else 'those items'} before quantifying with "
                f"{'it' if n == 1 else 'them'}."
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
