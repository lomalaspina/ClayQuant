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
from dataclasses import dataclass, field
from pathlib import Path

from .crystal import AtomSite, Crystal

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
        x\s+(?P<x>.+?)\s+
        y\s+(?P<y>.+?)\s+
        z\s+(?P<z>.+?)\s+
        occ\s+(?P<species>\S+)\s+(?P<occ>.+?)\s+
        beq\s+(?P<beq>.+?)\s*$""",
    re.VERBOSE,
)
_PO_RE = re.compile(r"PO\([^)]*?,,\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)")
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


@dataclass
class PhaseDefinition:
    """One phase read from a TOPAS structure library."""

    name: str
    cell: dict[str, float]
    space_group: str
    sites: list[AtomSite]
    icsd: int | None = None
    po_hkl: tuple[int, int, int] | None = None
    source_item: str = ""
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


def write_phase_database(
    xml_path: str | Path, output: str | Path, verbose: bool = False
) -> dict[str, int]:
    """Import a macro library and write a ClayQuant phase database.

    Returns counts of what happened, and prints per-phase problems when
    ``verbose``.
    """
    skipped: list[str] = []
    phases = parse_macro_library(xml_path, skipped=skipped)
    records = []
    failures: list[str] = list(skipped)
    for phase in phases:
        try:
            symops = symmetry_operations(phase.space_group)
        except ValueError as exc:
            failures.append(f"{phase.name}: {exc}")
            continue
        records.append(phase.to_dict(symops))
        if verbose:
            for warning in phase.warnings:
                print(f"  note: {warning}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(
            {
                "source": str(Path(xml_path).name),
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
    return {
        "parsed": len(phases) + len(skipped),
        "written": len(records),
        "failed": len(failures),
        "clay": sum(1 for record in records if record["is_clay"]),
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
    parser.add_argument("xml", type=Path, help="the jEdit macro menu XML file")
    parser.add_argument("-o", "--out", type=Path, default=Path("structures/phases.json"))
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    counts = write_phase_database(arguments.xml, arguments.out, verbose=not arguments.quiet)
    if not arguments.quiet:
        print(
            f"\n{counts['written']} phases written to {arguments.out} "
            f"({counts['clay']} classified as clay minerals, {counts['failed']} failed)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
