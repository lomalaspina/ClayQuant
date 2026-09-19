"""Library of clay layer models.

Two kinds of entry are provided:

* Layers derived from crystal structures supplied as CIF files.  The ICSD
  entries requested for ClayQuant are listed in :data:`CIF_SOURCES`.  ICSD data
  is licensed and is *not* redistributed with this package: export the CIFs
  yourself and place them in the structure directory (see
  :func:`structure_directory`).

* Layers published directly as one-dimensional models, currently the ethylene
  glycol-montmorillonite complex of Reynolds (1965), whose parameters are
  transcribed in :data:`REYNOLDS_1965_EG_SMECTITE_ROWS`.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from .crystal import Crystal, LayerModel, read_cif

__all__ = [
    "CIF_SOURCES",
    "CifSource",
    "structure_directory",
    "structure_directories",
    "find_structure_file",
    "describe_structure_search",
    "load_crystal",
    "load_layer",
    "chlorite_crystal",
    "chlorite_layer",
    "CHLORITE_OCTAHEDRA",
    "use_refined_structures",
    "clear_refined_structures",
    "refined_structures",
    "air_dried_smectite_layer",
    "eg_smectite_layer",
    "available_phases",
    "REYNOLDS_1965_EG_SMECTITE_ROWS",
    "AIR_DRIED_SMECTITE_D001",
    "REYNOLDS_1965_D001",
    "WATER_PER_CELL",
]


@dataclass(frozen=True)
class CifSource:
    """A crystal structure ClayQuant expects to find as a CIF file."""

    key: str
    filename: str
    icsd: int
    description: str
    layers_per_cell: int
    synonyms: tuple[str, ...] = ()
    """Other names the same mineral goes under, for finding a file.

    A structure exported from somewhere else is named by whoever exported it,
    and TOPAS writing a refined structure out names it after the phase as the
    refinement called it.  A chlorite refined as "Clinochlore" arrives as
    ``Clinochlore.cif``, which carries neither the ICSD code nor the key, so
    without this it is passed over in silence and the published structure used
    instead - the user drops in the refined file, sees no change, and has
    nothing to go on.
    """


CIF_SOURCES: dict[str, CifSource] = {
    source.key: source
    for source in [
        CifSource(
            key="illite",
            filename="illite_ICSD_90144.cif",
            icsd=90144,
            description="Illite, C2/c, Gualtieri (2000) J. Appl. Cryst. 33, 267-278",
            layers_per_cell=2,
        ),
        CifSource(
            key="chlorite",
            filename="chlorite_ICSD_164234.cif",
            icsd=164234,
            synonyms=("clinochlore",),
            description=(
                "Clinochlore IIb-4, C-1, Zanazzi, Comodi, Nazzareni & Andreozzi (2009) "
                "Eur. J. Mineral. 21, 581-589"
            ),
            layers_per_cell=1,
        ),
        CifSource(
            key="kaolinite_1M",
            filename="kaolinite_1M_ICSD_63192.cif",
            icsd=63192,
            description="Kaolinite, C1, Bish & Von Dreele (1989) Clays Clay Miner. 37, 289-296",
            layers_per_cell=1,
        ),
        CifSource(
            key="kaolinite_2M",
            filename="kaolinite_2M_ICSD_30285.cif",
            icsd=30285,
            description="Kaolinite 2M, C1c1, Gruner (1932) Z. Kristallogr. 83, 75-88",
            layers_per_cell=2,
        ),
    ]
}


def structure_directories() -> list[Path]:
    """Every directory searched for CIF files, in order of precedence.

    ``$CLAYQUANT_STRUCTURE_DIR`` first, then a ``structures`` directory beside
    the working directory, the working directory itself, the ``structures``
    directory of a source checkout, and finally the package data directory.
    Several are searched rather than one because the program is as likely to be
    started from the home directory as from the project, and a file that is
    plainly there should not be reported missing over a detail of where the
    shell happened to be.
    """
    candidates: list[Path] = []
    override = os.environ.get("CLAYQUANT_STRUCTURE_DIR")
    if override:
        candidates.append(Path(override).expanduser())
    cwd = Path.cwd()
    candidates.extend([cwd / "structures", cwd])
    # models.py -> clayquant -> src -> the checkout root, when run from source.
    candidates.append(Path(__file__).resolve().parents[2] / "structures")
    candidates.append(Path(str(resources.files("clayquant.data").joinpath("structures"))))

    seen: set[Path] = set()
    ordered: list[Path] = []
    for directory in candidates:
        try:
            resolved = directory.resolve()
        except OSError:  # a path that cannot be resolved is simply not searched
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(directory)
    return ordered


def structure_directory() -> Path:
    """The first searched directory that exists, for messages and for writing."""
    for directory in structure_directories():
        if directory.is_dir():
            return directory
    return Path.cwd() / "structures"


def _normalised(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_structure_file(source: "CifSource") -> Path | None:
    """Locate the CIF of ``source``, or ``None`` if no file matches it.

    The expected file name is matched first, but it is not required.  An export
    is named by whoever made it - ``Kaolinite_1M_63192.cif`` rather than
    ``kaolinite_1M_ICSD_63192.cif``, or the same name in different case on a
    case-sensitive file system - and a structure that is plainly present should
    be used rather than reported missing over its spelling.  So a file is
    accepted when its name carries the ICSD code, or when it carries the phase
    name (``kaolinite_2M`` and ``kaolinite_1M`` being distinguished by the
    polytype, which is part of the key), or one of the mineral's
    :attr:`CifSource.synonyms`.

    The four are tried in that order, and each is tried against every file in a
    directory before the next directory is looked at, so a precise name always
    beats a looser one and a nearer directory beats a further one.
    """
    canonical = source.filename.lower()
    code = str(source.icsd)
    names = [_normalised(source.key), *(_normalised(name) for name in source.synonyms)]

    for directory in structure_directories():
        if not directory.is_dir():
            continue
        try:
            files = sorted(entry for entry in directory.iterdir()
                           if entry.is_file() and entry.suffix.lower() == ".cif")
        except OSError:
            continue
        for entry in files:
            if entry.name.lower() == canonical:
                return entry
        for entry in files:
            if code in _normalised(entry.stem):
                return entry
        for name in names:
            for entry in files:
                if name in _normalised(entry.stem):
                    return entry
    return None


CHLORITE_OCTAHEDRA: dict[str, tuple[str, ...]] = {
    "2:1": ("Mg1", "Fe1", "Mg2", "Fe2"),
    "hydroxide": ("Mg3", "Fe3", "Mg4", "Fe4"),
}
"""The two octahedral sheets of the chlorite structure, by site label.

A chlorite layer carries two of them: the octahedral sheet of the 2:1 layer, at
the layer origin, and the interlayer hydroxide ("brucite") sheet at half the
repeat.  Iron substitutes for magnesium in both, in proportions that differ
between the two and between deposits, and because the sheets sit at different
heights the two substitutions act differently on the basal orders - which is
what makes them separable from a measured basal series
(:func:`clayquant.composition.fit_chlorite_iron`).
"""


def chlorite_crystal(
    iron_2to1: float,
    iron_hydroxide: float,
    octahedral_b: float | None = None,
) -> Crystal:
    """The chlorite structure with its octahedral iron set to these fractions.

    Each fraction is the iron occupancy of one sheet, with magnesium taking the
    rest, so ``(0.0, 0.0)`` is an iron-free clinochlore and ``(1.0, 1.0)`` a
    fully ferrous chamosite-like end member.  Everything else - the cell, the
    positions, the tetrahedral Si/Al - is the published structure's.

    The substitution changes the cell mass as well as the basal intensities, so
    a weight percent computed from a fitted composition is consistent with it
    (Sec. 2.13).

    ``octahedral_b`` replaces the displacement parameter of every octahedral
    site, in A^2.  It is one value for all of them rather than one per species
    because the magnesium and the iron of a substituted site occupy the *same*
    site: they share its environment, so they share its displacement parameter,
    and letting them differ would be both unphysical and degenerate with the
    occupancy that shares the site.  The published structure already pairs them
    this way; this keeps that true of anything derived from it.  ``None`` leaves
    the published values alone.
    """
    for name, value in (("iron_2to1", iron_2to1), ("iron_hydroxide", iron_hydroxide)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} is an occupancy and must lie in [0, 1], not {value}")
    if octahedral_b is not None and octahedral_b < 0.0:
        raise ValueError(
            f"a displacement parameter cannot be negative, and {octahedral_b} is"
        )

    base = load_crystal("chlorite")
    iron_of = {"2:1": float(iron_2to1), "hydroxide": float(iron_hydroxide)}
    sheet_of = {
        label: sheet for sheet, labels in CHLORITE_OCTAHEDRA.items() for label in labels
    }
    sites = []
    for site in base.sites:
        sheet = sheet_of.get(site.label)
        if sheet is None:
            sites.append(site)
            continue
        iron = iron_of[sheet]
        fraction = iron if site.species.startswith("Fe") else 1.0 - iron
        sites.append(dataclasses.replace(
            site,
            occupancy=fraction,
            b_iso=site.b_iso if octahedral_b is None else float(octahedral_b),
        ))
    described = (
        f"{base.source}, octahedral iron set to {iron_2to1:.3f} (2:1 sheet) "
        f"and {iron_hydroxide:.3f} (hydroxide sheet)"
    )
    if octahedral_b is not None:
        described += f", octahedral B = {octahedral_b:.3f} A^2 on every such site"
    return dataclasses.replace(
        base,
        sites=sites,
        name=f"chlorite Fe {iron_2to1:.2f}/{iron_hydroxide:.2f}",
        source=described,
    )


def chlorite_layer(
    iron_2to1: float, iron_hydroxide: float, octahedral_b: float | None = None
) -> LayerModel:
    """One folded layer of :func:`chlorite_crystal`."""
    crystal = chlorite_crystal(iron_2to1, iron_hydroxide, octahedral_b=octahedral_b)
    return crystal.layer_model(
        layers_per_cell=CIF_SOURCES["chlorite"].layers_per_cell, name=crystal.name
    )


def available_phases() -> dict[str, bool]:
    """Map each expected phase key to whether its structure can be loaded.

    A structure supplied by :func:`use_refined_structures` counts as available:
    it needs no file, and a refinement is a legitimate source of a structure.
    """
    return {
        key: key in _REFINED or find_structure_file(source) is not None
        for key, source in CIF_SOURCES.items()
    }


def describe_structure_search() -> str:
    """What was looked for and where, for an error or a status message."""
    lines = []
    for key, source in CIF_SOURCES.items():
        found = find_structure_file(source)
        lines.append(f"  {key}: {found if found else 'not found'} (ICSD {source.icsd})")
    searched = "\n".join(f"  {directory}" for directory in structure_directories())
    return "Searched:\n" + searched + "\nStructures:\n" + "\n".join(lines)


_REFINED: dict[str, Crystal] = {}
"""Structures to use in place of the published ones; see :func:`use_refined_structures`."""


def use_refined_structures(structures: dict[str, Crystal]) -> dict[str, Crystal]:
    """Use these structures in place of the CIFs, keyed as in :data:`CIF_SOURCES`.

    A published structure is of somebody else's specimen.  Where a refinement of
    *this* specimen exists, its cell and its refined occupancies describe the
    material in the beam, and a clay is exactly where that matters most: the
    octahedral Fe-for-Mg and Fe-for-Al substitutions are refinable against the
    basal intensities, they differ from one deposit to the next, and iron
    scatters about twice as strongly as the magnesium it replaces, so an
    occupancy taken from a published structure can distort a calculated basal
    series and with it every intensity ratio that follows.

    Returns what was replaced, so a caller can put it back.  Both this and
    :func:`clear_refined_structures` clear the caches of
    :func:`load_crystal` and :func:`load_layer`, since those hold structures
    loaded under the previous setting.
    """
    unknown = sorted(set(structures) - set(CIF_SOURCES))
    if unknown:
        raise KeyError(
            f"not structures the library asks for: {unknown}; "
            f"expected among {sorted(CIF_SOURCES)}"
        )
    replaced = dict(_REFINED)
    _REFINED.update(structures)
    load_crystal.cache_clear()
    load_layer.cache_clear()
    return replaced


def clear_refined_structures() -> None:
    """Go back to the published structures."""
    _REFINED.clear()
    load_crystal.cache_clear()
    load_layer.cache_clear()


def refined_structures() -> dict[str, Crystal]:
    """Which structures are currently overridden."""
    return dict(_REFINED)


@lru_cache(maxsize=None)
def load_crystal(key: str) -> Crystal:
    """Load one of the :data:`CIF_SOURCES` structures.

    A structure registered through :func:`use_refined_structures` is returned in
    place of the CIF, and is not required to be present on disk.
    """
    if key in _REFINED:
        return _REFINED[key]
    try:
        source = CIF_SOURCES[key]
    except KeyError:
        raise KeyError(
            f"unknown phase {key!r}; expected one of {sorted(CIF_SOURCES)}"
        ) from None
    path = find_structure_file(source)
    if path is None:
        raise FileNotFoundError(
            f"No CIF for {key} was found.\n"
            f"ClayQuant does not redistribute ICSD data. Export ICSD {source.icsd} "
            f"({source.description}) as CIF and put it in {structure_directory()}, or set "
            f"$CLAYQUANT_STRUCTURE_DIR. The name need only carry the ICSD code or the phase "
            f"name; {source.filename} is the expected spelling.\n"
            + describe_structure_search()
        )
    return read_cif(path)


@lru_cache(maxsize=None)
def load_layer(key: str) -> LayerModel:
    """Load a phase as a single-layer one-dimensional model."""
    source = CIF_SOURCES[key]
    return load_crystal(key).layer_model(layers_per_cell=source.layers_per_cell, name=key)


# --------------------------------------------------------------------------- #
# Ethylene glycol-montmorillonite, Reynolds (1965)
# --------------------------------------------------------------------------- #

REYNOLDS_1965_D001 = 16.86
"""Measured basal spacing of the EG-montmorillonite complex in A (Reynolds 1965)."""

REYNOLDS_1965_EG_SMECTITE_ROWS: list[tuple[float, str, float, float]] = [
    # z [A], species, occupancy, B [A^2].  Transcribed from Table 1 of Reynolds
    # (1965), which lists one half of the unit cell; the rows below are mirrored
    # about the octahedral plane at z = 0 by eg_smectite_layer().
    #
    # Table 1 quotes the content of each atomic plane, except the octahedral
    # plane at z = 0, which lies on the mirror and is quoted as half its content
    # (1.54 Al + 0.16 Fe + 0.33 Mg, i.e. two of the cell's four octahedral
    # cations); its occupancies are therefore doubled here.  The glycol rows
    # "H2C-HO" and "CH2-OH" are half-molecules, so each contributes C + O + 3 H;
    # hydroxyls are O + H and interlayer water is O + 2 H.  Together these give
    # 504 electrons per unit cell, against the "~500" of the paper, for the cell
    # formula 2{Al2Si4O10(OH)2 . 1.7 (CH2OH)2 . 0.8 H2O . 0.2 Ca} with the
    # octahedral composition of Kerr et al. (1950) for Clay Spur bentonite.
    #
    # Reproducing the paper's Table 2 from these rows gives R(|Fo|) = 0.039 over
    # the 13 observed orders, against the R = 0.042 of the paper's own best
    # model; see tests/test_reynolds_1965.py.  Neutral-atom scattering factors
    # were used in the original refinement, so neutral species are used here.
    (0.00, "Al", 3.08, 1.68),  # octahedral sheet, on the mirror plane
    (0.00, "Fe", 0.32, 1.68),
    (0.00, "Mg", 0.66, 1.68),
    (1.06, "O", 6.00, 1.68),  # 4 apical oxygens + 2 hydroxyls
    (1.06, "H", 2.00, 1.68),
    (2.70, "Si", 4.00, 1.68),  # tetrahedral sheet
    (3.27, "O", 6.00, 1.68),  # basal oxygens
    (6.12, "C", 1.70, 11.0),  # first glycol sheet, H2C-HO
    (6.12, "O", 1.70, 11.0),
    (6.12, "H", 5.10, 11.0),
    (7.07, "C", 1.70, 11.0),  # second glycol sheet, CH2-OH
    (7.07, "O", 1.70, 11.0),
    (7.07, "H", 5.10, 11.0),
    (7.94, "O", 0.80, 1.68),  # interlayer water and exchangeable calcium
    (7.94, "H", 1.60, 1.68),
    (7.94, "Ca", 0.20, 1.68),
]


AIR_DRIED_SMECTITE_D001 = 12.4
"""Layer repeat of an air-dried smectite in A, before glycolation.

A smectite in the laboratory atmosphere carries interlayer water, and how much
depends on the exchangeable cation and on the humidity: a Na-smectite holds one
water layer at about 12.4 A, a Ca- or Mg-smectite two at about 15 A.  12.4 A is
the usual figure for a routine air-dried mount and is what this defaults to; it
is a parameter of :func:`clayquant.library.build_library` so that a specimen
known to be Ca-saturated can be given 15 A instead.

The exact value matters much less than it looks, and that is the point of it.
What the air-dried mount is asked here is whether a candidate expandable phase
*changes* between the two treatments, and every spacing in the 12.4-15 A range
differs from the 16.86 A glycol complex by far more than a measurement's noise:
see :func:`clayquant.treatment.air_dried_observation` for the measured
sensitivity.  The discrimination is not delicate.
"""

WATER_PER_CELL = 4.0
"""Interlayer H2O per ``O20(OH)4`` layer in a one-water-layer smectite.

Four per cell is the usual figure for the 12.4 A one-layer hydrate.  It is the
least certain number in :func:`air_dried_smectite_layer` - the interlayer of an
air-dried smectite is disordered and how much water it holds depends on the
humidity of the room - so the sensitivity to it is measured rather than
assumed; see ``tests/test_air_dried_smectite.py``.
"""


def air_dried_smectite_layer(
    thickness: float = AIR_DRIED_SMECTITE_D001,
    water: float = WATER_PER_CELL,
) -> LayerModel:
    """The Reynolds smectite with the glycol replaced by one water layer.

    Glycolation changes the interlayer and leaves the 2:1 layer alone, so this
    keeps Reynolds' 2:1 rows exactly as they are - octahedral sheet on the
    mirror, the apical oxygens and hydroxyls at 1.06 A, tetrahedral silicon at
    2.70 A, basal oxygens at 3.27 A - and replaces his three interlayer planes
    (two glycol sheets and a water/calcium plane) with a single plane of water
    and the exchangeable cation at the middle of the collapsed interlayer.

    Two details are easy to get wrong and are worth stating.  The 2:1 rows are
    *not* rescaled with the repeat distance: drying empties the interlayer, it
    does not compress the tetrahedral and octahedral sheets, and scaling them
    would put the structure factor wrong.  And the interlayer plane sits at
    exactly ``thickness / 2``, which is on the cell boundary, so its occupancies
    are halved here: :meth:`LayerModel.from_table` mirrors every row with
    ``z != 0`` about ``z = 0``, and the copies at ``+d/2`` and ``-d/2`` are one
    repeat apart - the same plane, counted once.

    This is a model of a collapsed layer, not a measurement of one: Reynolds
    (1965) refined the glycol complex, and no comparable table exists here for
    the water complex that replaces it.  It is fit for the one purpose it is put
    to, which is to predict how far the 00l series of an interstratified stack
    moves between the two treatments - an effect governed by the layer
    *spacing*, which is known, and only modulated by the interlayer contents,
    which are not.  It is not fit to quantify an air-dried mount on its own, and
    nothing in ClayQuant asks it to.
    """
    thickness = float(thickness)
    if thickness <= 2.0 * 3.27:
        raise ValueError(
            f"an air-dried repeat of {thickness:g} A leaves no interlayer: the 2:1 layer "
            f"alone reaches {2.0 * 3.27:.2f} A"
        )
    interlayer = thickness / 2.0
    rows: list[tuple[float, str, float, float]] = [
        row for row in REYNOLDS_1965_EG_SMECTITE_ROWS if row[0] <= 3.27
    ]
    # Halved, because the mirror plane at z = 0 gives this row a twin at -d/2
    # that is the same plane in the cell below.
    # Occupancies follow Reynolds' convention, in which the table is half the
    # cell and every row is mirrored: his 0.20 Ca is 0.40 per cell.  There is
    # one interlayer per repeat, split between the two cell edges, so a row here
    # carries half of what that interlayer holds - hence ``water / 2`` oxygens
    # for ``water`` molecules per cell, and the same 0.20 Ca as his own row.
    rows += [
        (interlayer, "O", 0.5 * float(water), 1.68),
        (interlayer, "H", 1.0 * float(water), 1.68),
        (interlayer, "Ca", 0.20, 1.68),
    ]
    return LayerModel.from_table(
        thickness=thickness,
        rows=rows,
        name="smectite_air",
        source=(
            "2:1 layer of Reynolds, R.C. Jr. (1965) Am. Mineral. 50, 990-1001, Table 1, "
            "with the glycol interlayer replaced by one water layer"
        ),
        mirror=True,
    )


def eg_smectite_layer(thickness: float = REYNOLDS_1965_D001) -> LayerModel:
    """Ethylene glycol-montmorillonite layer after Reynolds (1965).

    Parameters
    ----------
    thickness:
        Layer repeat distance in A.  The default is the measured 16.86 A of the
        original paper; 16.9-17.1 A is also used in the literature for the
        two-layer glycol complex.
    """
    return LayerModel.from_table(
        thickness=thickness,
        rows=REYNOLDS_1965_EG_SMECTITE_ROWS,
        name="smectite_EG",
        source=(
            "Reynolds, R.C. Jr. (1965) An X-ray study of an ethylene glycol-montmorillonite "
            "complex. Am. Mineral. 50, 990-1001, Table 1"
        ),
        mirror=True,
    )
