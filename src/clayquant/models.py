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
    "illite_crystal",
    "chlorite_layer",
    "CHLORITE_OCTAHEDRA",
    "CHLORITE_PUBLISHED_IRON",
    "CHLORITE_HYDROXYL",
    "FIBRE_AXES",
    "FIBRE_ORIENTATION",
    "habit_axis",
    "MINERAL_HABIT",
    "MINERAL_CLEAVAGE",
    "ILLITE_OCTAHEDRON_TOLERANCE",
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
    cod: int | None = None
    """COD number, for a structure that comes from the Crystallography Open
    Database rather than from the ICSD.  ``icsd`` is then 0 and this carries the
    identifier, so that a message about a missing file names the right database
    and the right number to go and fetch."""
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


    @property
    def reference(self) -> str:
        """How to name this structure's source in a message to the user."""
        if self.cod:
            return f"COD {self.cod}"
        return f"ICSD {self.icsd}"


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
"""Where each phase's structure comes from, and how many layers its cell holds.

The layer count is what lets a published cell be rescaled to a different basal
spacing: a 2M structure holds two layers per cell, so its d(001) is twice the
layer repeat.  Getting it wrong rescales by a factor of two.
"""


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
    code = str(source.cod or source.icsd)
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
"""The octahedral site labels of each chlorite sheet, as the CIF names them.

Which sheet a site belongs to is read off its z coordinate in ICSD 164234 - the
2:1 layer's octahedra sit at z ~ 0 and the interlayer hydroxide sheet's at
z ~ 0.5 - and it is recorded here rather than re-derived because the two sheets
act on the basal orders in opposite directions and mixing them up would be
invisible in the result.
"""

CHLORITE_PUBLISHED_IRON: tuple[float, float] = (0.0877, 0.0580)
"""The octahedral iron of ICSD 164234 itself, as (2:1 sheet, hydroxide sheet).

The structure the chlorite entries are calculated from is not an iron-free
clinochlore.  It is a single-crystal refinement at 298 K of one particular
specimen, and its formula is H16 Al2.884 Fe0.874 Mg11.126 O36 Si5.116 - the
iron is already there, on four sites, and refined:

====  =========  ========  ========
site  Wyckoff    sheet     Fe occ.
====  =========  ========  ========
Fe1   2 a        2:1         0.087
Fe2   4 i        2:1         0.088
Fe3   4 i        OH          0.060
Fe4   2 h        OH          0.054
====  =========  ========  ========

Averaged over each sheet by site multiplicity, (2x0.087 + 4x0.088)/6 = 0.0877
and (4x0.060 + 2x0.054)/6 = 0.0580.

This matters because :func:`chlorite_crystal` *sets* the occupancies rather than
adding to them, so ``chlorite_crystal(0.0, 0.0)`` is not the published structure
- it is a magnesium end member nobody refined.  The library called that entry by
the plain name ``chlorite`` and its comment said the published structure keeps
its plain name, which was not true, and the composition axis built around the
refinement did not contain the refinement: it held the 2:1 sheet at zero while
the published value is 0.0877.  The difference is not cosmetic.  Calculated on a
240 mm instrument, the published structure gives basal orders 0.348 / 1 / 0.523 /
0.608 / 0.166 and the magnesium end member 0.389 / 1 / 0.694 / 0.676 / 0.203,
against 0.423 and 0.594 measured for the 003 on two chlorite standards: the
refinement is much the closer of the two, and it was the one not on the axis.
"""
"""The two octahedral sheets of the chlorite structure, by site label.

A chlorite layer carries two of them: the octahedral sheet of the 2:1 layer, at
the layer origin, and the interlayer hydroxide ("brucite") sheet at half the
repeat.  Iron substitutes for magnesium in both, in proportions that differ
between the two and between deposits, and because the sheets sit at different
heights the two substitutions act differently on the basal orders - which is
what makes them separable from a measured basal series
(:func:`clayquant.composition.fit_chlorite_iron`).
"""


ILLITE_OCTAHEDRON_TOLERANCE = 0.05
"""How near a layer plane an aluminium site must sit to be the octahedral one.

In fractional coordinates of the two-layer cell, whose layers are centred at
z = 0 and z = 1/2.  The illite structure puts its octahedral aluminium at
z = 0.007 and its tetrahedral aluminium at z = 0.137, so the two are separated
by twenty times this tolerance and the assignment is not delicate; it is written
by height rather than by site label so that another illite CIF, whose labels
will not be the same, is still read correctly or refused outright.
"""


FIBRE_AXES: dict[str, tuple[int, int, int]] = {
    "sepiolite": (0, 0, 1),
    "palygorskite": (0, 0, 1),
}
"""Minerals whose crystallites are needles, and the direction they are long in.

In *direct* space, because that is what the long axis of a needle is - see
:meth:`clayquant.crystal.Crystal.angle_to_direction`.  Both chain clays are
elongated along ``c``.

They are here rather than in :data:`MINERAL_HABIT` because a needle and a plate
settle differently and the March model treats them as opposite cases.  A plate
lies on its face, which is axially symmetric *compression* of the orientation
distribution: the pole of that face points along the specimen normal, and
``r < 1``.  A needle lies with its long axis in the specimen plane and is free
to roll about it, which is *expansion*: the fibre axis is pushed away from the
specimen normal, and ``r > 1`` (Dollase 1986, J. Appl. Cryst. 19, 267-272).

The distinction is not academic and the duality makes it exact.  A needle along
``c`` lying in the plane can only diffract from planes whose normal is
perpendicular to ``c``, and ``g . c = l``, so the set it shows is ``hk0``
precisely, whatever the cell angles.  Measured instead from ``c*`` - which for
the monoclinic palygorskite here is 17 degrees away - the 110 comes out at 76
degrees rather than 90 and is under-enhanced.

Measured on the sepiolite standard: with the 110 pole and ``r < 1`` the best
agreement with the measured pattern is a cosine of 0.623, and the fitted ``r``
runs to the bottom of its range because the pattern then collapses to the single
110 line and nothing is left to determine ``r`` with.  About the fibre axis with
``r > 1`` the best is 0.756, at ``r = 1.5``.
"""


FIBRE_ORIENTATION = 1.5
"""The March-Dollase value a needle is calculated at, everywhere.

Above 1 because a needle lies with its long axis *in* the specimen plane, which
is the expansion case of the March model and not the compression case of a
plate.  One value rather than a range because an oriented basal scan cannot
measure it: over the whole needle range the calculated pattern keeps its shape
to a cosine of 0.998 while the mass it implies scales as ``r^3``, so a spanned
``r`` is a free multiplier on the weight rather than a parameter the fit
determines (Sec. A.65).  1.5 is where agreement with the sepiolite standard is
best; the optimum is shallow and what the value fixes is the basis the weight
percent is on.
"""


MINERAL_HABIT: dict[str, tuple[int, int, int]] = {
    # Sepiolite and palygorskite were here once, on the (110) pole with r < 1.
    # They are needles, not plates, and they are now in FIBRE_AXES; the
    # amphiboles below keep this pole because a prism thick enough to settle on
    # a face really is plate-like, and that was measured rather than assumed.
    # The amphiboles, for the same reason and with the same pole.  They are
    # prismatic, they settle on a prism face, and (110) is the face they settle
    # on, so an oriented mount shows the 8.4 A (110) and very little else.  On a
    # real separate the 10.4 deg line stood at 9592 counts while the 35.3 and
    # 32.9 deg lines - 59 and 54 per cent of it in a random powder - held 127
    # and 157.  Every orientation that fits that is a (110) pole at r = 0.3-0.4
    # and no random-powder amphibole comes close (Sec. A.48).
    # Riebeckite on (020) and hornblende on (110), which is not a guess from
    # the shape of an amphibole but what the refinements of these two give.
    # They are different directions and the distinction is measurable: (020) is
    # the b normal, (110) the prism face, and the two predict different
    # intensities for every line but the one they share.
    "riebeckite": (0, 2, 0),
    "hornblende": (1, 1, 0),
    "actinolite": (1, 1, 0),
    "tremolite": (1, 1, 0),
    "glaucophane": (1, 1, 0),
    "anthophyllite": (1, 1, 0),
    "cummingtonite": (1, 1, 0),
    "grunerite": (1, 1, 0),
}
"""Preferred-orientation pole per mineral, for minerals that are not platelets.

Keyed on a lower-case mineral name.  A mineral absent from this table is taken
to be equant, which is the right assumption for quartz, the feldspars and the
carbonates in a clay mount, and a layer silicate is flattened on 001, which the
clay library assumes throughout without needing an entry here.

Sepiolite and palygorskite are the reason the table exists.  They are clay
minerals but not layer silicates: chain silicates whose crystallites are laths,
elongated along c, which settle on a side face rather than on a basal plane.
Measured on the sepiolite standard, no orientation about c* improves the fit
at all - every value below 1 makes it worse, which is what "this is not a
platelet" looks like - while orientation about the 110 pole does improve it
(Sec. A.37).
"""


MINERAL_CLEAVAGE: dict[str, tuple[int, int, int]] = {
    # Feldspars: perfect on (001), good on (010).
    "albite": (0, 0, 1), "anorthite": (0, 0, 1), "labradorite": (0, 0, 1),
    "microcline": (0, 0, 1), "orthoclase": (0, 0, 1), "sanidine": (0, 0, 1),
    # Carbonates, on the rhombohedron.
    "calcite": (1, 0, 4), "dolomite": (1, 0, 4), "ankerite": (1, 0, 4),
    "magnesite": (1, 0, 4), "siderite": (1, 0, 4),
    # Sheet minerals that are not clays, and sulphates that cleave as well.
    "muscovite": (0, 0, 1), "biotite": (0, 0, 1), "phlogopite": (0, 0, 1),
    "talc": (0, 0, 1), "pyrophyllite": (0, 0, 1), "graphite": (0, 0, 1),
    "gibbsite": (0, 0, 1), "brucite": (0, 0, 1), "boehmite": (0, 2, 0),
    "gypsum": (0, 1, 0), "barite": (0, 0, 1),
}
"""Accompanying minerals that cleave, and the pole they would lie down on.

Distinct from :data:`MINERAL_HABIT`, and the difference matters.  A sepiolite
crystallite *is* a lath and is oriented whether anyone asks or not.  A feldspar
is equant until it is broken, and then it breaks on (001); whether the flakes in
a particular mount have settled flat is a question about that mount, so it is
offered rather than assumed.

Offered, and not to everything, because giving an equant mineral an orientation
it does not have is not a harmless extra parameter.  On a real separate,
allowing every accompanying mineral to orient let quartz be fitted at r = 0.3 -
a basal series is enhanced as r^-3, so that is 37 times the mass for the same
scattering - and quartz went from 17 per cent of the specimen to 64, at a
R_wp *better* by 0.007.  Quartz has no cleavage and its grains in a clay
separate are equant; the fit was not measuring a texture but spending a
parameter.  Restricting the offer to minerals that actually cleave leaves quartz
where it was and still closes the albite line (Sec. A.45).
"""


CHLORITE_HYDROXYL: dict[str, tuple[str, ...]] = {
    "oxygen": ("O7", "O8", "O9"),
    "hydrogen": ("H2", "H3", "H4"),
}
"""The hydroxyl groups of the interlayer hydroxide sheet, by site label.

Not the 2:1 layer's own hydroxyl (O6, H1), which sits inside the layer and
survives to a higher temperature than the interlayer sheet does.
"""



FIBROUS_CIF_SOURCES: dict[str, CifSource] = {
    source.key: source
    for source in [
        CifSource(
            key="sepiolite",
            filename="sepiolite_COD_9014723.cif",
            icsd=0,
            cod=9014723,
            description=(
                "Sepiolite, Pncn, Post, Bish & Heaney (2007) Am. Mineral. 92, 91-97"
            ),
            layers_per_cell=1,
        ),
        CifSource(
            key="palygorskite",
            filename="palygorskite_COD_1533365.cif",
            icsd=0,
            cod=1533365,
            description=(
                "Palygorskite, C2/m, Giustetto & Chiari (2004) Eur. J. Mineral. 16, 521-532"
            ),
            layers_per_cell=1,
            synonyms=("attapulgite",),
        ),
    ]
}
"""The two channel clays, which are built differently from everything else.

A sepiolite or a palygorskite is not a stack of layers with an interlayer: it is
a 2:1 ribbon structure with channels running along the fibre, and the channels
hold zeolitic water.  That has three consequences and all of them matter here.

Its strongest reflection is the 110 and not a 00l, so it is built from the
three-dimensional structure like any other mineral rather than from a layer
model, and its texture pole is the fibre axis - :data:`MINERAL_HABIT` already
carries (110) for both.  There is no layer spacing to span and no
interstratification to model.

And the channels take water but not ethylene glycol, so a channel clay does not
move between the air-dried and the glycolated mount.  That is what separates it
from a smectite, whose 001 goes from about 12.4 to 16.9 A, and it is a positive
test rather than an absence: see
:func:`clayquant.treatment.channel_clay_evidence`.
"""



ALL_CIF_SOURCES: dict[str, CifSource] = {**CIF_SOURCES, **FIBROUS_CIF_SOURCES}
"""Every structure ClayQuant knows how to load, basal clays and channel clays.

:data:`CIF_SOURCES` alone drives the interstratified and basal-series machinery,
which the channel clays have no part in; this is for looking a structure up by
name.
"""


def illite_crystal(potassium: float = 1.0, iron: float = 0.0) -> Crystal:
    """The illite structure with this interlayer potassium and octahedral iron.

    Two substitutions, and between them they account for the basal series of a
    measured illite, which the published structure does not.

    The published structure (ICSD 90144) carries K at full occupancy, which is a
    muscovite: an illite is defined by having less, 0.75 to 0.9 of an atom per
    O10(OH)2.  The K sits at z = 1/4 of the two-layer cell, exactly between the
    layers, so its phase factor alternates: it subtracts from the 10 A
    reflection, adds to the 5 A one and subtracts from the 3.33 A one.  Removing
    some of it therefore lowers the higher orders against the first.

    ``iron`` is the fraction of the octahedral aluminium replaced by Fe(3+),
    which an illite carries in the range 0.05 to 0.2.  That sheet lies at the
    middle of the 2:1 layer, so it is the term the layer is nearly symmetric
    about, and adding an atom with three times aluminium's electrons there
    reweights the whole series - most strongly the 5 A order, which the
    potassium barely moves.

    Measured on an illite standard on this instrument, the basal series is
    1 : 0.164 : 0.477.  The published structure calculates 1 : 0.507 : 0.868 -
    the 5 A order three times too strong and the 3.33 A order nearly twice.
    Lowering the potassium to 0.6 brings the 3.33 A order to 0.581 and leaves
    the 5 A order at 0.449, so it cannot be the whole explanation; a proper
    illite interlayer of 0.9 with 0.15 of octahedral iron gives
    1 : 0.173 : 0.449, which is the measurement to within 6 per cent on both.  Sec. A.35 of the manual sets out what else was tried -
    crystallite thickness, microstrain, specimen length, layer spacing and the
    height of the potassium itself - and why none of them does it.
    """
    value = float(potassium)
    if not 0.0 <= value <= 1.0:
        raise ValueError("the potassium occupancy must lie in [0, 1]")
    fraction = float(iron)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("the octahedral iron fraction must lie in [0, 1]")
    base = load_crystal("illite")
    if not any(site.label.startswith("K") for site in base.sites):
        raise ValueError("the illite structure has no interlayer potassium site")

    def is_octahedral(site) -> bool:
        if not site.species.startswith("Al"):
            return False
        # Distance to the nearer of the two layer planes, z = 0 and z = 1/2.
        height = abs((site.z % 0.5 + 0.5) % 0.5)
        return min(height, 0.5 - height) < ILLITE_OCTAHEDRON_TOLERANCE

    if fraction and not any(is_octahedral(site) for site in base.sites):
        raise ValueError("the illite structure has no octahedral aluminium site")

    sites = []
    for site in base.sites:
        if site.label.startswith("K"):
            sites.append(dataclasses.replace(site, occupancy=value))
        elif fraction and is_octahedral(site):
            sites.append(dataclasses.replace(
                site, occupancy=site.occupancy * (1.0 - fraction)))
            sites.append(dataclasses.replace(
                site, label=f"{site.label}_Fe", species="Fe3+",
                occupancy=site.occupancy * fraction))
        else:
            sites.append(site)
    return dataclasses.replace(base, sites=sites)


def chlorite_crystal(
    iron_2to1: float,
    iron_hydroxide: float,
    octahedral_b: float | None = None,
    dehydroxylation: float = 0.0,
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

    ``dehydroxylation`` is how far the *interlayer* hydroxide sheet has lost its
    water, 0 for the mineral as measured air dried and 1 for a sheet that has
    given up all of it.  Heating an oriented mount to 550 C is what this is for.
    The reaction is 2 OH(-) -> O(2-) + H2O, so half the sheet's oxygen leaves
    with all of its hydrogen, and the occupancies follow that: the oxygen sites
    of :data:`CHLORITE_HYDROXYL` scale by ``1 - delta/2`` and the hydrogen sites
    by ``1 - delta``.  The 2:1 layer's own hydroxyl (O6, H1) is untouched, as it
    is in the mineral at this temperature.

    The consequence for the pattern is large and is the reason heating is worth
    calculating rather than guessing at.  A chlorite's 001 reflects the
    *contrast* between the two sheets and its even orders their sum, so emptying
    the hydroxide sheet raises the 001 several times over while the even orders
    collapse.  Calculated on the published structure, 002/001 falls from 1.23 to
    0.07 across the range while the 001 grows by 5.9; measured on two heated
    chlorite standards, 002/001 is 0.168 and 0.752 against 1.862 and 2.289 air
    dried, and the 001 grows by 3.51 and 2.18.  Both specimens therefore sit
    inside this range, at about 0.7 and 0.2, and neither is fully
    dehydroxylated after 1.5 h - which is why this is a parameter to span and
    not a second structure.

    The cell mass falls with the water, so a weight percent computed from a
    heated mount is on the dehydroxylated mass; convert to the mineral as
    weighed by dividing by the mass ratio the two structures report.
    """
    for name, value in (("iron_2to1", iron_2to1), ("iron_hydroxide", iron_hydroxide)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} is an occupancy and must lie in [0, 1], not {value}")
    if octahedral_b is not None and octahedral_b < 0.0:
        raise ValueError(
            f"a displacement parameter cannot be negative, and {octahedral_b} is"
        )
    delta = float(dehydroxylation)
    if not 0.0 <= delta <= 1.0:
        raise ValueError(
            f"dehydroxylation runs from 0 (as measured) to 1 (no water left), not {delta}"
        )

    base = load_crystal("chlorite")
    iron_of = {"2:1": float(iron_2to1), "hydroxide": float(iron_hydroxide)}
    sheet_of = {
        label: sheet for sheet, labels in CHLORITE_OCTAHEDRA.items() for label in labels
    }
    water = {
        label: 1.0 - 0.5 * delta for label in CHLORITE_HYDROXYL["oxygen"]
    } | {
        label: 1.0 - delta for label in CHLORITE_HYDROXYL["hydrogen"]
    }
    if delta and not any(site.label in water for site in base.sites):
        raise ValueError("the chlorite structure has no interlayer hydroxyl sites")
    sites = []
    for site in base.sites:
        sheet = sheet_of.get(site.label)
        if sheet is None:
            if delta and site.label in water:
                site = dataclasses.replace(
                    site, occupancy=site.occupancy * water[site.label])
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
    if delta:
        described += (
            f", interlayer hydroxide sheet {delta:.3f} dehydroxylated "
            "(2 OH -> O + H2O)"
        )
    return dataclasses.replace(
        base,
        sites=sites,
        name=(f"chlorite Fe {iron_2to1:.2f}/{iron_hydroxide:.2f}"
              + (f" dehydrox {delta:.2f}" if delta else "")),
        source=described,
    )


def chlorite_layer(
    iron_2to1: float,
    iron_hydroxide: float,
    octahedral_b: float | None = None,
    dehydroxylation: float = 0.0,
) -> LayerModel:
    """One folded layer of :func:`chlorite_crystal`."""
    crystal = chlorite_crystal(iron_2to1, iron_hydroxide, octahedral_b=octahedral_b,
                               dehydroxylation=dehydroxylation)
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
        lines.append(f"  {key}: {found if found else 'not found'} ({source.reference})")
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
        source = ALL_CIF_SOURCES[key]
    except KeyError:
        raise KeyError(
            f"unknown phase {key!r}; expected one of {sorted(ALL_CIF_SOURCES)}"
        ) from None
    path = find_structure_file(source)
    if path is None:
        raise FileNotFoundError(
            f"No CIF for {key} was found.\n"
            f"ClayQuant does not redistribute licensed structure data. Export "
            f"{source.reference} "
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
"""The glycolated smectite layer, as Reynolds (1965) tabulated it.

Am. Mineral. 50, 990-1001.  Transcribed rather than recalculated: the electron
density of an ethylene glycol complex is not something this program models from
first principles, and substituting a guess for the table would be inventing the
one layer the whole expandable-clay method rests on.
"""


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



# --------------------------------------------------------------------------- #
# Trioctahedral smectite (saponite)
# --------------------------------------------------------------------------- #

MIRROR_PLANE_TOLERANCE = 0.05
"""How close to the octahedral plane a site must be to count as sitting on it, in A.

A site refined at ``z = 0.99987`` is on the mirror plane and folds to a height
of about minus two thousandths of an angstrom.  Discarding it as negative loses
a third of the octahedral sheet - four cations where there should be six - and
the layer is then dioctahedral by accident.  The tolerance is far larger than
the rounding it guards against and far smaller than the 1.05 A to the next
plane, so nothing else can fall inside it.
"""


def trioctahedral_two_one_rows() -> list[tuple[float, str, float, float]]:
    """The 2:1 sheet of the chlorite refinement, as half-layer rows.

    A saponite is a trioctahedral smectite: a talc-like 2:1 layer with an
    expandable interlayer.  There is no saponite among the structures this
    program is given, and inventing a table for one is exactly what must not
    happen - but the 2:1 layer is not something that needs inventing, because
    the chlorite refinement already contains one.  A chlorite is a 2:1 layer
    alternating with a hydroxide sheet, and :data:`CHLORITE_OCTAHEDRA` and
    :data:`CHLORITE_HYDROXYL` already say which sites are which.  Take the 2:1
    sites and leave the hydroxide sheet behind, and what is left is a published
    trioctahedral 2:1 layer with published heights, occupancies and displacement
    parameters.

    It expands to six octahedral cations, eight tetrahedral and O20(OH)4, which
    is the trioctahedral 2:1 formula; and its planes agree with the ones
    Reynolds (1965) refined for the dioctahedral layer to within 0.06 A -
    octahedral sheet on the mirror, apical oxygens and hydroxyls at 1.05-1.10
    against his 1.06, tetrahedral cations at 2.74 against his 2.70, basal
    oxygens at 3.33 against his 3.27.  The two refinements describe the same
    object and disagree about it by less than a twentieth of an angstrom.

    What differs between them, and what a basal series is actually sensitive to,
    is the octahedral sheet: six magnesium against four aluminium is 72
    electrons against 52 in the plane that sits at the centre of the layer.  The
    tetrahedral sheet differs too - this one carries 1.44 Al per four cations
    against Reynolds' none - but silicon and aluminium differ by one electron in
    fourteen, so that difference is worth about two parts in a hundred of one
    plane and is not what distinguishes the two minerals in a diffractogram.

    The layer charge is therefore chlorite's, not a saponite analysis, and the
    exchangeable cation that balances it is Reynolds' 0.2 Ca rather than one
    computed from that charge.  Both affect the mass by a fraction of a per cent
    and neither affects the 00l intensities measurably, but they are assumptions
    and are named here rather than buried.
    """
    crystal = load_crystal("chlorite")
    hydroxide = (set(CHLORITE_OCTAHEDRA["hydroxide"])
                 | set(CHLORITE_HYDROXYL["oxygen"])
                 | set(CHLORITE_HYDROXYL["hydrogen"]))
    planes: dict[tuple[float, str], float] = {}
    b_isos: dict[tuple[float, str], float] = {}
    for site in crystal.expanded_sites():
        if site.label in hydroxide:
            continue
        height = site.z * crystal.d001
        if height > crystal.d001 / 2.0:
            height -= crystal.d001
        if height < -MIRROR_PLANE_TOLERANCE:
            continue            # the far side of the mirror; from_table adds it
        height = max(height, 0.0)
        key = (round(height, 2), site.species)
        planes[key] = planes.get(key, 0.0) + site.occupancy
        b_isos[key] = site.b_iso
    return [(z, species, occupancy, b_isos[(z, species)])
            for (z, species), occupancy in sorted(planes.items())]


SAPONITE_SOURCE = (
    "2:1 layer of ICSD 164234 (clinochlore) with its hydroxide sheet removed, "
    "interlayer of Reynolds, R.C. Jr. (1965) Am. Mineral. 50, 990-1001, Table 1"
)


def saponite_layer(thickness: float = REYNOLDS_1965_D001) -> LayerModel:
    """A glycolated trioctahedral smectite layer.

    The 2:1 sheet of :func:`trioctahedral_two_one_rows` carrying the ethylene
    glycol interlayer Reynolds (1965) refined - his three interlayer planes at
    6.12, 7.07 and 7.94 A, unchanged.  Glycolation acts on the interlayer and
    leaves the 2:1 layer alone, so the two pieces are independent and joining
    them is not a refinement of anything.

    Neither this layer nor :func:`air_dried_saponite_layer` has been checked
    against a measured trioctahedral smectite, because the saponite standard in
    this collection shows one broad 001 near 11.7 A and no usable higher order,
    and a single reflection constrains a spacing and not a layer model.
    """
    rows = trioctahedral_two_one_rows() + [
        row for row in REYNOLDS_1965_EG_SMECTITE_ROWS if row[0] > 3.27
    ]
    return LayerModel.from_table(thickness=thickness, rows=rows, name="saponite_EG",
                                 source=SAPONITE_SOURCE, mirror=True)


def air_dried_saponite_layer(
    thickness: float = AIR_DRIED_SMECTITE_D001,
    water: float = WATER_PER_CELL,
) -> LayerModel:
    """The same trioctahedral layer with one water layer in place of the glycol.

    Built exactly as :func:`air_dried_smectite_layer` is, and carrying the same
    caveats: the 2:1 rows are not rescaled with the repeat, and the interlayer
    plane at ``thickness / 2`` is halved because its mirror image one repeat
    away is the same plane.
    """
    thickness = float(thickness)
    if thickness <= 2.0 * 3.33:
        raise ValueError(
            f"an air-dried repeat of {thickness:g} A leaves no interlayer: the "
            f"trioctahedral 2:1 layer alone reaches {2.0 * 3.33:.2f} A"
        )
    interlayer = thickness / 2.0
    rows = trioctahedral_two_one_rows() + [
        (interlayer, "O", 0.5 * float(water), 1.68),
        (interlayer, "H", 1.0 * float(water), 1.68),
        (interlayer, "Ca", 0.20, 1.68),
    ]
    return LayerModel.from_table(thickness=thickness, rows=rows, name="saponite_air",
                                 source=SAPONITE_SOURCE + ", glycol replaced by one water layer",
                                 mirror=True)


def habit_axis(name: str, crystal=None) -> tuple[tuple[float, float, float], str] | None:
    """The texture axis of a mineral and which lattice it is given in.

    Returns ``(axis, space)`` with ``space`` either ``"direct"`` - a needle's
    long axis, from :data:`FIBRE_AXES` - or ``"reciprocal"`` - a plate or prism
    face normal, from :data:`MINERAL_HABIT`.  ``None`` for a mineral with no
    habit worth describing, which is most of them.

    Callers that cannot carry the space through should prefer this and pass the
    space on; one that takes only an axis will read a direct one as reciprocal,
    which is exact for an orthorhombic cell and 17 degrees out for the
    monoclinic palygorskite.
    """
    fibre = FIBRE_AXES.get(name.strip().lower())
    if fibre is not None:
        return tuple(float(v) for v in fibre), "direct"
    pole = habit_pole(name, crystal)
    return None if pole is None else (pole, "reciprocal")


def habit_pole(name: str, crystal=None) -> tuple[float, float, float] | None:
    """The texture pole of a mineral that has a crystal habit, or ``None``.

    The pole comes from :data:`MINERAL_HABIT`, keyed by name, and not from
    whatever axis the structure database happens to carry.  That distinction is
    the whole point and it was got wrong once: a refined preferred-orientation
    correction in somebody's TOPAS input is a fitting parameter, so taking it as
    a statement of habit would span quartz, while *failing* to find one would
    silently drop the habit of a mineral that has one - which is what happened
    to sepiolite and palygorskite, whose database entries carry no axis at all,
    so the mechanism written for them did nothing for them.

    ``crystal`` is accepted so a caller can pass what it has; it is used only as
    a fallback for a mineral named in :data:`MINERAL_HABIT` whose pole is
    recorded there as ``None``, which none currently is.
    """
    pole = MINERAL_HABIT.get(name.strip().lower())
    if pole is not None:
        return tuple(float(v) for v in pole)
    if crystal is not None and getattr(crystal, "po_axis", None) is not None:
        return None
    return None
