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
    "eg_smectite_layer",
    "available_phases",
    "REYNOLDS_1965_EG_SMECTITE_ROWS",
    "REYNOLDS_1965_D001",
]


@dataclass(frozen=True)
class CifSource:
    """A crystal structure ClayQuant expects to find as a CIF file."""

    key: str
    filename: str
    icsd: int
    description: str
    layers_per_cell: int


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
    polytype, which is part of the key).
    """
    canonical = source.filename.lower()
    code = str(source.icsd)
    key = _normalised(source.key)

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
        for entry in files:
            if key in _normalised(entry.stem):
                return entry
    return None


def available_phases() -> dict[str, bool]:
    """Map each expected CIF phase key to whether a matching file was found."""
    return {key: find_structure_file(source) is not None for key, source in CIF_SOURCES.items()}


def describe_structure_search() -> str:
    """What was looked for and where, for an error or a status message."""
    lines = []
    for key, source in CIF_SOURCES.items():
        found = find_structure_file(source)
        lines.append(f"  {key}: {found if found else 'not found'} (ICSD {source.icsd})")
    searched = "\n".join(f"  {directory}" for directory in structure_directories())
    return "Searched:\n" + searched + "\nStructures:\n" + "\n".join(lines)


@lru_cache(maxsize=None)
def load_crystal(key: str) -> Crystal:
    """Load one of the :data:`CIF_SOURCES` structures."""
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
