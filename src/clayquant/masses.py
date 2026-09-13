"""Standard atomic weights, for turning scale factors into masses.

A diffractometer measures scattered intensity, which is a property of electrons.
A quantitative analysis is wanted in mass.  The bridge is the mass of the unit
that was calculated: if the pattern of a phase is computed for one unit cell (or
for one layer, for an interstratified stack), then the fitted scale factor is
proportional to the number of those units in the beam, and multiplying by the
mass of one gives a quantity proportional to mass.  That is the same relation
Rietveld analysis uses as ``W_p proportional to S_p (ZMV)_p``, written for the
way ClayQuant normalises its calculated patterns.

The values are the standard atomic weights of the IUPAC 2021 table (Prohaska et
al., *Pure Appl. Chem.* **94**, 573-600), rounded to the precision that table
gives.  Where an element has no stable isotope the mass number of its
longest-lived one is used, in brackets in the source table; those elements do
not occur in rock-forming minerals and are included only so that a stray site in
a structure file does not stop an analysis without explanation.
"""

from __future__ import annotations

from .scattering import normalise_species

__all__ = ["ATOMIC_WEIGHTS", "atomic_weight", "element_of"]

ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.95, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 97.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.04,
    "Pa": 231.04, "U": 238.03, "Np": 237.0, "Pu": 244.0,
}
"""Standard atomic weight in g/mol, by element symbol."""


def element_of(species: str) -> str:
    """The element symbol of a species label such as ``Fe3+`` or ``O2-``."""
    normalised = normalise_species(species)
    symbol = normalised.rstrip("0123456789+-")
    if not symbol:
        raise KeyError(f"cannot read an element from {species!r}")
    return symbol


def atomic_weight(species: str) -> float:
    """Standard atomic weight of ``species`` in g/mol, ignoring its charge.

    The charge is ignored deliberately: an electron weighs about 1/1800 of a
    proton, so an ionisation state changes the mass by far less than the
    uncertainty in a site occupancy, while the scattering factor - where the
    charge does matter - is looked up separately.
    """
    element = element_of(species)
    try:
        return ATOMIC_WEIGHTS[element]
    except KeyError:
        raise KeyError(
            f"no atomic weight for element {element!r} (from species {species!r})"
        ) from None
