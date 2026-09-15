"""Mass attenuation coefficients, for the external-standard scale.

A weight percent from a fitted scale factor is relative: it says how the phases
of one mount divide that mount between them, and nothing about how much of the
mount was crystalline or whether the calculated pattern accounted for everything
the phase contributes (Sec. 2.13).  A standard measured through the same optics
fixes that, and the relation it enters is

    W(p) = S(p) (ZMV)(p) mu_m / K

with ``mu_m`` the specimen's *mass* attenuation coefficient and ``K`` an
instrument constant obtained once from the standard (O'Connor & Raven 1988).
The absorption term is there because the relation is for a flat plate thick
enough to absorb the beam completely: how deep the beam reaches, and so how much
material diffracts, is set by the specimen's own absorption.  That is why this
table is needed and why a thin film breaks the method rather than bending it -
see :func:`penetration_depth`.

Values are mass attenuation coefficients in cm^2/g at Cu K-alpha
(1.5418 A, 8.048 keV), from International Tables for Crystallography Volume C.
Only the elements that occur in the phases ClayQuant is used on are tabulated; an
element that is not here raises rather than being guessed at, because a silent
default would propagate into a weight percent.

Note the two discontinuities that matter for clay work.  Iron, manganese and
chromium sit just *below* the Cu K-alpha energy with their K edges, so they
absorb enormously - iron at 308 cm^2/g against magnesium at 38.6 - which is why
an iron-rich clay is not interchangeable with an iron-poor one in an
absorption-corrected analysis.  Nickel, copper and zinc sit just *above* it and
absorb little.
"""

from __future__ import annotations

import math

from .crystal import Crystal
from .masses import atomic_weight, element_of

__all__ = [
    "MASS_ATTENUATION_CU_KA",
    "mass_attenuation",
    "mass_attenuation_of",
    "penetration_depth",
    "thick_enough",
]

MASS_ATTENUATION_CU_KA: dict[str, float] = {
    "H": 0.435,
    "C": 4.60,
    "N": 7.52,
    "O": 11.5,
    "F": 15.95,
    "Na": 30.1,
    "Mg": 38.6,
    "Al": 48.6,
    "Si": 60.6,
    "P": 74.1,
    "S": 89.1,
    "Cl": 106.0,
    "K": 148.0,
    "Ca": 172.0,
    "Ti": 208.0,
    "V": 233.0,
    "Cr": 260.0,
    "Mn": 285.0,
    "Fe": 308.0,
    "Co": 313.0,
    # Nickel through zinc lie above the Cu K-alpha energy and absorb little.
    "Ni": 49.2,
    "Cu": 52.9,
    "Zn": 60.3,
    "Sr": 125.0,
    "Zr": 143.0,
    "Ba": 336.0,
    "Pb": 232.0,
}
"""Mass attenuation coefficient in cm^2/g at Cu K-alpha, by element symbol."""


def mass_attenuation(species: str) -> float:
    """The coefficient for a species as a CIF writes it, ``Fe3+`` included."""
    element = element_of(species)
    try:
        return MASS_ATTENUATION_CU_KA[element]
    except KeyError:
        raise KeyError(
            f"no Cu K-alpha mass attenuation coefficient for {element!r} (from {species!r}). "
            "It is not guessed at, because a wrong one propagates into a weight percent; add "
            "the value from International Tables Volume C to MASS_ATTENUATION_CU_KA."
        ) from None


def mass_attenuation_of(crystal: Crystal) -> float:
    """The coefficient of a phase, weighted by the mass fractions of its elements.

    A mass attenuation coefficient is additive in mass fraction, which is what
    makes it usable here: the coefficient of a mixture follows from the
    coefficients of its phases and their weight fractions the same way.
    """
    total_mass = 0.0
    weighted = 0.0
    for site in crystal.expanded_sites():
        mass = site.occupancy * atomic_weight(site.species)
        total_mass += mass
        weighted += mass * mass_attenuation(site.species)
    if total_mass <= 0.0:
        raise ValueError(f"{crystal.name or 'this structure'} has no mass to weight by")
    return weighted / total_mass


def penetration_depth(
    two_theta: float, mass_attenuation_coefficient: float, density: float
) -> float:
    """Depth in micrometres from which a fraction 1 - 1/e of the intensity comes.

    In Bragg-Brentano reflection the beam travels in and out through the same
    depth, so the linear attenuation counts twice and the depth is
    ``sin(theta) / (2 mu)``.  It shrinks towards low angle, which is the
    counter-intuitive part: the basal reflections a clay analysis lives on come
    from the shallowest material in the mount.
    """
    theta = math.radians(two_theta / 2.0)
    mu = mass_attenuation_coefficient * density          # cm^-1
    if mu <= 0.0:
        raise ValueError("a linear attenuation coefficient must be positive")
    return 1.0e4 * math.sin(theta) / (2.0 * mu)


def thick_enough(
    film_thickness_um: float,
    two_theta: float,
    mass_attenuation_coefficient: float,
    density: float,
    fraction: float = 0.98,
) -> bool:
    """Whether a film of this thickness absorbs ``fraction`` of what it would.

    The external-standard relation is for a specimen thick enough that its own
    absorption, not its mass, decides how much diffracts.  A pressed powder is;
    a smeared clay film on a glass slide may not be, and at the high-angle end
    of a scan least of all, since the penetration depth grows with angle.  Where
    this returns ``False`` the relation does not apply and the mount has to be
    weighed instead.
    """
    depth = penetration_depth(two_theta, mass_attenuation_coefficient, density)
    # Intensity from a film of thickness t, against an infinite one: 1 - exp(-t/depth).
    return (1.0 - math.exp(-film_thickness_um / depth)) >= fraction
