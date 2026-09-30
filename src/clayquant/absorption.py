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
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .crystal import Crystal
from .masses import atomic_weight, element_of

__all__ = [
    "MASS_ATTENUATION_CU_KA",
    "mass_attenuation",
    "mass_attenuation_of",
    "penetration_depth",
    "thick_enough",
    "thin_film_factor",
    "mass_per_area",
]

_CHECKED_CU_KA: dict[str, float] = {
    "H": 0.435,
    "C": 4.6,
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
    "Ni": 49.2,
    "Cu": 52.9,
    "Zn": 60.3,
    "Sr": 125.0,
    "Zr": 143.0,
    "Ba": 336.0,
    "Pb": 232.0,
}
"""International Tables Volume C, Table 4.2.4.3, for the rock-forming elements.

These are the ones the program is checked on, and they are kept exactly as they
are because of that check: the published coefficients of kaolinite, illite and
chlorite come out within 1 % on these values and 2 to 4 % out on any other
tabulation tried.
"""

_ELAM_CU_KA: dict[str, float] = {
    "He": 0.2917,
    "Li": 0.4993,
    "Be": 1.106,
    "B": 2.306,
    "Ne": 22.87,
    "Ar": 116.1,
    "Sc": 179.9,
    "Ga": 62.01,
    "Ge": 67.8,
    "As": 74.5,
    "Se": 79.82,
    "Br": 88.8,
    "Kr": 94.92,
    "Rb": 104.1,
    "Y": 123.8,
    "Nb": 144.5,
    "Mo": 154.1,
    "Tc": 165.6,
    "Ru": 175.6,
    "Rh": 188.4,
    "Pd": 198.5,
    "Ag": 213,
    "Cd": 221.8,
    "In": 235.3,
    "Sn": 246.2,
    "Sb": 259,
    "Te": 266,
    "I": 287.6,
    "Xe": 298.6,
    "Cs": 316.5,
    "La": 347.4,
    "Ce": 367.4,
    "Pr": 388.9,
    "Nd": 403.2,
    "Pm": 424.9,
    "Sm": 433.4,
    "Eu": 392.7,
    "Gd": 400.9,
    "Tb": 308.3,
    "Dy": 321.6,
    "Ho": 125.1,
    "Er": 131,
    "Tm": 137.6,
    "Yb": 142.2,
    "Lu": 149.1,
    "Hf": 154.7,
    "Ta": 161.4,
    "W": 168,
    "Re": 175.1,
    "Os": 180.9,
    "Ir": 188.6,
    "Pt": 195.7,
    "Au": 204.1,
    "Hg": 210.9,
    "Tl": 217.5,
    "Bi": 234.3,
    "Po": 245.7,
    "At": 256.2,
    "Rn": 253.8,
    "Fr": 264.5,
    "Ra": 272.7,
    "Ac": 283.7,
    "Th": 289.4,
    "Pa": 302.9,
    "U": 306.1,
}
"""Elam et al. at 8047.8 eV, for every remaining element up to uranium.

What a real phase database reaches and a clay analysis is not calibrated on: a
cassiterite, a tungsten carbide, a lithium-bearing amphibole.  Generated once
and stored, so nothing is looked up at runtime.

Kept as a separate table rather than merged by hand because the two sources
differ by up to 11 % on individual elements.  A coefficient from here carries
the accuracy of its source and not of the check above, and which one a value
came from should be answerable by looking.
"""

MASS_ATTENUATION_CU_KA: dict[str, float] = {**_ELAM_CU_KA, **_CHECKED_CU_KA}
"""Mass attenuation coefficient in cm^2/g at Cu K-alpha, by element symbol.

Every element from hydrogen to uranium, so that no structure can stop a fit for
want of a coefficient, from two sources with the seam between them marked in the
table above.

The rock-forming elements are International Tables Volume C, Table 4.2.4.3, and
they stay that way because they are the ones the program is checked on: the
published coefficients of kaolinite, illite and chlorite come out within 1 % on
these and 2 to 4 % out on any other tabulation tried.  The rest are Elam et al.
at 8047.8 eV, covering what a real phase database reaches - a cassiterite, a
tungsten carbide, a lithium-bearing amphibole - where no such check exists.

The series is not monotonic in atomic number and must never be interpolated.
Two edges cross the Cu K-alpha energy: the K edge between cobalt and nickel,
where the coefficient falls from 313 to 49, and the L3 edge among the heavy
elements, where tungsten sits below lanthanum before lead rises again.  An
element missing from a table like this cannot be estimated from its neighbours,
which is why the whole table is here and why an unknown one still raises.
"""


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


def mass_attenuation_of_layer(layer) -> float:
    """The coefficient of a :class:`clayquant.crystal.LayerModel`.

    The same mass-fraction weighting as :func:`mass_attenuation_of`, over a
    layer's sites rather than a cell's.  An interstratified entry has no unit
    cell to take a composition from - its unit is one average layer - so this is
    how a mixed-layer phase gets a coefficient at all.
    """
    total_mass = 0.0
    weighted = 0.0
    for species, occupancy in zip(layer.species, layer.occupancy):
        mass = float(occupancy) * atomic_weight(species)
        total_mass += mass
        weighted += mass * mass_attenuation(species)
    if total_mass <= 0.0:
        raise ValueError(f"{layer.name or 'this layer'} has no mass to weight by")
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


def mass_per_area(mass_g: float, area_cm2: float) -> float:
    """Mass per unit area of a mount, in g/cm^2.

    The quantity a weighed film enters every absorption expression through.  It
    is not the mass: two mounts carrying the same milligrams over different
    areas absorb differently and diffract differently, so the area the
    suspension dried over has to be measured too.
    """
    if mass_g <= 0.0:
        raise ValueError(f"a mount's mass must be positive, not {mass_g}")
    if area_cm2 <= 0.0:
        raise ValueError(f"a deposited area must be positive, not {area_cm2}")
    return mass_g / area_cm2


def thin_film_factor(
    two_theta: np.ndarray | float,
    mass_attenuation_coefficient: float,
    film_mass_per_area: float,
) -> np.ndarray:
    """How much of a thick specimen's intensity a weighed film delivers.

    Every calculated pattern in ClayQuant is for a flat plate thick enough to
    absorb the beam completely: that is what the Debye-Scherrer Lorentz factor
    and the beam-overflow correction of :class:`Divergence` describe between
    them, with the specimen's ``1 / 2 mu`` folded into the scale factor.  A clay
    film smeared on a glass slide is not that specimen.  It delivers a fraction

        A = 1 - exp(-2 mu_m W / sin(theta))

    of it, with ``W`` the film's mass per area and ``mu_m`` its mass attenuation
    coefficient - the same expression :func:`thick_enough` tests, written as the
    factor rather than as a yes or no.

    The angle dependence is the whole point, and it is not a rescaling.  The
    exponent carries ``1 / sin(theta)``, so a thin film keeps more of its
    low-angle intensity than of its high-angle intensity: at the limit the
    factor becomes ``2 mu_m W / sin(theta)``, which *rises* towards low angle.
    An illite film of 1.3 mg/cm^2 delivers 0.77 of its 10 A reflection, 0.52 of
    its 5 A one and 0.38 of its 3.33 A one, so its basal series is measured with
    the first order enhanced by a factor of 1.5 against the second - which is
    the same size as the structural effects the series is used to measure, and
    in the same direction.

    Multiply a calculated pattern by this to compare it with a measurement of a
    film of known mass.  ``W`` needs the deposited area as well as the mass;
    see :func:`mass_per_area`.

    This is the correction the external-standard relation of
    :class:`clayquant.quantification.ExternalStandard` cannot make.  That
    relation is for a specimen where absorption alone decides how much
    diffracts, and the mass has cancelled out of it; a weighed film is the other
    case, and the better one, because the mass is known rather than inferred.
    """
    if mass_attenuation_coefficient <= 0.0:
        raise ValueError("a mass attenuation coefficient must be positive")
    if film_mass_per_area < 0.0:
        raise ValueError("a mass per area cannot be negative")
    theta = np.radians(np.asarray(two_theta, dtype=float) / 2.0)
    sine = np.sin(theta)
    if np.any(sine <= 0.0):
        raise ValueError("every two-theta must lie strictly between 0 and 180 degrees")
    return 1.0 - np.exp(-2.0 * mass_attenuation_coefficient * film_mass_per_area / sine)


@dataclass(frozen=True)
class FilmAbsorption:
    """What a weighed film of known mass per area delivers, angle by angle.

    Every pattern ClayQuant calculates is for a plate thick enough to absorb the
    beam completely.  A clay film smeared on a glass slide is not that plate, and
    the difference is not a scale factor: it is

        A(theta) = 1 - exp(-2 mu_m W / sin(theta))

    which carries ``1 / sin(theta)`` and so keeps more of the film's low-angle
    intensity than of its high-angle intensity.  On the weighed standards it is
    what removes a bias that leaves every weighed standard's calculated basal
    ratio 32-104 % too high, and 60 % too high on average (Sec. A.46).

    The coefficient is the **mixture's**, not each phase's, and that is the
    whole of why this is one object for a mount rather than a property of an
    entry.  A photon on its way to a kaolinite crystallite buried in an illite
    film is attenuated by the illite it passes through, so what governs the
    correction is the average composition of the film, which is what the fit is
    trying to find.  :func:`fit_film_absorption` closes that loop.

    ``mass_per_area`` is in g/cm^2 and needs the area the suspension dried over,
    which has to be measured: two mounts of the same weight spread over different
    areas absorb differently.  On a 2.5 cm round mount the area is 4.91 cm^2.
    """

    mass_per_area: float
    mass_attenuation: float

    def __post_init__(self) -> None:
        if self.mass_per_area <= 0.0:
            raise ValueError(
                f"a film's mass per area must be positive, not {self.mass_per_area}; "
                "a mount that was not weighed has no film correction, and None is how "
                "that is said"
            )
        if self.mass_attenuation <= 0.0:
            raise ValueError(
                f"a mass attenuation coefficient must be positive, not "
                f"{self.mass_attenuation}"
            )

    def factor(self, two_theta: np.ndarray | float) -> np.ndarray:
        """The fraction of a thick specimen's intensity this film delivers."""
        return thin_film_factor(two_theta, self.mass_attenuation, self.mass_per_area)

    @property
    def opacity(self) -> float:
        """``2 mu_m W``, the exponent's numerator - how thick the film is, in effect.

        Small against ``sin(theta)`` everywhere means a film so thin that the
        factor is ``2 mu_m W / sin(theta)`` and the correction is at its most
        severe; large everywhere means a film already thick enough that there is
        no correction to make.  On these mounts it is 0.03-0.07, which is the
        awkward middle: the correction is large and it varies across the scan.
        """
        return 2.0 * self.mass_attenuation * self.mass_per_area


def mixture_mass_attenuation(weights: Mapping[str, float],
                             coefficients: Mapping[str, float]) -> float:
    """The coefficient of a mixture, weighted by mass fraction.

    A mass attenuation coefficient is additive in mass fraction - that is the
    property that makes it the useful one - so a mixture's follows from its
    phases' and their weight fractions.  Phases absent from ``coefficients`` are
    dropped from both sums rather than assigned a guess, because a wrong
    coefficient propagates straight into a weight percent; if that empties the
    sum, there is nothing to average and the caller is told so.
    """
    total = 0.0
    weighted = 0.0
    for phase, weight in weights.items():
        if weight <= 0.0:
            continue
        coefficient = coefficients.get(phase)
        if coefficient is None:
            continue
        total += weight
        weighted += weight * coefficient
    if total <= 0.0:
        raise ValueError(
            "no phase with a known mass attenuation coefficient carries any weight, so "
            "the mixture's coefficient cannot be averaged; pass one explicitly"
        )
    return weighted / total
