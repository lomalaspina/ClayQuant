"""X-ray fluorescence of the same mounts, as an independent composition check.

A basal series constrains a *combination* of substitutions and not either one
alone.  Taking potassium out of an illite lowers its higher basal orders;
putting iron into the octahedral sheet lowers them; and so does the film being
thin (:func:`clayquant.absorption.thin_film_factor`).  Nothing in a
diffractogram separates the three.  A fluorescence analysis of the same mount
does, because it measures the elements rather than their joint effect on an
intensity ratio - which is why it is worth the trouble even though, as this
module spends most of its length saying, it cannot be read naively.

Three things about a fluorescence analysis of a thin film on a glass slide have
to be respected or the numbers are worse than useless.

**It is normalised.**  The instrument reports concentrations summing to 100 per
cent, so there is no absolute intensity and a blank cannot simply be subtracted.
What can be done is to treat the measurement as a mixture of the film and the
slide and solve for the mixing fraction from an element the film has none of.

**The slide is most of it.**  On the mounts this was written for the glass
contributes half to nine tenths of the normalised signal, because a few
milligrams of clay is nearly transparent to the energies involved.  A sample
where the marker says nine tenths has no composition worth reading.

**Light elements are not measurable this way.**  Na, Mg, Al and Si fluoresce
between 1.0 and 1.7 keV, and both the film and the air path absorb those lines
heavily while passing K, Ca, Ti and Fe at 3.3 to 6.4 keV.  Corrected absolute
concentrations of the light elements come out impossible - 30 per cent
potassium in an illite - which is the honest signature of the method being
pushed past what it can do.  What survives is a *ratio of two heavy elements*,
and the reason it survives is worth stating: both are minor in the glass, so as
the mixing fraction is varied the ratio barely moves.  On the illite mount
Fe/K runs from 0.699 to 0.797 as the glass fraction goes from nothing to 0.85 -
a spread of 14 per cent against concentrations that change fivefold.

So this module offers exactly that: read the file, estimate how much of it is
the slide, and take ratios of elements heavy enough to mean something.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Measurement",
    "read_xrf",
    "RELIABLE_ELEMENTS",
    "GLASS_MARKERS",
    "illite_octahedral_iron",
]

RELIABLE_ELEMENTS: frozenset[str] = frozenset({"K", "Ca", "Ti", "V", "Cr", "Mn", "Fe",
                                               "Ni", "Cu", "Zn", "Rb", "Sr", "Y", "Zr"})
"""Elements whose ratios are worth reading off a thin film on glass.

Everything from potassium upwards, whose K-alpha lies above about 3 keV.  Below
that the film and the air path absorb the line itself, so the concentration
reported is not a concentration but an attenuation - see the module docstring.
"""

GLASS_MARKERS: tuple[str, ...] = ("Ca", "Na")
"""Elements a soda-lime slide has much of and a clay normally has none of.

Two rather than one so that they can be checked against each other: the mixing
fraction is only believable where independent markers agree.  Neither is safe
on every specimen - a montmorillonite is commonly calcium-exchanged, which
makes calcium its own as well as the slide's - and that failure is visible as
the two markers disagreeing, or as a marker implying more slide than there is
room for.
"""


@dataclass
class Measurement:
    """One fluorescence measurement, as the instrument reports it."""

    name: str
    values: dict[str, float] = field(default_factory=dict)
    """Concentration per material, in per cent.  A material below detection is
    absent from this mapping rather than stored as zero: the two are different
    statements and only one of them is a measurement."""

    sigmas: dict[str, float] = field(default_factory=dict)
    date: str = ""
    mode: str = ""
    """``"oxides"`` or ``"elements"``, from which list the instrument reported."""

    @property
    def total(self) -> float:
        return sum(self.values.values())

    def get(self, symbol: str) -> float | None:
        return self.values.get(symbol)

    def glass_fraction(self, blank: "Measurement", marker: str = "Ca") -> float | None:
        """The share of this measurement that is the slide, from one marker.

        ``None`` where either measurement lacks the marker.  A value near or
        above 1 does not mean the mount is empty; it means the marker is not
        the slide's alone, and the estimate has to be thrown away rather than
        believed.
        """
        here, there = self.get(marker), blank.get(marker)
        if not here or not there:
            return None
        return here / there

    def marker_agreement(self, blank: "Measurement") -> dict:
        """What each marker says, and how far apart they are.

        The check that decides whether a mixing fraction from this mount can be
        used at all.
        """
        fractions = {
            marker: self.glass_fraction(blank, marker)
            for marker in GLASS_MARKERS
        }
        usable = [value for value in fractions.values() if value is not None]
        spread = (abs(max(usable) - min(usable)) / max(usable)
                  if len(usable) > 1 and max(usable) > 0 else None)
        return {"fractions": fractions, "relative_spread": spread}

    def ratio(
        self,
        numerator: str,
        denominator: str,
        blank: "Measurement | None" = None,
        glass_fraction: float = 0.0,
    ) -> float:
        """A concentration ratio, optionally with the slide's share removed.

        Raises for an element the method cannot measure on a film, rather than
        returning a number that would be quoted later as though it meant
        something.
        """
        for symbol in (numerator, denominator):
            if symbol not in RELIABLE_ELEMENTS:
                raise ValueError(
                    f"{symbol!r} is not one of the elements a thin film on glass gives a "
                    "usable ratio for; its line is absorbed by the film and the air path, "
                    "so what is reported is an attenuation and not a concentration. "
                    f"Use one of {sorted(RELIABLE_ELEMENTS)}."
                )
        top, bottom = self.get(numerator), self.get(denominator)
        if not top or not bottom:
            raise ValueError(
                f"this measurement has no {numerator if not top else denominator}"
            )
        if blank is not None and glass_fraction:
            if not 0.0 <= glass_fraction < 1.0:
                raise ValueError("a glass fraction must lie in [0, 1)")
            top = (top - glass_fraction * (blank.get(numerator) or 0.0)) / (1.0 - glass_fraction)
            bottom = (bottom - glass_fraction * (blank.get(denominator) or 0.0)) / (1.0 - glass_fraction)
            if top <= 0.0 or bottom <= 0.0:
                raise ValueError(
                    "removing the slide's share leaves one of these elements at or below "
                    "zero, which means the marker over-estimated the slide for this mount"
                )
        return top / bottom

    def ratio_envelope(
        self,
        numerator: str,
        denominator: str,
        blank: "Measurement",
        fractions: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 0.85),
    ) -> tuple[float, float]:
        """The lowest and highest the ratio goes over a range of slide shares.

        The honest form of the answer.  The mixing fraction of a thin film on
        glass is not well known, so a single corrected number overstates what
        was measured; the width of this envelope is the uncertainty the ratio
        actually carries, and for two heavy elements it is narrow.
        """
        values = []
        for fraction in fractions:
            try:
                values.append(self.ratio(numerator, denominator, blank, fraction))
            except ValueError:
                continue
        if not values:
            raise ValueError("no slide share in this range leaves a usable ratio")
        return min(values), max(values)


def _symbol(material: str) -> str:
    """``Na(Sodium(Natrium))`` -> ``Na``; an oxide keeps its formula."""
    return material.split("(")[0].strip()


def read_xrf(path: str | Path) -> dict[str, Measurement]:
    """Read a fluorescence export: one row per material per measurement.

    The columns expected are ``Sample``, ``Analysis date``, ``Material``,
    ``Measurement result``, ``Unit`` and ``Sigma``, which is what the
    instrument writes.  A result of ``ND`` is left out rather than stored as
    zero.

    Measurements are keyed by the name in the file.  An instrument run twice
    over the same mount, once for oxides and once for elements, produces two
    entries, and which is which is taken from the name where it says so and
    from the materials otherwise.
    """
    path = Path(path)
    out: dict[str, Measurement] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Sample") or "").strip()
            material = _symbol(row.get("Material") or "")
            if not name or not material:
                continue
            measurement = out.setdefault(name, Measurement(name=name))
            measurement.date = measurement.date or (row.get("Analysis date") or "").strip()
            result = (row.get("Measurement result") or "").strip()
            if result and result.upper() != "ND":
                measurement.values[material] = float(result)
                try:
                    measurement.sigmas[material] = float(row.get("Sigma") or 0.0)
                except ValueError:
                    pass
    for measurement in out.values():
        oxide = any("O" in material and len(material) > 1 for material in measurement.values)
        measurement.mode = "oxides" if oxide else "elements"
    return out


def illite_octahedral_iron(
    iron_over_potassium: float,
    potassium: float,
    octahedral_sites: float = 2.0,
    potassium_sites: float = 1.0,
    iron_mass: float = 55.845,
    potassium_mass: float = 39.098,
) -> float:
    """Octahedral iron fraction implied by a measured Fe/K mass ratio.

    Per ``O10(OH)2`` an illite has two octahedral cations and at most one
    interlayer potassium, so with a fraction ``f`` of the octahedra iron and an
    occupancy ``k`` of the interlayer,

        Fe/K by mass = (2 f M_Fe) / (k M_K)

    and this inverts it for ``f`` at a given ``k``.  The ratio alone fixes only
    ``f/k``, so the potassium has to come from elsewhere - from what the mineral
    is, in the case this was written for, since an illite is defined by carrying
    0.75 to 0.9 of an atom and a phengite is identifiable in hand specimen.

    Two things make this an *upper* bound rather than a measurement.  Any iron
    in an accessory oxide is counted as octahedral here, and any potassium in a
    feldspar or a mica is counted as interlayer.  Where the same analysis shows
    titanium well above the slide's, accessory oxides are present and the bound
    is loose by however much iron went with them.
    """
    if iron_over_potassium <= 0.0:
        raise ValueError("a mass ratio must be positive")
    if not 0.0 < potassium <= 1.0:
        raise ValueError(f"a potassium occupancy must lie in (0, 1], not {potassium}")
    per_site = (octahedral_sites * iron_mass) / (potassium_sites * potassium_mass)
    return iron_over_potassium * potassium / per_site
