"""The library must be able to reproduce the standards it is checked against.

A reference library that cannot calculate the pure chlorite it was calibrated on
is not describing that chlorite, and every fit that uses it pays: on a real
separate the chlorite's unreachable 003 was taken up by illite/smectite and
chlorite/smectite entries, reporting an expandable component that is not there.

The axis spanned the 2:1 sheet alone first, which was worse than the published
structure everywhere, then the hydroxide sheet alone, which fitted the 002 and
the 004 and left the 003 unreachable at any value.  Neither tried the two
together.  These tests hold the two-sheet span in place and state what it is
measured to achieve, so a future narrowing has to argue with the numbers.
"""

import numpy as np
import pytest

from clayquant.emission import CU_KA_5LINE
from clayquant.library import CHLORITE_IRON, DEFAULT_PEAK_SHAPE, scaled_to_d001
from clayquant.models import CIF_SOURCES, chlorite_crystal
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, powder_pattern

WAVELENGTH = 1.540598
GRID = np.arange(4.0, 36.0, 0.02)
D001 = 14.15

# Measured on the glycol mounts, background removed and the angular offset
# solved out, as areas against the 002.  The mounts are the 4.91 cm2 discs the
# standards table records - 25 mm - on a 240 mm goniometer with a 0.5 deg slit,
# and the geometry matters here: beam overflow at the 001 moves that ratio by a
# third between a 25 and a 35 mm mount, so these numbers only mean anything
# against this geometry.
MEASURED = {
    "Chlorite 16":    (0.503, 1.000, 0.423, 0.617, 0.125),
    "Prochlorite 15": (0.425, 1.000, 0.594, 0.750, 0.169),
}


def an_instrument() -> Instrument:
    return Instrument(
        emission=CU_KA_5LINE, peak_shape=DEFAULT_PEAK_SHAPE, lp_mode="powder",
        divergence=Divergence(specimen_length=25.0, goniometer_radius=240.0,
                              divergence=0.5, shape="round"),
    )


def basal_ratios(two_one: float, hydroxide: float) -> np.ndarray:
    crystal = scaled_to_d001(chlorite_crystal(two_one, hydroxide),
                             CIF_SOURCES["chlorite"].layers_per_cell, D001)
    intensity = np.asarray(
        powder_pattern(crystal, GRID, an_instrument(),
                       r_march_dollase=0.3, name="chlorite").intensity)
    areas = []
    for order in range(1, 6):
        centre = 2.0 * np.degrees(np.arcsin(order * WAVELENGTH / (2.0 * D001)))
        half = 0.35 if order <= 2 else 0.45
        inside = (GRID >= centre - half) & (GRID <= centre + half)
        areas.append(float(np.trapezoid(np.clip(intensity[inside], 0.0, None),
                                        GRID[inside])))
    return np.asarray(areas) / areas[1]


def worst_error(calculated: np.ndarray, measured) -> float:
    """Largest relative error over the orders that are not the reference."""
    measured = np.asarray(measured)
    others = [0, 2, 3, 4]
    return float(np.max(np.abs(calculated[others] - measured[others]) / measured[others]))


@pytest.fixture(scope="module")
def reachable():
    return {pair: basal_ratios(*pair) for pair in CHLORITE_IRON}


def test_the_axis_spans_both_octahedral_sheets():
    """One sheet cannot do it: the two pull the odd and even orders in opposite
    directions, so a single axis trades one order against another."""
    assert len({two_one for two_one, _ in CHLORITE_IRON}) >= 3
    assert len({hydroxide for _, hydroxide in CHLORITE_IRON}) >= 3


@pytest.mark.parametrize("standard", sorted(MEASURED))
def test_each_chlorite_standard_is_within_reach(standard, reachable):
    measured = MEASURED[standard]
    best = min(worst_error(r, measured) for r in reachable.values())
    assert best <= 0.25, (
        f"no entry comes within 25 % of {standard} on every basal order; the "
        f"closest is {100 * best:.0f} % off, and a library that cannot calculate "
        f"its own standard will hand the difference to another phase"
    )


def test_the_003_of_both_standards_is_reachable(reachable):
    """The order that forced this change, and the one the kaolinite diagnostic
    rests on: it is the only chlorite reflection kaolinite does not share."""
    span = [r[2] for r in reachable.values()]
    for standard, measured in MEASURED.items():
        assert min(span) <= measured[2] <= max(span), (
            f"{standard}'s 003 at {measured[2]:.3f} of its 002 is outside the "
            f"{min(span):.3f}-{max(span):.3f} the axis reaches, so its collapse "
            f"on heating cannot be measured against a calculated chlorite"
        )


def test_a_hydroxide_only_axis_would_not_pass(reachable):
    """The regression guard proper: the axis this replaced, re-measured.

    Keeping it here means a future simplification back to one sheet fails with
    the reason attached rather than quietly losing the 003.
    """
    hydroxide_only = [basal_ratios(0.0, b)
                      for b in (0.0, 0.015, 0.035, 0.08, 0.15)]
    span = [r[2] for r in hydroxide_only]
    assert MEASURED["Chlorite 16"][2] < min(span), (
        "the hydroxide-only axis is supposed to be unable to reach this 003 - "
        "if it now can, the structure model changed and these numbers need redoing"
    )
    best = min(worst_error(r, MEASURED["Chlorite 16"]) for r in hydroxide_only)
    assert best > 0.4
