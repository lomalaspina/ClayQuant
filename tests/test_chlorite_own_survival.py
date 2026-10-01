"""How much of the chlorite 002 survives 550 C is measured, not assumed.

Kaolinite's share of the 7.15 A window is whatever the chlorite 002 under it did
not keep, so the share rests entirely on how much of that chlorite 002 was
expected to survive.  Assuming the two standards' absolute range - 32 to 72 % -
is right only for a chlorite that behaves like them, and chlorites vary far more
than that: on one metabasite separate whose chlorite kept barely a fifth of
itself, the standards' range put kaolinite at 55 to 80 % of the window when the
collapse was almost entirely the chlorite's own, and forcing the fit to honour it
cost ten points of Rwp.

The specimen's own 003 is the way out, and the only way out: it is a chlorite
reflection with no kaolinite under it, so measuring it in a kaolinite-bearing
specimen is not circular, which the 002 and the 004 both are.  The standards
then supply only the ratio between the two orders' survival, which is the stable
part of it.
"""

import numpy as np
import pytest

from clayquant.diagnostics import (
    CHLORITE_002_OVER_003_SURVIVAL,
    CHLORITE_002_SURVIVAL,
    chlorite_002_survival_from_003,
    kaolinite_evidence,
)
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 34.0, 0.0167)
BACKGROUND = 400.0
WAVELENGTH = 1.540596


def angle(d: float) -> float:
    return 2.0 * np.degrees(np.arcsin(WAVELENGTH / (2.0 * d)))


def peak(d: float, height: float, width: float = 0.14) -> np.ndarray:
    return height * np.exp(-4.0 * np.log(2.0) * ((GRID - angle(d)) / width) ** 2)


def mount(signal, seed: int = 20260930) -> Pattern:
    rng = np.random.default_rng(seed)
    return Pattern(two_theta=GRID,
                   intensity=rng.poisson(np.zeros_like(GRID) + signal + BACKGROUND).astype(float),
                   name="mount")


def quartz(strength: float = 3000.0) -> np.ndarray:
    """The reflection the two mounts are put on a common scale by."""
    return peak(4.257, strength, width=0.10)


def chlorite(strength: float, survival: float = 1.0) -> np.ndarray:
    """A chlorite basal series, every order above the 001 scaled by ``survival``.

    Heating enhances the 001 and takes the orders above it down together, which
    is what makes the 003 stand for the 002.
    """
    first = strength * (1.0 if survival >= 1.0 else 3.0)
    return (
        peak(14.20, first)
        + peak(7.10, strength * 2.0 * survival)
        + peak(4.733, strength * 1.05 * survival)
        + peak(3.550, strength * 1.4 * survival)
    )


def test_a_chlorite_that_collapses_hard_is_measured_as_such():
    air = mount(quartz() + chlorite(4000.0))
    heated = mount(quartz() + chlorite(4000.0, survival=0.20))
    least, most, note = chlorite_002_survival_from_003(air, heated)
    assert least < CHLORITE_002_SURVIVAL[0], note
    low, high = CHLORITE_002_OVER_003_SURVIVAL
    assert 0.20 * low == pytest.approx(least, abs=0.06), note
    assert 0.20 * high == pytest.approx(most, abs=0.06), note
    assert "003" in note


def test_a_chlorite_that_behaves_like_the_standards_lands_in_their_range():
    air = mount(quartz() + chlorite(4000.0))
    heated = mount(quartz() + chlorite(4000.0, survival=0.55))
    least, most, _ = chlorite_002_survival_from_003(air, heated)
    assert CHLORITE_002_SURVIVAL[0] <= least <= CHLORITE_002_SURVIVAL[1]
    assert CHLORITE_002_SURVIVAL[0] <= most <= CHLORITE_002_SURVIVAL[1]


def test_without_a_chlorite_003_the_standards_range_is_kept():
    """A pure kaolinite has no 003 to measure, and must not be given a made-up one."""
    kaolin = peak(7.16, 6000.0) + peak(3.58, 3000.0)
    air = mount(quartz() + kaolin)
    heated = mount(quartz())
    least, most, note = chlorite_002_survival_from_003(air, heated)
    assert (least, most) == CHLORITE_002_SURVIVAL
    assert "no chlorite 003" in note


def test_both_routes_are_reported_so_the_disagreement_is_visible():
    """The standards' route is kept beside the measured one, never replaced by it.

    The two can differ by a factor of three, and which is right turns on how a
    heated 003 at the detection limit is integrated.  Resolving that silently
    would hide the one thing the reader has to judge.
    """
    air = mount(quartz() + chlorite(4000.0))
    heated = mount(quartz() + chlorite(4000.0, survival=0.20))
    evidence = kaolinite_evidence(air, heated)
    assert evidence.collapse is not None
    assert evidence.measured_collapse is not None
    assert evidence.collapse.survival == CHLORITE_002_SURVIVAL
    assert evidence.measured_collapse.survival != CHLORITE_002_SURVIVAL
    assert evidence.survival_note
    # the chlorite accounts for more of the window on its own, so the measured
    # route asks less of kaolinite than the standards' route does
    assert evidence.measured_collapse.kaolinite_bounds[0] <= evidence.collapse.kaolinite_bounds[0]
