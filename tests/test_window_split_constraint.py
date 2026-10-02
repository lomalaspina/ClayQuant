"""The 7.15 A window is a partition, so both sides of it are restrained.

Saying only what kaolinite may take leaves the chlorite free, and one peak
divided between two minerals is not divided at all unless both shares are named.
On the separate this was built for, kaolinite held to 16 per cent of the window
sat beside a chlorite family taking 83 per cent of the same window.

Everything here is in peak area.  The restraint is on how the window is split;
what that works out to as a weight per cent depends on orientation and structure
factors, which the fit is still free to settle.
"""

import numpy as np
import pytest

from clayquant.library import PatternLibrary
from clayquant.pattern import Pattern
from clayquant.treatment import kaolinite_share_constraint

GRID = np.arange(4.0, 20.0, 0.02)
WINDOW = (11.6, 13.2)


def peak(centre: float, height: float, width: float = 0.2) -> np.ndarray:
    return height * np.exp(-4.0 * np.log(2.0) * ((GRID - centre) / width) ** 2)


def a_library() -> PatternLibrary:
    lib = PatternLibrary(two_theta=GRID, entries=[], metadata={})
    for name, phase, centre in (("kaolinite_1M PO=1", "kaolinite_1M", 12.37),
                                ("chlorite PO=1", "chlorite", 12.47),
                                ("C/S 0.95/0.05 PO=1", "C/S", 12.45),
                                ("illite PO=1", "illite", 8.85)):
        lib.add(Pattern(two_theta=GRID, intensity=peak(centre, 1000.0), name=name),
                phase=phase, march_dollase=1.0, unit_mass=1.0, unit_volume=1.0)
    return lib


def a_measurement(height: float = 5000.0) -> Pattern:
    return Pattern(two_theta=GRID, intensity=peak(12.42, height), name="mount")


def test_it_restrains_both_halves_of_the_window():
    lib = a_library()
    block = kaolinite_share_constraint(a_measurement(), lib, (0.2, 0.3))
    assert block is not None
    assert block.design.shape[0] == 2, "one row names kaolinite, the other the chlorite"
    assert block.target.shape == (2,)
    # the two targets are complementary shares of one window
    assert block.target.sum() == pytest.approx(
        block.target[0] / 0.25, rel=1e-6), "the shares must add to the whole window"


def test_the_chlorite_row_includes_the_mixed_layer_chlorite():
    """A restraint naming only the discrete chlorite is satisfied by moving the
    intensity into C/S, which is how a fit comes to report an expandable
    component that is not there."""
    lib = a_library()
    block = kaolinite_share_constraint(a_measurement(), lib, (0.2, 0.3))
    phases = [e.phase for e in lib.entries]
    chlorite_row = block.design[1]
    assert chlorite_row[phases.index("chlorite")] > 0.0
    assert chlorite_row[phases.index("C/S")] > 0.0
    assert chlorite_row[phases.index("kaolinite_1M")] == 0.0
    assert chlorite_row[phases.index("illite")] == 0.0


def test_the_kaolinite_row_names_only_kaolinite():
    lib = a_library()
    block = kaolinite_share_constraint(a_measurement(), lib, (0.2, 0.3))
    phases = [e.phase for e in lib.entries]
    row = block.design[0]
    assert row[phases.index("kaolinite_1M")] > 0.0
    assert row[phases.index("chlorite")] == 0.0
    assert row[phases.index("C/S")] == 0.0


def test_the_target_is_an_area_and_scales_with_the_measurement():
    """It is a share of the measured window, so doubling the counts doubles it.
    A weight per cent would not move."""
    lib = a_library()
    small = kaolinite_share_constraint(a_measurement(5000.0), lib, (0.2, 0.3))
    large = kaolinite_share_constraint(a_measurement(10000.0), lib, (0.2, 0.3))
    assert large.target[0] == pytest.approx(2.0 * small.target[0], rel=1e-6)


def test_the_one_sided_restraint_is_still_available():
    lib = a_library()
    block = kaolinite_share_constraint(a_measurement(), lib, (0.2, 0.3),
                                       constrain_chlorite=False)
    assert block.design.shape[0] == 1


def test_subsetting_keeps_every_row():
    """select_one_orientation narrows the library and must narrow the block with
    it; a block whose columns no longer match the library is silently inert."""
    lib = a_library()
    block = kaolinite_share_constraint(a_measurement(), lib, (0.2, 0.3))
    narrowed = block.subset(np.array([0, 1, 2]))
    assert narrowed.design.shape == (2, 3)
    assert np.array_equal(narrowed.target, block.target)
