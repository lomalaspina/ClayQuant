"""Preferred orientation about a pole that is not c*.

Every layer silicate is a platelet flattened on 001, so c* is the pole for all
of them and the clay library assumes it throughout.  Not every clay mineral is a
layer silicate: sepiolite and palygorskite are chain silicates whose
crystallites are laths, and a lath settles on a side face.  Forcing one to be a
random powder, or orienting it about c*, misplaces every line it has.
"""

import numpy as np
import pytest

from clayquant.crystal import Crystal, read_cif
from clayquant.models import MINERAL_HABIT, load_crystal
from clayquant.pattern import powder_pattern, reflections, two_theta_grid


def test_the_default_pole_is_c_star():
    crystal = load_crystal("illite")
    hkl = np.array([[0, 0, 2], [1, 1, 0], [1, 1, 2]])
    assert np.allclose(crystal.angle_to(hkl, (0, 0, 1)), crystal.angle_to_cstar(hkl))


def test_a_reflection_lies_on_its_own_pole():
    crystal = load_crystal("illite")
    for pole in ((0, 0, 1), (1, 1, 0), (1, 0, 2)):
        assert crystal.angle_to(np.array([pole]), pole)[0] == pytest.approx(0.0, abs=1e-6)


def test_the_opposite_reflection_is_at_180_degrees():
    crystal = load_crystal("illite")
    angle = crystal.angle_to(np.array([[0, 0, -2]]), (0, 0, 1))[0]
    assert angle == pytest.approx(np.pi, abs=1e-9)


def test_a_direction_of_zero_length_is_refused():
    with pytest.raises(ValueError):
        load_crystal("illite").angle_to(np.array([[0, 0, 1]]), (0, 0, 0))


def test_the_reflection_list_carries_the_chosen_pole():
    crystal = load_crystal("illite")
    about_c = reflections(crystal, 3.0)
    about_110 = reflections(crystal, 3.0, po_axis=(1, 1, 0))
    assert np.array_equal(about_c.hkl, about_110.hkl)
    assert not np.allclose(about_c.alpha, about_110.alpha)


def test_the_pole_changes_which_reflection_is_enhanced():
    """The whole point: r < 1 lifts whatever lies on the pole."""
    crystal = load_crystal("illite")
    grid = two_theta_grid(4.0, 40.0, 0.02)

    def height(pattern, centre, half=0.3):
        x = np.asarray(pattern.two_theta)
        y = np.asarray(pattern.intensity)
        inside = (x >= centre - half) & (x <= centre + half)
        return float(np.max(y[inside])) / float(np.max(y))

    basal = 2 * np.degrees(np.arcsin(1.540598 / (2 * crystal.d001 / 2)))
    about_c = powder_pattern(crystal, grid, r_march_dollase=0.3, po_axis=(0, 0, 1))
    about_110 = powder_pattern(crystal, grid, r_march_dollase=0.3, po_axis=(1, 1, 0))
    # The 002 of the two-layer cell is the 10 A basal reflection, and it lies on
    # c*.  Orienting about anything else must cost it its dominance.
    assert height(about_c, basal) > height(about_110, basal)


def test_the_habit_table_names_the_fibrous_clays_and_nothing_else():
    assert MINERAL_HABIT["sepiolite"] == (1, 1, 0)
    assert MINERAL_HABIT["palygorskite"] == (1, 1, 0)
    assert "quartz" not in MINERAL_HABIT
    assert "illite" not in MINERAL_HABIT


def test_a_crystal_carries_no_habit_unless_given_one():
    assert load_crystal("illite").po_axis is None
    assert Crystal(a=5, b=5, c=5, alpha=90, beta=90, gamma=90, sites=[]).po_axis is None
