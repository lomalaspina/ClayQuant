"""A needle is not a plate, and the March model treats them as opposite cases.

The defect these tests pin was not a wrong number but an unidentifiable one.
Spanning ``r`` downwards about the 110 pole of a sepiolite collapses its pattern
to that single line; the shape then stops depending on ``r`` while the mass goes
on scaling as ``r^3``, so the weight is free over three orders of magnitude and
the fit takes it.  On the sepiolite standard it reported a phase carrying 46 per
cent of the calculated intensity at 0.3 per cent by weight.
"""
import math

import numpy as np
import pytest

from clayquant.library import FIBROUS_ORIENTATION, FIBROUS_SPACINGS, scaled_in_plane
from clayquant.models import FIBRE_AXES, MINERAL_HABIT, habit_axis, habit_pole, load_crystal
from clayquant.optics import march_dollase
from clayquant.pattern import Instrument, powder_pattern, reflections
from clayquant.profile import PeakShape


def _instrument():
    return Instrument(peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0))


def test_the_chain_clays_are_needles_not_plates():
    for key in ("sepiolite", "palygorskite"):
        axis, space = habit_axis(key)
        assert space == "direct"
        assert axis == (0.0, 0.0, 1.0)
        assert key not in MINERAL_HABIT, "the old 110 plate pole must be gone"


def test_the_amphiboles_keep_their_plate_pole():
    """That one was measured on a real separate; only the chain clays change."""
    axis, space = habit_axis("hornblende")
    assert space == "reciprocal"
    assert axis == (1.0, 1.0, 0.0)


def test_a_needle_axis_puts_hk0_at_ninety_degrees_exactly():
    """g . c = l by duality, so l = 0 is the set perpendicular to the fibre.

    This is why the axis has to be given in direct space.  For the monoclinic
    palygorskite the 110 is 90 degrees from ``c`` and only 76 from ``c*``.
    """
    for key in ("sepiolite", "palygorskite"):
        crystal = load_crystal(key)
        hk0 = np.array([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, 0.0]])
        direct = np.degrees(crystal.angle_to_direction(hk0, (0.0, 0.0, 1.0)))
        assert np.allclose(direct, 90.0, atol=1e-6), key


def test_the_reciprocal_axis_would_get_the_monoclinic_one_wrong():
    crystal = load_crystal("palygorskite")
    hkl = np.array([[1.0, 1.0, 0.0]])
    reciprocal = float(np.degrees(crystal.angle_to(hkl, (0.0, 0.0, 1.0)))[0])
    direct = float(np.degrees(crystal.angle_to_direction(hkl, (0.0, 0.0, 1.0)))[0])
    assert direct == pytest.approx(90.0, abs=1e-6)
    assert abs(reciprocal - 90.0) > 10.0


def test_expansion_enhances_the_fibre_normal_set():
    """r > 1 is the needle case: maximum at 90 degrees, suppression at 0."""
    along = float(march_dollase(np.array([0.0]), 2.0)[0])
    across = float(march_dollase(np.array([math.pi / 2.0]), 2.0)[0])
    assert across > 1.0 > along
    # and r < 1 is the plate case, the other way round
    assert float(march_dollase(np.array([0.0]), 0.5)[0]) > 1.0


def test_the_orientation_is_one_value_and_it_is_the_needle_case():
    """Not an axis: see the next test for why there is nothing to span."""
    assert isinstance(FIBROUS_ORIENTATION, float)
    assert FIBROUS_ORIENTATION > 1.0


def test_the_orientation_of_a_channel_clay_is_not_measurable_here():
    """The whole reason it is fixed rather than spanned.

    Over the needle range the calculated pattern keeps its shape while the mass
    it implies scales as r^3.  An axis in r would therefore not be a parameter
    the fit measures but a free multiplier on the weight.
    """
    crystal = scaled_in_plane(load_crystal("sepiolite"), 12.35)
    grid = np.arange(4.0, 34.0, 0.02)

    def shape(r):
        y = np.asarray(powder_pattern(
            crystal, grid, _instrument(), r_march_dollase=r,
            po_axis=(0.0, 0.0, 1.0), po_axis_space="direct").intensity)
        return y / np.linalg.norm(y)

    reference = shape(FIBROUS_ORIENTATION)
    for r in (1.25, 2.0, 3.0, 5.0, 10.0):
        assert float(shape(r) @ reference) > 0.995, r


def test_the_plate_model_collapsed_the_pattern_to_one_line():
    """The symptom that gave the old defect away."""
    crystal = scaled_in_plane(load_crystal("sepiolite"), 12.35)
    grid = np.arange(4.0, 34.0, 0.02)
    y = np.asarray(powder_pattern(
        crystal, grid, _instrument(), r_march_dollase=0.1, po_axis=(1.0, 1.0, 0.0)).intensity)
    peak = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    assert int(np.count_nonzero(peak & (y[1:-1] >= 0.10 * y.max()))) == 1


def test_the_axis_space_reaches_the_pattern():
    crystal = load_crystal("palygorskite")
    grid = np.arange(4.0, 34.0, 0.05)
    reciprocal = np.asarray(powder_pattern(
        crystal, grid, _instrument(), r_march_dollase=2.0, po_axis=(0.0, 0.0, 1.0)).intensity)
    direct = np.asarray(powder_pattern(
        crystal, grid, _instrument(), r_march_dollase=2.0, po_axis=(0.0, 0.0, 1.0),
        po_axis_space="direct").intensity)
    assert not np.allclose(reciprocal, direct)


def test_the_axis_space_reaches_the_reflection_list():
    crystal = load_crystal("palygorskite")
    a = reflections(crystal, 3.0, po_axis=(0.0, 0.0, 1.0))
    b = reflections(crystal, 3.0, po_axis=(0.0, 0.0, 1.0), po_axis_space="direct")
    assert np.allclose(a.d, b.d) and np.allclose(a.f_squared, b.f_squared)
    assert not np.allclose(a.alpha, b.alpha)


def test_a_direction_of_zero_length_is_refused():
    with pytest.raises(ValueError, match="not a direction"):
        load_crystal("sepiolite").angle_to_direction(
            np.array([[1.0, 1.0, 0.0]]), (0.0, 0.0, 0.0))


def test_the_published_spacing_is_still_the_first_point():
    """Read from the structure, not written down beside it.

    This test held the literal 11.93 A of COD 9014723 and so had to be edited
    when the structure changed to 9010148 and its 12.01 A (Sec. A.70).  A test
    that has to be edited whenever the thing it checks changes is not checking
    anything: what matters is that the axis starts at the *loaded* structure's
    own spacing, whichever structure that is.
    """
    crystal = load_crystal("sepiolite")
    published = float(np.atleast_1d(
        crystal.d_spacing(np.array([[1.0, 1.0, 0.0]])))[0])
    assert FIBROUS_SPACINGS["sepiolite"][0] == pytest.approx(published, abs=0.02)
    assert FIBROUS_SPACINGS["sepiolite"] == tuple(sorted(FIBROUS_SPACINGS["sepiolite"]))
    assert habit_pole("sepiolite") is None, "no plate pole for a needle"
