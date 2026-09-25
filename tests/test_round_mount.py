"""A mount that is a disc, which is what an oriented separate is dried on.

The beam-overflow correction assumed a rectangular specimen.  On a disc the
strip's corners run off the edge before its middle does, so intensity is lost
sooner than the length alone says - and, more importantly, the length itself
has to be the mount's.  A 25 mm disc described as a 35 mm smear keeps the whole
beam where it really keeps nine tenths.
"""

import numpy as np
import pytest

from clayquant.optics import Divergence


def strip_length(two_theta, radius=240.0, divergence=0.5):
    return radius * np.radians(divergence) / np.sin(np.radians(two_theta) / 2.0)


def brute_force(two_theta, diameter, width, radius=240.0, divergence=0.5, n=1200):
    length = strip_length(two_theta, radius, divergence)
    x = np.linspace(-length / 2.0, length / 2.0, n)
    y = np.linspace(-width / 2.0, width / 2.0, n)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    return float(np.mean(gx**2 + gy**2 <= (diameter / 2.0) ** 2))


@pytest.fixture
def disc():
    return Divergence(specimen_length=25.0, goniometer_radius=240.0,
                      divergence=0.5, shape="round", beam_width=10.0)


def test_the_overlap_is_the_area_it_claims_to_be(disc):
    for two_theta in (4.0, 6.22, 8.84, 11.0):
        assert disc.factor(np.array([two_theta]))[0] == pytest.approx(
            brute_force(two_theta, 25.0, 10.0), abs=2e-3)


def test_a_disc_intercepts_less_than_a_rectangle_of_the_same_length(disc):
    rectangle = Divergence(specimen_length=25.0, goniometer_radius=240.0, divergence=0.5)
    angles = np.array([4.0, 6.22, 8.84])
    assert np.all(disc.factor(angles) < rectangle.factor(angles))
    # and not by much: this is a correction, not a different model
    assert np.all(disc.factor(angles) > 0.95 * rectangle.factor(angles))


def test_both_reach_full_illumination_and_stay_there(disc):
    high = np.array([15.0, 25.0, 40.0])
    assert np.allclose(disc.factor(high), 1.0)
    assert disc.full_illumination_two_theta < 15.0


def test_the_length_matters_far_more_than_the_shape():
    """The error being corrected: a 25 mm disc called a 35 mm smear."""
    disc = Divergence(specimen_length=25.0, goniometer_radius=240.0,
                      divergence=0.5, shape="round")
    wrong = Divergence(specimen_length=35.0, goniometer_radius=240.0, divergence=0.5)
    # the illite 001 and the chlorite 001
    assert disc.factor(np.array([8.84]))[0] == pytest.approx(0.895, abs=0.01)
    assert wrong.factor(np.array([8.84]))[0] == pytest.approx(1.0, abs=0.01)
    assert disc.factor(np.array([6.22]))[0] == pytest.approx(0.630, abs=0.01)
    assert wrong.factor(np.array([6.22]))[0] == pytest.approx(0.907, abs=0.01)


def test_the_mask_width_hardly_enters(disc):
    """What limits the strip at low angle is its length, not its width."""
    narrow = Divergence(specimen_length=25.0, goniometer_radius=240.0,
                        divergence=0.5, shape="round", beam_width=5.0)
    wide = Divergence(specimen_length=25.0, goniometer_radius=240.0,
                      divergence=0.5, shape="round", beam_width=15.0)
    angles = np.array([4.0, 8.84])
    assert np.all(np.abs(narrow.factor(angles) - wide.factor(angles)) < 0.06)


def test_a_shape_that_is_not_a_shape_is_refused():
    with pytest.raises(ValueError, match="rectangular"):
        Divergence(shape="oval")
    with pytest.raises(ValueError):
        Divergence(shape="round", beam_width=0.0)


def test_the_rectangular_model_is_untouched():
    """Everything calculated before this existed must still calculate the same."""
    before = Divergence(specimen_length=20.0, goniometer_radius=280.0, divergence=0.5)
    angles = np.array([5.0, 10.0, 20.0])
    expected = np.clip(20.0 * np.sin(np.radians(angles) / 2.0)
                       / (280.0 * np.radians(0.5)), 0.0, 1.0)
    assert np.allclose(before.factor(angles), expected)
    assert before.shape == "rectangular"
