"""Sepiolite and palygorskite, and the test that finds them.

A channel clay puts a reflection where an air-dried smectite puts its 001, and
for most of this program's history the library had no column that could carry
it - so a 12.4 A line that never moved was fitted as expandable clay, which is
the one thing it cannot be.  These tests pin both halves of the fix: that the
minerals are there, and that the measurement which distinguishes them is read.
"""
import math

import numpy as np
import pytest

from clayquant.library import FIBROUS_SPACINGS, scaled_in_plane
from clayquant.models import FIBROUS_CIF_SOURCES, habit_pole, load_crystal
from clayquant.pattern import reflections
from clayquant.quantification import CLAY_LIBRARY_PHASES
from clayquant.treatment import (
    CHANNEL_CLAY_110,
    ILLITE_001,
    PeakShift,
    ShiftEvidence,
    channel_clay_evidence,
)

WAVELENGTH = 1.540596


def _d110(crystal):
    """From the crystal's own metric: palygorskite is monoclinic, beta ~ 107 deg."""
    return float(np.atleast_1d(crystal.d_spacing(np.array([[1.0, 1.0, 0.0]])))[0])


def test_both_channel_clays_load_and_know_their_fibre_axis():
    for key in ("sepiolite", "palygorskite"):
        assert key in FIBROUS_CIF_SOURCES
        crystal = load_crystal(key)
        assert len(crystal.sites) > 0
        assert habit_pole(key) == (1.0, 1.0, 0.0)


def test_the_110_is_the_strongest_long_spacing_reflection():
    """Which is what they are recognised by, and what the spacing axis spans.

    Not the *longest*: sepiolite's 020 at b/2 is 13.5 A and palygorskite's 010
    is 17.9 A.  Both are far weaker, and the 010 is extinct.
    """
    for key, expected in (("sepiolite", 11.93), ("palygorskite", 10.37)):
        crystal = load_crystal(key)
        assert _d110(crystal) == pytest.approx(expected, abs=0.02)
        found = reflections(crystal, 2.5)
        long_lines = np.where(found.d > 8.0)[0]
        strongest = long_lines[int(np.argmax(found.f_squared[long_lines]))]
        assert float(found.d[strongest]) == pytest.approx(_d110(crystal), abs=0.02)
        assert {abs(int(v)) for v in found.hkl[strongest]} == {1, 1, 0}


def test_the_spacing_axis_moves_the_110_and_leaves_the_fibre_axis_alone():
    base = load_crystal("sepiolite")
    for spacing in FIBROUS_SPACINGS["sepiolite"]:
        scaled = scaled_in_plane(base, spacing)
        assert _d110(scaled) == pytest.approx(spacing, abs=1e-6)
        assert scaled.c == pytest.approx(base.c)


def test_the_published_spacing_is_the_first_point_on_the_axis():
    """A spanned axis must contain the structure it was spanned from."""
    assert FIBROUS_SPACINGS["sepiolite"][0] == pytest.approx(
        _d110(load_crystal("sepiolite")), abs=0.01)
    assert FIBROUS_SPACINGS["palygorskite"][0] == pytest.approx(
        _d110(load_crystal("palygorskite")), abs=0.01)


def test_the_axis_reaches_what_the_specimens_measure():
    """11.93 A alone would put the line a third of a degree from the data."""
    assert max(FIBROUS_SPACINGS["sepiolite"]) >= 12.5


def test_they_count_as_clay_minerals():
    assert {"sepiolite", "palygorskite"} <= CLAY_LIBRARY_PHASES


def _shift(d_air, d_glycol, height, uncertainty=0.02):
    def angle(d):
        return 2.0 * math.degrees(math.asin(WAVELENGTH / (2.0 * d)))
    return PeakShift(
        glycol_two_theta=angle(d_glycol), air_two_theta=angle(d_air),
        displacement=angle(d_glycol) - angle(d_air), height=height,
        uncertainty=uncertainty, d_glycol=d_glycol, d_air=d_air,
    )


def _evidence(shifts):
    return channel_clay_evidence(
        ShiftEvidence(shifts, 0.0, 0.0, 0.02, False, 0.0, [], ""))


def test_a_held_twelve_angstrom_line_is_a_channel_clay():
    found = _evidence([_shift(12.45, 12.44, 2000.0), _shift(10.01, 10.02, 9000.0)])
    assert found.found
    assert found.minerals == ["sepiolite"]


def test_a_smectite_that_glycolates_is_not_one():
    """The same window, the opposite answer, because this one moved."""
    found = _evidence([_shift(12.40, 16.90, 2000.0)])
    assert not found.found


def test_the_illite_001_is_not_mistaken_for_palygorskite():
    """It is held just as firmly and sits only 0.2 A below the window."""
    found = _evidence([_shift(ILLITE_001, ILLITE_001, 9000.0)])
    assert not found.found
    assert ILLITE_001 < CHANNEL_CLAY_110["palygorskite"][0]


def test_a_palygorskite_near_the_illite_is_reported_with_that_said():
    found = _evidence([_shift(10.22, 10.22, 5000.0)])
    assert found.found and found.minerals == ["palygorskite"]
    assert found.findings[0].near_illite
    assert "illite 001 is only" in found.status


def test_a_held_line_in_the_noise_is_not_a_mineral():
    found = _evidence([_shift(10.01, 10.02, 9000.0), _shift(12.45, 12.44, 20.0)])
    assert not found.found


def test_chlorite_and_kaolinite_do_not_trip_it():
    """They hold position too, and fall outside both windows."""
    found = _evidence([_shift(14.28, 14.28, 3000.0), _shift(7.13, 7.13, 4000.0)])
    assert not found.found


def test_no_matched_reflections_says_so_rather_than_claiming_nothing_held():
    found = _evidence([])
    assert not found.found
    assert "nothing could be held" in found.status
