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
from clayquant.models import FIBROUS_CIF_SOURCES, habit_axis, load_crystal
from clayquant.pattern import reflections
from clayquant.quantification import CLAY_LIBRARY_PHASES
from clayquant.treatment import (
    CHANNEL_CLAY_110,
    CHANNEL_CLAY_MAXIMUM_EXPANSION,
    ILLITE_001,
    PeakShift,
    ShiftEvidence,
    channel_clay_entries,
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
        assert habit_axis(key) == ((0.0, 0.0, 1.0), "direct")


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


# --------------------------------------------------------------------------- #
# a channel breathes: the criterion is physical, not statistical
# --------------------------------------------------------------------------- #

def test_the_sepiolite_standards_own_110_is_accepted():
    """The case the statistical criterion refused.

    12.095 to 12.320 A, +1.86 per cent, as the tallest line in the pattern.  It
    was read as having moved because 0.147 deg is nine times its 0.017 deg
    placement uncertainty - which is true, and is not the question.  A channel
    holds zeolitic water and the cell breathes with it; what a channel clay
    cannot do is take a glycol layer.
    """
    found = _evidence([_shift(12.095, 12.320, 14075.0, uncertainty=0.017)])
    assert found.found
    assert found.minerals == ["sepiolite"]
    assert "1.9 per cent" in found.status


def test_a_well_placed_line_is_not_punished_for_being_well_placed():
    """The same expansion, refused or allowed by how sharp the line is.

    Under the old criterion it was: the threshold moved with the uncertainty, so
    the better the measurement the less breathing it permitted.
    """
    for uncertainty in (0.002, 0.005, 0.017, 0.05):
        assert _evidence([_shift(12.10, 12.32, 9000.0, uncertainty)]).found


def test_glycol_uptake_is_still_refused_at_every_level():
    """One glycol layer takes 12.4 A to about 14.2, two to 16.9."""
    assert not _evidence([_shift(12.40, 14.20, 9000.0)]).found
    assert not _evidence([_shift(12.40, 16.90, 9000.0)]).found


def test_the_threshold_sits_between_the_two_populations():
    assert CHANNEL_CLAY_MAXIMUM_EXPANSION == pytest.approx(0.05)
    # this specimen's breathing, and the whole hydration range of the 110
    assert (12.320 - 12.095) / 12.095 < CHANNEL_CLAY_MAXIMUM_EXPANSION
    assert (12.52 - 11.93) / 11.93 < CHANNEL_CLAY_MAXIMUM_EXPANSION
    # and the smallest glycol expansion there is
    assert (14.2 - 12.4) / 12.4 > CHANNEL_CLAY_MAXIMUM_EXPANSION


def test_a_negative_tolerance_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        channel_clay_evidence(
            ShiftEvidence([_shift(12.1, 12.1, 9000.0)], 0.0, 0.0, 0.02, False, 0.0, [], ""),
            maximum_expansion=-0.1)


# --------------------------------------------------------------------------- #
# and the evidence decides whether the phase is in the library at all
# --------------------------------------------------------------------------- #

class _Entry:
    def __init__(self, phase):
        self.phase = phase


class _Library:
    def __init__(self, phases):
        self.entries = [_Entry(p) for p in phases]


PHASES = ("illite", "sepiolite", "palygorskite", "smectite_EG", "sepiolite")


def test_a_channel_clay_the_mounts_support_is_kept():
    evidence = _evidence([_shift(12.095, 12.320, 14075.0, 0.017)])
    kept = channel_clay_entries(_Library(PHASES), evidence)
    assert [PHASES[i] for i in kept] == ["illite", "sepiolite", "smectite_EG", "sepiolite"]


def test_a_channel_clay_with_no_evidence_is_dropped():
    """Palygorskite took 24 per cent of the Illite_10 standard without this.

    Nothing at all falls in either channel-clay window on eight of the nine
    standards, so the test is sharp rather than marginal.
    """
    evidence = _evidence([_shift(ILLITE_001, ILLITE_001, 9000.0)])
    assert not evidence.found
    kept = channel_clay_entries(_Library(PHASES), evidence)
    assert [PHASES[i] for i in kept] == ["illite", "smectite_EG"]


def test_a_test_that_could_not_be_made_is_not_a_negative_result():
    """No air-dried mount, so nothing is dropped."""
    kept = channel_clay_entries(_Library(PHASES), None)
    assert kept == list(range(len(PHASES)))


def test_only_the_channel_clays_are_ever_dropped():
    evidence = _evidence([_shift(ILLITE_001, ILLITE_001, 9000.0)])
    library = _Library(("illite", "chlorite", "kaolinite_1M", "I/S", "C/S",
                        "corrensite", "smectite_EG", "Quartz"))
    assert channel_clay_entries(library, evidence) == list(range(8))
