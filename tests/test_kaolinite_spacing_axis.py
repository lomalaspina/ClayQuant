"""The kaolinite basal axis, and why it stops where it does.

Every other clay in the library spans its basal spacing, on the reasoning that a
refined structure is one specimen's and not the mineral's.  The kaolinites were
left out of that, which left them no freedom at all: a pure dickite standard
fitted at Rwp 94.5 per cent with its calculated 001 at a tenth the measured
height, and a chlorite/smectite at 14.2 A took the misfit.

The floor of the axis is the interesting part.  7.10 A was tried and withdrawn:
a kaolinite there is very nearly a chlorite 002, and non-negative least squares
used it as one - a pure prochlorite standard came out 50 per cent kaolinite.
Dropping the value cost 9 points of Rwp and halved the error, which is the whole
argument for choosing a library by what it reports rather than by its residual.
"""

import pytest

from clayquant.library import HOST_THICKNESSES

KAOLINITES = ("kaolinite_1M", "kaolinite_2M")


def test_the_kaolinites_have_a_basal_axis_at_all():
    for key in KAOLINITES:
        assert key in HOST_THICKNESSES, (
            f"{key} has no spacing axis, so a kaolin whose 001 differs from the "
            f"published structure cannot be fitted at all"
        )
        assert len(HOST_THICKNESSES[key]) >= 2


def test_it_does_not_reach_down_into_the_chlorite_002():
    """7.10 A is where a kaolinite stops being distinguishable from a chlorite.

    A chlorite spanning 14.05 to 14.35 A puts its 002 between 7.025 and 7.175,
    so the overlap cannot be removed entirely - that is the minerals, not the
    table - but the bottom of the chlorite's range must not be reachable.
    """
    chlorite_002 = [value / 2.0 for value in HOST_THICKNESSES["chlorite"]]
    for key in KAOLINITES:
        assert min(HOST_THICKNESSES[key]) > min(chlorite_002) + 0.1, (
            "the kaolinite axis reaches a spacing only a chlorite has, and the "
            "fit will use it as one: with 7.10 A present a pure prochlorite "
            "standard fitted as 50 per cent kaolinite"
        )


def test_it_covers_the_kaolin_group_as_the_standards_measure_it():
    """Kaolinite 12, Kaolinite 43 and Dickite 7 sit at 7.14, 7.14 and 7.18 A
    once their angular offset is solved out; the axis has to bracket that."""
    for key in KAOLINITES:
        values = HOST_THICKNESSES[key]
        assert min(values) <= 7.14
        assert max(values) >= 7.18


def test_the_step_is_fine_enough_to_tell_the_standards_apart():
    """The three span 0.04 A between them.  A step of 0.05 is one step wide and
    cannot separate them at all, which is why this is finer than the other
    phases' proportional step."""
    for key in KAOLINITES:
        values = sorted(HOST_THICKNESSES[key])
        steps = [b - a for a, b in zip(values, values[1:])]
        assert max(steps) <= 0.035


def test_the_other_phases_keep_their_own_ranges():
    """The kaolinite entries are additions, not a rewrite of the table."""
    assert HOST_THICKNESSES["illite"] == (9.90, 9.95, 10.02, 10.10)
    assert HOST_THICKNESSES["chlorite"] == (14.05, 14.15, 14.25, 14.35)
