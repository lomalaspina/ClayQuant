"""Reading the air-dried mount as evidence of movement, not as a pattern to fit."""
import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.pattern import Pattern
from clayquant.treatment import (
    LEAST_MOVING_HYDRATION,
    expandable_bound,
    shift_evidence,
)

GRID = np.arange(3.0, 34.0, 0.0167)
QUARTZ = (20.859, 26.640)


def peak(center, height=1.0, width=0.06):
    return height * np.exp(-0.5 * ((GRID - center) / width) ** 2)


def scan(lines, scale=1.0, background=300.0, seed=0):
    """A scan with counting noise, background-subtracted as the workflow does.

    The background is subtracted because that is what reaches this code: the
    mounts are compared after step 3.  Leaving it in would also hide a real
    weakness - a floor taken as a share of absolute intensity lets noise on a
    background through - which the code now guards against by prominence.
    """
    values = np.full(GRID.shape, background, dtype=float)
    for centre, height in lines:
        values += scale * height * peak(centre, 1.0).ravel()
    rng = np.random.default_rng(seed)
    counts = rng.poisson(values).astype(float)
    return Pattern(two_theta=GRID, intensity=counts - background, name="s")


CLAY = [(8.84, 6000.0), (12.45, 3000.0), (17.75, 1500.0)]
REF = [(20.859, 1200.0), (26.640, 7000.0)]


def test_two_mounts_of_an_unexpandable_specimen_show_no_movement():
    air = scan(CLAY + REF, seed=1)
    glycol = scan(CLAY + REF, scale=1.1, seed=2)
    evidence = shift_evidence(air, glycol, reference=QUARTZ)
    assert not evidence.moved
    assert evidence.largest < 3.0 * evidence.uncertainty
    assert "Nothing moved" in evidence.status


def test_a_difference_in_brightness_is_not_movement():
    """The two mounts are separate preparations and rarely carry the same
    amount of material; a peak is not missing because it is smaller."""
    air = scan(CLAY + REF, seed=3)
    glycol = scan(CLAY + REF, scale=2.5, seed=4)
    assert not shift_evidence(air, glycol, reference=QUARTZ).moved


def test_a_basal_reflection_that_moves_is_found():
    air = scan([(7.30, 4000.0)] + CLAY[1:] + REF, seed=5)
    glycol = scan([(7.05, 4000.0)] + CLAY[1:] + REF, seed=6)
    evidence = shift_evidence(air, glycol, reference=QUARTZ)
    assert evidence.moved
    assert evidence.largest > 0.15
    assert "expandable clay" in evidence.status


def test_a_common_zero_offset_is_not_movement():
    """Each mount has its own displacement; the quartz lines take it out."""
    air = scan(CLAY + REF, seed=7)
    shifted = Pattern(two_theta=GRID + 0.08, intensity=air.intensity.copy(), name="eg")
    evidence = shift_evidence(air, shifted, reference=QUARTZ)
    assert evidence.offset == pytest.approx(0.08, abs=0.02)
    assert not evidence.moved


def a_library():
    entries = [LibraryEntry(name="illite", phase="illite", intensity=peak(8.84),
                            fraction=None, unit_mass=100.0, unit_volume=100.0)]
    for expandable in (0.0, 0.02, 0.05, 0.10, 0.30, 0.50):
        entries.append(LibraryEntry(
            name=f"I/S {1 - expandable:.2f}", phase="I/S", fraction=1.0 - expandable,
            intensity=peak(8.84 - 3.0 * expandable),
            unit_mass=100.0, unit_volume=100.0))
    return PatternLibrary(two_theta=GRID, entries=entries)


def test_movement_leaves_the_library_alone():
    air = scan([(7.30, 4000.0)] + CLAY[1:] + REF, seed=8)
    glycol = scan([(7.05, 4000.0)] + CLAY[1:] + REF, seed=9)
    bound = expandable_bound(shift_evidence(air, glycol, reference=QUARTZ), a_library())
    assert bound.unrestricted
    assert bound.excluded == []
    assert bound.maximum == 1.0


def test_no_movement_puts_a_ceiling_on_the_expandable_content():
    air = scan(CLAY + REF, seed=10)
    glycol = scan(CLAY + REF, scale=1.1, seed=11)
    evidence = shift_evidence(air, glycol, reference=QUARTZ)
    bound = expandable_bound(evidence, a_library())
    assert not bound.unrestricted
    assert 0.0 < bound.maximum < 0.5
    library = a_library()
    for index in bound.excluded:
        entry = library.entries[index]
        assert 1.0 - float(entry.fraction) > bound.maximum
    for index in bound.allowed:
        entry = library.entries[index]
        assert entry.fraction is None or 1.0 - float(entry.fraction) <= bound.maximum + 1e-9
    # a discrete illite has no expandable layer and is never excluded
    assert library.entries[0].name == "illite" and 0 in bound.allowed


def test_the_ceiling_follows_the_measurement():
    """A noisier pair hides more, so it must allow more."""
    quiet = shift_evidence(scan(CLAY + REF, seed=12),
                           scan(CLAY + REF, scale=1.1, seed=13), reference=QUARTZ)
    loose = expandable_bound(quiet, a_library(), margin=8.0)
    tight = expandable_bound(quiet, a_library(), margin=1.0)
    assert loose.maximum > tight.maximum
    assert len(loose.excluded) <= len(tight.excluded)


def test_the_least_moving_hydration_state_is_the_conservative_one():
    """The bound must not depend on guessing which hydration state the
    specimen was in, so it uses the state that moves least."""
    from clayquant.treatment import AIR_DRIED_STATES

    others = [s for name, s in AIR_DRIED_STATES.items() if s <= 15.0]
    assert LEAST_MOVING_HYDRATION == max(others)
    evidence = shift_evidence(scan(CLAY + REF, seed=14),
                              scan(CLAY + REF, scale=1.1, seed=15), reference=QUARTZ)
    # a state further from the glycol complex would move more and so allow less
    assert (expandable_bound(evidence, a_library(), hydrated=12.4).maximum
            < expandable_bound(evidence, a_library(), hydrated=15.0).maximum)


def test_the_status_says_what_was_measured_and_what_was_excluded():
    evidence = shift_evidence(scan(CLAY + REF, seed=16),
                              scan(CLAY + REF, scale=1.1, seed=17), reference=QUARTZ)
    status = expandable_bound(evidence, a_library()).status
    assert "Nothing moved" in status
    assert "expandable" in status and "left out of the fit" in status
