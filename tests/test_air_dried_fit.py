"""Fitting the glycol mount and the air-dried mount with one set of coefficients.

This is the measurement that bears on whether an expandable clay is there, and
these tests pin down both what it does and - just as much - what it does not do.
"""
import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import nnls_fit
from clayquant.pattern import Pattern
from clayquant.treatment import AIR_OBSERVATION_WEIGHT, air_dried_observation


GRID = np.arange(4.0, 34.0, 0.02)


def peak(center: float, height: float = 1.0, width: float = 0.12) -> np.ndarray:
    return height * np.exp(-0.5 * ((GRID - center) / width) ** 2)


def a_library() -> PatternLibrary:
    """An illite, an expandable phase that moves on glycolation, and a quartz.

    The expandable entry puts its 001 at 8.8 deg in the glycol mount and at
    10.6 deg in the air-dried one - the collapse - while the illite and the
    quartz sit still.
    """
    return PatternLibrary(
        two_theta=GRID,
        entries=[
            LibraryEntry(name="illite", phase="illite",
                         intensity=peak(8.84) + 0.5 * peak(26.6),
                         unit_mass=100.0, unit_volume=100.0),
            LibraryEntry(name="I/S", phase="I/S", fraction=0.5,
                         intensity=peak(8.84) + 0.5 * peak(26.6),
                         air_intensity=peak(10.6) + 0.5 * peak(28.1),
                         unit_mass=100.0, unit_volume=100.0),
            LibraryEntry(name="Quartz", phase="Quartz",
                         intensity=peak(20.86, 0.4) + peak(26.64),
                         unit_mass=60.0, unit_volume=113.0),
        ],
    )


def mounts(library, amounts, collapse=True):
    """A glycol and an air-dried mount from a known mixture."""
    glycol = np.zeros_like(GRID)
    air = np.zeros_like(GRID)
    for entry, amount in zip(library.entries, amounts):
        glycol += amount * entry.intensity
        counterpart = entry.intensity if (entry.air_intensity is None or not collapse) \
            else entry.air_intensity
        air += amount * counterpart
    return (Pattern(two_theta=GRID, intensity=glycol, name="eg"),
            Pattern(two_theta=GRID, intensity=air, name="air"))


def coefficients(library, glycol, constraints=()):
    return dict(zip(*(lambda r: (r.names, r.coefficients))(
        nnls_fit(glycol, library, constraints=list(constraints)))))


def test_a_library_without_counterparts_is_refused():
    """Falling back to the glycol patterns would assert what is being asked."""
    library = a_library()
    for entry in library.entries:
        entry.air_intensity = None
    glycol, air = mounts(library, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="no air-dried patterns"):
        air_dried_observation(air, glycol, library)


def test_the_two_mounts_agreeing_leaves_the_answer_alone():
    """A specimen with nothing expandable in it, measured twice."""
    library = a_library()
    glycol, air = mounts(library, [1.0, 0.0, 0.6])
    observation = air_dried_observation(air, glycol, library, scale=1.0)
    assert observation.constraint is not None
    before = coefficients(library, glycol)
    after = coefficients(library, glycol, (observation.constraint,))
    assert after["illite"] == pytest.approx(before["illite"], rel=1e-6)
    assert after["I/S"] == pytest.approx(0.0, abs=1e-9)


def test_an_expandable_phase_that_did_not_collapse_is_driven_out():
    """The whole point.  The glycol mount alone cannot tell the two apart -
    they are the same pattern - so it is free to use either; the air-dried
    mount, which shows no collapse, says it was the illite.
    """
    library = a_library()
    glycol, air = mounts(library, [1.0, 0.0, 0.6])
    # The glycol mount really cannot distinguish them.
    assert np.allclose(library.entries[0].intensity, library.entries[1].intensity)

    free = nnls_fit(glycol, library)
    observation = air_dried_observation(air, glycol, library, scale=1.0)
    restrained = nnls_fit(glycol, library, constraints=[observation.constraint])
    got = dict(zip(restrained.names, restrained.coefficients))
    assert got["I/S"] == pytest.approx(0.0, abs=1e-9)
    assert got["illite"] == pytest.approx(1.0, rel=1e-6)
    assert free.r_wp <= restrained.r_wp + 1e-9  # it can only cost on this mount


def test_a_real_expandable_phase_is_left_alone():
    """The other side of it: a specimen that does collapse must keep its I/S."""
    library = a_library()
    glycol, air = mounts(library, [0.0, 1.0, 0.6])
    observation = air_dried_observation(air, glycol, library, scale=1.0)
    got = coefficients(library, glycol, (observation.constraint,))
    assert got["I/S"] == pytest.approx(1.0, rel=1e-6)
    assert got["illite"] == pytest.approx(0.0, abs=1e-9)


def test_zero_weight_is_the_same_as_not_using_it():
    library = a_library()
    glycol, air = mounts(library, [0.4, 0.6, 0.6])
    observation = air_dried_observation(air, glycol, library, scale=1.0, weight=0.0)
    assert coefficients(library, glycol, (observation.constraint,)) == pytest.approx(
        coefficients(library, glycol)
    )


def test_the_shift_is_not_measured_unless_it_is_asked_for():
    """A quartz-free specimen will hand back a zero shift of several hundredths
    of a degree if asked, and a misaligned block wrecks the fit rather than
    restraining it, so the default trusts the caller's own zero correction.
    """
    library = a_library()
    glycol, air = mounts(library, [1.0, 0.0, 0.0])
    assert air_dried_observation(air, glycol, library, scale=1.0).shift == 0.0


def test_the_air_mount_is_scaled_not_the_design_weighted_by_it():
    """One coefficient must describe both mounts whatever the second mount's
    overall intensity, so a mount recorded at half the counts must give the
    same answer - and must not thereby count for half as much.
    """
    library = a_library()
    glycol, air = mounts(library, [1.0, 0.0, 0.6])
    faint = Pattern(two_theta=GRID, intensity=0.25 * air.intensity, name="air")
    full = air_dried_observation(air, glycol, library, scale=1.0)
    quarter = air_dried_observation(faint, glycol, library, scale=4.0)
    assert coefficients(library, glycol, (quarter.constraint,)) == pytest.approx(
        coefficients(library, glycol, (full.constraint,)), rel=1e-6
    )


def test_a_mount_that_does_not_cover_the_range_is_not_used():
    library = a_library()
    glycol, air = mounts(library, [1.0, 0.0, 0.6])
    short = Pattern(two_theta=GRID[:200], intensity=air.intensity[:200], name="air")
    observation = air_dried_observation(short, glycol, library, scale=1.0)
    assert observation.constraint is None
    assert "covers only" in observation.status


def test_the_default_weight_is_one_because_it_is_an_equal_measurement():
    assert AIR_OBSERVATION_WEIGHT == 1.0
