"""What the air-dried mount says about the expandable clays.

The quantitative fit is made on the glycolated mount, and on that one mount an
illite-rich interstratified pattern and an illite are close together.  What
tells them apart is the difference between the mounts, so the air-dried one
enters the fit as a restraint on the same coefficients rather than as a second
fit.  These tests pin Clayfit's windows and tests, the one place this departs
from Clayfit, and - because it is the thing that matters - the measured limit of
what the mechanism can do.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import nnls_fit
from clayquant.pattern import Pattern
from clayquant.treatment import (
    EXPANSION_CONSTRAINT_WEIGHT,
    INTERSTRATIFIED_SHIFT_WINDOW,
    MINIMUM_EXPANSION_GAIN,
    MINIMUM_PROFILE_AGREEMENT,
    SMECTITE_GLYCOL_001_WINDOW,
    ExtraObservation,
    expandable_entries,
    expansion_evidence,
)

GRID = np.arange(4.0, 34.0, 0.02)
QUARTZ_100 = 20.86


def peak(centre: float, height: float, width: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((GRID - centre) / (width / 2.355)) ** 2)


def entry(name, phase, intensity, fraction=None) -> LibraryEntry:
    top = float(np.max(intensity)) or 1.0
    return LibraryEntry(name=name, phase=phase, intensity=intensity / top,
                        normalization=top, fraction=fraction,
                        unit_mass=100.0, unit_volume=100.0)


def a_library() -> PatternLibrary:
    """Quartz, an illite, and a smectite whose 001 is at 5.2 deg."""
    return PatternLibrary(two_theta=GRID, entries=[
        entry("Quartz", "Quartz", peak(QUARTZ_100, 1.0, 0.10) + peak(26.64, 2.0, 0.10)),
        entry("illite", "illite", peak(8.84, 1.0, 0.25) + peak(17.7, 0.3, 0.25),
              fraction=1.0),
        entry("smectite_EG", "smectite_EG", peak(5.2, 1.0, 0.45) + peak(10.4, 0.2, 0.45),
              fraction=0.0),
    ])


def mounts(smectite_in_glycol: float, seed: int = 1):
    """A pair of mounts: quartz and illite in both, smectite only in glycol."""
    library = a_library()
    quartz, illite, smectite = (item.intensity for item in library.entries)
    rng = np.random.default_rng(seed)
    air_counts = 60.0 + 4000.0 * quartz + 3000.0 * illite
    glycol_counts = 60.0 + 4000.0 * quartz + 3000.0 * illite + \
        smectite_in_glycol * smectite
    air = Pattern(two_theta=GRID,
                  intensity=air_counts + rng.normal(0.0, np.sqrt(air_counts)),
                  name="air")
    glycol = Pattern(two_theta=GRID,
                     intensity=glycol_counts + rng.normal(0.0, np.sqrt(glycol_counts)),
                     name="glycol")
    return library, air, glycol


# --------------------------------------------------------------------------
# Clayfit's windows and tests
# --------------------------------------------------------------------------

def test_the_windows_are_clayfits():
    assert SMECTITE_GLYCOL_001_WINDOW == (4.75, 5.55)
    assert INTERSTRATIFIED_SHIFT_WINDOW == (9.10, 9.90)
    assert EXPANSION_CONSTRAINT_WEIGHT == pytest.approx(0.25)
    assert MINIMUM_EXPANSION_GAIN == pytest.approx(0.08)
    assert MINIMUM_PROFILE_AGREEMENT == pytest.approx(0.10)


def test_only_the_entries_with_expandable_layers_are_restrained():
    library = a_library()
    assert expandable_entries(library) == [2]      # the smectite alone


def test_real_expansion_is_detected_and_supported():
    library, air, glycol = mounts(smectite_in_glycol=3000.0)
    evidence = expansion_evidence(glycol, air, library)
    assert evidence.supported
    assert evidence.gain > MINIMUM_EXPANSION_GAIN
    assert evidence.agreement > MINIMUM_PROFILE_AGREEMENT
    assert evidence.peak_two_theta == pytest.approx(5.2, abs=0.15)
    assert evidence.constraint is not None
    assert "restrained to account for it" in evidence.status


def test_identical_mounts_are_evidence_against_expansion_not_an_absence_of_it():
    """The one departure from Clayfit, and the reason for it."""
    library, air, _ = mounts(smectite_in_glycol=0.0)
    evidence = expansion_evidence(glycol=air, air=air, library=library)
    assert not evidence.supported
    # Clayfit leaves the fit free here.  This does not: the target is the
    # measured expansion, which is zero, so the term says the expandable
    # component must be near zero.
    assert evidence.constraint is not None
    assert "evidence against expandable clay" in evidence.status
    np.testing.assert_allclose(evidence.constraint.target, 0.0, atol=1e-9)


def test_clayfits_own_behaviour_is_available():
    library, air, _ = mounts(smectite_in_glycol=0.0)
    evidence = expansion_evidence(glycol=air, air=air, library=library, when="supported")
    assert evidence.constraint is None
    assert "Clayfit's behaviour" in evidence.status


def test_an_unknown_mode_is_refused():
    library, air, glycol = mounts(0.0)
    with pytest.raises(ValueError, match='"always" or "supported"'):
        expansion_evidence(glycol, air, library, when="sometimes")


def test_the_constraint_acts_on_the_expandable_entries_alone():
    library, air, glycol = mounts(smectite_in_glycol=3000.0)
    evidence = expansion_evidence(glycol, air, library)
    design = evidence.constraint.design
    # Quartz and illite are zero in it; the smectite is its own column.
    np.testing.assert_allclose(design[:, 0], 0.0)
    np.testing.assert_allclose(design[:, 1], 0.0)
    assert float(np.max(design[:, 2])) > 0.0


def test_only_the_diagnostic_points_count():
    library, air, glycol = mounts(smectite_in_glycol=3000.0)
    weights = expansion_evidence(glycol, air, library).constraint.point_weights
    inside = (
        ((GRID >= SMECTITE_GLYCOL_001_WINDOW[0]) & (GRID <= SMECTITE_GLYCOL_001_WINDOW[1]))
        | ((GRID >= INTERSTRATIFIED_SHIFT_WINDOW[0]) & (GRID <= INTERSTRATIFIED_SHIFT_WINDOW[1]))
    )
    np.testing.assert_array_equal(weights.astype(bool), inside)


def test_a_range_that_excludes_the_windows_says_so():
    library, air, glycol = mounts(0.0)
    evidence = expansion_evidence(glycol, air, library, range_two_theta=(12.0, 30.0))
    assert evidence.constraint is None
    assert "excludes the diagnostic windows" in evidence.status


def test_a_library_with_nothing_expandable_has_nothing_to_restrain():
    library = PatternLibrary(two_theta=GRID, entries=[
        entry("illite", "illite", peak(8.84, 1.0, 0.25), fraction=1.0),
    ])
    _, air, glycol = mounts(0.0)
    evidence = expansion_evidence(glycol, air, library)
    assert evidence.constraint is None
    assert "nothing to restrain" in evidence.status


# --------------------------------------------------------------------------
# What it does to a fit
# --------------------------------------------------------------------------

def share_of(result, phase: str) -> float:
    return float(sum(fraction for name, fraction
                     in zip(result.phases, result.scattering_fraction)
                     if name == phase))


def test_a_smectite_with_no_expansion_behind_it_is_held_down():
    """The defect this exists to remove, on a specimen that has no smectite."""
    library, air, glycol = mounts(smectite_in_glycol=0.0)
    evidence = expansion_evidence(glycol, air, library)
    free = nnls_fit(glycol, library)
    held = nnls_fit(glycol, library, constraints=[evidence.constraint])
    assert share_of(held, "smectite_EG") < share_of(free, "smectite_EG")


def test_a_smectite_that_is_really_there_survives_the_restraint():
    """The complement, without which the restraint would just delete smectite."""
    library, air, glycol = mounts(smectite_in_glycol=4000.0)
    evidence = expansion_evidence(glycol, air, library)
    assert evidence.supported
    free = nnls_fit(glycol, library)
    held = nnls_fit(glycol, library, constraints=[evidence.constraint])
    assert share_of(held, "smectite_EG") > 0.5 * share_of(free, "smectite_EG")


def test_the_pattern_statistics_are_still_the_pattern_s():
    """So a restrained fit stays comparable with an unrestrained one."""
    library, air, glycol = mounts(smectite_in_glycol=3000.0)
    evidence = expansion_evidence(glycol, air, library)
    held = nnls_fit(glycol, library, constraints=[evidence.constraint])
    inside = held.mask
    observed = held.observed[inside]
    residual = observed - held.calculated[inside]
    weights = 1.0 / np.clip(observed, 1.0, None)
    expected = float(np.sqrt(np.sum(weights * residual ** 2)
                             / np.sum(weights * observed ** 2)))
    assert held.r_wp == pytest.approx(expected, rel=1e-9)
    assert held.metadata["constraints"][0]["name"] == "air-to-glycol expansion"


def test_a_constraint_of_the_wrong_width_is_refused():
    library, air, glycol = mounts(0.0)
    bad = ExtraObservation(target=np.zeros(GRID.size),
                           design=np.zeros((GRID.size, 2)), name="wrong")
    with pytest.raises(ValueError, match="describes 2 entries"):
        nnls_fit(glycol, library, constraints=[bad])


def test_a_zero_weighted_constraint_is_the_same_as_none():
    library, air, glycol = mounts(smectite_in_glycol=3000.0)
    evidence = expansion_evidence(glycol, air, library, weight=0.0)
    held = nnls_fit(glycol, library, constraints=[evidence.constraint])
    free = nnls_fit(glycol, library)
    np.testing.assert_allclose(held.coefficients, free.coefficients)
    assert held.metadata["constraints"] == []


# --------------------------------------------------------------------------
# The measured limit of it
# --------------------------------------------------------------------------

def test_clayfits_windows_cannot_separate_an_illite_rich_stack_from_an_illite():
    """Measured on the real library, and the reason her I/S is not fixed by this.

    The area an interstratified composition puts in Clayfit's 4.75-5.55 deg
    window falls from 80 % smectite down to about 25 % and then *rises* again,
    so between an illite-rich stack and a pure illite the window has no signal
    to give.  A restraint built on it therefore cannot hold down a 90/10
    illite/smectite, whatever the air-dried mount shows.
    """
    smectite_rich = peak(5.2, 1.0, 0.5) + peak(10.4, 0.2, 0.5)
    illite_rich = peak(8.84, 1.0, 0.28) + peak(5.2, 0.03, 0.9)
    pure_illite = peak(8.84, 1.0, 0.25)
    inside = ((GRID >= SMECTITE_GLYCOL_001_WINDOW[0])
              & (GRID <= SMECTITE_GLYCOL_001_WINDOW[1]))

    def area(curve):
        return float(np.trapezoid(curve[inside] / curve.max(), GRID[inside]))

    # The window separates a smectite-rich stack from the rest...
    assert area(smectite_rich) > 5.0 * area(illite_rich)
    # ...and does not separate an illite-rich stack from an illite.
    assert area(illite_rich) == pytest.approx(area(pure_illite), abs=0.02)
