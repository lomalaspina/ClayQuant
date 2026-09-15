"""The instrument constant, and the factor k it yields for a pure-phase mount.

A pure mount has a known weight fraction, so the external-standard relation can
be read backwards: whatever it takes to make the fitted mass agree with the
weight actually present is the factor by which the calculated pattern fails to
describe the phase.  Tested here on synthetic measurements, where the truth is
set: a phase built at the same scale as the standard must give k = 1, and one
built at half of it must give 0.5.

What synthetic data cannot test is the assumption underneath - that both mounts
are thick enough for absorption rather than mass to decide how much diffracts.
That is what :mod:`clayquant.absorption` reports on, and why a k that comes out
the same for chemically different phases says something about the mounts rather
than the minerals.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.library import PatternLibrary
from clayquant.models import available_phases, load_crystal
from clayquant.nnls import nnls_fit
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, Pattern, powder_pattern
from clayquant.profile import PeakShape
from clayquant.quantification import ExternalStandard, calibration_factor

pytestmark = pytest.mark.skipif(
    not all(available_phases().values()), reason="ICSD CIF files are not installed"
)

GRID = np.arange(4.0, 39.0, 0.02)
INSTRUMENT = Instrument(
    peak_shape=PeakShape(u=0.004, v=-0.001, w=0.002, eta=0.5, size_ab=600.0),
    divergence=Divergence(),
)


def single_phase(name, key, scale=1.0):
    """A library of one pattern, and a measurement of that pattern times ``scale``."""
    crystal = load_crystal(key)
    pattern = powder_pattern(crystal, GRID, INSTRUMENT, r_march_dollase=1.0, name=name)
    library = PatternLibrary(two_theta=GRID, entries=[], metadata={})
    library.add(pattern, phase=name, march_dollase=1.0,
                unit_mass=crystal.cell_mass, unit_volume=crystal.volume)
    measured = Pattern(GRID, scale * pattern.intensity, name=f"{name} x {scale:g}")
    return library, measured, crystal


def fit_of(name, key, scale=1.0):
    library, measured, crystal = single_phase(name, key, scale)
    return nnls_fit(measured, library, range_two_theta=(4.0, 39.0)), crystal


def test_a_phase_measured_like_the_standard_gives_a_factor_of_one():
    result, _ = fit_of("illite", "illite")
    standard = ExternalStandard.from_fit(result, "illite", 40.0)
    assert calibration_factor(result, "illite", 40.0, standard) == pytest.approx(1.0)


def test_half_the_intensity_gives_half_the_factor():
    strong, _ = fit_of("illite", "illite", scale=1.0)
    weak, _ = fit_of("illite", "illite", scale=0.5)
    standard = ExternalStandard.from_fit(strong, "illite", 40.0)
    assert calibration_factor(weak, "illite", 40.0, standard) == pytest.approx(0.5, rel=1e-3)


def test_the_absorption_coefficient_enters_linearly():
    """It is the term that makes two different minerals comparable at all."""
    result, _ = fit_of("illite", "illite")
    standard = ExternalStandard.from_fit(result, "illite", 30.0)
    assert calibration_factor(result, "illite", 60.0, standard) == pytest.approx(2.0)


def test_a_standard_of_known_impurity_is_accounted_for():
    """A standard that is 90 % pure is not a 100 % standard."""
    result, _ = fit_of("illite", "illite")
    pure = ExternalStandard.from_fit(result, "illite", 40.0, weight_fraction=1.0)
    impure = ExternalStandard.from_fit(result, "illite", 40.0, weight_fraction=0.9)
    assert impure.constant == pytest.approx(pure.constant / 0.9)
    # And the same correction on the specimen side cancels it.
    assert calibration_factor(result, "illite", 40.0, impure, weight_fraction=0.9) == \
        pytest.approx(calibration_factor(result, "illite", 40.0, pure))


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_an_impossible_weight_fraction_is_refused(bad):
    result, _ = fit_of("illite", "illite")
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        ExternalStandard.from_fit(result, "illite", 40.0, weight_fraction=bad)
    standard = ExternalStandard.from_fit(result, "illite", 40.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        calibration_factor(result, "illite", 40.0, standard, weight_fraction=bad)


def test_a_phase_the_fit_did_not_find_is_an_error_not_a_zero():
    result, _ = fit_of("illite", "illite")
    with pytest.raises(ValueError, match="no mass"):
        ExternalStandard.from_fit(result, "chlorite", 40.0)
    standard = ExternalStandard.from_fit(result, "illite", 40.0)
    with pytest.raises(ValueError, match="no mass"):
        calibration_factor(result, "chlorite", 40.0, standard)


def test_the_conditions_of_the_standard_are_checked_against_the_specimen():
    """The failure that cost a first attempt at this: a different slit.

    A standard is only a standard through identical optics, and a fixed
    divergence slit of 0.25 deg against 0.5 deg is not even a constant factor -
    it overflows the specimen at low angle and not at high.
    """
    result, _ = fit_of("illite", "illite")
    standard = ExternalStandard.from_fit(
        result, "illite", 40.0,
        conditions={"divergence /deg": 0.5, "counting time /s": 19.685, "mask": "14.0 mm"},
    )
    assert standard.comparable_with(
        {"divergence /deg": 0.5, "counting time /s": 19.685, "mask": "14.0 mm"}
    ) == []
    differences = standard.comparable_with(
        {"divergence /deg": 0.25, "counting time /s": 25.4, "mask": "14.0 mm"}
    )
    assert any("divergence" in note for note in differences)
    assert any("counting time" in note for note in differences)
    assert not any("mask" in note for note in differences)


def test_conditions_that_were_never_recorded_say_so():
    result, _ = fit_of("illite", "illite")
    standard = ExternalStandard.from_fit(result, "illite", 40.0)
    notes = standard.comparable_with({"divergence /deg": 0.5})
    assert notes and "not recorded" in notes[0]
