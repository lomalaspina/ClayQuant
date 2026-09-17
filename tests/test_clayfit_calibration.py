"""Carrying Clayfit's family calibration over, and what it must refuse to do.

The factors are a real calibration - determined against mixtures held in the
older Clayfit4 database - but they are expressed in Clayfit's units, where each
family's profile is multiplied by its factor and the fitted coefficient is read
as the amount.  Re-anchoring them here is arithmetic with one assumption, and
the assumption fails loudly in one case that these tests pin: a family whose
pattern is held at a different orientation than the one Clayfit anchored on is
not comparable, and importing its factor anyway is wrong by ``r ** -3``.
"""

import numpy as np
import pytest

from clayquant.library import PatternLibrary
from clayquant.pattern import Pattern
from clayquant.quantification import (
    CLAYFIT_ANCHOR_ORIENTATION,
    CLAYFIT_SCALE_FACTORS,
    Calibration,
    clayfit_weight_fractions,
)

GRID = np.arange(4.0, 40.0, 0.02)


def a_library(orientations=(0.1, 1.0), phases=("illite", "chlorite"), smectite_r=0.1):
    library = PatternLibrary(two_theta=GRID, entries=[], metadata={})
    for index, phase in enumerate(phases):
        for r in orientations:
            y = np.exp(-0.5 * ((GRID - (8.0 + 3.0 * index)) / 0.2) ** 2) / r**3
            library.add(Pattern(two_theta=GRID, intensity=y, name=f"{phase}-{r}"),
                        phase=phase, march_dollase=r,
                        unit_mass=400.0 + 100.0 * index, unit_volume=500.0 + 100.0 * index)
    y = np.exp(-0.5 * ((GRID - 5.2) / 0.3) ** 2) / smectite_r**3
    library.add(Pattern(two_theta=GRID, intensity=y, name="smectite_EG"),
                phase="smectite_EG", march_dollase=smectite_r,
                unit_mass=700.0, unit_volume=900.0)
    return library


class Fit:
    def __init__(self, coefficients):
        self.coefficients = np.asarray(coefficients, dtype=float)


def test_the_factors_are_the_ones_clayfit_publishes():
    # Transcribed from Clayfit5's oriented_clay.py; a typo here is silent, so
    # it is pinned.
    assert CLAYFIT_SCALE_FACTORS["chlorite"] == pytest.approx(358.0343993649871)
    assert CLAYFIT_SCALE_FACTORS["illite"] == pytest.approx(113.15656786986817)
    assert CLAYFIT_SCALE_FACTORS["kaolinite_1M"] == pytest.approx(180.223981959315)
    assert CLAYFIT_SCALE_FACTORS["kaolinite_2M"] == pytest.approx(104.678273593158)
    assert CLAYFIT_SCALE_FACTORS["smectite_EG"] == pytest.approx(754.5132206968542)
    assert CLAYFIT_ANCHOR_ORIENTATION == pytest.approx(0.1)


def test_the_interstratified_series_are_not_calibrated():
    # Clayfit gives one factor per composition for its mixed-layer profiles, not
    # one per family, so there is nothing to carry over for I/S or C/S.
    assert "I/S" not in CLAYFIT_SCALE_FACTORS
    assert "C/S" not in CLAYFIT_SCALE_FACTORS


def test_a_calibration_comes_out_with_a_factor_per_calibrated_phase():
    calibration = Calibration.from_clayfit(a_library())
    assert set(calibration.factors) == {"illite", "chlorite", "smectite_EG"}
    assert all(value > 0.0 for value in calibration.factors.values())


def test_the_factors_are_normalised_to_average_one():
    calibration = Calibration.from_clayfit(a_library())
    values = list(calibration.factors.values())
    assert sum(values) / len(values) == pytest.approx(1.0)


def test_a_phase_the_library_holds_at_another_orientation_is_refused():
    # The glycol smectite as a random powder: its basal series is a thousand
    # times weaker per unit amount than at r = 0.1, so importing Clayfit's
    # factor for it would be wrong by that much.  It came out at k = 0.0003.
    calibration = Calibration.from_clayfit(a_library(smectite_r=1.0))
    assert "smectite_EG" not in calibration.factors
    assert "smectite_EG" in calibration.source
    assert "r = 0.1" in calibration.source


def test_a_phase_clayfit_does_not_calibrate_keeps_a_factor_of_one():
    calibration = Calibration.from_clayfit(a_library(phases=("illite", "I/S")))
    assert "I/S" not in calibration.factors
    assert calibration.factor("I/S") == pytest.approx(1.0)


def test_the_source_names_the_calibration_and_its_limits():
    source = Calibration.from_clayfit(a_library()).source
    assert "Clayfit" in source and "Clayfit4" in source
    assert "computed basis" in source


def test_a_library_with_no_calibrated_phase_says_so():
    library = a_library(phases=("I/S",), smectite_r=1.0)
    library.entries = [entry for entry in library.entries if entry.phase == "I/S"]
    with pytest.raises(ValueError, match="cannot"):
        Calibration.from_clayfit(library)


def test_weight_fractions_follow_the_scale_factors_inversely():
    # Two phases fitted with the same coefficient and the same anchor differ in
    # amount exactly as their factors differ, inversely: that is what the
    # factors mean.
    library = a_library(orientations=(0.1,), phases=("illite", "chlorite"), smectite_r=0.1)
    library.entries = [entry for entry in library.entries if entry.phase != "smectite_EG"]
    weights = clayfit_weight_fractions(Fit([1.0, 1.0]), library)
    ratio = weights["illite"] / weights["chlorite"]
    expected = CLAYFIT_SCALE_FACTORS["chlorite"] / CLAYFIT_SCALE_FACTORS["illite"]
    assert ratio == pytest.approx(expected, rel=1e-6)


def test_weight_fractions_sum_to_one():
    library = a_library(orientations=(0.1,), smectite_r=0.1)
    weights = clayfit_weight_fractions(Fit([1.0, 1.0, 1.0]), library)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_a_fitted_phase_with_no_factor_raises_rather_than_being_dropped():
    # Dropping it would renormalise the rest to 100 % between them, and the
    # answer would look complete with a mineral missing from it.
    library = a_library(orientations=(0.1,), phases=("illite", "I/S"), smectite_r=0.1)
    with pytest.raises(KeyError, match="I/S"):
        clayfit_weight_fractions(Fit([1.0, 1.0, 1.0]), library)


def test_a_phase_that_was_not_fitted_is_simply_absent():
    library = a_library(orientations=(0.1,), smectite_r=0.1)
    weights = clayfit_weight_fractions(Fit([1.0, 0.0, 0.0]), library)
    assert set(weights) == {"illite"}


def test_the_library_records_the_smectite_orientation_it_used():
    from clayquant.library import build_library, two_theta_grid

    grid = two_theta_grid(4.0, 20.0, 0.05)
    library = build_library(
        grid=grid, orientations=(1.0,), illite_smectite=(), chlorite_smectite=(),
        chlorite_iron=(), csds_means=(15.0,), smectite_orientation=0.2,
        host_thicknesses={},
    )
    assert library.metadata["smectite_orientation"] == pytest.approx(0.2)
    smectite = [entry for entry in library.entries if entry.phase == "smectite_EG"]
    assert len(smectite) == 1
    assert smectite[0].march_dollase == pytest.approx(0.2)
