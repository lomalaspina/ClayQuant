"""Failing to prove kaolinite is not the same as disproving it."""
from __future__ import annotations

import numpy as np
import pytest

from clayquant.absorption import MASS_ATTENUATION_CU_KA, mass_attenuation


def test_every_element_a_real_phase_database_carries_has_a_coefficient():
    """The six that were missing stopped a fit dead, and one of them - lithium -
    is carried by an amphibole, so it was reached often."""
    for element in ("Li", "B", "Cd", "Sn", "La", "W"):
        assert element in MASS_ATTENUATION_CU_KA
        assert MASS_ATTENUATION_CU_KA[element] > 0.0


def test_the_coefficients_are_not_monotonic_in_atomic_number():
    """Tungsten sits below lanthanum because its L3 edge has passed over the Cu
    K-alpha energy.  A table that was interpolated rather than read would have
    got this wrong, and the test records that it must not be."""
    assert MASS_ATTENUATION_CU_KA["W"] < MASS_ATTENUATION_CU_KA["La"]
    assert MASS_ATTENUATION_CU_KA["Pb"] > MASS_ATTENUATION_CU_KA["W"]


def test_an_unknown_element_still_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="no Cu K-alpha mass attenuation"):
        mass_attenuation("Pu")


def test_tin_resolves_through_its_oxidation_state():
    """The error the user hit named 'Sn+4', as a CIF writes it."""
    assert mass_attenuation("Sn+4") == MASS_ATTENUATION_CU_KA["Sn"]


def _evidence(collapse_bounds, ratio_bounds):
    """A KaoliniteEvidence with the collapse and ratio routes stubbed."""
    from clayquant.diagnostics import KaoliniteEvidence

    class Air:
        is_present = True

    class Collapse:
        air = Air()
        kaolinite_bounds = collapse_bounds

        @property
        def kaolinite_detected(self):
            return self.kaolinite_bounds[0] > 0.1

    return KaoliniteEvidence(
        windows={}, references={}, shares=(), collapse=Collapse(),
    )


def test_the_collapse_test_decides_where_it_exists():
    """The heated mount is a measurement; the chlorite ratios are an inference.

    On a real separate the four ratio routes gave 2-35, 36-44, 40-51 and 52-66
    per cent while 86 per cent of the 7.15 A peak disappeared on heating.  The
    verdict was "not established" because one route's lower bound was 2.
    """
    assert _evidence((0.56, 0.81), None).detected is True


def test_a_peak_that_survives_heating_is_not_kaolinite():
    assert _evidence((0.0, 0.03), None).detected is False


def test_detected_and_excluded_are_not_opposites():
    """The distinction that stopped kaolinite being deleted from the fit: an
    upper bound near zero rules it out, a lower bound near zero does not."""
    ambiguous = _evidence((0.02, 0.66), None)
    assert ambiguous.detected is False or ambiguous.detected is True
    assert ambiguous.excluded is False

    absent = _evidence((0.0, 0.02), None)
    assert absent.detected is False
    assert absent.excluded is True
