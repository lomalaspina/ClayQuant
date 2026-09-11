"""Validation against Reynolds (1965), Am. Mineral. 50, 990-1001.

The paper refines a one-dimensional model of an ethylene glycol-montmorillonite
complex against 13 observed basal reflections and publishes both the observed F
factors and the resulting calculated intensities (its Table 2).  Reproducing
those numbers from the transcribed model exercises the whole basal-reflection
chain: scattering factors, the layer structure factor, and the random-powder
Lorentz-polarization factor the paper used.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.models import REYNOLDS_1965_D001, eg_smectite_layer
from clayquant.optics import lorentz_polarization

CU_KA1 = 1.540596

ORDERS = np.arange(1, 15)

# Table 2, "observed" column; 00 14 was not detected.
F_OBSERVED = np.array(
    [63.4, 24.6, 36.2, 20.9, 70.9, 47.2, 6.5, 38.7, 33.1, 3.6, 23.8, 24.2, 24.8, np.nan]
)

# Table 2, "calculated intensity" column, normalised to I(001) = 100.
I_CALCULATED = np.array(
    [100.0, 3.43, 3.48, 0.66, 4.91, 1.35, 0.021, 0.549, 0.308, 0.0070, 0.090, 0.085, 0.056, 0.031]
)


@pytest.fixture(scope="module")
def basal():
    layer = eg_smectite_layer()
    d = REYNOLDS_1965_D001 / ORDERS
    two_theta = np.degrees(2.0 * np.arcsin(CU_KA1 / (2.0 * d)))
    f = layer.structure_factor(1.0 / d)
    intensity = np.abs(f) ** 2 * lorentz_polarization(two_theta, mode="powder")
    return layer, f, 100.0 * intensity / intensity[0]


def test_electron_count(basal):
    layer, _, _ = basal
    assert layer.electrons == pytest.approx(504.0, abs=5.0)


def test_reliability_factor_matches_paper(basal):
    _, f, _ = basal
    amplitude = np.abs(f)
    scale = np.nansum(amplitude * F_OBSERVED) / np.nansum(amplitude**2)
    r_factor = np.nansum(np.abs(F_OBSERVED - scale * amplitude)) / np.nansum(F_OBSERVED)
    assert r_factor < 0.05


def test_calculated_intensities_match_paper(basal):
    _, _, intensity = basal
    ratio = intensity / I_CALCULATED
    # A constant offset is expected: the paper used the scattering factors of
    # Ibers (1962), this package those of Waasmaier & Kirfel (1995).  What must
    # hold is that the ratio is constant, i.e. relative intensities agree.
    assert np.all(ratio > 0.8)
    assert np.all(ratio < 1.1)
    assert np.std(ratio) < 0.05


def test_strong_orders_ordering(basal):
    _, _, intensity = basal
    # 001 strongest, then the characteristic 005 > 003 ~ 002 sequence of the
    # two-layer glycol complex.
    assert intensity[0] > intensity[4] > intensity[5]
    assert intensity[4] > intensity[2] > intensity[3]
