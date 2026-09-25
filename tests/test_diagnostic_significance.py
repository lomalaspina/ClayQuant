"""A diagnostic window has to hold a reflection before its loss means anything.

``measure_peak`` used to report a height and an area from any window at all:
after a baseline is removed the tallest noise point is positive, and the area
integrates only the points above zero, so an empty window has a positive area by
construction.  The collapse test then read noise as kaolinite.  On nine real clay
standards - two chlorites, two kaolinites, a dickite, an illite, two
montmorillonites and a sepiolite - four of the nine were reported as containing
kaolinite with nothing at 7.15 A at all.
"""

import numpy as np
import pytest

from clayquant.diagnostics import (
    CHLORITE_002_SURVIVAL,
    MINIMUM_SIGMAS,
    kaolinite_collapse,
    measure_peak,
)
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 30.0, 0.0167)
WINDOW = (11.6, 13.2)


def background(level: float = 900.0, hump: float = 6000.0) -> np.ndarray:
    """A flat level plus low-angle air scatter, which is what leaves a pedestal."""
    return level + hump * np.exp(-((GRID - 3.0) / 6.0) ** 2)


def scan(signal: np.ndarray | float = 0.0, seed: int = 20260925) -> Pattern:
    rng = np.random.default_rng(seed)
    return Pattern(
        two_theta=GRID,
        intensity=rng.poisson(signal + background()).astype(float),
        name="scan",
    )


def peak(centre: float, height: float, width: float = 0.12) -> np.ndarray:
    return height * np.exp(-4.0 * np.log(2.0) * ((GRID - centre) / width) ** 2)


def test_an_empty_window_holds_no_reflection():
    metrics = measure_peak(scan(), WINDOW)
    assert not metrics.is_present
    assert metrics.significance < metrics.minimum_sigmas


def test_an_empty_window_still_has_a_positive_area():
    """Which is exactly why the area cannot be the test."""
    metrics = measure_peak(scan(), WINDOW)
    assert metrics.area > 0.0
    assert metrics.height > 0.0


def test_a_real_reflection_is_found_with_room_to_spare():
    metrics = measure_peak(scan(peak(12.4, 4000.0)), WINDOW)
    assert metrics.is_present
    assert metrics.significance > 10.0 * metrics.minimum_sigmas
    assert metrics.d_spacing == pytest.approx(7.14, abs=0.03)


def test_the_threshold_allows_for_the_window_having_been_searched():
    """The largest of n noise points stands above zero with no peak present."""
    metrics = measure_peak(scan(), WINDOW)
    assert metrics.minimum_sigmas > MINIMUM_SIGMAS
    assert metrics.averaged_points > 1


def test_a_weak_reflection_near_the_threshold_is_not_claimed():
    """Between noise and a reflection there is a size that says nothing."""
    weak = measure_peak(scan(peak(12.4, 30.0)), WINDOW)
    assert weak.significance < 3.0 * weak.minimum_sigmas


def test_an_empty_window_reports_no_kaolinite_and_no_chlorite():
    result = kaolinite_collapse(scan(seed=1), scan(seed=2), scale=1.0)
    assert not result.kaolinite_detected
    assert not result.chlorite_indicated
    assert np.isnan(result.kaolinite_bounds[0])
    assert "above the noise" in result.summary() or "No reflection" in result.summary()


def test_a_peak_that_wholly_disappears_is_all_kaolinite():
    result = kaolinite_collapse(
        scan(peak(12.4, 4000.0), seed=1), scan(seed=2), scale=1.0
    )
    assert result.kaolinite_detected
    assert result.kaolinite_bounds == (1.0, 1.0)
    assert not result.chlorite_indicated


def test_a_peak_that_wholly_survives_is_no_kaolinite():
    signal = peak(12.4, 4000.0)
    result = kaolinite_collapse(scan(signal, seed=1), scan(signal, seed=2), scale=1.0)
    assert not result.kaolinite_detected
    assert result.chlorite_indicated
    assert result.kaolinite_bounds[0] == 0.0


def test_a_peak_that_half_survives_cannot_be_called_kaolinite():
    """A chlorite 002 loses up to two thirds of itself, so half is within it."""
    result = kaolinite_collapse(
        scan(peak(12.4, 4000.0), seed=1), scan(peak(12.4, 2000.0), seed=2), scale=1.0
    )
    least, most = result.kaolinite_bounds
    assert least == 0.0
    assert most > 0.0
    assert not result.kaolinite_detected


def test_the_chlorite_survival_range_is_a_measurement():
    low, high = CHLORITE_002_SURVIVAL
    assert 0.0 < low < high < 1.0


def test_no_threshold_measures_whatever_is_in_the_window():
    """The old behaviour stays reachable, for data that is already clean."""
    metrics = measure_peak(scan(), WINDOW, minimum_sigmas=0.0)
    assert metrics.is_present
