"""Calibrating the angular axis from a basal series, with no quartz to do it with.

A specimen off the focusing circle moves every line by -(2s/R) cos(theta).  Over
a 4 to 40 deg clay scan that is very nearly a constant - 0.25 mm on a 240 mm
goniometer shifts every reflection by between 0.119 and 0.113 deg - so it is
*not* separable from a zero error there, and these tests do not pretend
otherwise.  What they pin down is the thing that matters: the correction that
makes the orders of one basal series consistent with each other.

That is the gap this fills.  estimate_zero_error needs a known reference line,
so a separate with no quartz was simply left uncorrected; three kaolin standards
measured here showed it, giving 7.197, 7.161 and 7.148 A for one spacing from
their first three orders.  A basal series is its own reference, because its
orders have to agree.
"""

import numpy as np
import pytest

from clayquant.calibration import (
    CU_KA1,
    RATIONAL_SERIES_SPREAD,
    DisplacementResult,
    apply_displacement,
    apply_zero_error,
    basal_series_positions,
    displacement_shift,
    estimate_displacement,
)
from clayquant.pattern import Pattern

RADIUS = 240.0


def angle(d: float, wavelength: float = CU_KA1) -> float:
    return 2.0 * np.degrees(np.arcsin(wavelength / (2.0 * d)))


def a_basal_series(spacing: float, displacement: float = 0.0, zero: float = 0.0,
                   orders: int = 3, radius: float = RADIUS) -> Pattern:
    """A clean 00l series, then displaced and offset the way a real mount is."""
    grid = np.arange(4.0, 44.0, 0.01)
    counts = np.zeros_like(grid)
    for order in range(1, orders + 1):
        sine = order * CU_KA1 / (2.0 * spacing)
        if not 0.0 < sine < 1.0:
            continue
        true = 2.0 * np.degrees(np.arcsin(sine))
        # where the instrument would actually record it
        seen = true - displacement_shift(true, displacement, radius) + zero
        counts += (6000.0 / order) * np.exp(
            -4.0 * np.log(2.0) * ((grid - seen) / 0.12) ** 2)
    return Pattern(two_theta=grid, intensity=counts + 50.0, name="series",
                   metadata={"goniometer_radius": radius})


def test_the_shift_is_largest_at_low_angle_and_fades():
    low = float(displacement_shift(10.0, 0.2, RADIUS))
    high = float(displacement_shift(120.0, 0.2, RADIUS))
    assert low > 0.0
    assert abs(high) < 0.6 * abs(low)
    assert float(displacement_shift(180.0, 0.2, RADIUS)) == pytest.approx(0.0, abs=1e-9)


def test_over_a_clay_scan_it_is_nearly_a_constant():
    """The honest limit of this correction, asserted so nobody re-reads it as more.

    cos(theta) runs from 0.9985 to 0.9468 between 6 and 38 deg, so the shape the
    displacement adds over the constant is a twentieth of a peak width.  A zero
    error of the mean value fits the same data almost as well, and no fit of
    this scan can say which of the two it was.
    """
    angles = np.array([6.2, 12.4, 18.8, 24.9, 31.3, 37.6])
    curved = displacement_shift(angles, 0.25, RADIUS)
    assert float(np.mean(curved)) == pytest.approx(0.117, abs=0.002)
    assert float(np.ptp(curved)) < 0.007, "the two are nearly degenerate here"


def test_it_reports_the_equivalent_zero_error():
    found = estimate_displacement(a_basal_series(7.15, displacement=0.25))
    assert found.detected
    # negated, because apply_zero_error subtracts
    assert found.equivalent_zero_error == pytest.approx(-0.117, abs=0.01)


def test_it_recovers_a_displacement_it_was_given():
    pattern = a_basal_series(7.15, displacement=0.25)
    found = estimate_displacement(pattern)
    assert found.detected, found.note
    assert found.displacement == pytest.approx(0.25, abs=0.02)
    assert found.spacing == pytest.approx(7.15, abs=0.005)
    assert found.spread_after < 0.1 * found.spread_before


def test_it_recovers_zero_from_an_undisplaced_mount():
    found = estimate_displacement(a_basal_series(7.15, displacement=0.0))
    assert found.detected
    assert found.displacement == pytest.approx(0.0, abs=0.02)


def test_the_uncorrected_orders_disagree_and_the_corrected_ones_do_not():
    """The diagnostic, stated as the thing a user can see for themselves."""
    pattern = a_basal_series(7.15, displacement=0.25)
    positions = basal_series_positions(pattern, (11.6, 13.0))
    assert len(positions) == 3
    implied = [order * CU_KA1 / (2.0 * np.sin(np.radians(p / 2.0)))
               for order, p in enumerate(positions, start=1)]
    assert implied[0] > implied[1] > implied[2], "apparent d must fall with order"
    found = estimate_displacement(pattern)
    corrected = apply_displacement(pattern, found.displacement)
    after = basal_series_positions(corrected, (11.6, 13.0))
    recovered = [order * CU_KA1 / (2.0 * np.sin(np.radians(p / 2.0)))
                 for order, p in enumerate(after, start=1)]
    assert max(recovered) - min(recovered) < 0.01


def test_a_zero_error_present_as_well_is_taken_up_into_the_solution():
    """It solves one angular correction, not two, and does not pretend to split them.

    With a 0.20 mm displacement and a +0.06 deg zero error both in the data, the
    solve returns neither number: it returns the single correction that makes
    the orders agree, which is what the axis needs.  Reading the millimetres back
    as the mount's position would be wrong, and the docstring says so.
    """
    pattern = a_basal_series(7.15, displacement=0.20, zero=0.06)
    found = estimate_displacement(pattern)
    assert found.detected
    assert found.displacement != pytest.approx(0.20, abs=0.03)
    # what it does guarantee: the corrected orders agree
    corrected = apply_displacement(pattern, found.displacement)
    after = basal_series_positions(corrected, (11.6, 13.0))
    implied = [order * CU_KA1 / (2.0 * np.sin(np.radians(p / 2.0)))
               for order, p in enumerate(after, start=1)]
    assert max(implied) - min(implied) < 0.01


def test_one_reflection_is_refused_rather_than_guessed():
    pattern = a_basal_series(7.15, displacement=0.25, orders=1)
    found = estimate_displacement(pattern)
    assert not found.detected
    assert "one reflection cannot" in found.note
    assert found.displacement == 0.0


def test_an_absurd_solution_is_refused():
    """Two reflections that are not orders of one series will reconcile at some
    enormous displacement; reporting it would dress a misidentification up as a
    correction."""
    grid = np.arange(4.0, 44.0, 0.01)
    counts = np.zeros_like(grid)
    for centre in (12.4, 22.0):
        counts += 5000.0 * np.exp(-4.0 * np.log(2.0) * ((grid - centre) / 0.12) ** 2)
    pattern = Pattern(two_theta=grid, intensity=counts + 50.0, name="not a series",
                      metadata={"goniometer_radius": RADIUS})
    found = estimate_displacement(pattern)
    assert found.displacement == 0.0


def test_applying_it_records_what_was_done():
    pattern = a_basal_series(7.15, displacement=0.25)
    moved = apply_displacement(pattern, 0.25)
    assert moved.metadata["displacement"] == pytest.approx(0.25)
    assert moved.metadata["displacement_radius"] == pytest.approx(RADIUS)
    assert moved.intensity.tolist() == pattern.intensity.tolist(), "counts are not resampled"


def test_the_radius_comes_from_the_file_when_it_is_there():
    pattern = a_basal_series(7.15, displacement=0.25, radius=200.0)
    pattern.metadata["goniometer_radius"] = 200.0
    found = estimate_displacement(pattern)
    assert found.radius == pytest.approx(200.0)
    assert found.displacement == pytest.approx(0.25, abs=0.02)


def test_both_corrections_compose_on_the_mount_state():
    from clayquant.gui.state import MountState

    state = MountState(raw=a_basal_series(7.15, displacement=0.25, zero=0.06))
    state.displacement = 0.25
    state.zero_error = 0.06
    corrected = state.corrected()
    positions = basal_series_positions(corrected, (11.6, 13.0))
    implied = [order * CU_KA1 / (2.0 * np.sin(np.radians(p / 2.0)))
               for order, p in enumerate(positions, start=1)]
    assert max(implied) - min(implied) < 0.01
    assert float(np.mean(implied)) == pytest.approx(7.15, abs=0.01)


def an_irrational_series(first: float = 14.5, ratios=(1.0, 1.95, 2.85)) -> Pattern:
    """A mixed-layer basal series: peaks that are not orders of one spacing.

    An interstratified clay is defined by this - the 001, 002 and 003 of an I/S
    do not share a layer repeat - so no angular correction reconciles them, and a
    solver that returns one anyway is describing noise.
    """
    grid = np.arange(3.0, 44.0, 0.01)
    counts = np.zeros_like(grid)
    for n, ratio in enumerate(ratios, start=1):
        sine = ratio * CU_KA1 / (2.0 * first)
        if not 0.0 < sine < 1.0:
            continue
        centre = 2.0 * np.degrees(np.arcsin(sine))
        counts += (6000.0 / n) * np.exp(
            -4.0 * np.log(2.0) * ((grid - centre) / 0.3) ** 2)
    return Pattern(two_theta=grid, intensity=counts + 50.0, name="mixed layer",
                   metadata={"goniometer_radius": RADIUS})


def test_an_interstratified_series_is_refused_rather_than_corrected():
    """The failure this guard was written from.

    Applied blind, the solve made an illite and two montmorillonites worse by 2
    to 4 points of Rwp, and all three were exactly the standards whose orders
    would not reconcile: 0.0165, 0.040 and 0.044 A against 0.0003 to 0.0036 for
    the five rational ones.
    """
    found = estimate_displacement(an_irrational_series(ratios=(1.0, 1.97, 2.93)),
                                  first=(5.5, 6.6), window=0.6)
    assert len(found.orders) >= 3, "the guard needs an order it did not fit with"
    assert not found.detected
    assert found.displacement == 0.0
    assert "interstratified" in found.note
    assert found.spread_after > RATIONAL_SERIES_SPREAD


def test_two_orders_alone_are_not_evidence_and_say_so():
    """Two orders and two unknowns reconcile exactly whatever the peaks are, so
    a zero residual there is arithmetic rather than a check."""
    found = estimate_displacement(an_irrational_series(), first=(5.5, 6.6))
    assert len(found.orders) == 2
    assert found.detected
    assert not found.verified
    assert "exact system" in found.note


def test_a_rational_series_is_not_caught_by_the_guard():
    found = estimate_displacement(a_basal_series(7.15, displacement=0.25))
    assert found.detected
    assert found.verified, "three orders, so the answer was checked against one of them"
    assert found.spread_after < RATIONAL_SERIES_SPREAD
