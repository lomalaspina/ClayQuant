"""The load-time triplet screen, as Clayfit does it.

What has to hold is that the accompanying minerals can be identified from the
three raw scans alone - before the zero error is set, before a background is
chosen and without a reference library - because that is where Clayfit puts the
step and it is a much more useful place for it.  Two things make it possible and
both are tested here: each mount is aligned on its own quartz lines, and the
test is which peaks the three treatments share rather than how well one of them
can be explained.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from clayquant.crystal import AtomSite, Crystal
from clayquant.detection import (
    CLAYFIT_MATCH_TOLERANCE,
    CLAYFIT_MAX_ZERO_SHIFT,
    QUARTZ_CALIBRATION,
    TreatmentPeaks,
    evidence_score,
    match_across_treatments,
    quartz_zero_shift,
    screen_treatments,
    treatment_peaks,
)
from clayquant.pattern import Pattern

STEP = 0.02
GRID = np.arange(3.0, 40.0, STEP)


def gaussian(centre: float, height: float, width: float = 0.09) -> np.ndarray:
    return height * np.exp(-0.5 * ((GRID - centre) / (width / 2.355)) ** 2)


def synthetic_mount(
    lines: list[tuple[float, float]],
    clay_at: float,
    shift: float = 0.0,
    seed: int = 1,
) -> Pattern:
    """One mount: fixed accompanying lines, one clay line that moves, noise.

    ``shift`` displaces every measured angle, which is what a zero error and a
    specimen displacement do and what the screen has to remove for itself.
    """
    rng = np.random.default_rng(seed)
    intensity = 150.0 + 3000.0 / GRID ** 1.5
    for centre, height in lines:
        intensity = intensity + gaussian(centre - shift, height)
    intensity = intensity + gaussian(clay_at - shift, 9000.0, width=0.30)
    return Pattern(two_theta=GRID.copy(),
                   intensity=intensity + rng.normal(0.0, np.sqrt(intensity)),
                   name="synthetic")


QUARTZ_LINES = [(QUARTZ_CALIBRATION[0], 1800.0), (QUARTZ_CALIBRATION[1], 9000.0)]


def triplet(shifts=(0.0, 0.0, 0.0), extra=()):
    """Three mounts whose clay peak moves and whose quartz does not."""
    lines = QUARTZ_LINES + list(extra)
    return {
        "air": synthetic_mount(lines, 8.8, shifts[0], seed=1),
        "glycol": synthetic_mount(lines, 5.2, shifts[1], seed=2),
        "heated": synthetic_mount(lines, 8.85, shifts[2], seed=3),
    }


# --------------------------------------------------------------------------
# Peaks and the quartz alignment
# --------------------------------------------------------------------------

def test_the_peaks_of_a_scan_are_found_without_a_background_being_chosen():
    peaks = treatment_peaks(synthetic_mount(QUARTZ_LINES, 8.8))
    assert len(peaks) >= 3
    for reference in QUARTZ_CALIBRATION:
        assert np.min(np.abs(peaks.two_theta - reference)) < 0.05
    assert np.all(peaks.prominence > 0.0)
    assert np.all(peaks.width > 0.0)


def test_the_shift_of_one_scan_is_measured_from_its_own_quartz_pair():
    for imposed in (0.0, 0.07, -0.12, 0.31, -0.42):
        peaks = treatment_peaks(synthetic_mount(QUARTZ_LINES, 8.8, shift=imposed))
        shift, indices = quartz_zero_shift(peaks)
        assert shift is not None
        assert len(indices) == 2, imposed
        # The screen returns the correction, so it comes back with the sign that
        # puts the measured angles back where they belong.
        assert shift == pytest.approx(imposed, abs=0.03)


def test_a_shift_beyond_the_search_window_is_not_invented():
    peaks = treatment_peaks(synthetic_mount(QUARTZ_LINES, 8.8,
                                            shift=CLAYFIT_MAX_ZERO_SHIFT + 0.3))
    shift, indices = quartz_zero_shift(peaks)
    assert shift is None or len(indices) < 2


def test_a_pair_whose_two_lines_disagree_is_rejected():
    """Two unrelated peaks near the quartz angles are not the quartz doublet."""
    peaks = TreatmentPeaks(
        two_theta=np.array([QUARTZ_CALIBRATION[0] - 0.40,
                            QUARTZ_CALIBRATION[1] + 0.40]),
        prominence=np.array([1000.0, 1000.0]),
        width=np.array([0.1, 0.1]),
    )
    shift, indices = quartz_zero_shift(peaks)
    # They disagree by 0.8 deg, so no pair is usable and the low line is
    # returned alone, as a starting value.
    assert len(indices) <= 1


def test_no_quartz_at_all_gives_no_shift():
    peaks = TreatmentPeaks(np.array([7.0, 12.0]), np.array([10.0, 10.0]),
                           np.array([0.1, 0.1]))
    assert quartz_zero_shift(peaks) == (None, ())


# --------------------------------------------------------------------------
# Matching a phase across the treatments
# --------------------------------------------------------------------------

def peaks_at(angles, prominence=1000.0) -> TreatmentPeaks:
    angles = np.asarray(angles, dtype=float)
    return TreatmentPeaks(angles, np.full(angles.size, prominence),
                          np.full(angles.size, 0.1))


def test_a_line_counts_only_when_every_mount_has_it():
    positions = np.array([10.0, 20.0, 30.0])
    heights = np.array([1.0, 0.5, 0.5])
    everywhere = {name: peaks_at([10.0, 20.0, 30.0]) for name in ("a", "b", "c")}
    shifts = {name: 0.0 for name in everywhere}
    assert match_across_treatments("P", positions, heights, everywhere, shifts).score \
        == pytest.approx(1.0)

    # Take the 30 deg line out of one mount and the phase loses it, however
    # well the other two agree.
    partial = dict(everywhere, c=peaks_at([10.0, 20.0]))
    evidence = match_across_treatments("P", positions, heights, partial, shifts)
    assert evidence.n_matched == 2
    assert evidence.score == pytest.approx(0.75)


def test_each_measured_peak_is_claimed_once():
    """Otherwise one strong peak answers every line that lies near it."""
    positions = np.array([20.00, 20.05, 20.09])
    heights = np.array([1.0, 1.0, 1.0])
    mounts = {name: peaks_at([20.02]) for name in ("a", "b")}
    shifts = {name: 0.0 for name in mounts}
    # The three lines cluster within 0.10 deg, so they are merged to one; if
    # they were not, one measured peak would answer all three.
    evidence = match_across_treatments("P", positions, heights, mounts, shifts)
    assert evidence.n_expected == 1
    assert evidence.n_matched == 1


def test_a_second_line_cannot_reuse_the_peak_the_first_took():
    positions = np.array([20.00, 20.20])
    heights = np.array([1.0, 1.0])
    mounts = {name: peaks_at([20.10]) for name in ("a", "b")}
    shifts = {name: 0.0 for name in mounts}
    evidence = match_across_treatments("P", positions, heights, mounts, shifts,
                                       tolerance=0.15)
    assert evidence.n_matched == 1


def test_the_mounts_own_shifts_are_applied_before_matching():
    positions = np.array([10.0, 20.0])
    heights = np.array([1.0, 1.0])
    mounts = {"a": peaks_at([10.3, 20.3]), "b": peaks_at([9.7, 19.7])}
    shifts = {"a": -0.3, "b": 0.3}
    assert match_across_treatments("P", positions, heights, mounts, shifts).score \
        == pytest.approx(1.0)
    # Without them the same peaks are half a degree from the lines and nothing
    # matches, which is the state the screen would be in before the zero error.
    assert match_across_treatments("P", positions, heights, mounts,
                                   {"a": 0.0, "b": 0.0}).n_matched == 0


def test_weak_lines_are_left_out_of_both_sides_of_the_coverage():
    positions = np.array([10.0, 20.0, 30.0])
    heights = np.array([1.0, 1.0, 0.01])
    mounts = {name: peaks_at([10.0, 20.0]) for name in ("a", "b")}
    shifts = {name: 0.0 for name in mounts}
    evidence = match_across_treatments("P", positions, heights, mounts, shifts)
    # The 1% line is below the floor, so its absence is not held against the
    # phase and its intensity is not in the denominator either.
    assert evidence.n_expected == 2
    assert evidence.score == pytest.approx(1.0)


def test_a_coherent_cell_scaling_moves_every_line_together():
    positions = np.array([10.0, 20.0, 30.0])
    heights = np.array([1.0, 1.0, 1.0])
    scale = 1.01
    moved = [2.0 * math.degrees(math.asin(scale * math.sin(math.radians(p / 2.0))))
             for p in positions]
    mounts = {name: peaks_at(moved) for name in ("a", "b")}
    shifts = {name: 0.0 for name in mounts}
    assert match_across_treatments("P", positions, heights, mounts, shifts).n_matched < 3
    assert match_across_treatments("P", positions, heights, mounts, shifts,
                                   scale=scale).score == pytest.approx(1.0)


def test_the_ranking_rewards_resting_on_several_lines():
    one = match_across_treatments(
        "one", np.array([20.0]), np.array([1.0]),
        {"a": peaks_at([20.0]), "b": peaks_at([20.0])}, {"a": 0.0, "b": 0.0})
    four = match_across_treatments(
        "four", np.array([10.0, 20.0, 30.0, 35.0]), np.ones(4),
        {"a": peaks_at([10.0, 20.0, 30.0, 35.0]),
         "b": peaks_at([10.0, 20.0, 30.0, 35.0])}, {"a": 0.0, "b": 0.0})
    assert one.score == pytest.approx(four.score)          # coverage cannot tell them apart
    assert evidence_score(four) > evidence_score(one)      # the score can
    assert evidence_score(four) == pytest.approx(2.0 * evidence_score(one))


# --------------------------------------------------------------------------
# The screen itself
# --------------------------------------------------------------------------

def quartz_like() -> Crystal:
    """The quartz cell, whose 100 and 101 are the calibration pair."""
    return Crystal(
        a=4.9134, b=4.9134, c=5.4052,
        alpha=90.0, beta=90.0, gamma=120.0,
        sites=[AtomSite("Si", 0.4697, 0.0, 1.0 / 3.0),
               AtomSite("O", 0.4135, 0.2669, 0.1191)],
        symops=["x, y, z", "-y, x-y, z+1/3", "y-x, -x, z+2/3",
                "y, x, -z", "x-y, -y, -z+2/3", "-x, y-x, -z+1/3"],
        name="Quartz",
    )


def test_the_screen_finds_a_phase_on_uncalibrated_scans():
    crystal = quartz_like()
    screen = screen_treatments(triplet(shifts=(0.09, -0.07, 0.15)), {"Quartz": crystal})
    assert screen.calibrated
    assert [evidence.name for evidence in screen.findings][:1] == ["Quartz"]
    # Each mount is aligned on itself, so the three shifts come back separately
    # and near what was imposed.
    assert screen.shifts["air"] == pytest.approx(0.09, abs=0.04)
    assert screen.shifts["glycol"] == pytest.approx(-0.07, abs=0.04)
    assert screen.shifts["heated"] == pytest.approx(0.15, abs=0.04)


def test_the_clay_peak_that_moves_is_not_reported_as_a_stable_phase():
    """The whole basis of the test: the clays move and nothing else does."""
    screen = screen_treatments(triplet(), {"Quartz": quartz_like()})
    for evidence in screen.findings:
        matched = [match.expected_two_theta for match in evidence.matches if match.matched]
        # 5.2 and 8.8 deg are where the synthetic clay sits in the glycol and
        # air-dried mounts; nothing may be matched there.
        assert not any(abs(angle - 5.2) < 0.3 or abs(angle - 8.8) < 0.3
                       for angle in matched), evidence.name


def test_a_mount_whose_alignment_disagrees_is_reported():
    screen = screen_treatments(triplet(shifts=(0.02, 0.02, 0.45)),
                               {"Quartz": quartz_like()})
    assert "outlier" in screen.note
    assert "heated" in screen.note


def test_without_quartz_the_screen_declines_rather_than_guesses():
    mounts = {
        name: synthetic_mount([(15.0, 4000.0), (31.0, 2000.0)], clay, seed=seed)
        for name, clay, seed in (("air", 8.8, 1), ("glycol", 5.2, 2), ("heated", 8.85, 3))
    }
    screen = screen_treatments(mounts, {"Quartz": quartz_like()})
    assert screen.findings == []
    assert "Quartz was not found" in screen.note
    assert not screen.calibrated


def test_a_flat_pattern_yields_nothing_rather_than_raising():
    flat = Pattern(two_theta=GRID.copy(), intensity=np.full(GRID.size, 100.0),
                   name="flat")
    screen = screen_treatments({"air": flat, "glycol": flat}, {"Quartz": quartz_like()})
    assert screen.findings == []
    assert screen.note


def test_the_quartz_width_comes_back_as_a_starting_value():
    screen = screen_treatments(triplet(), {"Quartz": quartz_like()})
    # The synthetic lines are 0.09 deg wide; the percentile smoothing broadens
    # them a little, so this is a bound rather than an equality.
    assert 0.05 < screen.width < 0.25


def test_clays_are_left_to_the_clay_library():
    crystals = {"Quartz": quartz_like()}
    screen = screen_treatments(triplet(), crystals, include_clays=False)
    assert all(not evidence.is_clay for evidence in screen.findings)


def test_the_tolerances_are_clayfits():
    assert CLAYFIT_MATCH_TOLERANCE == pytest.approx(0.11)
    assert CLAYFIT_MAX_ZERO_SHIFT == pytest.approx(0.50)
    assert QUARTZ_CALIBRATION[0] == pytest.approx(20.86, abs=0.01)
    assert QUARTZ_CALIBRATION[1] == pytest.approx(26.64, abs=0.01)
