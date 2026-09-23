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

from clayquant.calibration import CU_KA1, reference_two_theta
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

# A clay separate with no quartz in it whatever.  Kaolinite alone puts two lines
# inside the half-degree window around the 20.86 deg quartz 100 - the 4.36 A
# band and the 4.18 A line - and one inside the window around the 26.64 deg
# quartz 101; the illite/smectite adds a third order near 3.5 A.  Nothing here
# is quartz, so nothing here may be calibrated on.
KAOLINITE_1M = [(4.46, 1200.0), (4.36, 1500.0), (4.18, 1800.0), (3.84, 700.0),
                (3.57, 6000.0), (3.37, 900.0), (2.56, 900.0), (2.49, 800.0)]
ILLITE_SMECTITE = [(5.25, 1500.0), (4.48, 600.0), (3.50, 2500.0)]


def at(d: float) -> float:
    """Where a d-spacing falls, in degrees."""
    return reference_two_theta(d, CU_KA1)


def clay_only_mount(d001: float = 10.5, seed: int = 11, shift: float = 0.0) -> Pattern:
    """One mount of a quartz-free specimen: kaolinite, and a clay that moves."""
    lines = [(at(d), height) for d, height in KAOLINITE_1M + ILLITE_SMECTITE]
    return synthetic_mount(lines, at(d001), shift=shift, seed=seed)


def glycol_smectite_mount(seed: int = 21) -> Pattern:
    """A glycolated smectite, whose 17 A series puts its 004 at 4.25 A.

    Within 0.01 A of quartz 100, so that order sits on top of the 20.86 deg
    calibration line: the tidiest way for a specimen with no quartz in it to
    look as though it had some.
    """
    lines = [(at(17.0 / order), height)
             for order, height in ((2, 1500.0), (3, 2500.0), (4, 1200.0), (5, 900.0))]
    return synthetic_mount(lines, at(17.0), seed=seed)


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
        zero = quartz_zero_shift(peaks)
        assert zero.shift is not None
        assert len(zero.used) == 2, imposed
        # Two lines agreed, so this one may be applied without asking.
        assert zero.confirmed
        # The screen returns the correction, so it comes back with the sign that
        # puts the measured angles back where they belong.
        assert zero.shift == pytest.approx(imposed, abs=0.03)


def test_a_shift_beyond_the_search_window_is_not_invented():
    peaks = treatment_peaks(synthetic_mount(QUARTZ_LINES, 8.8,
                                            shift=CLAYFIT_MAX_ZERO_SHIFT + 0.3))
    zero = quartz_zero_shift(peaks)
    assert not zero.confirmed


def test_a_pair_whose_two_lines_disagree_is_rejected():
    """Two unrelated peaks near the quartz angles are not the quartz doublet."""
    peaks = TreatmentPeaks(
        two_theta=np.array([QUARTZ_CALIBRATION[0] - 0.40,
                            QUARTZ_CALIBRATION[1] + 0.40]),
        prominence=np.array([1000.0, 1000.0]),
        width=np.array([0.1, 0.1]),
    )
    zero = quartz_zero_shift(peaks)
    # They disagree by 0.8 deg, so no pair is usable.  The low line is still
    # reported, because the operator may know the specimen has quartz in it,
    # but it is not confirmed and no caller may apply it unasked.
    assert not zero.confirmed
    assert len(zero.used) <= 1


def test_no_quartz_at_all_gives_no_shift():
    peaks = TreatmentPeaks(np.array([7.0, 12.0]), np.array([10.0, 10.0]),
                           np.array([0.1, 0.1]))
    zero = quartz_zero_shift(peaks)
    assert (zero.shift, zero.used, zero.confirmed) == (None, (), False)
    assert zero.note


def test_a_specimen_with_no_quartz_in_it_gets_no_confident_shift():
    """The failure this guards against, on a pattern built without quartz.

    Kaolinite puts its 4.18 A line 0.4 deg from the quartz 100, and read as
    quartz that is a zero shift of -0.38 deg on a specimen that has nothing to
    calibrate on.  A wrong shift does not fail loudly - it quietly misaligns
    whatever it is applied to - so what matters is that nothing here comes back
    confirmed.
    """
    zero = quartz_zero_shift(treatment_peaks(clay_only_mount()))
    assert not zero.confirmed
    assert len(zero.used) <= 1
    # It rested on one peak, and that peak is kaolinite's.
    assert zero.shift == pytest.approx(at(4.255) - at(4.18), abs=0.05)
    # The 26.64 deg window holds kaolinite's 3.37 A line and it agrees with
    # nothing, which is the evidence against quartz: the 101 is the stronger of
    # the two reflections, so a specimen showing the 100 has to show it.
    assert "none agrees with it" in zero.note


def test_a_lone_line_is_reported_rather_than_discarded():
    """A one-line match is worth showing; it is not worth applying.

    The operator may know the specimen has quartz in it and that the 101 fell
    outside the scan, and then the number is usable - by them, deliberately.
    """
    zero = quartz_zero_shift(treatment_peaks(clay_only_mount()))
    assert zero.shift is not None
    assert not zero.confirmed
    assert zero.note


def test_a_basal_order_that_lands_on_the_quartz_line_is_named_as_one():
    """The scan explains the peak itself: it is the 004 of its own 17 A series."""
    zero = quartz_zero_shift(treatment_peaks(glycol_smectite_mount()))
    assert not zero.confirmed
    assert "basal series" in zero.note
    assert "order 4" in zero.note


def test_quartz_among_crowding_clay_lines_is_still_confirmed():
    """The test has to be strong enough to refuse, and no stronger.

    The same quartz-free pattern with quartz added: two lines are there, they
    agree, and the clay lines in both windows do not stop them being found.
    """
    lines = [(at(d), height) for d, height in KAOLINITE_1M + ILLITE_SMECTITE]
    mount = synthetic_mount(lines + QUARTZ_LINES, at(10.5), shift=0.12, seed=11)
    zero = quartz_zero_shift(treatment_peaks(mount))
    assert zero.confirmed
    assert len(zero.used) == 2
    assert zero.shift == pytest.approx(0.12, abs=0.03)


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
    assert "No quartz pair was found" in screen.note
    assert not screen.calibrated


def test_a_quartz_free_specimen_is_declined_by_the_screen_as_well():
    """The screen is built on the peaks the mounts share, so it needs the shift.

    A shift invented from a kaolinite line would move one mount's peaks a few
    tenths of a degree away from the others' and corrupt exactly the comparison
    the screen is.  It declines, and says what it refused, so the operator can
    set the zero error themselves if they know better.
    """
    mounts = {"air": clay_only_mount(10.5, seed=11),
              "glycol": clay_only_mount(17.0, seed=12),
              "heated": clay_only_mount(10.0, seed=13)}
    screen = screen_treatments(mounts, {"Quartz": quartz_like()})
    assert screen.findings == []
    assert screen.shifts == {}
    assert not screen.calibrated
    assert sorted(screen.unconfirmed) == ["air", "glycol", "heated"]
    assert all(value == pytest.approx(-0.38, abs=0.05)
               for value in screen.unconfirmed.values())
    assert "not applied" in screen.note


def test_a_mount_whose_shift_is_not_confirmed_is_left_out_and_named():
    """One mount of the three has no quartz; the other two keep working."""
    mounts = {"air": synthetic_mount(QUARTZ_LINES, 8.8, 0.09, seed=1),
              "glycol": synthetic_mount(QUARTZ_LINES, 5.2, -0.07, seed=2),
              "heated": clay_only_mount(10.0, seed=13)}
    screen = screen_treatments(mounts, {"Quartz": quartz_like()})
    assert sorted(screen.shifts) == ["air", "glycol"]
    assert sorted(screen.unconfirmed) == ["heated"]
    assert not screen.calibrated
    assert "heated" in screen.note
    # The quartz that really is in the other two is still found on them.
    assert [evidence.name for evidence in screen.findings][:1] == ["Quartz"]


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
