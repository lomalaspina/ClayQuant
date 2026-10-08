"""A background that does not reach the data hands the fit counts no mineral is under.

This is the fault that is invisible in a residual plot: running *above* the data
clips the reflections and an operator sees it at once, running *below* leaves a
floor standing under every point of a region and it looks like intensity.  On the
mount that raised it the default model left 165 to 240 counts under 20-34 deg,
and every reported number moved - the kaolinite bound from the heated mount went
from 82-92 to 97-99 per cent, the chlorite from 18.7 to 25.7 per cent of the
clay, the accessory minerals from 60 to 15 per cent of the weight.  None of that
is a fitting error and none of it is visible without measuring the background
against the pattern's own floor.

Three estimators were tried before this one and the first two are in the
docstrings because they look right and are not: the band minimum is biased low by
two to three sigma of counting noise, and the clipped mean counts diffuse
scattering as background and condemns every model including the good ones.
"""

import numpy as np
import pytest

from clayquant.background import (
    PEDESTAL_BAND,
    PEDESTAL_NOTICE,
    PEDESTAL_QUANTILE,
    PEDESTAL_SHARE,
    PEDESTAL_SIGMAS,
    background_pedestal,
    clayfit_background,
)

GRID = np.arange(4.0, 34.0, 0.02)
PEAKS = ((8.8, 4000.0, 0.08), (12.4, 900.0, 0.10), (26.6, 3000.0, 0.07))


def a_pattern(baseline, peaks=PEAKS, seed=7) -> np.ndarray:
    """Counts: a background, some reflections, and Poisson noise."""
    counts = np.asarray(baseline, dtype=float).copy()
    for centre, height, width in peaks:
        counts += height * np.exp(-0.5 * ((GRID - centre) / width) ** 2)
    return np.random.default_rng(seed).poisson(counts).astype(float)


def flat(level: float = 500.0) -> np.ndarray:
    return np.full(GRID.shape, float(level))


def a_hump(level: float = 500.0, height: float = 300.0) -> np.ndarray:
    return level + height * np.exp(-0.5 * ((GRID - 26.0) / 5.0) ** 2)


def falling(amplitude: float = 1400.0, floor: float = 300.0) -> np.ndarray:
    return amplitude * np.exp(-0.12 * GRID) + floor


def test_a_background_that_follows_the_floor_is_sound():
    truth = flat()
    found = background_pedestal(GRID, a_pattern(truth), truth)
    assert found.sound
    assert not found.found
    assert not found.over
    assert all(abs(band.excess) < PEDESTAL_SIGMAS * band.noise for band in found.bands)


def test_a_monotonic_model_over_a_humped_floor_is_caught_and_diagnosed():
    """The fault that raised this, in its simplest form, and the reason for it."""
    counts = a_pattern(a_hump())
    found = background_pedestal(GRID, counts, falling(), model="Exponential")
    assert found.found
    worst = found.worst
    assert worst is not None
    assert worst.low >= 20.0, "the hump is near 26 deg and that is where it must point"
    assert worst.excess > PEDESTAL_SIGMAS * worst.noise
    assert found.monotonic and found.floor_rise > 100.0
    assert "only ever falls" in found.status
    assert "no shape for" in found.status
    assert "Exponential" in found.status


def test_a_background_wrong_by_the_same_amount_everywhere_is_not_detected():
    """A known and deliberate blind spot, stated so nobody relies on the opposite.

    The floor of a real clay pattern stands above even a correct background,
    because part of what is there is diffuse scattering and no measurement
    separates it from background that was left behind.  Subtracting the median
    band is what makes the shape fault visible, and it is also what makes a
    uniform offset invisible: a model that is 200 counts low *everywhere* has the
    same profile as one that is right, and this cannot tell them apart.

    That is the rarer fault and the less damaging one - it inflates everything
    rather than distorting one region against another - but it is not caught, and
    a model that passes here has not been certified correct.
    """
    truth = flat()
    counts = a_pattern(truth)
    found = background_pedestal(GRID, counts, truth - 200.0)
    assert not found.found
    assert all(abs(band.excess) < PEDESTAL_SIGMAS * band.noise for band in found.bands)
    # The raw floor still carries it, for anyone who wants to look.
    assert all(band.gap == pytest.approx(200.0, abs=25.0) for band in found.bands)


def test_the_raw_floor_is_unbiased_where_the_band_minimum_would_not_be():
    """The estimator, pinned.  The minimum of a band is not its floor.

    A few hundred Poisson samples have their minimum two to three and a half
    sigma below the mean, so measuring the floor that way reports a correct
    background as 40 to 80 counts too low and a 200-count pedestal as about 130.
    The quantile has an exact Gaussian offset instead, whatever the sample size.
    """
    truth = flat()
    counts = a_pattern(truth)
    exact = background_pedestal(GRID, counts, truth)
    assert all(abs(band.gap) < 15.0 for band in exact.bands), (
        "a correct background must read as a floor of zero, not of minus sixty"
    )
    low = background_pedestal(GRID, counts, truth - 200.0)
    assert all(band.gap == pytest.approx(200.0, abs=25.0) for band in low.bands)
    # And the minimum-based reading, for the record: biased by several sigma.
    for lo in np.arange(4.0, 32.0, PEDESTAL_BAND):
        here = (GRID >= lo) & (GRID <= lo + PEDESTAL_BAND)
        by_minimum = float(np.min(counts[here] - truth[here]))
        assert by_minimum < -30.0


def test_a_background_above_the_data_is_reported_as_clipping():
    truth = flat()
    counts = a_pattern(truth)
    found = background_pedestal(GRID, counts, truth + 120.0)
    assert found.over, "a background above the data destroys counts and must be said"
    assert not found.sound
    deepest = min(found.over, key=lambda band: band.sigmas)
    assert deepest.gap == pytest.approx(-120.0, abs=25.0)
    assert "taken away rather than fitted" in found.status


def test_half_a_correct_band_lies_below_its_own_background():
    """Why the clipped fraction is reported and never tested.

    Noise about a curve puts half the points under it.  An earlier version made
    'a quarter of the band subtracted to zero' the test for over-subtraction,
    which fires on every background that is right.
    """
    truth = flat()
    found = background_pedestal(GRID, a_pattern(truth), truth)
    assert all(0.3 < band.clipped < 0.7 for band in found.bands)
    assert found.sound


def test_both_thresholds_have_to_be_crossed():
    truth = flat()
    # Significant against the noise, and a small part of a band full of reflection.
    counts = a_pattern(truth, peaks=((20.0, 400000.0, 1.5),))
    model = truth.copy()
    model[(GRID >= 18.0) & (GRID <= 22.0)] -= 150.0
    found = background_pedestal(GRID, counts, model)
    band = next(b for b in found.bands if b.low <= 20.0 < b.high)
    assert band.excess > PEDESTAL_SIGMAS * band.noise
    assert band.share < PEDESTAL_SHARE
    assert not band.flagged


def test_a_thin_fault_that_passes_every_band_is_still_noticed():
    counts = a_pattern(flat())
    model = flat().copy()
    model[GRID >= 24.0] -= 70.0
    found = background_pedestal(GRID, counts, model)
    assert found.share >= PEDESTAL_NOTICE or found.found
    assert not found.sound


def test_the_range_restricts_what_is_measured():
    counts = a_pattern(flat())
    model = flat().copy()
    model[GRID < 10.0] -= 400.0
    whole = background_pedestal(GRID, counts, model)
    above = background_pedestal(GRID, counts, model, range_two_theta=(12.0, 34.0))
    assert whole.found
    assert not above.found, "the fitted range is what the fit sees, and all it sees"
    assert above.bands[0].low >= 12.0


def test_a_short_remainder_is_absorbed_rather_than_reported():
    """Half a degree's worth of points does not hold a floor."""
    grid = np.arange(4.0, 4.0 + 2.5 * PEDESTAL_BAND, 0.02)
    counts = np.full(grid.shape, 500.0)
    found = background_pedestal(grid, counts, np.full(grid.shape, 500.0))
    assert len(found.bands) == 2
    assert found.bands[-1].high - found.bands[-1].low > PEDESTAL_BAND


def test_it_takes_a_callable_or_an_array():
    counts = a_pattern(flat())
    fit = clayfit_background(GRID, counts)
    assert background_pedestal(GRID, counts, fit).bands
    assert background_pedestal(GRID, counts, fit(GRID)).bands


def test_the_quantile_is_low_enough_to_stay_under_the_reflections():
    assert 0.0 < PEDESTAL_QUANTILE <= 0.10, (
        "above a tenth the floor starts to climb into the weak reflections; at or "
        "below zero it is the minimum again, with the bias that carries"
    )


def test_it_refuses_what_it_cannot_measure():
    counts = a_pattern(flat())
    with pytest.raises(ValueError, match="same shape"):
        background_pedestal(GRID, counts[:-1], flat())
    with pytest.raises(ValueError, match="same points"):
        background_pedestal(GRID, counts, flat()[:-1])
    with pytest.raises(ValueError, match="band width"):
        background_pedestal(GRID, counts, flat(), band=0.0)


def test_the_panel_colours_by_what_it_found():
    """The Background tab shows the measurement whatever it says, and colours it.

    Red where a band is flagged, because that is the fit being handed counts no
    mineral is under; amber where no band crosses both thresholds and enough of
    the pattern is pedestal to matter; quiet where the curve is sound.  Shown in
    all three cases because the operator is choosing between six models and this
    is the number that separates them.
    """
    from clayquant.gui.app import pedestal_panel
    from clayquant.pattern import Pattern

    counts = a_pattern(a_hump())
    pattern = Pattern(GRID, counts, name="glycol")

    red = pedestal_panel(pattern, lambda x: falling())
    assert red.style["border"].endswith("#d9534f")
    assert "only ever falls" in red.children[0].children

    quiet = pedestal_panel(pattern, lambda x: a_hump())
    assert quiet.style["border"].endswith("#cfe0cf")

    # Per-band counts are always listed, so two models can be compared by eye.
    assert "Counts above the pattern's own floor" in quiet.children[1].children


def test_the_panel_never_breaks_the_view():
    """A guard that can raise is a guard that takes the tab down with it."""
    from clayquant.gui.app import pedestal_panel
    from clayquant.pattern import Pattern

    def explode(_angles):
        raise RuntimeError("no background today")

    panel = pedestal_panel(Pattern(GRID, a_pattern(flat()), name="glycol"), explode)
    assert "could not be measured" in panel.children
