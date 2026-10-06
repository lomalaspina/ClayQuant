"""The one way of calling nnls_fit that is silently wrong.

The weights are 1/raw counting statistics, and `raw` is whatever was handed in.
A measurement whose background has already been removed, passed without the
background that removed it, therefore weights the flat regions between the peaks
- where only noise is left - far above the peaks.  nnls_fit's own comment said
why it must be the raw counts; nothing checked that it had them.

It cost a whole investigation.  On the sepiolite standard, fitted this way, the
110 came out 4 per cent explained, R_wp 87 per cent, and the specimen read as 65
per cent illite; with the raw counts and their background it is 86 per cent
explained, R_wp 39, and 66 per cent sepiolite.  Nothing about the model had been
wrong (Sec. A.70).
"""
import numpy as np
import pytest

from clayquant.background import clayfit_background
from clayquant.library import PatternLibrary
from clayquant.nnls import PRESUBTRACTED_ZERO_FRACTION, nnls_fit
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 34.0, 0.02)


def a_peak(centre, height, width=0.3):
    return height * np.exp(-0.5 * ((GRID - centre) / width) ** 2)


@pytest.fixture
def library():
    lib = PatternLibrary(two_theta=GRID)
    lib.add(Pattern(two_theta=GRID, intensity=a_peak(7.15, 1.0), name="sepiolite"),
            phase="sepiolite", strip_continuum=False)
    lib.add(Pattern(two_theta=GRID, intensity=a_peak(8.80, 1.0), name="illite"),
            phase="illite", strip_continuum=False)
    return lib


def a_measurement():
    """Counts on a sloping background, as a diffractometer records them."""
    background = 600.0 + 4000.0 * np.exp(-(GRID - 4.0) / 3.0)
    return Pattern(two_theta=GRID,
                   intensity=background + a_peak(7.15, 12000.0) + a_peak(8.80, 2000.0),
                   name="raw")


def test_the_guard_fires_on_a_pattern_whose_background_is_gone(library):
    raw = a_measurement()
    fit = clayfit_background(raw.two_theta, raw.intensity)
    subtracted = Pattern(two_theta=GRID, name="sub",
                         intensity=fit.subtract(raw.two_theta, raw.intensity))
    note = nnls_fit(subtracted, library,
                    range_two_theta=(4.0, 34.0)).metadata["presubtracted"]
    assert note
    assert "at or below zero" in note
    assert "not trustworthy" in note


def test_the_guard_is_silent_when_the_background_is_passed(library):
    raw = a_measurement()
    fit = clayfit_background(raw.two_theta, raw.intensity)
    result = nnls_fit(raw, library, background=fit, range_two_theta=(4.0, 34.0))
    assert result.metadata["presubtracted"] == ""


def test_the_guard_is_silent_on_raw_counts_with_no_background_asked_for(library):
    """A caller fitting counts without subtracting anything is not making this
    mistake, and must not be told they are."""
    result = nnls_fit(a_measurement(), library, range_two_theta=(4.0, 34.0))
    assert result.metadata["presubtracted"] == ""


def test_the_threshold_sits_between_raw_and_subtracted_patterns():
    """Measured on the standards: 0.00 per cent of raw points are at or below
    zero, against 5.9 to 14.0 per cent of subtracted ones."""
    assert PRESUBTRACTED_ZERO_FRACTION == pytest.approx(0.01)
    assert PRESUBTRACTED_ZERO_FRACTION < 0.059
    assert PRESUBTRACTED_ZERO_FRACTION > 0.0


def test_the_fit_still_runs_and_is_not_refused(library):
    """Reported rather than raised, which is the treatment `crowded` gets: a
    caller who means to accept the weighting may."""
    raw = a_measurement()
    fit = clayfit_background(raw.two_theta, raw.intensity)
    subtracted = Pattern(two_theta=GRID, name="sub",
                         intensity=fit.subtract(raw.two_theta, raw.intensity))
    result = nnls_fit(subtracted, library, range_two_theta=(4.0, 34.0))
    assert np.isfinite(result.r_wp)
    assert len(result.coefficients) == 2


def test_the_wrong_call_really_does_bias_against_the_peaks(library):
    """Not a style point: the misweighted fit explains much less of the strong
    peak than the correct one, which is what the warning claims."""
    raw = a_measurement()
    fit = clayfit_background(raw.two_theta, raw.intensity)
    subtracted = Pattern(two_theta=GRID, name="sub",
                         intensity=fit.subtract(raw.two_theta, raw.intensity))

    def explained(result):
        window = (result.two_theta >= 6.5) & (result.two_theta <= 7.8)
        return (float(np.sum(result.calculated[window]))
                / float(np.sum(result.observed[window])))

    right = explained(nnls_fit(raw, library, background=fit,
                               range_two_theta=(4.0, 34.0)))
    wrong = explained(nnls_fit(subtracted, library, range_two_theta=(4.0, 34.0)))
    assert right > 0.9
    assert wrong < right
