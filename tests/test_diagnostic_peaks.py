"""Making an expandable clay justify the reflection that identifies it.

A non-negative fit will take a little of a broad mixed-layer pattern because it
improves a strong reflection the pattern overlaps, and pay for it by predicting
a low-angle reflection the measurement does not contain.  That reflection is the
whole evidence for the expandable clay, so a fit which predicts it without its
being there has reported a phase it has not seen.  These tests pin the screen
that catches it and, as important, that it leaves a real one alone.

The background here is *known* rather than estimated.  That is deliberate: a
background left a few per cent high puts a pedestal under the diagnostic window
which a matched filter reads as support, and these tests are about the decision,
not about the background estimator.  What it costs is worth saying out loud -
the screen is only as good as the background beneath the diagnostic band.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import DIAGNOSTIC_WINDOWS, nnls_fit, screen_diagnostic_peaks
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 30.0, 0.02)
BACKGROUND = 400.0
RANGE = (4.0, 29.0)


def peak(centre: float, height: float, width: float = 0.12) -> np.ndarray:
    return height * np.exp(-4.0 * np.log(2.0) * ((GRID - centre) / width) ** 2)


@dataclass
class KnownBackground:
    """The background the counts were generated with, subtracted exactly."""

    level: float

    def subtract(self, two_theta, intensity):
        return np.asarray(intensity, dtype=float) - self.level


ILLITE = peak(8.85, 1.0) + peak(17.8, 0.45) + peak(26.8, 0.30)
DIAGNOSTIC = peak(9.60, 0.07)
EXPANDABLE = ILLITE + peak(12.40, 0.60) + DIAGNOSTIC


def library() -> PatternLibrary:
    """An illite, and an I/S which is it plus a strong 12.4 peak and its order.

    This is the failure mode rather than a contrived one.  The fit needs the
    expandable pattern for the 12.4 deg reflection, which is strong and which
    nothing else in the library has; the weak diagnostic order at 9.6 deg comes
    along as a prediction, and the residual is too pleased about the strong peak
    to object to the weak one.
    """
    return PatternLibrary(
        two_theta=GRID,
        entries=[
            LibraryEntry(
                name="I/S 1.00/0.00", phase="I/S", intensity=ILLITE / ILLITE.max(),
                march_dollase=0.3, fraction=1.0, unit_mass=100.0, unit_volume=100.0,
            ),
            LibraryEntry(
                name="I/S 0.90/0.10", phase="I/S",
                intensity=EXPANDABLE / EXPANDABLE.max(),
                march_dollase=0.3, fraction=0.90, unit_mass=100.0, unit_volume=100.0,
            ),
        ],
    )


def measurement(signal: np.ndarray, seed: int = 20260923) -> Pattern:
    """Poisson counts on signal plus background, as the instrument records them."""
    rng = np.random.default_rng(seed)
    return Pattern(
        two_theta=GRID,
        intensity=rng.poisson(signal + BACKGROUND).astype(float),
        name="mount",
    )


def without_expandable_layers() -> np.ndarray:
    """Everything the expandable entry has except the order that identifies it."""
    return 900.0 * (ILLITE + peak(12.40, 0.60))


def with_expandable_layers() -> np.ndarray:
    return 900.0 * EXPANDABLE


def screen(pattern, lib, **kwargs):
    kwargs.setdefault("background", KnownBackground(BACKGROUND))
    return screen_diagnostic_peaks(pattern, lib, range_two_theta=RANGE, **kwargs)


def fit_of(pattern, lib):
    return nnls_fit(
        pattern, lib, background=KnownBackground(BACKGROUND), range_two_theta=RANGE
    )


def test_the_fit_does_reach_for_an_expandable_clay_it_should_not():
    """The behaviour the screen exists for, shown before it is screened."""
    lib = library()
    fit = fit_of(measurement(without_expandable_layers()), lib)
    assert fit.coefficients[1] > 0.0


def test_an_expandable_clay_the_measurement_does_not_show_is_removed():
    lib = library()
    pattern = measurement(without_expandable_layers())
    kept = screen(pattern, lib, result=fit_of(pattern, lib))
    assert [item.name for item in kept.rejected] == ["I/S 0.90/0.10"]
    assert kept.kept.tolist() == [0]
    assert "removed" in kept.status
    rejection = kept.rejected[0]
    assert rejection.position == pytest.approx(9.60, abs=0.05)
    assert rejection.supported < rejection.required


def test_a_real_expandable_clay_is_kept():
    lib = library()
    kept = screen(measurement(with_expandable_layers()), lib)
    assert kept.rejected == ()
    assert kept.kept.tolist() == [0, 1]
    assert "supported by its own diagnostic" in kept.status


def test_the_screened_fit_is_the_one_returned():
    lib = library()
    kept = screen(measurement(without_expandable_layers()), lib)
    assert kept.result.names == ["I/S 1.00/0.00"]
    assert kept.result.coefficients.size == 1


def test_an_end_member_is_never_screened():
    """Fraction 1.0 carries no expandable layers, so it has nothing to justify."""
    kept = screen(measurement(without_expandable_layers()), library())
    assert all(item.name != "I/S 1.00/0.00" for item in kept.rejected)


def test_a_phase_with_no_window_is_left_alone():
    lib = library()
    for entry in lib.entries:
        entry.phase = "C/S"
    kept = screen(measurement(without_expandable_layers()), lib)
    assert kept.rejected == ()
    assert kept.kept.tolist() == [0, 1]


def test_the_diagnostic_is_what_the_host_cannot_produce():
    """Not the entry's strongest peak in the window, which is the host's."""
    from clayquant.nnls import _diagnostic_peak, _host_excess

    lib = library()
    entry = lib.entries[1]
    excess = _host_excess(GRID, entry.intensity, lib, entry)
    index = _diagnostic_peak(GRID, excess, DIAGNOSTIC_WINDOWS["I/S"])
    assert GRID[index] == pytest.approx(9.60, abs=0.03)
    # The entry's own maximum in that window is the illite-like 001, which the
    # host explains and which would therefore reject every entry.
    window = (GRID >= 8.75) & (GRID <= 10.40)
    assert GRID[window][int(np.argmax(entry.intensity[window]))] == pytest.approx(
        8.85, abs=0.03
    )


def test_the_windows_name_only_the_phases_with_a_band():
    assert set(DIAGNOSTIC_WINDOWS) == {"I/S", "smectite_EG"}
    for low, high in DIAGNOSTIC_WINDOWS.values():
        assert low < high


def test_a_supplied_fit_is_reused():
    lib = library()
    pattern = measurement(with_expandable_layers())
    first = fit_of(pattern, lib)
    assert screen(pattern, lib, result=first).result is first


def test_bad_thresholds_are_refused():
    lib = library()
    pattern = measurement(without_expandable_layers())
    with pytest.raises(ValueError):
        screen(pattern, lib, half_width=0.0)
    with pytest.raises(ValueError):
        screen(pattern, lib, minimum_support=-1.0)
