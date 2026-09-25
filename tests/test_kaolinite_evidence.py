"""Kaolinite measured against the chlorite's own reflections, without heating.

Both windows kaolinite occupies are shared with chlorite - its 001 at 12.36 deg
against the chlorite 002 at 12.46, its 002 at 24.85 against the chlorite 004 at
25.06 - so neither can be read alone.  What can be read alone is the chlorite 001
at 6.22 deg and its 003 at 18.73, which kaolinite does not have.  Each of those
times a ratio measured on chlorite standards is the chlorite's share of an
overlapped window, and the rest is kaolinite.
"""

import numpy as np
import pytest

from clayquant.diagnostics import (
    CHLORITE_RATIOS,
    kaolinite_evidence,
)
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 34.0, 0.0167)
BACKGROUND = 400.0
WAVELENGTH = 1.540596


def angle(d: float) -> float:
    return 2.0 * np.degrees(np.arcsin(WAVELENGTH / (2.0 * d)))


def peak(d: float, height: float, width: float = 0.14) -> np.ndarray:
    return height * np.exp(-4.0 * np.log(2.0) * ((GRID - angle(d)) / width) ** 2)


def chlorite(strength: float = 4000.0) -> np.ndarray:
    """A chlorite basal series at the ratios the standards measure."""
    return (
        peak(14.20, strength)
        + peak(7.10, strength * 2.0)
        + peak(4.733, strength * 1.05)
        + peak(3.550, strength * 1.4)
    )


def kaolinite(strength: float = 4000.0) -> np.ndarray:
    return peak(7.16, strength) + peak(3.58, strength * 0.5)


def scan(signal, seed: int = 20260925) -> Pattern:
    rng = np.random.default_rng(seed)
    values = np.zeros_like(GRID) + signal
    return Pattern(
        two_theta=GRID,
        intensity=rng.poisson(values + BACKGROUND).astype(float),
        name="mount",
    )


def test_a_pure_chlorite_is_not_called_kaolinite():
    evidence = kaolinite_evidence(scan(chlorite()))
    least, most = evidence.bounds
    assert least == 0.0
    assert not evidence.detected


def test_a_pure_kaolinite_is_all_kaolinite():
    """No chlorite reflection at all, so nothing in the window is chlorite."""
    evidence = kaolinite_evidence(scan(kaolinite()))
    least, most = evidence.bounds
    assert least > 0.9
    assert evidence.detected
    assert all(share.limited for share in evidence.usable)


def test_kaolinite_over_a_chlorite_is_established():
    evidence = kaolinite_evidence(scan(chlorite(2000.0) + kaolinite(6000.0)))
    least, _ = evidence.bounds
    assert evidence.detected
    assert least > 0.1


def test_an_empty_pattern_gives_no_estimate():
    evidence = kaolinite_evidence(scan(0.0))
    assert evidence.usable == ()
    assert np.isnan(evidence.bounds[0])
    assert not evidence.detected
    assert "no kaolinite and no chlorite" in evidence.summary()


def test_both_windows_and_both_references_are_tried():
    evidence = kaolinite_evidence(scan(chlorite() + kaolinite(1500.0)))
    pairs = {(share.window, share.reference) for share in evidence.usable}
    assert pairs == set(CHLORITE_RATIOS)


def test_disagreeing_references_are_reported_as_such():
    """A chlorite 001 too big for its own 003 cannot be believed."""
    signal = chlorite() + peak(14.20, 20000.0) + kaolinite(3000.0)
    evidence = kaolinite_evidence(scan(signal))
    assert not evidence.references_agree
    assert "do not overlap" in evidence.summary()


def test_agreeing_references_say_nothing_about_disagreement():
    evidence = kaolinite_evidence(scan(chlorite()))
    assert evidence.references_agree
    assert "do not overlap" not in evidence.summary()


def test_the_collapse_test_is_reported_beside_it_when_a_heated_mount_is_given():
    evidence = kaolinite_evidence(
        scan(chlorite() + kaolinite(6000.0), seed=1),
        scan(chlorite(), seed=2),
        scale=1.0,
    )
    assert evidence.collapse is not None
    assert "collapsed on heating" in evidence.summary() or "gone after heating" in evidence.summary()


def test_the_ratios_come_from_two_standards_and_bracket():
    for (window, reference), (low, high) in CHLORITE_RATIOS.items():
        assert window in {"7.15", "3.58"}
        assert reference in {"001", "003"}
        assert 0.0 < low < high


def test_the_bound_is_the_envelope_of_the_estimates():
    evidence = kaolinite_evidence(scan(chlorite() + kaolinite(3000.0)))
    least, most = evidence.bounds
    lows = [share.kaolinite[0] for share in evidence.usable]
    highs = [share.kaolinite[1] for share in evidence.usable]
    assert least == pytest.approx(min(lows))
    assert most == pytest.approx(max(highs))
