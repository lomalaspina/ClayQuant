"""Whether a specimen swelled, decided on eight pure standards of known answer.

This verdict is not a report, it is a switch: when nothing moved, the air-dried
restraint puts a ceiling on how much expandable layer the fit may use, and when
something moved the fit is left alone.  So a false positive silently removes the
one thing stopping a mixed-layer entry from standing in for a mineral that is
actually there - which is how a pure chlorite came to be fitted as 41 per cent
illite/smectite.

Two faults were found by running the standards through it, and both are the same
fault in different clothes: a wide search window plus "take the tallest thing in
it" picks the wrong line.
"""

import numpy as np
import pytest

from clayquant.pattern import Pattern
from clayquant.treatment import REFERENCE_MINIMUM_SHARE, shift_evidence

GRID = np.arange(3.0, 40.0, 0.0167)
WAVELENGTH = 1.540596


def angle(d: float) -> float:
    return 2.0 * np.degrees(np.arcsin(WAVELENGTH / (2.0 * d)))


def peak(d: float, height: float, width: float = 0.15) -> np.ndarray:
    return height * np.exp(-4.0 * np.log(2.0) * ((GRID - angle(d)) / width) ** 2)


def mount(signal, name="mount") -> Pattern:
    return Pattern(two_theta=GRID, intensity=signal, name=name)


def test_a_weak_line_cannot_set_the_mount_offset():
    """The pure kaolinite case.

    Its window around the quartz 101 held a kaolinite line at 1.7 per cent of
    the pattern, and the two scans placed that broad weak feature 0.056 deg
    apart.  Subtracting that as though it were a calibration turned the
    kaolinite 001's true -0.003 deg into +0.053 and marked a kaolinite as
    swelling.
    """
    strong = peak(7.16, 40000.0)
    # a weak feature near the quartz 101, placed differently in the two scans
    air = mount(strong + peak(3.325, 650.0, width=0.30), "air")
    glycol = mount(strong + peak(3.332, 650.0, width=0.30), "glycol")
    evidence = shift_evidence(air, glycol)
    assert evidence.offset == pytest.approx(0.0, abs=1e-9), (
        "a line at under %.0f %% of the pattern was allowed to set the offset"
        % (100 * REFERENCE_MINIMUM_SHARE)
    )
    assert not evidence.significant


def test_a_strong_reference_line_still_sets_it():
    """The threshold must not throw away a real quartz line."""
    quartz = peak(3.3435, 8000.0)
    air = mount(peak(7.16, 10000.0) + quartz, "air")
    glycol = mount(peak(7.16, 10000.0) + peak(3.3375, 8000.0), "glycol")
    evidence = shift_evidence(air, glycol)
    assert evidence.offset != 0.0


def test_a_reflection_that_shrank_is_not_swelling():
    """The pure illite case.

    Glycolation replaces interlayer water with a larger complex, so an
    expandable 001 moves to larger d.  Illite 10's only movement was a 10.073 A
    line going to 10.055 - a reflection that shrank - and counting its magnitude
    read a pure illite as swelling.
    """
    quartz = peak(3.3435, 6000.0)
    air = mount(peak(10.073, 9000.0) + quartz, "air")
    glycol = mount(peak(10.00, 9000.0) + quartz, "glycol")
    evidence = shift_evidence(air, glycol)
    assert not evidence.significant, "a shrinking reflection was read as swelling"


def test_a_reflection_that_expanded_is_swelling():
    quartz = peak(3.3435, 6000.0)
    air = mount(peak(14.2, 9000.0) + quartz, "air")
    glycol = mount(peak(16.9, 9000.0) + quartz, "glycol")
    evidence = shift_evidence(air, glycol)
    assert evidence.significant


def test_a_peak_that_appears_is_evidence_whatever_its_direction():
    """What the two montmorillonite standards actually rest on.

    Neither shows a displacement the measurement can distinguish from its own
    placement; both show a new 17 A line.  The direction rule filters
    displacements and must not touch appearances.
    """
    quartz = peak(3.3435, 6000.0)
    air = mount(peak(10.0, 4000.0) + quartz, "air")
    glycol = mount(peak(10.0, 4000.0) + peak(16.9, 9000.0) + quartz, "glycol")
    evidence = shift_evidence(air, glycol)
    assert evidence.appeared
    assert evidence.significant
