"""A pure mineral standard has no quartz, and must still be analysable.

The kaolinite and expandable steps put the two mounts on a common intensity
scale through a reflection the treatment does not affect, quartz 100 by default.
A pure dickite, kaolinite or chlorite standard contains no quartz at all, and
both steps used to raise rather than proceed - failing on exactly the specimens
the diagnostics should be easiest on, and the ones the standards' own ranges are
measured from.

Two mounts of one separate at the same settings are already on a common scale,
so 1.0 is the honest default.  What must not happen is that it is assumed
silently: the note says it was assumed, and the manual factor is there for
mounts that really do differ.
"""

import numpy as np
import pytest

from clayquant.diagnostics import (
    common_scale,
    kaolinite_collapse,
    scale_to_reference,
    smectite_swelling,
)
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 34.0, 0.0167)
WAVELENGTH = 1.540596


def angle(d: float) -> float:
    return 2.0 * np.degrees(np.arcsin(WAVELENGTH / (2.0 * d)))


def peak(d: float, height: float, width: float = 0.14) -> np.ndarray:
    return height * np.exp(-4.0 * np.log(2.0) * ((GRID - angle(d)) / width) ** 2)


def mount(signal, seed: int = 20261001) -> Pattern:
    rng = np.random.default_rng(seed)
    return Pattern(two_theta=GRID,
                   intensity=rng.poisson(np.zeros_like(GRID) + signal + 400.0).astype(float),
                   name="standard")


def a_pure_kaolinite(strength: float = 20000.0) -> np.ndarray:
    return peak(7.16, strength) + peak(3.58, strength * 0.25)


def test_the_collapse_test_runs_on_a_standard_with_no_quartz():
    air = mount(a_pure_kaolinite())
    heated = mount(np.zeros_like(GRID))
    result = kaolinite_collapse(air, heated)
    assert result.scale == pytest.approx(1.0)
    assert result.collapse_fraction > 0.9
    assert "common scale" in result.scale_note


def test_the_swelling_test_runs_on_a_standard_with_no_quartz():
    air = mount(peak(14.0, 9000.0))
    glycol = mount(peak(17.0, 9000.0))
    result = smectite_swelling(air, glycol)
    assert result.scale == pytest.approx(1.0)
    assert result.expandable_detected
    assert "common scale" in result.scale_note


def test_a_measured_reference_leaves_no_note():
    quartz = peak(4.257, 5000.0, width=0.10)
    air = mount(quartz + a_pure_kaolinite())
    heated = mount(quartz)
    result = kaolinite_collapse(air, heated)
    assert result.scale_note == ""
    assert result.scale == pytest.approx(1.0, abs=0.15)


def test_the_scale_is_still_measured_where_a_reference_exists():
    """The fallback must not shadow a real measurement: a mount at half the
    intensity of the other has to come back as a factor of two, not as 1.0."""
    air = mount(peak(4.257, 6000.0, width=0.10))
    heated = mount(peak(4.257, 3000.0, width=0.10))
    scale, note = common_scale(air, heated)
    assert scale == pytest.approx(2.0, rel=0.15)
    assert note == ""


def test_scale_to_reference_still_raises_for_a_caller_that_cannot_proceed():
    air = mount(a_pure_kaolinite())
    heated = mount(np.zeros_like(GRID))
    with pytest.raises(ValueError, match="no reference reflection"):
        scale_to_reference(air, heated)
    assert scale_to_reference(air, heated, fallback=1.0) == pytest.approx(1.0)
