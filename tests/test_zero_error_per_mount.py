"""Each mount keeps its own zero error, and the shared slider must not blur them.

Air and glycol commonly agree while the heated mount differs: it was taken out,
fired, and put back, so it carries its own displacement.  The three zero errors
are therefore three numbers.  The slider that sets them is one control, and that
mismatch is where the bug was - selecting a different mount left the previous
mount's value in the slider, and the callback wrote it into the mount just
selected.
"""

import pytest
from dash import no_update

from clayquant.gui.app import ZERO_ERROR_RELOADS_ON, zero_error_exchange
from clayquant.gui.state import MountState
from clayquant.pattern import Pattern

import numpy as np


def a_mount(zero_error=0.0):
    pattern = Pattern(two_theta=np.arange(4.0, 40.0, 0.02),
                      intensity=np.ones(1800), name="m")
    return MountState(raw=pattern, zero_error=zero_error)


def test_moving_the_slider_stores_it_on_the_mount():
    store, slider = zero_error_exchange("zero-slider", stored=0.0, slider=-0.06)
    assert store == pytest.approx(-0.06)
    assert slider is no_update


def test_changing_the_mount_reads_from_it_instead():
    """The bug: this used to write -0.06 into the mount just selected."""
    store, slider = zero_error_exchange("zero-mount", stored=0.02, slider=-0.06)
    assert store == pytest.approx(0.02)
    assert slider == pytest.approx(0.02)


def test_changing_the_mount_when_they_already_agree_leaves_the_slider_alone():
    store, slider = zero_error_exchange("zero-mount", stored=-0.06, slider=-0.06)
    assert store == pytest.approx(-0.06)
    assert slider is no_update


def test_loading_reads_too():
    store, slider = zero_error_exchange("load-status", stored=0.0, slider=-0.06)
    assert store == pytest.approx(0.0)
    assert slider == pytest.approx(0.0)


def test_the_reference_line_only_redraws():
    """Changing which line is used is not a reason to move anything."""
    store, slider = zero_error_exchange("zero-reference", stored=0.02, slider=-0.06)
    assert store == pytest.approx(-0.06)
    assert slider is no_update


def test_which_controls_read_rather_than_write():
    assert set(ZERO_ERROR_RELOADS_ON) == {"zero-mount", "load-status"}


def test_the_sequence_that_was_reported():
    """Calibrate air, calibrate heat, click back to air: air keeps its own."""
    mounts = {"air": a_mount(), "glycol": a_mount(), "heat": a_mount()}
    slider = 0.0

    # the user calibrates the air mount
    mounts["air"].zero_error, out = zero_error_exchange("zero-slider", 0.0, -0.06)
    if out is not no_update:
        slider = out

    # selects the heated mount: it reads 0, so the slider goes to 0
    mounts["heat"].zero_error, out = zero_error_exchange(
        "zero-mount", mounts["heat"].zero_error, slider)
    if out is not no_update:
        slider = out
    assert slider == pytest.approx(0.0)

    # and calibrates it to its own, different value
    mounts["heat"].zero_error, out = zero_error_exchange("zero-slider", slider, -0.14)
    if out is not no_update:
        slider = out
    slider = -0.14

    # now back to air, which must still be -0.06 and not -0.14
    mounts["air"].zero_error, out = zero_error_exchange(
        "zero-mount", mounts["air"].zero_error, slider)
    assert mounts["air"].zero_error == pytest.approx(-0.06)
    assert out == pytest.approx(-0.06)
    assert mounts["heat"].zero_error == pytest.approx(-0.14)
    assert mounts["glycol"].zero_error == pytest.approx(0.0)


def test_the_mount_applies_its_own_shift_to_its_own_pattern():
    """And the storage was always per mount; only the callback blurred it."""
    air, heat = a_mount(-0.06), a_mount(-0.14)
    # apply_zero_error removes the error, so a negative one moves the pattern up
    assert air.corrected().two_theta[0] == pytest.approx(air.raw.two_theta[0] + 0.06)
    assert heat.corrected().two_theta[0] == pytest.approx(heat.raw.two_theta[0] + 0.14)
    # and the two differ, which is the whole point
    assert air.corrected().two_theta[0] != pytest.approx(heat.corrected().two_theta[0])
