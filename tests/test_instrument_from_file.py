"""The instrument to calculate with comes from the measurement, not from a default.

Two values that decide calculated intensities are recorded in the data file and
were being ignored: the goniometer radius and the divergence slit.  Calculating
with 280 mm on a 240 mm instrument mis-states how much of the beam a flat
specimen intercepts at low angle, and that lands on the 001 reflections the
oriented-mount method rests on.  The peak widths are not in the file at all and
are measured off the pattern.
"""

import numpy as np
import pytest

from clayquant.gui.app import DEFAULT_PEAK_SHAPE, gui_instrument, instrument_from_measurement
from clayquant.gui.state import STATE
from clayquant.library import HOST_THICKNESSES
from clayquant.pattern import Pattern
from clayquant.profile import pseudo_voigt


def a_pattern(radius=240.0, slit=0.25, widths=0.12):
    x = np.arange(3.0, 40.0, 0.0167)
    y = np.zeros_like(x)
    for centre in (8.9, 12.5, 20.9, 26.6, 31.5, 36.5):
        profile = pseudo_voigt(x - centre, widths, 0.5)
        y += 1000.0 * profile / profile.max()
    return Pattern(
        two_theta=x,
        intensity=y + 20.0,
        name="synthetic",
        metadata={"goniometer_radius": radius, "divergence_slit": slit,
                  "wavelength": 1.540596},
    )


@pytest.fixture(autouse=True)
def _clean_state():
    STATE.instrument = None
    STATE.instrument_note = ""
    yield
    STATE.instrument = None
    STATE.instrument_note = ""


def test_the_radius_and_the_slit_come_from_the_file():
    instrument, note = instrument_from_measurement(a_pattern(radius=240.0, slit=0.25))
    assert instrument.divergence.goniometer_radius == pytest.approx(240.0)
    assert instrument.divergence.divergence == pytest.approx(0.25)
    assert "240 mm" in note


def test_a_file_without_the_geometry_falls_back_rather_than_failing():
    pattern = a_pattern()
    pattern.metadata.pop("goniometer_radius")
    pattern.metadata.pop("divergence_slit")
    instrument, _ = instrument_from_measurement(pattern)
    assert instrument.divergence.goniometer_radius == pytest.approx(280.0)
    assert instrument.divergence.divergence == pytest.approx(0.5)


def test_the_width_is_measured_off_the_pattern():
    instrument, note = instrument_from_measurement(a_pattern(widths=0.18))
    width = float(instrument.peak_shape.fwhm(np.array([26.6]), 1.540596)[0])
    assert width == pytest.approx(0.18, abs=0.02)
    assert "scaled by" in note


def test_the_slit_changes_how_much_beam_the_specimen_intercepts_at_low_angle():
    # This is the whole reason the slit has to be read: at 8.9 deg a 0.5 deg
    # slit overflows a 20 mm specimen and a 0.25 deg slit does not, so the 001
    # reflection is attenuated in one case and not in the other.
    wide, _ = instrument_from_measurement(a_pattern(slit=0.5))
    narrow, _ = instrument_from_measurement(a_pattern(slit=0.25))
    angle = np.array([8.9])
    assert float(wide.divergence.factor(angle)[0]) < float(narrow.divergence.factor(angle)[0])
    assert float(narrow.divergence.factor(angle)[0]) == pytest.approx(1.0)


def test_the_session_instrument_is_what_the_gui_calculates_with():
    assert gui_instrument().peak_shape == DEFAULT_PEAK_SHAPE
    STATE.instrument, STATE.instrument_note = instrument_from_measurement(a_pattern())
    assert gui_instrument() is STATE.instrument


def test_a_reset_forgets_the_instrument():
    STATE.instrument, STATE.instrument_note = instrument_from_measurement(a_pattern())
    STATE.reset()
    assert STATE.instrument is None
    assert gui_instrument().peak_shape == DEFAULT_PEAK_SHAPE


def test_the_basal_spacings_span_what_real_clays_do():
    # A chlorite at 14.15 A and an illite at 9.90 A are both ordinary, and both
    # were outside the spanned range: the fit then used a chlorite/smectite
    # entry at 14.2 A as a stand-in, which reports an expandable component that
    # is not in the specimen.
    assert min(HOST_THICKNESSES["chlorite"]) <= 14.05
    assert max(HOST_THICKNESSES["chlorite"]) >= 14.35
    assert min(HOST_THICKNESSES["illite"]) <= 9.90
    assert max(HOST_THICKNESSES["illite"]) >= 10.10


def test_a_library_built_with_another_instrument_is_reported():
    from clayquant.library import PatternLibrary, describe_instrument_mismatch
    from clayquant.optics import Divergence
    from clayquant.pattern import Instrument
    from clayquant.profile import PeakShape

    library = PatternLibrary(
        two_theta=np.arange(4.0, 40.0, 0.02),
        entries=[],
        metadata={
            "peak_shape": {"u": 0.02, "v": -0.005, "w": 0.01, "eta": 0.6,
                           "size_c": None, "size_ab": 400.0},
            "geometry": {"specimen_length": 20.0, "goniometer_radius": 280.0,
                         "divergence": 0.5},
        },
    )
    same = Instrument(peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0),
                      divergence=Divergence(20.0, 280.0, 0.5))
    assert describe_instrument_mismatch(library, same) == ""

    other = Instrument(peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0),
                       divergence=Divergence(20.0, 240.0, 0.5))
    complaint = describe_instrument_mismatch(library, other)
    assert "goniometer radius 280" in complaint

    wider = Instrument(peak_shape=PeakShape(u=0.08, v=-0.02, w=0.04, eta=0.6, size_ab=400.0),
                       divergence=Divergence(20.0, 280.0, 0.5))
    assert "peak width at 26 deg" in describe_instrument_mismatch(library, wider)


def test_a_library_that_records_no_instrument_says_it_cannot_be_checked():
    from clayquant.library import PatternLibrary, describe_instrument_mismatch
    from clayquant.pattern import Instrument

    library = PatternLibrary(two_theta=np.arange(4.0, 40.0, 0.02), entries=[], metadata={})
    assert "does not record" in describe_instrument_mismatch(library, Instrument())
