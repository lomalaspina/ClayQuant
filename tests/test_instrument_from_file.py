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
from clayquant.profile import PeakShape, pseudo_voigt


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


def test_a_freshly_built_library_records_what_it_was_built_with():
    """The round trip, because the diagnosis depends on it.

    The warning that a library does not match a measurement is only as good as
    what the file remembers, and the three things that decide whether the
    reference peaks are the right shape - the layer spacings spanned, the width
    model and the geometry - all have to survive save and load.
    """
    import tempfile
    from pathlib import Path as _Path

    from clayquant.library import PatternLibrary, build_library, two_theta_grid

    grid = two_theta_grid(4.0, 20.0, 0.05)
    built = build_library(
        grid=grid,
        instrument=instrument_from_measurement(a_pattern(radius=240.0, slit=0.25))[0],
        orientations=(1.0, 0.5), illite_smectite=(), chlorite_smectite=(),
        chlorite_iron=(), csds_means=(15.0,), smectite_orientation=0.3,
    )
    path = _Path(tempfile.mkdtemp()) / "library.npz"
    built.save(path)
    loaded = PatternLibrary.load(path)
    assert loaded.metadata["geometry"]["goniometer_radius"] == pytest.approx(240.0)
    assert loaded.metadata["geometry"]["divergence"] == pytest.approx(0.25)
    assert loaded.metadata["smectite_orientation"] == pytest.approx(0.3)
    assert loaded.metadata["host_thicknesses"]["chlorite"] == pytest.approx(
        list(HOST_THICKNESSES["chlorite"])
    )
    assert loaded.metadata["peak_shape"]["w"] == pytest.approx(
        built.metadata["peak_shape"]["w"]
    )


def test_a_library_built_with_the_defaults_is_recognised_as_such():
    # The mistake this catches: pressing Build before loading a measurement, so
    # the library takes the 280 mm default on a 240 mm instrument.  Nothing in
    # the file says "built blind", so it is recognised by the defaults being
    # exactly what they are.
    from clayquant.library import PatternLibrary, describe_instrument_mismatch

    library = PatternLibrary(
        two_theta=np.arange(4.0, 40.0, 0.02), entries=[],
        metadata={
            "peak_shape": {"u": 0.02, "v": -0.005, "w": 0.01, "eta": 0.6,
                           "size_c": None, "size_ab": 400.0},
            "geometry": {"specimen_length": 20.0, "goniometer_radius": 280.0,
                         "divergence": 0.5},
        },
    )
    instrument, _ = instrument_from_measurement(a_pattern(radius=240.0, slit=0.25))
    complaint = describe_instrument_mismatch(library, instrument)
    assert "goniometer radius 280" in complaint
    assert "divergence slit 0.5" in complaint


def _text(node) -> str:
    if isinstance(node, str):
        return node
    children = getattr(node, "children", None)
    if children is None:
        return ""
    if isinstance(children, (list, tuple)):
        return "\n".join(_text(child) for child in children)
    return _text(children)


def a_library(radius=240.0, slit=0.5, width=0.09):
    from clayquant.library import PatternLibrary

    return PatternLibrary(
        two_theta=np.arange(4.0, 40.0, 0.02), entries=[],
        metadata={
            "peak_shape": {"u": 0.0, "v": 0.0, "w": width**2, "eta": 0.5,
                           "size_c": None, "size_ab": None},
            "geometry": {"specimen_length": 20.0, "goniometer_radius": radius,
                         "divergence": slit},
        },
    )


def test_the_strip_says_so_when_nothing_is_loaded():
    from clayquant.gui.app import instrument_strip

    said = _text(instrument_strip())
    assert "none loaded" in said
    assert "280 mm" in said


def test_the_strip_shows_the_measurement_geometry_and_widths():
    from clayquant.gui.app import instrument_strip

    STATE.instrument, STATE.instrument_note = instrument_from_measurement(
        a_pattern(radius=240.0, slit=0.25, widths=0.11)
    )
    said = _text(instrument_strip())
    assert "240 mm radius" in said
    assert "0.25" in said
    assert "0.11" in said


def test_the_strip_reports_a_library_that_disagrees():
    from clayquant.gui.app import instrument_strip

    STATE.instrument, _ = instrument_from_measurement(a_pattern(radius=240.0, slit=0.25))
    STATE.library = a_library(radius=280.0, slit=0.5)
    try:
        said = _text(instrument_strip())
        assert "do not match" in said
        assert "280 mm radius" in said and "240 mm radius" in said
    finally:
        STATE.library = None


def test_the_strip_is_quiet_when_the_two_agree():
    from clayquant.gui.app import instrument_strip

    pattern = a_pattern(radius=240.0, slit=0.25, widths=0.12)
    STATE.instrument, _ = instrument_from_measurement(pattern)
    width = float(STATE.instrument.peak_shape.fwhm(np.array([26.0]), 1.540596)[0])
    STATE.library = a_library(radius=240.0, slit=0.25, width=width)
    try:
        assert "do not match" not in _text(instrument_strip())
    finally:
        STATE.library = None


def test_the_strip_says_when_a_library_remembers_nothing():
    from clayquant.gui.app import instrument_strip
    from clayquant.library import PatternLibrary

    STATE.instrument, _ = instrument_from_measurement(a_pattern())
    STATE.library = PatternLibrary(
        two_theta=np.arange(4.0, 40.0, 0.02), entries=[], metadata={})
    try:
        assert "does not record" in _text(instrument_strip())
    finally:
        STATE.library = None


def test_the_library_cli_takes_its_instrument_from_a_scan():
    """The whole point of this change: the icon must not need the GUI.

    The library is built from a desktop icon before ClayQuant is opened, so the
    instrument cannot come from a loaded mount - it has to come from a scan the
    icon asks for.  Anything else makes a matched library depend on doing two
    things in the right order, which is not a requirement to put on anyone.
    """
    import tempfile
    from pathlib import Path as _Path

    from clayquant.library import PatternLibrary, main

    folder = _Path(tempfile.mkdtemp())
    scan = folder / "scan.xy"
    pattern = a_pattern(radius=240.0, slit=0.25, widths=0.13)
    scan.write_text(
        "\n".join(f"{angle:.4f} {value:.3f}"
                  for angle, value in zip(pattern.two_theta, pattern.intensity)),
        encoding="utf-8",
    )
    out = folder / "library.npz"
    assert main([
        "--measurement", str(scan), "-o", str(out), "--stop", "14", "--step", "0.1",
        "--csds-means", "15", "--quiet",
    ]) == 0
    library = PatternLibrary.load(out)
    # The .xy format carries no geometry, so the radius falls back; the widths
    # are measured from the peaks either way, and that is the part no file
    # records and no default can supply.
    assert library.metadata["peak_shape"]["w"] > 0.0
    modelled = float(PeakShape(
        u=library.metadata["peak_shape"]["u"],
        v=library.metadata["peak_shape"]["v"],
        w=library.metadata["peak_shape"]["w"],
        size_ab=library.metadata["peak_shape"]["size_ab"],
    ).fwhm(np.array([26.0]), 1.540596)[0])
    assert modelled == pytest.approx(0.13, abs=0.03)
