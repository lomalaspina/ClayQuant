"""Reading a fluorescence analysis of the same mounts, and what it can be asked.

Every caution in the module is here as a test, because the failure mode of this
measurement is not an error but a plausible-looking number: a normalised
analysis of a thin film on glass will happily report 30 per cent potassium in
an illite.
"""

import pytest

from clayquant.xrf import (
    GLASS_MARKERS,
    RELIABLE_ELEMENTS,
    Measurement,
    illite_octahedral_iron,
    read_xrf,
)

FILE = """\
Sample,Analysis date,Material,Measurement result,Unit,Sigma,3Sigma,Judge
film,23/09/2026 14:11:44,Na(Sodium(Natrium)),ND,%,0.7,2.1,
film,23/09/2026 14:11:44,K(Potassium(Kalium)),16.3,%,0.1,0.2,
film,23/09/2026 14:11:44,Ca(Calcium),18.5,%,0.1,0.2,
film,23/09/2026 14:11:44,Fe(Iron),11.4,%,0.0,0.1,
film,23/09/2026 14:11:44,Al(Aluminium(Aluminum)),11.0,%,0.1,0.2,
glass,23/09/2026 14:50:15,Na(Sodium(Natrium)),5.4,%,0.4,1.3,
glass,23/09/2026 14:50:15,K(Potassium(Kalium)),3.1,%,0.0,0.1,
glass,23/09/2026 14:50:15,Ca(Calcium),35.5,%,0.1,0.2,
glass,23/09/2026 14:50:15,Fe(Iron),0.6,%,0.0,0.0,
glass,23/09/2026 14:50:15,Al(Aluminium(Aluminum)),0.5,%,0.0,0.1,
film_oxides,23/09/2026 14:28:13,K2O,10.2,%,0.0,0.1,
film_oxides,23/09/2026 14:28:13,Fe2O3,6.8,%,0.0,0.0,
"""


@pytest.fixture
def analysis(tmp_path):
    path = tmp_path / "standards.csv"
    path.write_text(FILE)
    return read_xrf(path)


def test_the_file_is_read_by_measurement(analysis):
    assert set(analysis) == {"film", "glass", "film_oxides"}
    assert analysis["film"].values["K"] == pytest.approx(16.3)
    assert analysis["film"].sigmas["K"] == pytest.approx(0.1)
    assert analysis["film"].date.startswith("23/09/2026")


def test_below_detection_is_absent_and_not_zero():
    """Two different statements, and only one of them is a measurement."""
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "a.csv"
        path.write_text(FILE)
        film = read_xrf(path)["film"]
    assert "Na" not in film.values
    assert film.get("Na") is None


def test_the_two_report_modes_are_distinguished(analysis):
    assert analysis["film"].mode == "elements"
    assert analysis["film_oxides"].mode == "oxides"


def test_the_slide_share_comes_from_a_marker(analysis):
    film, glass = analysis["film"], analysis["glass"]
    assert film.glass_fraction(glass, "Ca") == pytest.approx(18.5 / 35.5)
    # The film's sodium was below detection, so that marker says nothing.
    assert film.glass_fraction(glass, "Na") is None


def test_a_marker_the_mount_lacks_is_reported_not_guessed(analysis):
    report = analysis["film"].marker_agreement(analysis["glass"])
    assert set(report["fractions"]) == set(GLASS_MARKERS)
    assert report["fractions"]["Na"] is None
    assert report["relative_spread"] is None


def test_a_light_element_ratio_is_refused(analysis):
    """The number would look fine and mean nothing."""
    with pytest.raises(ValueError, match="absorbed by the film"):
        analysis["film"].ratio("Fe", "Al")
    assert "Al" not in RELIABLE_ELEMENTS
    assert "Si" not in RELIABLE_ELEMENTS


def test_a_heavy_ratio_barely_moves_with_the_slide_share(analysis):
    """Which is the whole reason it is usable."""
    film, glass = analysis["film"], analysis["glass"]
    low, high = film.ratio_envelope("Fe", "K", glass)
    assert low == pytest.approx(11.4 / 16.3, rel=1e-6)
    assert (high - low) / high < 0.15
    # while the concentrations behind it change several fold
    assert (film.ratio("Fe", "K", glass, 0.85)
            == pytest.approx(high, rel=1e-6))


def test_an_impossible_slide_share_is_refused(analysis):
    film, glass = analysis["film"], analysis["glass"]
    with pytest.raises(ValueError, match="must lie"):
        film.ratio("Fe", "K", glass, 1.0)


def test_over_correcting_is_reported_rather_than_returned_negative():
    film = Measurement(name="mont", values={"K": 1.9, "Ca": 29.9, "Fe": 2.8})
    glass = Measurement(name="glass", values={"K": 3.1, "Ca": 35.5, "Fe": 0.6})
    # A calcium-exchanged smectite: the marker credits the slide with more
    # potassium than the measurement contains.
    with pytest.raises(ValueError, match="over-estimated"):
        film.ratio("Fe", "K", glass, film.glass_fraction(glass, "Ca"))


def test_the_iron_fraction_follows_from_the_ratio_and_the_interlayer():
    # Fe/K = (2 f M_Fe)/(k M_K), so f = (Fe/K) k M_K / (2 M_Fe).
    assert illite_octahedral_iron(0.699, 0.9) == pytest.approx(
        0.699 * 0.9 * 39.098 / (2.0 * 55.845))
    assert illite_octahedral_iron(0.699, 0.9) > illite_octahedral_iron(0.699, 0.8)
    assert 0.19 < illite_octahedral_iron(0.699, 0.8) < 0.21


def test_an_impossible_occupancy_or_ratio_is_refused():
    with pytest.raises(ValueError):
        illite_octahedral_iron(0.0, 0.9)
    for potassium in (0.0, 1.5):
        with pytest.raises(ValueError):
            illite_octahedral_iron(0.7, potassium)
