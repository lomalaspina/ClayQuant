"""From fitted scale factors to weight percent.

The chain has four links and an error in any one of them scales a phase without
looking wrong: the pattern is calculated for one unit of the phase, stored
divided by a normalisation, fitted with a coefficient, and converted back with
the mass of that unit.  The test that catches a broken link is to build a
specimen of known masses out of the library's own patterns and ask for the
masses back.

That tests the arithmetic, not the mineralogy.  What it cannot test is the
assumption underneath: that the calculated pattern accounts for everything the
phase contributes.  In an oriented mount it does not - texture differs between
phases and between preparations - which is what :class:`Calibration` is for, and
why every factor in it is 1 until somebody measures them.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.library import build_library
from clayquant.models import available_phases
from clayquant.nnls import nnls_fit
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, Pattern
from clayquant.profile import PeakShape
from clayquant.quantification import Calibration, quantify

pytestmark = pytest.mark.skipif(
    not all(available_phases().values()), reason="ICSD CIF files are not installed"
)

GRID = np.arange(3.008, 40.0, 0.0334)


@pytest.fixture(scope="module")
def library():
    return build_library(
        grid=GRID,
        instrument=Instrument(
            peak_shape=PeakShape(u=0.004, v=-0.001, w=0.002, eta=0.5, size_ab=600.0),
            divergence=Divergence(),
        ),
        orientations=(0.3,),
        illite_smectite=(0.80,),
        chlorite_smectite=(0.90,),
        csds_means=(15.0,),
        host_thicknesses={},
    )


def specimen(library, masses: dict[str, float], constant: float = 1e-6) -> Pattern:
    """A pattern from known masses of library entries.

    A stored pattern is the calculated one divided by ``normalization``, and the
    calculated one is for a single unit of mass ``unit_mass``.  So a mass ``m``
    of an entry contributes ``m * normalization * stored / unit_mass``, times one
    constant - the instrument, the irradiated volume, the counting time - that
    every phase in a measurement shares.
    """
    intensity = np.zeros_like(library.two_theta)
    for name, mass in masses.items():
        entry = library.entries[library.names.index(name)]
        intensity += mass * entry.normalization * entry.intensity / entry.unit_mass
    return Pattern(library.two_theta, constant * intensity, name="synthetic mixture")


def phase_of(library, name: str) -> str:
    return library.entries[library.names.index(name)].phase


def fit_specimen(library, masses, calibration=None):
    result = nnls_fit(specimen(library, masses), library, range_two_theta=(4.0, 34.0))
    assert result.r_wp < 1e-6, "the specimen is made of library patterns; the fit must be exact"
    return quantify(result, calibration=calibration)


def test_known_masses_come_back(library):
    masses = {
        "illite PO=0.3": 40.0,
        "kaolinite_1M PO=0.3": 25.0,
        "chlorite PO=0.3": 20.0,
        "I/S 0.80/0.20 PO=0.3": 15.0,
    }
    quantification = fit_specimen(library, masses)
    assert quantification.weights_available
    computed = {share.phase: share.weight for share in quantification.shares}
    for name, mass in masses.items():
        assert computed[phase_of(library, name)] == pytest.approx(mass, abs=0.05), name
    assert sum(share.weight for share in quantification.shares) == pytest.approx(100.0)


def test_the_instrument_constant_cancels(library):
    """Counting twice as long must not double the weight percent."""
    masses = {"illite PO=0.3": 60.0, "kaolinite_1M PO=0.3": 40.0}
    first = fit_specimen(library, masses)
    second = quantify(
        nnls_fit(specimen(library, masses, constant=2e-6), library, range_two_theta=(4.0, 34.0))
    )
    a = {s.phase: s.weight for s in first.shares}
    b = {s.phase: s.weight for s in second.shares}
    for phase, value in a.items():
        assert b[phase] == pytest.approx(value, abs=1e-6)


def test_a_heavy_phase_is_not_confused_with_an_abundant_one(library):
    """Chlorite scatters more per gram than kaolinite; the mass says otherwise.

    Equal masses must come back equal, which they only do if the conversion uses
    the mass of the calculated unit rather than the scattered intensity.
    """
    quantification = fit_specimen(
        library, {"chlorite PO=0.3": 50.0, "kaolinite_1M PO=0.3": 50.0}
    )
    weights = {share.phase: share.weight for share in quantification.shares}
    assert weights["chlorite"] == pytest.approx(50.0, abs=0.05)
    assert weights["kaolinite_1M"] == pytest.approx(50.0, abs=0.05)
    # And the share of scattered intensity is *not* 50/50, which is the point.
    scattering = {share.phase: share.scattering for share in quantification.shares}
    assert abs(scattering["chlorite"] - scattering["kaolinite_1M"]) > 0.02


def test_the_clay_basis_leaves_the_accompanying_minerals_out(library):
    masses = {"illite PO=0.3": 50.0, "kaolinite_1M PO=0.3": 30.0, "I/S 0.80/0.20 PO=0.3": 20.0}
    quantification = fit_specimen(library, masses)
    clay_weights = quantification.computed_weight_percent(clay_basis=True)
    assert sum(clay_weights.values()) == pytest.approx(100.0)


def test_a_calibration_factor_moves_one_phase(library):
    masses = {"illite PO=0.3": 50.0, "kaolinite_1M PO=0.3": 50.0}
    uncalibrated = fit_specimen(library, masses)
    calibrated = fit_specimen(
        library, masses, calibration=Calibration({"kaolinite_1M": 2.0}, source="a test")
    )
    before = {s.phase: s.weight for s in uncalibrated.shares}
    after = {s.phase: s.weight for s in calibrated.shares}
    assert before["kaolinite_1M"] == pytest.approx(50.0, abs=0.05)
    assert after["kaolinite_1M"] == pytest.approx(100.0 * 2 / 3, abs=0.05)
    assert after["illite"] == pytest.approx(100.0 / 3, abs=0.05)


def test_the_default_calibration_changes_nothing(library):
    masses = {"illite PO=0.3": 70.0, "chlorite PO=0.3": 30.0}
    plain = fit_specimen(library, masses)
    with_default = fit_specimen(library, masses, calibration=Calibration())
    assert [s.weight for s in plain.shares] == pytest.approx([s.weight for s in with_default.shares])


def test_factors_are_solved_from_a_known_composition():
    computed = {"illite": 60.0, "kaolinite_1M": 40.0}
    known = {"illite": 50.0, "kaolinite_1M": 50.0}
    calibration = Calibration.from_known_composition(computed, known)
    # Only ratios matter, so they are normalised to average 1.
    assert sum(calibration.factors.values()) == pytest.approx(2.0)
    assert calibration.factor("kaolinite_1M") > calibration.factor("illite")
    assert calibration.factor("anything else") == 1.0


def test_a_calibration_survives_a_round_trip(tmp_path):
    original = Calibration({"illite": 0.8, "chlorite": 1.25}, source="five known mixtures")
    path = original.to_json(tmp_path / "calibration.json")
    restored = Calibration.from_json(path)
    assert restored.factors == original.factors
    assert restored.source == original.source


def test_a_library_without_masses_reports_no_weights(library):
    """An older library still fits; it just cannot say what anything weighs."""
    from dataclasses import replace

    stripped = replace(
        library,
        entries=[replace(entry, unit_mass=None) for entry in library.entries],
    )
    masses = {"illite PO=0.3": 50.0, "kaolinite_1M PO=0.3": 50.0}
    result = nnls_fit(specimen(library, masses), stripped, range_two_theta=(4.0, 34.0))
    quantification = quantify(result)
    assert not quantification.weights_available
    assert all(share.weight == 0.0 for share in quantification.shares)
    assert all(row["weight_percent"] == "" for row in quantification.table())
    assert any(share.scattering > 0 for share in quantification.shares), "the fit still works"
