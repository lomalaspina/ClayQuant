"""An unweighed mount's mass, inferred from counts per gram.

The film correction needs grams.  Most specimens were never weighed - measured
months ago, or by somebody else - so for the correction to be usable at all the
mass has to come out of the measurement.
"""
from __future__ import annotations

import numpy as np
import pytest

from clayquant.absorption import FilmAbsorption
from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import FitResult, nnls_fit, solve_film_and_mass
from clayquant.pattern import Pattern
from clayquant.quantification import InstrumentConstant, mass_on_the_plate


def _result(phases, masses, coefficients) -> FitResult:
    n = len(phases)
    zeros = np.zeros(4)
    return FitResult(
        two_theta=np.linspace(4, 40, 4),
        observed=zeros, calculated=zeros,
        coefficients=np.array(coefficients, dtype=float),
        names=[f"{p} {i}" for i, p in enumerate(phases)],
        phases=list(phases),
        r_wp=0.1, r_p=0.1,
        scattering_fraction=np.ones(n) / n,
        amplitude_fraction=np.ones(n) / n,
        mask=np.ones(4, dtype=bool),
        relative_mass=np.array(masses, dtype=float),
    )


def test_the_constant_is_per_phase_because_averaging_destroys_it():
    """On the weighed standards the constant is consistent within a mineral and
    varies 3.8-fold between minerals.  A mean over phases mis-predicted every
    mass by 0.71 to 4.12 times, which is why this is a mapping and not a
    number."""
    constant = InstrumentConstant(per_phase={"kaolinite_1M": 1.7e6, "chlorite": 4.6e6})
    assert constant.constant_for({"kaolinite_1M": 1.0}) == pytest.approx(1.7e6)
    assert constant.constant_for({"chlorite": 1.0}) == pytest.approx(4.6e6)
    # a mixture lies between, weighted by mass
    mixed = constant.constant_for({"kaolinite_1M": 1.0, "chlorite": 1.0})
    assert 1.7e6 < mixed < 4.6e6


def test_a_phase_with_no_measured_constant_is_dropped_not_invented():
    constant = InstrumentConstant(per_phase={"illite": 1.2e6})
    assert constant.constant_for(
        {"illite": 1.0, "something else": 5.0}
    ) == pytest.approx(1.2e6)


def test_a_specimen_of_wholly_uncalibrated_phases_refuses_rather_than_guesses():
    constant = InstrumentConstant(per_phase={"illite": 1.2e6})
    with pytest.raises(ValueError, match="cannot be inferred"):
        constant.constant_for({"quartz": 1.0})


def test_mass_on_the_plate_inverts_the_weighed_mount_calculation():
    constant = InstrumentConstant(per_phase={"illite": 2.0e6})
    result = _result(["illite"], [4.0e3], [1.0])
    assert mass_on_the_plate(result, constant) == pytest.approx(2.0e-3)


def test_the_orientation_divides_out_at_r_cubed():
    """A basal series is enhanced as r**-3, so a constant held at r = 1 has to be
    put back on the fit's own orientation before it means anything."""
    constant = InstrumentConstant(per_phase={"illite": 2.0e6})
    result = _result(["illite"], [4.0e3], [1.0])
    at_half = mass_on_the_plate(result, constant, orientation=0.5)
    assert at_half == pytest.approx(2.0e-3 / 0.5 ** 3)


def test_a_fit_with_no_mass_is_an_error_not_a_zero():
    constant = InstrumentConstant(per_phase={"illite": 2.0e6})
    with pytest.raises(ValueError, match="no mass"):
        mass_on_the_plate(_result(["illite"], [0.0], [0.0]), constant)


def test_from_weighed_mounts_averages_repeats_of_one_phase():
    from clayquant.quantification import WeighedMount

    mounts = [
        (WeighedMount(constant=1.0e6, mass=1e-3, orientation=1.0), "kaolinite_1M"),
        (WeighedMount(constant=2.0e6, mass=1e-3, orientation=1.0), "kaolinite_1M"),
    ]
    constants = InstrumentConstant.from_weighed_mounts(mounts)
    assert constants.per_phase["kaolinite_1M"] == pytest.approx(1.5e6)


def _library() -> PatternLibrary:
    grid = np.linspace(4.0, 40.0, 361)

    def peak(centre):
        return np.exp(-0.5 * ((grid - centre) / 0.25) ** 2)

    return PatternLibrary(
        two_theta=grid,
        entries=[
            LibraryEntry(name="illite a", phase="illite", intensity=peak(8.8),
                         normalization=1.0, unit_mass=100.0, unit_volume=500.0,
                         mass_attenuation=42.0),
            LibraryEntry(name="illite b", phase="illite", intensity=peak(17.7),
                         normalization=1.0, unit_mass=100.0, unit_volume=500.0,
                         mass_attenuation=42.0),
        ],
    )


def test_the_solver_recovers_a_mass_it_was_not_told():
    """The round trip: build a pattern from a film of known mass, then let the
    solver find that mass knowing only the counts per gram."""
    library = _library()
    truth_g = 2.6e-3
    area = 4.91
    truth = FilmAbsorption(mass_per_area=truth_g / area, mass_attenuation=42.0)
    synthetic = library.for_film(truth)
    observed = Pattern(
        two_theta=library.two_theta,
        intensity=sum(e.intensity for e in synthetic.entries),
    )

    def fit(lib):
        return nnls_fit(observed, lib), 1.0

    # calibrate the constant from this very mount, so the solver has a true one
    seed, _ = fit(library.for_film(truth))
    per_gram = float(np.sum(seed.relative_mass)) / truth_g
    constant = InstrumentConstant(per_phase={"illite": per_gram})

    _result_, film, grams, history = solve_film_and_mass(
        library, fit, constant, area_cm2=area, first_guess_g=8.0e-3,
    )
    assert grams == pytest.approx(truth_g, rel=0.1)
    assert len(history) >= 2
    # and it got there from a guess three times too big
    assert history[0] == pytest.approx(8.0e-3)


def test_the_solver_refuses_a_library_without_attenuation_coefficients():
    grid = np.linspace(4.0, 40.0, 50)
    library = PatternLibrary(
        two_theta=grid,
        entries=[LibraryEntry(name="a", phase="a", intensity=np.ones_like(grid))],
    )
    observed = Pattern(two_theta=grid, intensity=np.ones_like(grid))
    with pytest.raises(ValueError, match="no mass attenuation coefficients"):
        solve_film_and_mass(
            library, lambda lib: (nnls_fit(observed, lib), 1.0),
            InstrumentConstant(per_phase={"a": 1.0}), area_cm2=4.91,
        )
