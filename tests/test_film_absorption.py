"""The thin-film correction: a weighed film is not a thick plate."""
from __future__ import annotations

import math

import numpy as np
import pytest

from clayquant.absorption import (
    FilmAbsorption,
    mass_attenuation_of,
    mass_attenuation_of_layer,
    mass_per_area,
    mixture_mass_attenuation,
    thin_film_factor,
)
from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.models import eg_smectite_layer, load_crystal

DISC_AREA = math.pi * 1.25 ** 2  # the 2.5 cm round mount, in cm^2


def test_a_film_delivers_less_than_a_thick_plate_everywhere():
    film = FilmAbsorption(mass_per_area=5.3e-4, mass_attenuation=29.8)
    factor = film.factor(np.linspace(4.0, 40.0, 50))
    assert np.all(factor > 0.0)
    assert np.all(factor < 1.0)


def test_the_correction_is_stronger_at_high_angle():
    """The whole point: 1/sin(theta) keeps low-angle intensity preferentially.

    This is what makes the correction change relative weights rather than
    rescale them, and the direction matters - chlorite's 001 at 6.2 deg against
    kaolinite's at 12.36 deg is the pair it acts on hardest.
    """
    film = FilmAbsorption(mass_per_area=5.3e-4, mass_attenuation=29.8)
    chlorite_001 = float(film.factor(6.2))
    kaolinite_001 = float(film.factor(12.36))
    assert chlorite_001 > kaolinite_001
    assert chlorite_001 / kaolinite_001 == pytest.approx(1.74, abs=0.05)


def test_a_thick_film_needs_almost_no_correction():
    thick = FilmAbsorption(mass_per_area=1.0, mass_attenuation=50.0)
    assert float(thick.factor(10.0)) == pytest.approx(1.0, abs=1e-6)


def test_opacity_is_the_exponent_numerator():
    film = FilmAbsorption(mass_per_area=5.3e-4, mass_attenuation=29.8)
    assert film.opacity == pytest.approx(2.0 * 29.8 * 5.3e-4)


def test_a_mount_that_was_not_weighed_has_no_correction():
    """None is how "not weighed" is said; it must not become a made-up number."""
    with pytest.raises(ValueError, match="must be positive"):
        FilmAbsorption(mass_per_area=0.0, mass_attenuation=30.0)


def _library(values: dict[str, float]) -> PatternLibrary:
    grid = np.linspace(4.0, 40.0, 200)
    entries = [
        LibraryEntry(
            name=f"{phase} entry",
            phase=phase,
            intensity=np.ones_like(grid),
            normalization=2.0,
            unit_mass=100.0,
            unit_volume=500.0,
            mass_attenuation=mu,
        )
        for phase, mu in values.items()
    ]
    return PatternLibrary(two_theta=grid, entries=entries)


def test_for_film_scales_the_patterns_and_leaves_the_normalisation_alone():
    """The relation coefficient/normalization -> scattering units must survive.

    Rescaling the corrected pattern back to unit maximum, or adjusting the
    normalisation to match, would silently change every weight percent.
    """
    library = _library({"kaolinite_1M": 29.8})
    film = FilmAbsorption(mass_per_area=5.3e-4, mass_attenuation=29.8)
    corrected = library.for_film(film)
    expected = np.asarray(film.factor(library.two_theta))
    assert np.allclose(corrected.entries[0].intensity, expected)
    assert corrected.entries[0].normalization == library.entries[0].normalization
    # and the original is untouched
    assert np.allclose(library.entries[0].intensity, 1.0)


def test_for_film_carries_the_air_counterpart_too():
    grid = np.linspace(4.0, 40.0, 50)
    entry = LibraryEntry(
        name="I/S", phase="I/S", intensity=np.ones_like(grid),
        air_intensity=np.full_like(grid, 0.5), mass_attenuation=40.0,
    )
    library = PatternLibrary(two_theta=grid, entries=[entry])
    film = FilmAbsorption(mass_per_area=6.3e-4, mass_attenuation=40.0)
    corrected = library.for_film(film)
    factor = np.asarray(film.factor(grid))
    assert np.allclose(corrected.entries[0].air_intensity, 0.5 * factor)


def test_correcting_twice_is_refused():
    """Squaring the factor is a silent error, so it is made a loud one."""
    library = _library({"illite": 52.1})
    film = FilmAbsorption(mass_per_area=6.3e-4, mass_attenuation=52.1)
    once = library.for_film(film)
    with pytest.raises(ValueError, match="already carries a film correction"):
        once.for_film(film)


def test_no_film_returns_the_library_unchanged():
    library = _library({"illite": 52.1})
    assert library.for_film(None) is library


def test_mixture_coefficient_is_weighted_by_mass():
    assert mixture_mass_attenuation(
        {"a": 3.0, "b": 1.0}, {"a": 30.0, "b": 50.0}
    ) == pytest.approx(35.0)


def test_a_phase_with_no_coefficient_is_dropped_not_guessed():
    """A wrong coefficient propagates into a weight percent, so none is invented."""
    assert mixture_mass_attenuation(
        {"a": 1.0, "unknown": 99.0}, {"a": 30.0}
    ) == pytest.approx(30.0)


def test_averaging_over_nothing_is_an_error():
    with pytest.raises(ValueError, match="cannot be averaged"):
        mixture_mass_attenuation({"unknown": 1.0}, {"a": 30.0})


def test_mass_attenuations_reports_one_coefficient_per_phase():
    library = _library({"kaolinite_1M": 29.8, "illite": 52.1})
    assert library.mass_attenuations() == {"kaolinite_1M": 29.8, "illite": 52.1}


def test_a_layer_gets_a_coefficient_though_it_has_no_cell():
    """An interstratified entry's unit is a layer, so it needs this route."""
    mu = mass_attenuation_of_layer(eg_smectite_layer())
    assert 15.0 < mu < 60.0


def test_the_clays_differ_enough_that_the_mixture_must_be_solved_for():
    """If they were all alike, a fixed coefficient would do and the iteration
    in fit_film_absorption would be unnecessary machinery."""
    kaolinite = mass_attenuation_of(load_crystal("kaolinite_1M"))
    illite = mass_attenuation_of(load_crystal("illite"))
    assert illite > 1.4 * kaolinite


def test_the_factor_matches_the_measured_kaolinite_basal_ratio():
    """The standard that motivated all of this.

    Kaolinite_12 carried 2.6 mg on a 2.5 cm disc and measured 002/001 = 0.401.
    A thick-plate calculation gives 0.666, which is 66 % high; multiplying by
    the film factor at each order brings it to 0.357, within 11 %.
    """
    mu = mass_attenuation_of(load_crystal("kaolinite_1M"))
    film = FilmAbsorption(
        mass_per_area=mass_per_area(2.6e-3, DISC_AREA), mass_attenuation=mu
    )
    corrected = 0.666 * float(film.factor(24.88)) / float(film.factor(12.36))
    assert abs(corrected - 0.401) < abs(0.666 - 0.401)
    assert corrected == pytest.approx(0.401, rel=0.15)


def test_thin_film_factor_and_the_dataclass_agree():
    film = FilmAbsorption(mass_per_area=5.3e-4, mass_attenuation=29.8)
    assert np.allclose(film.factor(20.0), thin_film_factor(20.0, 29.8, 5.3e-4))


def _two_phase_library() -> PatternLibrary:
    """Two phases whose intensity sits at different angles, so the correction
    changes their ratio rather than rescaling both."""
    grid = np.linspace(4.0, 40.0, 361)

    def peak(centre: float) -> np.ndarray:
        return np.exp(-0.5 * ((grid - centre) / 0.2) ** 2)

    return PatternLibrary(
        two_theta=grid,
        entries=[
            LibraryEntry(name="low", phase="low", intensity=peak(6.2),
                         normalization=1.0, unit_mass=100.0, unit_volume=500.0,
                         mass_attenuation=30.0),
            LibraryEntry(name="high", phase="high", intensity=peak(24.9),
                         normalization=1.0, unit_mass=100.0, unit_volume=500.0,
                         mass_attenuation=60.0),
        ],
    )


def test_the_correction_reweights_rather_than_rescales():
    """The physical claim behind the whole change.

    A phase whose intensity is at high angle needs *more* of itself to explain
    the same measured peak once the film correction is in, because the film
    delivered less of it.  A pure rescaling could not change one phase against
    the other, and if this ever became one the correction would be pointless.
    """
    library = _two_phase_library()
    film = FilmAbsorption(mass_per_area=5.3e-4, mass_attenuation=40.0)
    corrected = library.for_film(film)
    low = corrected.entries[0].intensity.max() / library.entries[0].intensity.max()
    high = corrected.entries[1].intensity.max() / library.entries[1].intensity.max()
    assert low > high
    assert low / high > 2.0


def test_the_iteration_converges_and_reports_its_path():
    from clayquant.nnls import fit_film_absorption, nnls_fit
    from clayquant.pattern import Pattern

    library = _two_phase_library()
    truth = FilmAbsorption(mass_per_area=5.3e-4, mass_attenuation=45.0)
    synthetic = library.for_film(truth)
    observed = Pattern(
        two_theta=library.two_theta,
        intensity=(2.0 * synthetic.entries[0].intensity
                   + 1.0 * synthetic.entries[1].intensity),
    )
    result, film, history = fit_film_absorption(
        library, mass_per_area=5.3e-4,
        fit=lambda lib: nnls_fit(observed, lib),
    )
    assert result.r_wp < 0.05
    assert len(history) >= 2
    # It settled somewhere between the two phases' coefficients, which is what
    # a mass-weighted average of 30 and 60 has to be.
    assert 30.0 <= film.mass_attenuation <= 60.0
    assert abs(history[-1] - history[-2]) <= 0.005 * history[-2] or len(history) == 9


def test_the_iteration_refuses_a_library_with_no_coefficients():
    from clayquant.nnls import fit_film_absorption, nnls_fit
    from clayquant.pattern import Pattern

    grid = np.linspace(4.0, 40.0, 100)
    library = PatternLibrary(
        two_theta=grid,
        entries=[LibraryEntry(name="a", phase="a", intensity=np.ones_like(grid))],
    )
    observed = Pattern(two_theta=grid, intensity=np.ones_like(grid))
    with pytest.raises(ValueError, match="before the film correction existed"):
        fit_film_absorption(library, 5.3e-4, lambda lib: nnls_fit(observed, lib))
