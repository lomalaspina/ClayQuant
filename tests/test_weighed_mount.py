"""The absolute scale from a mount that was weighed.

A relative analysis cannot check itself: it says how the phases of one mount
divide that mount between them, whatever the calculated patterns get wrong.
Weighing the material closes that, and with several mounts it becomes a test -
the instrument did not change between them, so a mount that disagrees about the
constant is a mineral whose reference pattern is wrong by that factor.
"""

import numpy as np
import pytest

from clayquant.absorption import (
    mass_attenuation_of,
    mass_per_area,
    thin_film_factor,
)
from clayquant.models import load_crystal
from clayquant.nnls import FitResult
from clayquant.quantification import WeighedMount, agreement_between


def fit(masses: dict[str, float]) -> FitResult:
    """A FitResult carrying nothing but the relative masses of some phases."""
    phases = list(masses)
    n = len(phases)
    return FitResult(
        two_theta=np.array([10.0]),
        observed=np.array([1.0]),
        calculated=np.array([1.0]),
        coefficients=np.ones(n),
        names=list(phases),
        phases=list(phases),
        scattering_fraction=np.full(n, 1.0 / n),
        amplitude_fraction=np.full(n, 1.0 / n),
        r_wp=0.0,
        r_p=0.0,
        mask=np.array([True]),
        relative_mass=np.array([masses[name] for name in phases]),
        metadata={"measurement": "test"},
    )


def test_the_constant_is_the_fitted_mass_per_gram():
    mount = WeighedMount.from_fit(fit({"illite": 4.0, "quartz": 1.0}), mass=0.0050)
    assert mount.constant == pytest.approx(5.0 / 0.0050)
    assert set(mount.phases) == {"illite", "quartz"}


def test_only_the_named_phases_are_counted():
    """For a mount where part of what was weighed is not being quantified."""
    mount = WeighedMount.from_fit(
        fit({"illite": 4.0, "quartz": 1.0}), mass=0.0050, phases=["illite"])
    assert mount.constant == pytest.approx(4.0 / 0.0050)
    assert mount.phases == ("illite",)


def test_a_phase_weighs_what_the_constant_says():
    result = fit({"illite": 4.0, "quartz": 1.0})
    mount = WeighedMount.from_fit(result, mass=0.0050)
    assert mount.mass_of(result, "illite") == pytest.approx(0.0040)
    assert mount.mass_of(result, "quartz") == pytest.approx(0.0010)
    assert (mount.mass_of(result, "illite")
            + mount.mass_of(result, "quartz")) == pytest.approx(mount.mass)


def test_an_impossible_mass_or_area_is_refused():
    result = fit({"illite": 1.0})
    for mass in (0.0, -1.0):
        with pytest.raises(ValueError):
            WeighedMount.from_fit(result, mass=mass)
    with pytest.raises(ValueError):
        WeighedMount.from_fit(result, mass=0.001, area=0.0)


def test_a_fit_with_no_mass_has_no_constant_in_it():
    with pytest.raises(ValueError):
        WeighedMount.from_fit(fit({"illite": 0.0}), mass=0.001)


def test_the_mass_per_area_needs_the_area():
    result = fit({"illite": 1.0})
    assert WeighedMount.from_fit(result, mass=0.008).mass_per_area is None
    assert WeighedMount.from_fit(result, mass=0.008, area=4.0).mass_per_area == pytest.approx(0.002)


def test_mounts_that_agree_and_a_mount_that_does_not():
    mounts = [
        WeighedMount(constant=1000.0, mass=0.005, measurement="illite"),
        WeighedMount(constant=1020.0, mass=0.004, measurement="kaolinite"),
        WeighedMount(constant=3000.0, mass=0.002, measurement="chlorite"),
    ]
    report = agreement_between(mounts)
    assert report["ratio_max_to_min"] == pytest.approx(3.0)
    # The chlorite's reference pattern would be wrong by its departure.
    assert report["departures"]["chlorite"] > 1.5
    assert report["departures"]["illite"] < 1.0


# --------------------------------------------------------------------------- #
# The film's own absorption
# --------------------------------------------------------------------------- #


def test_a_massless_film_delivers_nothing_and_a_heavy_one_everything():
    assert thin_film_factor(10.0, 45.0, 0.0) == pytest.approx(0.0)
    assert thin_film_factor(10.0, 45.0, 10.0) == pytest.approx(1.0)


def test_the_factor_rises_towards_low_angle_relative_to_high():
    """A thin film keeps more of its low-angle intensity, because the beam's
    path through it is longest there."""
    mu = mass_attenuation_of(load_crystal("illite"))
    factor = thin_film_factor(np.array([8.84, 17.72, 26.72]), mu,
                              mass_per_area(0.008, 6.0))
    assert factor[0] > factor[1] > factor[2]


def test_the_correction_lowers_the_calculated_higher_orders():
    """Which is the same direction as taking potassium out or putting iron in,
    and the reason the deposited area has to be measured before either is
    believed."""
    mu = mass_attenuation_of(load_crystal("illite"))
    orders = np.array([8.84, 17.72, 26.72])
    thick = thin_film_factor(orders, mu, mass_per_area(0.008, 0.01))
    film = thin_film_factor(orders, mu, mass_per_area(0.008, 6.0))
    assert np.allclose(thick, 1.0, atol=1e-6)
    assert film[1] / film[0] < 0.8
    assert film[2] / film[0] < film[1] / film[0]


def test_a_bad_coefficient_or_angle_is_refused():
    with pytest.raises(ValueError):
        thin_film_factor(10.0, 0.0, 0.001)
    with pytest.raises(ValueError):
        thin_film_factor(0.0, 45.0, 0.001)
    with pytest.raises(ValueError):
        mass_per_area(0.0, 1.0)


# --------------------------------------------------------------------------- #
# The orientation, without which two mounts are not on one scale
# --------------------------------------------------------------------------- #


def oriented_fit(masses: dict[str, float], r: float) -> FitResult:
    result = fit(masses)
    result.march_dollase = np.full(len(masses), r)
    return result


def test_the_orientation_is_taken_from_the_fit():
    mount = WeighedMount.from_fit(oriented_fit({"illite": 4.0}, 0.3), mass=0.005)
    assert mount.orientation == pytest.approx(0.3)


def test_an_orientation_can_be_given_instead():
    mount = WeighedMount.from_fit(fit({"illite": 4.0}), mass=0.005, orientation=0.2)
    assert mount.orientation == pytest.approx(0.2)


def test_the_comparable_constant_divides_the_orientation_out():
    """A basal series is enhanced by r^-3, so the mass behind a given measured
    intensity goes as r^3 and two mounts at different r are not on one scale."""
    mount = WeighedMount.from_fit(oriented_fit({"illite": 4.0}, 0.5), mass=0.005)
    assert mount.comparable_constant == pytest.approx(mount.constant / 0.125)


def test_without_an_orientation_nothing_is_corrected():
    """A caller that did not record it gets what it asked for, not a guess."""
    mount = WeighedMount.from_fit(fit({"illite": 4.0}), mass=0.005)
    assert mount.orientation is None
    assert mount.comparable_constant == pytest.approx(mount.constant)


def test_two_mounts_at_different_orientations_agree_once_corrected():
    """The 393-fold spread over nine standards was mostly this."""
    one = WeighedMount(constant=1.0 * 0.1**3, mass=0.003, orientation=0.1,
                       measurement="a")
    other = WeighedMount(constant=1.0 * 0.5**3, mass=0.003, orientation=0.5,
                         measurement="b")
    raw = max(one.constant, other.constant) / min(one.constant, other.constant)
    assert raw == pytest.approx(125.0)
    assert agreement_between([one, other])["ratio_max_to_min"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# What is left once the orientation is out: the per-phase factor
# --------------------------------------------------------------------------- #


def test_a_phase_that_sits_low_gets_a_factor_above_one():
    from clayquant.quantification import calibration_from_weighed_mounts

    mounts = [
        (WeighedMount(constant=2.0, mass=0.003, orientation=1.0, measurement="k1"),
         "kaolinite_1M"),
        (WeighedMount(constant=2.0, mass=0.003, orientation=1.0, measurement="k2"),
         "kaolinite_1M"),
        (WeighedMount(constant=0.5, mass=0.003, orientation=1.0, measurement="s"),
         "smectite_EG"),
    ]
    calibration = calibration_from_weighed_mounts(mounts)
    assert calibration.factors["smectite_EG"] > 1.0
    assert calibration.factors["kaolinite_1M"] < 1.0
    # the ratio is what the mounts measured, four to one
    assert (calibration.factors["smectite_EG"]
            / calibration.factors["kaolinite_1M"]) == pytest.approx(4.0)


def test_the_factors_average_one_because_only_ratios_matter():
    from clayquant.quantification import calibration_from_weighed_mounts

    mounts = [
        (WeighedMount(constant=c, mass=0.003, orientation=1.0, measurement=name), phase)
        for c, name, phase in ((3.0, "a", "chlorite"), (1.0, "b", "illite"),
                               (2.0, "c", "kaolinite_1M"))
    ]
    factors = calibration_from_weighed_mounts(mounts).factors
    assert sum(1.0 / f for f in factors.values()) / len(factors) == pytest.approx(1.0)


def test_repeat_mounts_of_one_phase_are_averaged():
    from clayquant.quantification import calibration_from_weighed_mounts

    mounts = [
        (WeighedMount(constant=1.0, mass=0.003, orientation=1.0, measurement="a"), "chlorite"),
        (WeighedMount(constant=3.0, mass=0.003, orientation=1.0, measurement="b"), "chlorite"),
        (WeighedMount(constant=2.0, mass=0.003, orientation=1.0, measurement="c"), "illite"),
    ]
    calibration = calibration_from_weighed_mounts(mounts)
    assert calibration.factors["chlorite"] == pytest.approx(calibration.factors["illite"])
    assert "chlorite (2)" in calibration.source


def test_a_calibration_needs_a_mount_with_something_in_it():
    from clayquant.quantification import calibration_from_weighed_mounts

    with pytest.raises(ValueError):
        calibration_from_weighed_mounts(
            [(WeighedMount(constant=0.0, mass=0.003), "illite")])
