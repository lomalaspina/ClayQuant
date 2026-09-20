"""Only the glycolated mount is fitted, and why.

An air-dried smectite interlayer holds zero, one, two or three layers of water
- about 9.6, 12.4, 15 or 18 A - depending on the exchangeable cation and on the
humidity, and a real specimen carries several of those states at once,
interstratified within single crystallites.  There is no single air-dried
structure to calculate.  Glycol solvation exists precisely to remove that
variability, which is what makes the glycolated mount the one a pattern can be
calculated for.
"""
import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import nnls_fit, select_one_per_family
from clayquant.pattern import Pattern
from clayquant.treatment import AIR_DRIED_STATES

GRID = np.arange(4.0, 34.0, 0.02)


def peak(center: float, height: float = 1.0, width: float = 0.10) -> np.ndarray:
    return height * np.exp(-0.5 * ((GRID - center) / width) ** 2)


def a_library() -> PatternLibrary:
    return PatternLibrary(
        two_theta=GRID,
        entries=[
            LibraryEntry(name="illite", phase="illite",
                         intensity=peak(8.84) + 0.4 * peak(26.6),
                         unit_mass=100.0, unit_volume=100.0),
            LibraryEntry(name="smectite_EG", phase="smectite_EG", fraction=0.0,
                         intensity=peak(5.24) + 0.3 * peak(15.8),
                         air_intensity=peak(7.13) + 0.3 * peak(21.5),
                         unit_mass=100.0, unit_volume=100.0),
        ],
    )


def a_mount():
    library = a_library()
    values = sum(entry.intensity for entry in library.entries)
    return Pattern(two_theta=GRID, intensity=values, name="eg")


def test_the_glycol_view_is_the_library_itself():
    library = a_library()
    assert library.for_treatment("glycol") is library
    assert library.for_treatment() is library


def test_the_air_dried_mount_is_refused_and_says_why():
    """Not "unsupported" - it is not a thing that can be done."""
    with pytest.raises(ValueError, match="cannot be fitted"):
        a_library().for_treatment("air")
    with pytest.raises(ValueError, match="layers of water"):
        nnls_fit(a_mount(), a_library(), treatment="air")
    with pytest.raises(ValueError, match="glycol solvation is for"):
        select_one_per_family(a_mount(), a_library(), families=["", "smectite"],
                              treatment="air")


def test_the_refusal_names_the_alternative():
    """A refusal that does not say what to do instead is not much use."""
    with pytest.raises(ValueError, match="shift_evidence"):
        a_library().for_treatment("air")


def test_the_heated_mount_is_refused_too():
    with pytest.raises(ValueError, match="no model for a mount heated"):
        a_library().for_treatment("heated")
    with pytest.raises(ValueError, match="kaolinite diagnostic"):
        nnls_fit(a_mount(), a_library(), treatment="heated")


def test_an_unknown_treatment_is_refused():
    with pytest.raises(ValueError, match="unknown treatment"):
        a_library().for_treatment("boiled")


def test_the_hydration_states_are_the_reason_and_are_recorded():
    """The spacings the argument rests on, so they can be checked."""
    assert AIR_DRIED_STATES["1 water layer"] == pytest.approx(12.4)
    assert AIR_DRIED_STATES["2 water layers"] == pytest.approx(15.0)
    # every one of them differs from the glycol complex by more than the
    # measurement's resolution, which is why the movement is detectable at all
    assert all(abs(spacing - 16.86) > 0.5 for spacing in AIR_DRIED_STATES.values())


def test_the_glycol_mount_is_fitted_exactly_as_before():
    library = a_library()
    result = nnls_fit(a_mount(), library)
    got = dict(zip(result.names, result.coefficients))
    assert got["smectite_EG"] == pytest.approx(1.0, rel=1e-6)
    assert got["illite"] == pytest.approx(1.0, rel=1e-6)
