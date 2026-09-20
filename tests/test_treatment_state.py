"""The library must be used in the state the specimen was in.

ClayQuant's patterns are calculated for the glycolated mount - the smectite is
Reynolds' (1965) ethylene glycol complex at 16.86 A, and so is the smectite
inside every I/S and C/S built from it.  Fitting an air-dried mount with those
is fitting the wrong interlayer, and these tests pin down that it no longer
happens silently.
"""
import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import nnls_fit, select_one_per_family
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 34.0, 0.02)


def peak(center: float, height: float = 1.0, width: float = 0.10) -> np.ndarray:
    return height * np.exp(-0.5 * ((GRID - center) / width) ** 2)


def a_library() -> PatternLibrary:
    """An illite that never moves and a smectite that collapses on drying."""
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


def mount(library, amounts, treatment):
    values = np.zeros_like(GRID)
    for entry, amount in zip(library.entries, amounts):
        pattern = (entry.intensity if treatment == "glycol" or entry.air_intensity is None
                   else entry.air_intensity)
        values += amount * pattern
    return Pattern(two_theta=GRID, intensity=values, name=treatment)


def test_the_glycol_view_is_the_library_itself():
    library = a_library()
    assert library.for_treatment("glycol") is library
    assert library.for_treatment() is library


def test_the_air_view_swaps_only_what_changes():
    library = a_library()
    air = library.for_treatment("air")
    assert np.allclose(air.entries[0].intensity, library.entries[0].intensity)
    assert np.allclose(air.entries[1].intensity, library.entries[1].air_intensity)
    # and it leaves everything a coefficient depends on alone
    for before, after in zip(library.entries, air.entries):
        assert after.normalization == before.normalization
        assert after.unit_mass == before.unit_mass
        assert after.fraction == before.fraction


def test_the_heated_mount_is_refused_rather_than_fitted_wrong():
    """Heating takes the interlayer to 10 A and destroys the kaolinite; none of
    that is calculated, so fitting it with the glycol patterns would repeat the
    error on a larger scale."""
    with pytest.raises(ValueError, match="no model for a mount heated"):
        a_library().for_treatment("heated")
    with pytest.raises(ValueError, match="no model for a mount heated"):
        nnls_fit(mount(a_library(), [1.0, 1.0], "glycol"), a_library(), treatment="heated")


def test_a_glycol_only_library_refuses_the_air_mount():
    """Falling back to the glycol patterns is the error itself, so it refuses."""
    library = a_library()
    library.entries[1].air_intensity = None
    with pytest.raises(ValueError, match="only the glycolated patterns"):
        library.for_treatment("air")


def test_an_unknown_treatment_is_refused():
    with pytest.raises(ValueError, match="unknown treatment"):
        a_library().for_treatment("boiled")


def test_the_air_mount_fitted_as_glycol_loses_the_smectite():
    """The measurement that prompted this, in miniature.

    The air-dried mount of a specimen that is half smectite has no intensity
    at 5.24 deg - the smectite is collapsed - so a fit with the glycol patterns
    cannot use the smectite entry at all, and reports a specimen without one.
    """
    library = a_library()
    air = mount(library, [1.0, 1.0], "air")

    wrong = nnls_fit(air, library, treatment="glycol")
    got = dict(zip(wrong.names, wrong.coefficients))
    assert got["smectite_EG"] == pytest.approx(0.0, abs=1e-9)

    right = nnls_fit(air, library, treatment="air")
    got = dict(zip(right.names, right.coefficients))
    assert got["smectite_EG"] == pytest.approx(1.0, rel=1e-6)
    assert got["illite"] == pytest.approx(1.0, rel=1e-6)
    assert right.r_wp < wrong.r_wp


def test_the_glycol_mount_is_unaffected():
    """The default path must be exactly what it was."""
    library = a_library()
    glycol = mount(library, [1.0, 1.0], "glycol")
    got = dict(zip(*(lambda r: (r.names, r.coefficients))(nnls_fit(glycol, library))))
    assert got["smectite_EG"] == pytest.approx(1.0, rel=1e-6)
    assert got["illite"] == pytest.approx(1.0, rel=1e-6)


def test_the_family_search_uses_the_treatment_too():
    """It has its own design matrix, and it decided which entries survive."""
    library = a_library()
    air = mount(library, [1.0, 1.0], "air")
    chosen = select_one_per_family(air, library, families=["", "smectite"],
                                   treatment="air")
    got = dict(zip(chosen.result.names, chosen.result.coefficients))
    assert got["smectite_EG"] == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(ValueError, match="no model for a mount heated"):
        select_one_per_family(air, library, families=["", "smectite"], treatment="heated")
