"""One linked layer spacing per family, refined rather than spanned.

The library spans the layer spacing on a grid, four values per host, and a
specimen's spacing is not on the grid.  A pattern calculated at 10.02 A fitted to
a specimen at 9.97 A is wrong by most of a peak width at the 002, and a
non-negative fit answers a peak in the wrong place by taking less of that pattern
and making up the difference from whatever else has intensity nearby - which is
how a mis-set spacing becomes a mis-stated phase.  These tests pin the refinement
that covers the space between the grid points.
"""

import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import (
    MAXIMUM_LATTICE_DEVIATION,
    apply_lattice_scales,
    bragg_resample,
    orientation_families,
    select_one_per_family,
    select_with_lattice_scaling,
)
from clayquant.pattern import Pattern

WAVELENGTH = 1.540596
GRID = np.arange(4.0, 34.0, 0.02)
BACKGROUND = 200.0


def angle_of(d: float) -> float:
    return 2.0 * np.degrees(np.arcsin(WAVELENGTH / (2.0 * d)))


def spacing_at(two_theta: float) -> float:
    return WAVELENGTH / (2.0 * np.sin(np.radians(two_theta / 2.0)))


def basal_series(d001: float, heights=(1.0, 0.5, 0.3), width: float = 0.12):
    series = np.zeros_like(GRID)
    for order, height in enumerate(heights, start=1):
        series += height * np.exp(
            -4.0 * np.log(2.0) * ((GRID - angle_of(d001 / order)) / width) ** 2
        )
    return series


def peak_angles(profile: np.ndarray, windows) -> list[float]:
    found = []
    for low, high in windows:
        inside = np.flatnonzero((GRID > low) & (GRID < high))
        found.append(float(GRID[inside[int(np.argmax(profile[inside]))]]))
    return found


WINDOWS = ((7.0, 11.0), (16.0, 19.0), (25.0, 29.0))


def test_every_order_moves_with_one_spacing():
    moved = bragg_resample(GRID, basal_series(10.0), 1.01)
    spacings = [spacing_at(angle) for angle in peak_angles(moved, WINDOWS)]
    assert spacings[0] == pytest.approx(10.1, abs=0.02)
    assert spacings[1] == pytest.approx(5.05, abs=0.01)
    assert spacings[2] == pytest.approx(3.367, abs=0.01)


def test_a_scale_of_one_changes_nothing():
    series = basal_series(10.0)
    assert np.array_equal(bragg_resample(GRID, series, 1.0), series)


def test_the_peak_height_is_preserved():
    series = basal_series(10.0)
    for scale in (0.98, 1.02):
        moved = bragg_resample(GRID, series, scale)
        assert float(moved.max()) == pytest.approx(float(series.max()), rel=1e-9)


def test_areas_are_not_preserved_and_that_is_the_point():
    """Heights are kept so the library's unit-maximum normalisation still holds."""
    series = basal_series(10.0)
    moved = bragg_resample(GRID, series, 1.02)
    assert float(np.trapezoid(moved, GRID)) != pytest.approx(
        float(np.trapezoid(series, GRID)), rel=1e-6
    )


def test_a_bad_scale_is_refused():
    series = basal_series(10.0)
    for scale in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            bragg_resample(GRID, series, scale)


def library(d001: float = 10.00) -> PatternLibrary:
    """Two orientations of one illite, and an unstretchable accompanying mineral."""
    entries = []
    for r, heights in ((0.3, (1.0, 0.50, 0.30)), (1.0, (1.0, 0.30, 0.18))):
        series = basal_series(d001, heights)
        entries.append(
            LibraryEntry(
                name=f"illite PO={r:g}", phase="illite", intensity=series / series.max(),
                march_dollase=r, thickness=d001, unit_mass=100.0, unit_volume=100.0,
            )
        )
    quartz = np.exp(-4.0 * np.log(2.0) * ((GRID - 26.64) / 0.1) ** 2)
    entries.append(
        LibraryEntry(
            name="Quartz", phase="Quartz", intensity=quartz, march_dollase=1.0,
            unit_mass=60.0, unit_volume=113.0,
        )
    )
    return PatternLibrary(two_theta=GRID, entries=entries)


def specimen(d001: float) -> Pattern:
    rng = np.random.default_rng(20260923)
    series = basal_series(d001, (1.0, 0.50, 0.30))
    signal = 4000.0 * series / series.max()
    return Pattern(
        two_theta=GRID,
        intensity=rng.poisson(signal + BACKGROUND).astype(float),
        name="mount",
    )


class KnownBackground:
    def subtract(self, two_theta, intensity):
        return np.asarray(intensity, dtype=float) - BACKGROUND


def test_a_spacing_between_the_grid_points_is_recovered():
    lib = library(10.00)
    measured = specimen(10.06)
    families = orientation_families(lib)
    chosen = select_with_lattice_scaling(
        measured, lib, families,
        background=KnownBackground(), range_two_theta=(4.0, 33.0),
    )
    assert chosen.lattice_scales["illite"] == pytest.approx(1.006, abs=0.003)


def test_refining_the_spacing_fits_better_than_the_grid_alone():
    lib = library(10.00)
    measured = specimen(10.06)
    families = orientation_families(lib)
    arguments = dict(
        background=KnownBackground(), range_two_theta=(4.0, 33.0),
    )
    plain = select_one_per_family(measured, lib, families=families, **arguments)
    refined = select_with_lattice_scaling(measured, lib, families, **arguments)
    assert refined.result.r_wp < plain.result.r_wp


def test_an_accompanying_mineral_is_never_stretched():
    """Quartz is fitted from its published cell and has no business moving."""
    lib = library(10.00)
    families = orientation_families(lib)
    moved = apply_lattice_scales(lib, [e.phase for e in lib.entries], {"illite": 1.02})
    quartz = lib.entries[2]
    assert np.array_equal(moved.entries[2].intensity, quartz.intensity)
    assert not np.array_equal(moved.entries[0].intensity, lib.entries[0].intensity)


def test_no_deviation_leaves_the_selection_alone():
    lib = library(10.00)
    measured = specimen(10.06)
    families = orientation_families(lib)
    arguments = dict(background=KnownBackground(), range_two_theta=(4.0, 33.0))
    plain = select_one_per_family(measured, lib, families=families, **arguments)
    off = select_with_lattice_scaling(
        measured, lib, families, maximum_deviation=0.0, **arguments
    )
    assert off.lattice_scales == {}
    assert off.result.r_wp == pytest.approx(plain.result.r_wp)


def test_the_library_the_result_came_from_is_returned():
    lib = library(10.00)
    chosen = select_with_lattice_scaling(
        specimen(10.06), lib, orientation_families(lib),
        background=KnownBackground(), range_two_theta=(4.0, 33.0),
    )
    assert chosen.library is not None
    assert len(chosen.library.entries) == len(lib.entries)


def test_a_bad_deviation_is_refused():
    lib = library(10.00)
    with pytest.raises(ValueError):
        select_with_lattice_scaling(
            specimen(10.0), lib, orientation_families(lib), maximum_deviation=1.5
        )
    assert 0.0 < MAXIMUM_LATTICE_DEVIATION < 1.0
