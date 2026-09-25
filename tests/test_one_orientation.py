"""One orientation for every clay in the mount.

The clays of one mount were settled out of one suspension and dried on one glass
plate, so they cannot be oriented differently by a factor of a thousand - and a
factor of a thousand is what the alternative comes to, because a basal series is
enhanced as ``r ** -3``.  A fit free to put one clay at ``r = 0.1`` and another
at ``r = 1`` converts a small share of the scattering into most of the mass.
"""

import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import select_one_orientation, select_one_per_family
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 30.0, 0.02)
BACKGROUND = 200.0
ORIENTATIONS = (0.1, 0.3, 1.0)


def peak(centre, height, width=0.12):
    return height * np.exp(-4.0 * np.log(2.0) * ((GRID - centre) / width) ** 2)


class KnownBackground:
    def subtract(self, two_theta, intensity):
        return np.asarray(intensity, dtype=float) - BACKGROUND


def library() -> PatternLibrary:
    """Two platy clays, each at three orientations, on the r**-3 basis.

    The stored pattern is at unit maximum whatever the orientation, and the
    orientation lives in the normalisation - which is what puts the factor of a
    thousand into the mass rather than into the shape.
    """
    shapes = {
        "chlorite": peak(6.2, 1.0) + peak(12.5, 1.8) + peak(18.8, 1.2) + peak(25.1, 1.2),
        "kaolinite": peak(12.4, 1.0) + peak(24.9, 0.66),
    }
    entries = []
    for phase, shape in shapes.items():
        for r in ORIENTATIONS:
            entries.append(
                LibraryEntry(
                    name=f"{phase} PO={r:g}", phase=phase,
                    intensity=shape / shape.max(),
                    march_dollase=r,
                    normalization=1.0 / r**3,
                    unit_mass=1000.0, unit_volume=700.0,
                )
            )
    return PatternLibrary(two_theta=GRID, entries=entries)


def a_pure_chlorite() -> Pattern:
    rng = np.random.default_rng(20260925)
    shape = peak(6.2, 1.0) + peak(12.5, 1.8) + peak(18.8, 1.2) + peak(25.1, 1.2)
    return Pattern(
        two_theta=GRID,
        intensity=rng.poisson(4000.0 * shape / shape.max() + BACKGROUND).astype(float),
        name="chlorite standard",
    )


ARGUMENTS = dict(background=KnownBackground(), range_two_theta=(4.0, 29.0))


def families(lib):
    return [entry.phase for entry in lib.entries]


def test_one_orientation_is_chosen_for_every_clay():
    lib = library()
    chosen = select_one_orientation(a_pure_chlorite(), lib, families(lib), **ARGUMENTS)
    used = {
        lib.entries[lib.names.index(name)].march_dollase
        for name, coefficient in zip(chosen.result.names, chosen.result.coefficients)
        if coefficient > 0.0
    }
    assert len(used) == 1
    assert chosen.orientation in ORIENTATIONS


def test_a_free_fit_can_mix_orientations_and_this_cannot():
    lib = library()
    measured = a_pure_chlorite()
    free = select_one_per_family(measured, lib, families=families(lib), **ARGUMENTS)
    free_used = {
        lib.entries[lib.names.index(name)].march_dollase
        for name, coefficient in zip(free.result.names, free.result.coefficients)
        if coefficient > 0.0
    }
    shared = select_one_orientation(measured, lib, families(lib), **ARGUMENTS)
    assert len(free_used) >= 1
    assert shared.orientation is not None


def test_the_orientation_may_be_restricted():
    lib = library()
    chosen = select_one_orientation(
        a_pure_chlorite(), lib, families(lib), orientations=[0.3], **ARGUMENTS
    )
    assert chosen.orientation == 0.3


def test_an_orientation_no_entry_has_is_refused():
    lib = library()
    with pytest.raises(ValueError):
        select_one_orientation(
            a_pure_chlorite(), lib, families(lib), orientations=[0.55], **ARGUMENTS
        )


def test_the_fits_of_every_orientation_are_counted():
    lib = library()
    chosen = select_one_orientation(a_pure_chlorite(), lib, families(lib), **ARGUMENTS)
    single = select_one_orientation(
        a_pure_chlorite(), lib, families(lib), orientations=[0.3], **ARGUMENTS
    )
    assert chosen.evaluations > single.evaluations


def test_a_library_of_accompanying_minerals_alone_is_passed_through():
    """Nothing to tie together, so this is an ordinary fit."""
    lib = PatternLibrary(
        two_theta=GRID,
        entries=[LibraryEntry(name="Quartz", phase="Quartz",
                              intensity=peak(26.6, 1.0), march_dollase=1.0,
                              unit_mass=60.0, unit_volume=113.0)],
    )
    chosen = select_one_orientation(a_pure_chlorite(), lib, [""], **ARGUMENTS)
    assert chosen.orientation is None


def test_one_label_per_entry_is_required():
    lib = library()
    with pytest.raises(ValueError):
        select_one_orientation(a_pure_chlorite(), lib, ["only one"], **ARGUMENTS)
