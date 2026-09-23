"""Choosing the composition first and the crystallite thickness second.

Clayfit settles the orientation with every interstratified pattern held at one
thickness, and only then offers each chosen composition its other thicknesses.
The argument is about how a coordinate search fails rather than about physics: a
search that has to settle twenty coupled choices can settle them worse than one
that settles fifteen and then five.
"""

import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import select_in_two_stages, select_one_per_family
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 30.0, 0.02)
BACKGROUND = 300.0


def peak(centre, height, width=0.12):
    return height * np.exp(-4.0 * np.log(2.0) * ((GRID - centre) / width) ** 2)


class KnownBackground:
    def subtract(self, two_theta, intensity):
        return np.asarray(intensity, dtype=float) - BACKGROUND


def library() -> PatternLibrary:
    """Two compositions at two orientations and three crystallite thicknesses."""
    entries = []
    for fraction in (1.00, 0.90):
        for r in (0.3, 1.0):
            for mean in (5.0, 15.0, 50.0):
                width = 0.30 - 0.004 * mean
                series = (
                    peak(8.85, 1.0, width)
                    + peak(17.8, 0.45 * r, width)
                    + (peak(9.60, 0.30, width) if fraction < 1.0 else 0.0)
                )
                entries.append(
                    LibraryEntry(
                        name=f"I/S {fraction:.2f}/{1 - fraction:.2f} PO={r:g} N={mean:g}",
                        phase="I/S", intensity=series / series.max(),
                        march_dollase=r, fraction=fraction, csds_mean=mean,
                        unit_mass=100.0, unit_volume=100.0,
                    )
                )
    return PatternLibrary(two_theta=GRID, entries=entries)


def measurement(entry, seed: int = 20260923) -> Pattern:
    rng = np.random.default_rng(seed)
    return Pattern(
        two_theta=GRID,
        intensity=rng.poisson(3000.0 * entry.intensity + BACKGROUND).astype(float),
        name="mount",
    )


def families(lib):
    import re

    return [re.sub(r"\s*PO=[-+0-9.eE]+", "", entry.name) for entry in lib.entries]


ARGUMENTS = dict(background=KnownBackground(), range_two_theta=(4.0, 29.0))


def test_both_searches_find_the_composition_that_made_the_measurement():
    lib = library()
    truth = lib.entries[7]
    measured = measurement(truth)
    joint = select_one_per_family(measured, lib, families=families(lib), **ARGUMENTS)
    staged = select_in_two_stages(measured, lib, families(lib), **ARGUMENTS)
    for chosen in (joint, staged):
        best = max(zip(chosen.result.names, chosen.result.coefficients),
                   key=lambda row: row[1])[0]
        assert best.startswith("I/S 0.90/0.10")


def test_the_second_stage_can_reach_a_thickness_the_first_never_offered():
    """The first stage holds one thickness; the answer must not be stuck there."""
    lib = library()
    # The thinnest crystallite, which is not the one the first stage holds.
    truth = next(e for e in lib.entries
                 if e.fraction == 0.90 and e.march_dollase == 0.3 and e.csds_mean == 5.0)
    staged = select_in_two_stages(measurement(truth), lib, families(lib), **ARGUMENTS)
    best = max(zip(staged.result.names, staged.result.coefficients),
               key=lambda row: row[1])[0]
    assert "N=5" in best


def test_one_thickness_makes_it_the_same_search():
    lib = library()
    single = PatternLibrary(
        two_theta=GRID,
        entries=[e for e in lib.entries if e.csds_mean == 15.0],
    )
    labels = families(single)
    measured = measurement(single.entries[0])
    joint = select_one_per_family(measured, single, families=labels, **ARGUMENTS)
    staged = select_in_two_stages(measured, single, labels, **ARGUMENTS)
    assert staged.result.r_wp == pytest.approx(joint.result.r_wp)


def test_the_fits_of_both_stages_are_counted():
    lib = library()
    staged = select_in_two_stages(measurement(lib.entries[7]), lib, families(lib),
                                  **ARGUMENTS)
    assert staged.evaluations > 0


def test_one_label_per_entry_is_required():
    lib = library()
    with pytest.raises(ValueError):
        select_in_two_stages(measurement(lib.entries[0]), lib, ["only one"])
