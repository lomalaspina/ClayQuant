"""One pattern per clay family, as Clayfit chooses them.

The problem this solves is measured in Sec. A.20 of the manual: an
over-complete clay library takes intensity from the accompanying minerals,
because a family of 1440 broad patterns with 1440 free coefficients can absorb
a sharp reflection that is not the family's.  Clayfit does not have the problem
because it fits one member per family.  These tests pin the search, the
objective, the margin it reports, and - the point of the whole thing - that it
gives a swallowed accompanying mineral back.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import (
    FamilySelection,
    clay_families,
    nnls_fit,
    select_one_per_family,
)
from clayquant.pattern import Pattern

GRID = np.arange(4.0, 34.0, 0.02)


def peak(centre: float, height: float, width: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((GRID - centre) / (width / 2.355)) ** 2)


def entry(name: str, phase: str, intensity: np.ndarray, **kwargs) -> LibraryEntry:
    top = float(np.max(intensity))
    return LibraryEntry(
        name=name, phase=phase,
        intensity=intensity / top if top > 0 else intensity,
        normalization=top or 1.0, unit_mass=100.0, unit_volume=100.0, **kwargs
    )


def broad_family(phase: str, count: int, centre: float = 8.8) -> list[LibraryEntry]:
    """A family of broad patterns differing only slightly - an I/S series.

    Broad and numerous is what makes a family able to swallow something else.
    """
    return [
        entry(f"{phase} {index}", phase,
              peak(centre + 0.04 * index, 1.0, 1.4 + 0.05 * index)
              + peak(2.0 * centre + 0.08 * index, 0.3, 1.8),
              march_dollase=0.1, fraction=0.5 + 0.4 * index / max(count - 1, 1))
        for index in range(count)
    ]


def sharp(name: str, centre: float) -> LibraryEntry:
    """A discrete accompanying mineral: two sharp lines."""
    return entry(name, name, peak(centre, 1.0, 0.10) + peak(centre + 5.8, 0.45, 0.10))


# --------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------

def test_clay_families_labels_clays_by_phase_and_leaves_the_rest_fixed():
    library = PatternLibrary(two_theta=GRID, entries=(
        broad_family("I/S", 3) + [entry("illite PO=0.1", "illite", peak(8.8, 1.0, 0.3))]
        + [sharp("Quartz", 20.86)]
    ))
    labels = clay_families(library)
    assert labels == ["I/S", "I/S", "I/S", "illite", ""]


def test_a_phase_can_be_pinned_into_every_fit():
    library = PatternLibrary(two_theta=GRID, entries=broad_family("I/S", 2))
    assert clay_families(library, fixed={"I/S"}) == ["", ""]


def test_a_wrong_number_of_labels_is_refused():
    library = PatternLibrary(two_theta=GRID, entries=broad_family("I/S", 3))
    measured = Pattern(two_theta=GRID, intensity=np.ones_like(GRID), name="m")
    with pytest.raises(ValueError, match="one label per"):
        select_one_per_family(measured, library, families=["I/S", "I/S"])


def test_an_empty_library_is_refused():
    library = PatternLibrary(two_theta=GRID, entries=[])
    measured = Pattern(two_theta=GRID, intensity=np.ones_like(GRID), name="m")
    with pytest.raises(ValueError, match="empty"):
        select_one_per_family(measured, library)


# --------------------------------------------------------------------------
# The search
# --------------------------------------------------------------------------

def two_small_families():
    entries = [
        entry("A0", "alpha", peak(10.0, 1.0, 0.3)),
        entry("A1", "alpha", peak(12.0, 1.0, 0.3)),
        entry("B0", "beta", peak(20.0, 1.0, 0.3)),
        entry("B1", "beta", peak(25.0, 1.0, 0.3)),
    ]
    return PatternLibrary(two_theta=GRID, entries=entries)


def measured_from(library, coefficients, seed=0):
    total = sum(c * e.intensity for c, e in zip(coefficients, library.entries))
    rng = np.random.default_rng(seed)
    counts = 200.0 + 4000.0 * total
    return Pattern(two_theta=GRID,
                   intensity=counts + rng.normal(0.0, np.sqrt(counts)),
                   name="synthetic")


def test_a_small_problem_is_tried_exhaustively_and_picks_the_right_members():
    library = two_small_families()
    # Built from A1 and B0 only.
    measured = measured_from(library, [0.0, 1.0, 1.0, 0.0])
    picked = select_one_per_family(
        measured, library,
        families=["alpha", "alpha", "beta", "beta"],
    )
    assert isinstance(picked, FamilySelection)
    assert picked.exhaustive
    assert picked.evaluations == 4          # 2 x 2, every one of them
    assert picked.chosen == {"alpha": "A1", "beta": "B0"}


def test_the_fit_holds_only_the_entries_that_were_kept():
    library = two_small_families()
    measured = measured_from(library, [0.0, 1.0, 1.0, 0.0])
    picked = select_one_per_family(measured, library,
                                   families=["alpha", "alpha", "beta", "beta"])
    assert picked.result.names == ["A1", "B0"]
    # And it is an ordinary FitResult, so everything downstream takes it.
    from clayquant.quantification import quantify
    quantified = quantify(picked.result)
    assert {share.phase for share in quantified.shares} == {"alpha", "beta"}


def test_the_fixed_entries_are_in_every_fit():
    library = PatternLibrary(two_theta=GRID, entries=[
        entry("A0", "alpha", peak(10.0, 1.0, 0.3)),
        entry("A1", "alpha", peak(12.0, 1.0, 0.3)),
        sharp("Quartz", 20.86),
    ])
    measured = measured_from(library, [1.0, 0.0, 1.0])
    picked = select_one_per_family(measured, library, families=["alpha", "alpha", ""])
    assert "Quartz" in picked.result.names
    assert picked.chosen == {"alpha": "A0"}


def test_the_coordinate_search_reaches_the_exhaustive_answer():
    """On a problem small enough to do both, so the answer is known."""
    library = PatternLibrary(two_theta=GRID, entries=(
        [entry(f"A{i}", "alpha", peak(8.0 + 0.7 * i, 1.0, 0.25)) for i in range(6)]
        + [entry(f"B{i}", "beta", peak(20.0 + 0.7 * i, 1.0, 0.25)) for i in range(6)]
        + [entry(f"C{i}", "gamma", peak(28.0 + 0.4 * i, 1.0, 0.25)) for i in range(6)]
    ))
    families = ["alpha"] * 6 + ["beta"] * 6 + ["gamma"] * 6
    coefficients = [0.0] * 18
    coefficients[3] = coefficients[6 + 4] = coefficients[12 + 1] = 1.0
    measured = measured_from(library, coefficients, seed=5)

    every = select_one_per_family(measured, library, families=families)
    searched = select_one_per_family(measured, library, families=families,
                                     max_exhaustive=0)
    assert every.exhaustive and not searched.exhaustive
    assert searched.chosen == every.chosen
    assert searched.evaluations < every.evaluations


def test_the_search_is_deterministic():
    library = PatternLibrary(two_theta=GRID, entries=(
        broad_family("I/S", 9) + broad_family("C/S", 9, centre=6.3)
    ))
    families = ["I/S"] * 9 + ["C/S"] * 9
    measured = measured_from(library, [0.0] * 4 + [1.0] + [0.0] * 13, seed=2)
    first = select_one_per_family(measured, library, families=families,
                                  max_exhaustive=0)
    second = select_one_per_family(measured, library, families=families,
                                   max_exhaustive=0)
    assert first.chosen == second.chosen
    assert first.objective == second.objective
    assert first.evaluations == second.evaluations


# --------------------------------------------------------------------------
# The margin
# --------------------------------------------------------------------------

def test_the_margin_is_how_much_worse_the_next_best_member_is():
    library = two_small_families()
    measured = measured_from(library, [0.0, 1.0, 1.0, 0.0])
    picked = select_one_per_family(measured, library,
                                   families=["alpha", "alpha", "beta", "beta"])
    # Both families have a genuine alternative that fits much worse: these
    # members are peaks two degrees apart, so the choice is settled.
    assert picked.margins["alpha"] > 0.1
    assert picked.margins["beta"] > 0.1


def test_a_family_of_one_has_no_margin_to_report():
    library = PatternLibrary(two_theta=GRID, entries=[
        entry("A0", "alpha", peak(10.0, 1.0, 0.3)),
        entry("S", "smectite_EG", peak(5.2, 1.0, 0.6)),
    ])
    measured = measured_from(library, [1.0, 1.0])
    picked = select_one_per_family(measured, library,
                                   families=["alpha", "smectite_EG"])
    assert math.isinf(picked.margins["alpha"])
    assert math.isinf(picked.margins["smectite_EG"])


def test_indistinguishable_members_are_reported_as_a_small_margin():
    """Two identical columns: the fit cannot choose, and it says so."""
    shape = peak(10.0, 1.0, 0.3)
    library = PatternLibrary(two_theta=GRID, entries=[
        entry("A0", "alpha", shape.copy()),
        entry("A1", "alpha", shape.copy()),
        entry("B0", "beta", peak(20.0, 1.0, 0.3)),
    ])
    measured = measured_from(library, [1.0, 0.0, 1.0])
    picked = select_one_per_family(measured, library,
                                   families=["alpha", "alpha", "beta"])
    assert picked.margins["alpha"] == pytest.approx(0.0, abs=1e-9)
    assert "not determined" in picked.summary()


def test_the_summary_names_what_was_chosen():
    library = two_small_families()
    measured = measured_from(library, [0.0, 1.0, 1.0, 0.0])
    picked = select_one_per_family(measured, library,
                                   families=["alpha", "alpha", "beta", "beta"])
    text = picked.summary()
    assert "A1" in text and "B0" in text
    assert "Rwp" in text


# --------------------------------------------------------------------------
# What it is for
# --------------------------------------------------------------------------

def test_a_broad_family_no_longer_swallows_a_sharp_accompanying_mineral():
    """The defect of Sec. A.20, reproduced and then removed.

    A family of forty broad patterns with forty free coefficients can build a
    sharp peak out of differences between them; the same family restricted to
    one pattern cannot, and the accompanying mineral gets its intensity back.
    """
    quartz = sharp("Quartz", 20.86)
    library = PatternLibrary(two_theta=GRID,
                             entries=broad_family("I/S", 40) + [quartz])
    families = ["I/S"] * 40 + [""]
    # A specimen that really holds one interstratified pattern and quartz.
    coefficients = [0.0] * 41
    coefficients[17] = 1.0
    coefficients[40] = 0.8
    measured = measured_from(library, coefficients, seed=11)

    free = nnls_fit(measured, library)
    picked = select_one_per_family(measured, library, families=families)

    def share_of(result, phase):
        total = 0.0
        for name, fraction in zip(result.phases, result.scattering_fraction):
            if name == phase:
                total += float(fraction)
        return total

    free_quartz = share_of(free, "Quartz")
    picked_quartz = share_of(picked.result, "Quartz")
    # The free fit gives quartz less than the restricted one does, because the
    # interstratified family took some of it.
    assert picked_quartz > free_quartz
    # And the restricted fit uses one interstratified pattern, not forty.
    assert sum(1 for phase, coefficient
               in zip(picked.result.phases, picked.result.coefficients)
               if phase == "I/S" and coefficient > 0.0) == 1
    assert sum(1 for phase, coefficient
               in zip(free.phases, free.coefficients)
               if phase == "I/S" and coefficient > 0.0) > 1


def test_the_objective_is_the_one_the_fit_is_judged_by():
    """So the choice is made on the same quantity as Rwp, not another."""
    library = two_small_families()
    measured = measured_from(library, [0.0, 1.0, 1.0, 0.0])
    picked = select_one_per_family(measured, library,
                                   families=["alpha", "alpha", "beta", "beta"])
    inside = picked.result.mask
    observed = picked.result.observed[inside]
    residual = observed - picked.result.calculated[inside]
    weights = 1.0 / np.clip(observed, 1.0, None)
    assert picked.objective == pytest.approx(float(np.sum(weights * residual ** 2)),
                                             rel=1e-6)


def test_the_metadata_records_how_the_choice_was_made():
    library = two_small_families()
    measured = measured_from(library, [0.0, 1.0, 1.0, 0.0])
    picked = select_one_per_family(measured, library,
                                   families=["alpha", "alpha", "beta", "beta"])
    recorded = picked.result.metadata["family_selection"]
    assert recorded["families"] == 2
    assert recorded["combinations"] == 4
    assert recorded["exhaustive"] is True
    assert set(recorded["margins"]) == {"alpha", "beta"}


def test_a_library_with_no_families_falls_back_to_an_ordinary_fit():
    library = PatternLibrary(two_theta=GRID, entries=[sharp("Quartz", 20.86)])
    measured = measured_from(library, [1.0])
    picked = select_one_per_family(measured, library, families=[""])
    assert picked.chosen == {}
    assert picked.result.names == ["Quartz"]
