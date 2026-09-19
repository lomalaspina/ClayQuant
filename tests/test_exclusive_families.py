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
    orientation_families,
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


# --------------------------------------------------------------------------
# One orientation per composition
# --------------------------------------------------------------------------

def a_composition(phase: str, thickness: float, centre: float,
                  orientations=(0.1, 0.3, 0.6, 1.0)) -> list[LibraryEntry]:
    """One composition at several orientations.

    A March-Dollase parameter scales a basal series without changing its shape,
    so the members here differ in height and in how the orders fall away - which
    is what the fit sees - and share everything else.
    """
    out = []
    for r in orientations:
        intensity = sum(
            peak(order * centre, (0.35 / r) ** (order - 1), 0.25)
            for order in (1, 2, 3)
        )
        out.append(entry(f"{phase} PO={r:g} d={thickness:g}", phase, intensity,
                         march_dollase=r, thickness=thickness))
    return out


def test_a_composition_is_one_family_whatever_its_orientation():
    library = PatternLibrary(two_theta=GRID, entries=(
        a_composition("illite", 9.95, 8.88) + a_composition("illite", 10.10, 8.75)
        + [sharp("Quartz", 20.86)]
    ))
    labels = orientation_families(library)
    assert labels[:4] == ["illite d=9.95"] * 4
    assert labels[4:8] == ["illite d=10.1"] * 4
    assert labels[8] == ""
    # Two compositions of one phase are two families, not one: a specimen may
    # hold two illite populations of different spacing, and that is not the
    # double count the orientation is.
    assert len(set(labels) - {""}) == 2


def test_the_interstratified_composition_is_part_of_the_family_key():
    library = PatternLibrary(two_theta=GRID, entries=[
        entry("I/S 0.90 PO=0.1", "I/S", peak(8.8, 1.0, 1.0), march_dollase=0.1,
              fraction=0.90, csds_mean=49.9, thickness=9.9),
        entry("I/S 0.90 PO=0.5", "I/S", peak(8.8, 0.6, 1.0), march_dollase=0.5,
              fraction=0.90, csds_mean=49.9, thickness=9.9),
        entry("I/S 0.50 PO=0.1", "I/S", peak(7.9, 1.0, 1.4), march_dollase=0.1,
              fraction=0.50, csds_mean=49.9, thickness=9.9),
    ])
    labels = orientation_families(library)
    assert labels[0] == labels[1]          # same composition, two orientations
    assert labels[2] != labels[0]          # different host fraction, own family
    assert "0.90 host" in labels[0] and "N=49.9" in labels[0] and "d=9.9" in labels[0]


def test_two_orientations_of_one_composition_cannot_both_be_fitted():
    """The defect this exists to remove.

    A free fit will happily take some of a composition at r = 0.1 and some at
    r = 1 to fake an intermediate orientation.  That double counts the randomly
    oriented crystallites, which the r = 0.1 distribution already contains, and
    it makes the reported r the mean of two values that is the orientation of
    nothing.
    """
    entries = a_composition("illite", 9.95, 8.88) + [sharp("Quartz", 20.86)]
    library = PatternLibrary(two_theta=GRID, entries=entries)
    # A specimen built from two orientations at once - exactly what must not
    # come back out.
    coefficients = [1.0, 0.0, 0.0, 0.8, 0.5]
    measured = measured_from(library, coefficients, seed=7)

    free = nnls_fit(measured, library)
    used = sum(1 for phase, coefficient in zip(free.phases, free.coefficients)
               if phase == "illite" and coefficient > 0.0)
    assert used > 1, "the free fit should mix orientations, or this proves nothing"

    picked = select_one_per_family(measured, library,
                                   families=orientation_families(library))
    used = sum(1 for phase, coefficient
               in zip(picked.result.phases, picked.result.coefficients)
               if phase == "illite" and coefficient > 0.0)
    assert used == 1
    # And the reported orientation is then one of the library's own values
    # rather than an average of several.
    from clayquant.quantification import quantify

    illite = next(share for share in quantify(picked.result).shares
                  if share.phase == "illite")
    assert illite.orientation in (0.1, 0.3, 0.6, 1.0)


def test_the_screen_does_not_change_the_answer():
    """It only removes families an unrestricted fit gave nothing to."""
    # Three compositions the specimen holds and two it does not, so the screen
    # has something to remove.
    library = PatternLibrary(two_theta=GRID, entries=(
        a_composition("illite", 9.95, 8.88)
        + a_composition("chlorite", 14.2, 6.2)
        + a_composition("kaolinite_2M", 7.15, 12.4)
        + a_composition("I/S", 9.9, 5.1)
        + a_composition("C/S", 14.4, 30.2)
        + [sharp("Quartz", 20.86)]
    ))
    families = orientation_families(library)
    coefficients = ([0.0, 1.0, 0.0, 0.0] + [0.0] * 4 + [0.0, 0.0, 0.7, 0.0]
                    + [0.0] * 8 + [0.6])
    measured = measured_from(library, coefficients, seed=3)
    screened = select_one_per_family(measured, library, families=families)
    whole = select_one_per_family(measured, library, families=families, screen=False)
    assert screened.chosen == whole.chosen
    assert screened.evaluations <= whole.evaluations


def test_a_family_with_nothing_in_the_fitted_range_is_screened_out():
    """Which is what makes the search affordable on the real library.

    There the screen leaves 15 families of 176 and the search takes two seconds
    instead of four minutes.  A synthetic problem small enough to check by hand
    does not reproduce that on its own: an over-complete non-negative fit of a
    noisy pattern gives almost every column some small coefficient, so the
    screen only bites where a composition really has nothing to offer.
    """
    library = PatternLibrary(two_theta=GRID, entries=(
        a_composition("illite", 9.95, 8.88)
        + a_composition("C/S", 14.4, 30.5)      # every line above the fitted range
        + [sharp("Quartz", 20.86)]
    ))
    families = orientation_families(library)
    coefficients = [0.0, 1.0, 0.0, 0.0] + [0.0] * 4 + [0.6]
    measured = measured_from(library, coefficients, seed=4)
    picked = select_one_per_family(measured, library, families=families,
                                   range_two_theta=(4.0, 25.0))
    assert picked.screened_out == 1
    assert "C/S d=14.4" not in picked.chosen
    assert picked.reinstated == ()


def test_a_family_the_screen_drops_can_be_reinstated():
    """The screen's assumption is checked rather than trusted.

    A composition that took nothing when every column was available may be
    wanted once the others are restricted to one orientation each, so each
    screened-out family is offered back and kept if it improves the fit.
    """
    library = PatternLibrary(two_theta=GRID, entries=(
        a_composition("illite", 9.95, 8.88) + a_composition("I/S", 9.9, 8.6)
        + [sharp("Quartz", 20.86)]
    ))
    families = orientation_families(library)
    coefficients = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.6]
    measured = measured_from(library, coefficients, seed=9)
    picked = select_one_per_family(measured, library, families=families)
    # Whatever the screen did, the answer holds one orientation per composition
    # and the metadata says what happened.
    recorded = picked.result.metadata["family_selection"]
    assert recorded["screened_out"] >= 0
    assert isinstance(recorded["reinstated"], list)
    for phase in ("illite", "I/S"):
        used = sum(1 for name, coefficient
                   in zip(picked.result.phases, picked.result.coefficients)
                   if name == phase and coefficient > 0.0)
        assert used <= 1, phase
