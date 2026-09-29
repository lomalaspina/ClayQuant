"""The mixed-layer series stop short of their pure end members.

A 1.00 entry is a second pure illite or chlorite.  The library already holds
one as a discrete phase, the two are near-duplicates, and because they carry
different phase names no family rule can merge them: the fit divides one
mineral between two names and the quantification reports it twice.
"""
from __future__ import annotations

import numpy as np
import pytest

from clayquant.library import (
    CHLORITE_SMECTITE_FRACTIONS,
    ILLITE_SMECTITE_FRACTIONS,
    LibraryEntry,
    PatternLibrary,
)
from clayquant.nnls import clay_families, orientation_families, parameters_at_an_edge


def test_neither_series_reaches_its_pure_end_member():
    assert 1.0 not in ILLITE_SMECTITE_FRACTIONS
    assert 1.0 not in CHLORITE_SMECTITE_FRACTIONS


def test_the_series_still_reach_close_enough_to_describe_a_near_pure_clay():
    """Dropping the end member must not leave a gap a real specimen falls into."""
    assert max(ILLITE_SMECTITE_FRACTIONS) >= 0.99
    assert max(CHLORITE_SMECTITE_FRACTIONS) >= 0.95


def test_every_remaining_series_entry_is_genuinely_expandable():
    """The air-dried restraint acts on entries with fraction < 1, and now that is
    all of them - which is what the series is for."""
    assert all(f < 1.0 for f in ILLITE_SMECTITE_FRACTIONS)
    assert all(f < 1.0 for f in CHLORITE_SMECTITE_FRACTIONS)


def _library_with_top_entry_only() -> PatternLibrary:
    grid = np.linspace(4.0, 40.0, 60)
    entries = [
        LibraryEntry(name=f"I/S {f:.2f}/{1 - f:.2f} PO=0.3", phase="I/S",
                     intensity=np.full_like(grid, 0.5), fraction=f)
        for f in (0.95, 0.99)
    ]
    return PatternLibrary(two_theta=grid, entries=entries)


def test_a_pure_illite_piling_on_the_top_entry_is_flagged_not_hidden():
    """The cost of dropping 1.00, made detectable.

    A pure illite can now only be described as 0.99, which reports 1 % of a
    smectite that is not there.  That is a bound rather than a measurement, and
    the edge detector is what says so - the honest reading of the flag being
    that the discrete illite phase is the entry for this specimen.
    """
    from clayquant.nnls import FitResult

    library = _library_with_top_entry_only()
    zeros = np.zeros_like(library.two_theta)
    result = FitResult(
        two_theta=library.two_theta,
        observed=zeros,
        calculated=zeros,
        coefficients=np.array([0.02, 0.98]),
        names=[e.name for e in library.entries],
        phases=[e.phase for e in library.entries],
        r_wp=0.1,
        r_p=0.1,
        scattering_fraction=np.array([0.02, 0.98]),
        amplitude_fraction=np.array([0.02, 0.98]),
        mask=np.ones_like(library.two_theta, dtype=bool),
    )
    flagged = parameters_at_an_edge(result, library)
    assert any("fraction" in message for message in flagged), flagged


def test_the_two_family_rules_still_cannot_merge_across_phase_names():
    """Why dropping the entry was the fix rather than a family change.

    Families group entries within a phase, so a 'C/S' entry and a 'chlorite'
    entry are never in one family however the label is built.  This records the
    constraint that made the duplication unfixable any other way.
    """
    grid = np.linspace(4.0, 40.0, 20)
    library = PatternLibrary(
        two_theta=grid,
        entries=[
            LibraryEntry(name="C/S 1.00/0.00 PO=0.3", phase="C/S",
                         intensity=np.ones_like(grid), fraction=1.0),
            LibraryEntry(name="chlorite PO=0.3", phase="chlorite",
                         intensity=np.ones_like(grid)),
        ],
    )
    for rule in (clay_families, orientation_families):
        labels = rule(library)
        assert labels[0] != labels[1], rule.__name__
