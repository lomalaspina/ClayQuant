"""Reporting a parameter that the library stopped rather than determined.

The fit chooses among pre-computed patterns, so it cannot return a value the
library does not hold.  When a phase's coefficients collect on the end of a
spanned range, the honest reading is not that the parameter equals that value
but that the data wanted to go further and ran out of library.  In a table of
numbers that all look alike, that difference is invisible - and it is the same
signal that marks a refined parameter sitting on its limit, which is worth
distrusting wherever it appears.

The case that prompted this: a measured pure illite put 94 % of its illite onto
the 0.99 illite/smectite entry, the highest the series went, reporting 1 % of
smectite the specimen did not have and a bound dressed as a composition.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.library import build_library
from clayquant.models import available_phases
from clayquant.nnls import nnls_fit, parameters_at_an_edge
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, Pattern
from clayquant.profile import PeakShape

pytestmark = pytest.mark.skipif(
    not all(available_phases().values()), reason="ICSD CIF files are not installed"
)

GRID = np.arange(3.008, 40.0, 0.0334)


@pytest.fixture(scope="module")
def library():
    return build_library(
        grid=GRID,
        instrument=Instrument(
            peak_shape=PeakShape(u=0.004, v=-0.001, w=0.002, eta=0.5, size_ab=600.0),
            divergence=Divergence(),
        ),
        orientations=(0.3, 0.6, 1.0),
        illite_smectite=(0.60, 0.80, 1.00),
        chlorite_smectite=(0.90,),
        csds_means=(5.0, 15.0),
        host_thicknesses={},
    )


def fit_entry(library, name):
    entry = library.entries[library.names.index(name)]
    pattern = Pattern(library.two_theta, entry.normalization * entry.intensity, name=name)
    return nnls_fit(pattern, library, range_two_theta=(4.0, 34.0))


def test_the_library_reports_what_it_spans(library):
    axes = library.spanned()
    assert "I/S/fraction" in axes
    assert axes["I/S/fraction"] == [0.60, 0.80, 1.00]
    assert axes["illite/march_dollase"] == [0.3, 0.6, 1.0]
    # A parameter with one value only is not a span and is not reported.
    assert all(len(values) > 1 for values in axes.values())


def test_a_phase_name_with_a_slash_is_not_split_by_it(library):
    """"I/S" carries a slash itself, so the parameter is what follows the last one."""
    name = next(n for n in library.names if n.startswith("I/S 1.00"))
    notes = parameters_at_an_edge(fit_entry(library, name), library)
    assert notes, "the end member sits on the end of the fraction range by construction"
    assert any(note.startswith("I/S: ") for note in notes)
    assert not any(note.startswith("I: ") for note in notes)
    assert any("fraction" in note and "S/fraction" not in note for note in notes)


def test_an_end_of_range_value_is_reported(library):
    name = next(n for n in library.names if n.startswith("I/S 1.00"))
    notes = parameters_at_an_edge(fit_entry(library, name), library)
    fraction = next(note for note in notes if "fraction" in note)
    assert "highest" in fraction and "1" in fraction
    assert "bound here and not a measurement" in fraction


def test_a_value_inside_the_range_is_not_reported(library):
    """The middle of a span is a result, and must not be flagged as a bound."""
    name = next(n for n in library.names if n.startswith("I/S 0.80"))
    notes = parameters_at_an_edge(fit_entry(library, name), library)
    assert not any("fraction" in note for note in notes), notes


def test_a_scatter_across_the_range_is_not_reported(library):
    """Including an end value among others is not piling up on it."""
    names = [n for n in library.names if n.startswith("I/S") and " PO=0.3" in n]
    entries = [library.entries[library.names.index(n)] for n in names]
    total = sum(e.normalization * e.intensity for e in entries)
    result = nnls_fit(Pattern(library.two_theta, total, name="all three"),
                      library, range_two_theta=(4.0, 34.0))
    fraction_notes = [n for n in parameters_at_an_edge(result, library) if "fraction" in n]
    assert not fraction_notes, fraction_notes


def test_the_threshold_is_adjustable(library):
    name = next(n for n in library.names if n.startswith("I/S 1.00"))
    result = fit_entry(library, name)
    assert parameters_at_an_edge(result, library, threshold=0.99)
    assert not parameters_at_an_edge(result, library, threshold=1.01)


def test_the_fit_records_it_without_being_asked(library):
    name = next(n for n in library.names if n.startswith("I/S 1.00"))
    result = fit_entry(library, name)
    assert "parameters_at_an_edge" in result.metadata
    assert result.metadata["parameters_at_an_edge"] == parameters_at_an_edge(result, library)


def test_the_illite_series_reaches_its_pure_end_member():
    """The gap a measured pure illite standard exposed."""
    from clayquant.library import CHLORITE_SMECTITE_FRACTIONS, ILLITE_SMECTITE_FRACTIONS

    assert max(ILLITE_SMECTITE_FRACTIONS) == 1.00
    assert max(CHLORITE_SMECTITE_FRACTIONS) == 1.00
