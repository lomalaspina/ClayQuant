"""Reporting how much of an interstratified phase is actually the host.

The library spans illite/smectite from 0.20 to 0.99 illite layers, and the fit
groups every one of them under the phase name "I/S".  A fit that settles on the
0.99/0.01 entry has found a stack of essentially pure illite, and reporting that
as "I/S 63 %" reads as if 63 % of the specimen were an expandable clay.  On a
measurement of a pure illite standard, that is the difference between accepting
the material and rejecting it.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.library import build_library
from clayquant.models import available_phases
from clayquant.nnls import nnls_fit
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, Pattern
from clayquant.profile import PeakShape
from clayquant.quantification import quantify

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
        orientations=(0.3,),
        illite_smectite=(0.99, 0.60),
        chlorite_smectite=(0.90,),
        csds_means=(15.0,),
        host_thicknesses={},
    )


def fit_one(library, name):
    entry = library.entries[library.names.index(name)]
    pattern = Pattern(library.two_theta, entry.normalization * entry.intensity, name=name)
    result = nnls_fit(pattern, library, range_two_theta=(4.0, 34.0))
    return quantify(result)


def test_a_nearly_pure_stack_reports_its_composition(library):
    """The 0.99/0.01 entry must not be presented as an expandable clay."""
    name = next(n for n in library.names if n.startswith("I/S 0.99"))
    share = next(s for s in fit_one(library, name).shares if s.phase == "I/S")
    assert share.host_fraction == pytest.approx(0.99, abs=0.01)
    assert share.expandable_fraction == pytest.approx(0.01, abs=0.01)


def test_a_genuinely_mixed_stack_reports_its_own(library):
    name = next(n for n in library.names if n.startswith("I/S 0.60"))
    share = next(s for s in fit_one(library, name).shares if s.phase == "I/S")
    assert share.host_fraction == pytest.approx(0.60, abs=0.01)
    assert share.expandable_fraction == pytest.approx(0.40, abs=0.01)


def test_a_discrete_mineral_is_all_host(library):
    name = next(n for n in library.names if n.startswith("illite PO"))
    share = next(s for s in fit_one(library, name).shares if s.phase == "illite")
    assert share.host_fraction == pytest.approx(1.0)
    assert share.expandable_fraction == 0.0


def test_the_table_and_the_csv_carry_it(library, tmp_path):
    name = next(n for n in library.names if n.startswith("I/S 0.60"))
    quantification = fit_one(library, name)
    row = next(r for r in quantification.table() if r["phase"] == "I/S")
    assert row["host_fraction"] == pytest.approx(0.60, abs=0.01)
    path = quantification.to_csv(tmp_path / "shares.csv")
    header = next(line for line in path.read_text().splitlines() if line.startswith("phase,"))
    assert "host_fraction" in header


def test_a_library_without_the_field_still_quantifies(library):
    """An older .npz has no fractions; the report falls back to 1 rather than failing."""
    from dataclasses import replace

    stripped = replace(
        library, entries=[replace(entry, fraction=None) for entry in library.entries]
    )
    name = next(n for n in library.names if n.startswith("I/S 0.60"))
    entry = stripped.entries[stripped.names.index(name)]
    pattern = Pattern(stripped.two_theta, entry.normalization * entry.intensity, name=name)
    quantification = quantify(nnls_fit(pattern, stripped, range_two_theta=(4.0, 34.0)))
    share = next(s for s in quantification.shares if s.phase == "I/S")
    assert share.host_fraction == pytest.approx(1.0)


# --------------------------------------------------------------------------
# An end member of a series is the discrete mineral
# --------------------------------------------------------------------------

def test_an_end_member_is_reported_as_the_discrete_mineral():
    """``I/S 1.00/0.00`` is 100 % illite layers and 0 % smectite: it is illite."""
    from clayquant.quantification import reported_phase

    assert reported_phase("I/S", 1.0) == "illite"
    assert reported_phase("C/S", 1.0) == "chlorite"


def test_a_real_mixed_layer_keeps_its_name():
    from clayquant.quantification import reported_phase

    assert reported_phase("I/S", 0.99) == "I/S"
    assert reported_phase("I/S", 0.5) == "I/S"
    assert reported_phase("C/S", 0.85) == "C/S"


def test_a_phase_that_is_not_a_series_is_untouched():
    from clayquant.quantification import reported_phase

    for phase in ("illite", "chlorite", "kaolinite_1M", "smectite_EG", "Quartz"):
        assert reported_phase(phase, 1.0) == phase
        assert reported_phase(phase, None) == phase


def test_an_unrecorded_fraction_is_not_an_end_member():
    """A library written before the field existed carries no fraction at all.

    Reporting that as 1 would rename every one of its I/S entries to illite,
    including a genuine 60/40 stack, so an unknown fraction keeps the name.
    """
    import math

    from clayquant.quantification import reported_phase

    assert reported_phase("I/S", None) == "I/S"
    assert reported_phase("I/S", math.nan) == "I/S"


def test_the_grouping_uses_it():
    """End-to-end: an end member fitted from the library is reported as illite.

    The default library spans the series to 1.00, so this is not a contrived
    case - it is what a real fit does whenever the specimen's illite is better
    matched by an I/S end member than by a discrete illite entry.
    """
    spanning = build_library(
        grid=GRID,
        instrument=Instrument(
            peak_shape=PeakShape(u=0.004, v=-0.001, w=0.002, eta=0.5, size_ab=600.0),
            divergence=Divergence(),
        ),
        orientations=(0.3,),
        illite_smectite=(1.00, 0.60),
        chlorite_smectite=(0.90,),
        csds_means=(15.0,),
        host_thicknesses={},
    )
    name = next(n for n in spanning.names if n.startswith("I/S 1.00"))
    entry = spanning.entries[spanning.names.index(name)]
    assert entry.phase == "I/S" and entry.fraction == 1.0  # as the library stores it
    pattern = Pattern(spanning.two_theta, entry.normalization * entry.intensity, name=name)
    quantification = quantify(nnls_fit(pattern, spanning, range_two_theta=(4.0, 34.0)))
    illite = next(s for s in quantification.shares if s.phase == "illite")
    assert name in illite.entries, "the end member must be counted as illite"
    # and it must not also be counted under the series it was built in
    series = next((s for s in quantification.shares if s.phase == "I/S"), None)
    assert series is None or name not in series.entries
    assert illite.host_fraction == pytest.approx(1.0)
