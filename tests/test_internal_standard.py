"""A weighed standard turns a share of the fit into a share of the specimen.

The weight percent of a fit sums to 100 whatever was left out of it, so it
overstates every phase by the same unknown factor when the specimen holds
amorphous material or a mineral the library does not have.  Nothing inside the
fit reveals that factor.  A known weight of a standard does, and the tests here
are of that arithmetic and of the two ways it can fail to apply: a standard that
was not fitted, and a clay basis that is already renormalised.

The reference case is a laboratory one: 20 % of corundum weighed into a rock
powder before the mount is pressed, which is how the accompanying analyses of
these specimens were quantified.
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
        illite_smectite=(0.80,),
        chlorite_smectite=(0.90,),
        csds_means=(15.0,),
        host_thicknesses={},
    )


def specimen(library, masses: dict[str, float], constant: float = 1e-6) -> Pattern:
    intensity = np.zeros_like(library.two_theta)
    for name, mass in masses.items():
        entry = library.entries[library.names.index(name)]
        calculated = entry.normalization * entry.intensity
        intensity += mass * calculated / (entry.unit_mass * entry.unit_volume)
    return Pattern(library.two_theta, constant * intensity, name="synthetic mixture")


def fit(library, masses, **kwargs):
    result = nnls_fit(specimen(library, masses), library, range_two_theta=(4.0, 34.0))
    return quantify(result, **kwargs)


# Kaolinite stands in for the standard: it is a phase of the library whose mass
# the test controls, which is all the arithmetic needs.
STANDARD = "kaolinite_1M PO=0.3"


def test_a_wholly_fitted_specimen_is_already_absolute(library):
    """Nothing is missing, so the standard confirms the scale rather than fixing it."""
    masses = {STANDARD: 20.0, "illite PO=0.3": 50.0, "chlorite PO=0.3": 30.0}
    q = fit(library, masses, internal_standard=("kaolinite_1M", 20.0))
    assert q.absolute
    assert q.absolute_scale == pytest.approx(1.0, abs=0.002)
    assert q.unaccounted == pytest.approx(0.0, abs=0.2)
    weights = {s.phase: s.absolute_weight for s in q.shares}
    assert weights["illite"] == pytest.approx(50.0, abs=0.2)
    assert weights["chlorite"] == pytest.approx(30.0, abs=0.2)


def test_what_is_not_fitted_comes_back_as_unaccounted(library):
    """The case the standard exists for: a quarter of the specimen is invisible.

    The mount holds 25 % of something the fit has no pattern for - amorphous, or
    a mineral missing from the library.  The relative weight percent cannot show
    it and sums to 100 without it; against the standard it appears.
    """
    masses = {STANDARD: 15.0, "illite PO=0.3": 45.0, "chlorite PO=0.3": 15.0}
    # 75 g of specimen were fitted, so the standard is 15 % of 100 g, not of 75.
    q = fit(library, masses, internal_standard=("kaolinite_1M", 15.0))
    assert q.absolute_scale == pytest.approx(0.75, rel=0.01)
    assert q.unaccounted == pytest.approx(25.0, abs=0.3)
    weights = {s.phase: s.absolute_weight for s in q.shares}
    assert weights["illite"] == pytest.approx(45.0, abs=0.3)
    assert weights["chlorite"] == pytest.approx(15.0, abs=0.3)
    # The relative weights still sum to 100 and still overstate every phase.
    assert sum(s.weight for s in q.shares) == pytest.approx(100.0)
    assert weights["illite"] < next(s.weight for s in q.shares if s.phase == "illite")


def test_a_standard_that_was_not_fitted_leaves_the_analysis_relative(library):
    """Better to say the scale is unknown than to invent one."""
    masses = {"illite PO=0.3": 60.0, "chlorite PO=0.3": 40.0}
    q = fit(library, masses, internal_standard=("Corundum", 20.0))
    assert not q.absolute
    assert q.unaccounted == 0.0
    assert all(s.absolute_weight == 0.0 for s in q.shares)
    assert q.internal_standard == "Corundum", "what was asked for is still recorded"
    assert all(s.weight > 0.0 for s in q.shares), "the relative analysis is unaffected"


def test_an_impossible_standard_weight_is_refused(library):
    masses = {STANDARD: 20.0, "illite PO=0.3": 80.0}
    with pytest.raises(ValueError, match="between 0 and 100"):
        fit(library, masses, internal_standard=("kaolinite_1M", 120.0))


def test_the_absolute_column_appears_only_on_the_specimen_basis(library):
    """A clay-only table is renormalised to 100 % already; scaling it means nothing."""
    masses = {STANDARD: 20.0, "illite PO=0.3": 50.0, "chlorite PO=0.3": 30.0}
    q = fit(library, masses, internal_standard=("kaolinite_1M", 20.0))
    whole = q.table()
    assert all(isinstance(row["absolute_weight_percent"], float) for row in whole)
    clays = q.table(clay_basis=True)
    assert all(row["absolute_weight_percent"] == "" for row in clays)


def test_the_csv_says_what_the_standard_was(library, tmp_path):
    masses = {STANDARD: 15.0, "illite PO=0.3": 45.0, "chlorite PO=0.3": 15.0}
    q = fit(library, masses, internal_standard=("kaolinite_1M", 15.0))
    path = q.to_csv(tmp_path / "absolute.csv")
    header = "".join(line for line in path.read_text().splitlines(True) if line.startswith("#"))
    assert "kaolinite_1M" in header and "15%" in header
    assert "amorphous or missing from the fit" in header


def test_without_a_standard_nothing_changes(library):
    masses = {STANDARD: 20.0, "illite PO=0.3": 80.0}
    plain, asked = fit(library, masses), fit(library, masses, internal_standard=None)
    assert not plain.absolute
    assert [s.weight for s in plain.shares] == pytest.approx([s.weight for s in asked.shares])
    assert all(row["absolute_weight_percent"] == "" for row in plain.table())
