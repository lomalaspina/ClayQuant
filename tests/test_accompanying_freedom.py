"""Cell and orientation freedom for the accompanying minerals.

Both exist because a published cell is some other specimen's and a cleaved
flake can lie down.  Both are off by default, and the orientation one is
offered only to minerals that cleave - because giving an equant mineral a
texture it does not have is not a free parameter.  On a real separate,
offering it to everything let quartz be fitted at r = 0.3, which is 37 times
the mass for the same scattering, and took quartz from 17 per cent of the
specimen to 64 while lowering R_wp.
"""

import numpy as np
import pytest

from clayquant.bern import load_phase_database
from clayquant.emission import CU_KA_5LINE
from clayquant.gui.app import cell_scales_for, habit_of, may_orient
from clayquant.library import (
    ACCOMPANYING_CELL_SCALES,
    ACCOMPANYING_ORIENTATIONS,
    DEFAULT_PEAK_SHAPE,
    accompanying_patterns,
)
from clayquant.models import MINERAL_CLEAVAGE, MINERAL_HABIT, load_crystal
from clayquant.nnls import FitResult, best_variant_per_phase
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, two_theta_grid

GRID = two_theta_grid(5.0, 40.0, 0.05)


@pytest.fixture(scope="module")
def instrument():
    return Instrument(emission=CU_KA_5LINE, peak_shape=DEFAULT_PEAK_SHAPE,
                      lp_mode="powder",
                      divergence=Divergence(specimen_length=25.0, goniometer_radius=240.0,
                                            divergence=0.5, shape="round"))


def test_nothing_is_free_by_default():
    assert ACCOMPANYING_CELL_SCALES == (1.0,)
    assert ACCOMPANYING_ORIENTATIONS == (1.0,)
    assert cell_scales_for(0.0) == (1.0,)


def test_one_scale_and_one_orientation_give_one_pattern(instrument):
    crystal = load_crystal("kaolinite_1M")
    variants = accompanying_patterns("Kaolinite", crystal, GRID, instrument)
    assert len(variants) == 1
    pattern, r, scale, scaled = variants[0]
    assert pattern.name == "Kaolinite"
    assert (r, scale) == (1.0, 1.0)
    assert scaled is crystal


def test_a_span_is_symmetric_about_no_change():
    scales = cell_scales_for(0.01, steps=2)
    assert len(scales) == 5
    assert 1.0 in scales
    assert scales[0] == pytest.approx(0.99)
    assert scales[-1] == pytest.approx(1.01)


def test_a_scaled_cell_moves_every_line_by_the_same_relative_amount(instrument):
    crystal = load_crystal("kaolinite_1M")
    variants = accompanying_patterns("K", crystal, GRID, instrument,
                                     cell_scales=(1.0, 1.01))
    assert len(variants) == 2
    plain, scaled = variants[0][0], variants[1][0]

    def strongest(pattern):
        y = np.asarray(pattern.intensity)
        return float(GRID[int(np.argmax(y))])

    # a bigger cell puts every reflection at a lower angle
    assert strongest(scaled) < strongest(plain)
    assert variants[1][3].a == pytest.approx(crystal.a * 1.01)


def test_the_scaled_crystal_carries_its_own_mass_and_volume(instrument):
    """A weight percent from the unscaled volume would be wrong by the cube."""
    crystal = load_crystal("kaolinite_1M")
    _, _, _, scaled = accompanying_patterns("K", crystal, GRID, instrument,
                                            cell_scales=(1.01,))[0]
    assert scaled.volume == pytest.approx(crystal.volume * 1.01 ** 3, rel=1e-6)
    assert scaled.cell_mass == pytest.approx(crystal.cell_mass)


def test_the_variants_are_named_apart_only_when_there_are_several(instrument):
    crystal = load_crystal("kaolinite_1M")
    names = [p.name for p, _, _, _ in accompanying_patterns(
        "K", crystal, GRID, instrument, cell_scales=(1.0, 1.01),
        orientations=(0.5, 1.0))]
    assert names == ["K PO=0.5 cell=1", "K PO=1 cell=1",
                     "K PO=0.5 cell=1.01", "K PO=1 cell=1.01"]


def test_an_impossible_scale_is_refused(instrument):
    crystal = load_crystal("kaolinite_1M")
    with pytest.raises(ValueError, match="must be positive"):
        accompanying_patterns("K", crystal, GRID, instrument, cell_scales=(0.0,))
    with pytest.raises(ValueError, match="at least one"):
        accompanying_patterns("K", crystal, GRID, instrument, cell_scales=())


# --------------------------------------------------------------------------- #
# which minerals may orient
# --------------------------------------------------------------------------- #


def a_crystal(po_axis=None):
    from clayquant.crystal import Crystal
    return Crystal(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0,
                   sites=[], po_axis=po_axis)


def test_quartz_may_not_orient():
    """The trap this restriction exists for."""
    assert may_orient("Quartz", a_crystal(po_axis=(1.0, 0.0, 1.0))) is None
    assert "quartz" not in MINERAL_CLEAVAGE


def test_a_feldspar_may():
    assert may_orient("Albite", a_crystal()) == (0.0, 0.0, 1.0)
    assert may_orient("Microcline", a_crystal()) == (0.0, 0.0, 1.0)


def test_the_database_axis_wins_where_there_is_one():
    assert may_orient("Albite", a_crystal(po_axis=(0.0, 1.0, 0.0))) == (0.0, 1.0, 0.0)


def test_cleavage_and_habit_are_different_lists():
    """A lath is oriented whether asked or not; a flake only if asked."""
    for name in MINERAL_HABIT:
        assert name not in MINERAL_CLEAVAGE
    # The chain clays left this path when they became library phases with a
    # fibre texture of their own: a needle must not be spanned as a plate
    # (Sec. A.65).  The amphiboles, which really are prismatic, stay.
    assert habit_of("Hornblende", a_crystal()) is not None
    assert may_orient("Hornblende", a_crystal()) is None
    assert habit_of("Sepiolite", a_crystal()) is None


# --------------------------------------------------------------------------- #
# one variant per mineral
# --------------------------------------------------------------------------- #


def a_fit(rows):
    names = [name for name, _, _ in rows]
    phases = [phase for _, phase, _ in rows]
    shares = np.array([share for _, _, share in rows])
    n = len(rows)
    return FitResult(
        two_theta=np.array([10.0]), observed=np.array([1.0]),
        calculated=np.array([1.0]), coefficients=np.ones(n),
        names=names, phases=phases, scattering_fraction=shares,
        amplitude_fraction=shares, r_wp=0.0, r_p=0.0, mask=np.array([True]),
    )


def test_the_variant_with_the_most_scattering_wins():
    result = a_fit([("Quartz cell=0.995", "Quartz", 0.1),
                    ("Quartz cell=1", "Quartz", 0.6),
                    ("Quartz cell=1.005", "Quartz", 0.3)])
    assert best_variant_per_phase(result) == {"Quartz": "Quartz cell=1"}


def test_each_mineral_chooses_its_own():
    result = a_fit([("Quartz cell=1", "Quartz", 0.5),
                    ("Sekaninaite cell=1", "Sekaninaite", 0.1),
                    ("Sekaninaite cell=1.005", "Sekaninaite", 0.4)])
    assert best_variant_per_phase(result) == {
        "Quartz": "Quartz cell=1", "Sekaninaite": "Sekaninaite cell=1.005"}


def test_only_the_phases_asked_about():
    result = a_fit([("Quartz", "Quartz", 0.5), ("illite PO=0.2", "illite", 0.5)])
    assert best_variant_per_phase(result, {"Quartz"}) == {"Quartz": "Quartz"}


def test_a_phase_the_fit_left_out_is_absent():
    result = a_fit([("Quartz", "Quartz", 0.0)])
    result.coefficients = np.zeros(1)
    assert best_variant_per_phase(result) == {}


def test_the_real_database_gives_albite_a_pole_and_quartz_none():
    S = "/tmp/claude-0/-home-user-ClayQuant/e97a8962-878d-55e8-97a5-df9339d94a82/scratchpad"
    import pathlib
    path = pathlib.Path(S) / "new_phases.json"
    if not path.is_file():
        pytest.skip("the phase database this was measured on is not in this container")
    db = load_phase_database(path)
    assert may_orient("Quartz", db["Quartz"]) is None
    assert may_orient("Albite", db["Albite"]) is not None


def test_the_habit_orientation_range_stays_narrow():
    """Measured, not assumed.

    Widening it from 0.3 to 0.1 sent every feldspar and both amphiboles to the
    new lowest value, made R_wp worse (38.82 to 38.99 %) and took the albite
    from 6.8 to 0.5 wt %.  Mass goes as r**-3, so an extreme orientation buys
    intensity at almost no mass and a fit free to choose it will; the residual
    does not object.  A mineral pinned at the bottom of this range is reporting
    that its orientation is unmeasured, not that it equals that value.
    """
    from clayquant.gui.app import HABIT_ORIENTATIONS

    assert min(HABIT_ORIENTATIONS) >= 0.3
    assert max(HABIT_ORIENTATIONS) == 1.0
