"""The orientation a fixed-orientation phase is weighed on.

A pure basal series has the same shape at every March-Dollase ``r``, so the fit
cannot measure the texture of the glycolated smectite and the library stores it
at one assumed value.  That assumption is cubic in the weight percent, and left
at a random powder beside clays fitted at 0.1 it overstates the smectite by a
factor of a thousand.  These tests pin the factor, the correction and the
reporting.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.library import eg_smectite_layer, basal_pattern
from clayquant.mixed_layer import MixedLayerStack
from clayquant.nnls import FitResult
from clayquant.pattern import Instrument, two_theta_grid
from clayquant.quantification import (
    FIXED_ORIENTATION_PHASES,
    PhaseShare,
    quantify,
    rebase_fixed_orientation,
)


def test_orientation_scales_a_basal_series_without_changing_its_shape():
    """The fact the whole problem rests on, measured rather than assumed."""
    grid = two_theta_grid(2.0, 40.0, 0.02)
    layer = eg_smectite_layer()
    stack = MixedLayerStack(layer, layer, 1.0, name="smectite")
    patterns = {
        r: np.asarray(
            basal_pattern(stack, grid, Instrument(), r_march_dollase=r,
                          name=f"r={r}").intensity,
            dtype=float,
        )
        for r in (1.0, 0.3, 0.1)
    }
    reference = patterns[1.0]
    for r in (0.3, 0.1):
        # Intensity scales as r**-3 ...
        assert np.isclose(patterns[r].max() / reference.max(), r ** -3, rtol=1e-6)
        # ... and the shape is untouched, which is why the fit cannot choose r
        # and why ten library entries at ten orientations would be ten identical
        # columns.
        np.testing.assert_allclose(
            patterns[r] / patterns[r].max(), reference / reference.max(), atol=1e-10
        )


def clay_shares(smectite_r: float) -> list[PhaseShare]:
    """A fit in which the platy clays came out at 0.1 and smectite is stored at r."""
    return [
        PhaseShare(phase="illite", is_clay=True, coefficient=4.0, scattering=0.50,
                   amplitude=0.6, mass=50.0, orientation=0.1),
        PhaseShare(phase="chlorite", is_clay=True, coefficient=2.0, scattering=0.20,
                   amplitude=0.2, mass=20.0, orientation=0.1),
        PhaseShare(phase="I/S", is_clay=True, coefficient=2.0, scattering=0.29,
                   amplitude=0.1, mass=29.0, orientation=0.1),
        PhaseShare(phase="smectite_EG", is_clay=True, coefficient=1.0,
                   scattering=0.01, amplitude=0.05, mass=100.0,
                   orientation=smectite_r),
    ]


def test_a_smectite_stored_as_a_powder_is_rebased_by_the_cube():
    shares = clay_shares(1.0)
    rebased = rebase_fixed_orientation(shares)
    assert set(rebased) == {"smectite_EG"}
    assert rebased["smectite_EG"] == pytest.approx((1.0, 0.1))
    smectite = next(share for share in shares if share.phase == "smectite_EG")
    # (0.1 / 1.0)**3 = 1/1000, which is the whole of the effect.
    assert smectite.mass == pytest.approx(0.1)
    assert smectite.orientation == pytest.approx(0.1)


def test_the_other_clays_are_left_alone():
    shares = clay_shares(1.0)
    before = {share.phase: share.mass for share in shares}
    rebase_fixed_orientation(shares)
    for share in shares:
        if share.phase != "smectite_EG":
            assert share.mass == before[share.phase]


def test_a_smectite_already_on_the_measured_texture_is_not_touched():
    shares = clay_shares(0.1)
    assert rebase_fixed_orientation(shares) == {}
    assert next(s for s in shares if s.phase == "smectite_EG").mass == 100.0


def test_the_target_is_the_coefficient_weighted_mean_of_the_measured_clays():
    shares = clay_shares(1.0)
    shares[0].orientation = 0.2   # illite, coefficient 4
    shares[1].orientation = 0.5   # chlorite, coefficient 2
    shares[2].orientation = 0.5   # I/S, coefficient 2
    rebased = rebase_fixed_orientation(shares)
    expected = (4 * 0.2 + 2 * 0.5 + 2 * 0.5) / 8.0
    assert rebased["smectite_EG"][1] == pytest.approx(expected)


def test_nothing_happens_when_no_clay_orientation_was_measured():
    """A fit of the smectite alone has nothing to be put on the basis of."""
    shares = [PhaseShare(phase="smectite_EG", is_clay=True, coefficient=1.0,
                         scattering=1.0, amplitude=1.0, mass=100.0, orientation=1.0)]
    assert rebase_fixed_orientation(shares) == {}
    assert shares[0].mass == 100.0


def synthetic_fit(smectite_r: float) -> FitResult:
    """One specimen, described by a library that stored smectite at ``smectite_r``.

    The mass follows the orientation rather than being held fixed, because that
    is what changing the library would do: the pattern at ``r`` is ``r**-3``
    times as intense, so it is stored against an ``r**-3`` larger normalisation
    and the same fitted coefficient implies ``r**3`` times the mass.
    """
    names = ["illite PO=0.1", "chlorite PO=0.1", "smectite_EG"]
    phases = ["illite", "chlorite", "smectite_EG"]
    coefficients = np.array([4.0, 2.0, 1.0])
    smectite_mass = 100.0 * smectite_r ** 3
    return FitResult(
        names=names,
        phases=phases,
        coefficients=coefficients,
        calculated=np.zeros(10),
        two_theta=np.linspace(4.0, 34.0, 10),
        observed=np.zeros(10),
        mask=np.ones(10, dtype=bool),
        r_wp=0.4,
        r_p=0.4,
        scattering_fraction=np.array([0.6, 0.3, 0.1]),
        amplitude_fraction=np.array([0.57, 0.29, 0.14]),
        relative_mass=np.array([60.0, 30.0, smectite_mass]),
        march_dollase=np.array([0.1, 0.1, smectite_r]),
        fraction=np.array([1.0, 1.0, 1.0]),
    )


def test_quantify_rebases_by_default_and_says_that_it_did():
    powder = quantify(synthetic_fit(1.0))
    assert set(powder.rebased_orientation) == {"smectite_EG"}
    stored, target = powder.rebased_orientation["smectite_EG"]
    assert (stored, target) == pytest.approx((1.0, 0.1))
    smectite = next(s for s in powder.shares if s.phase == "smectite_EG")
    # 100 units of mass become 0.1, so the smectite goes from most of the
    # specimen to a trace of it - on the same 10 % of the scattering.
    assert smectite.clay_weight < 0.2
    assert smectite.clay_scattering == pytest.approx(0.1)


def test_the_library_basis_can_still_be_seen():
    powder = quantify(synthetic_fit(1.0), rebase_orientation=False)
    assert powder.rebased_orientation == {}
    smectite = next(s for s in powder.shares if s.phase == "smectite_EG")
    # 100 of 190 units of mass: the number that made 0.95 % of the scattering
    # read as 22 % of the clay weight on a real mount.
    assert smectite.clay_weight == pytest.approx(100.0 * 100.0 / 190.0)


def test_a_library_built_on_the_measured_texture_needs_no_rebasing():
    built = quantify(synthetic_fit(0.1))
    assert built.rebased_orientation == {}
    smectite = next(s for s in built.shares if s.phase == "smectite_EG")
    rebased = quantify(synthetic_fit(1.0))
    matching = next(s for s in rebased.shares if s.phase == "smectite_EG")
    # The two routes agree: building at 0.1 and rebasing from 1 to 0.1 are the
    # same statement about the specimen.
    assert smectite.clay_weight == pytest.approx(matching.clay_weight)


def test_the_library_is_built_on_the_measured_texture_by_default():
    import inspect

    from clayquant.library import build_library

    default = inspect.signature(build_library).parameters["smectite_orientation"].default
    assert default == pytest.approx(0.1)
    assert FIXED_ORIENTATION_PHASES == ("smectite_EG",)
