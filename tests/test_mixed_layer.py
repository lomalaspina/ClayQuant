"""Correctness checks for the Markov/matrix interstratification model.

The strongest available check is the one-component limit: a stack of identical
layers must reproduce the analytic Laue interference function, which is derived
independently of the matrix formalism.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.crystal import LayerModel
from clayquant.mixed_layer import (
    MixedLayerStack,
    fixed_csds,
    lognormal_csds,
    markov_transition,
    random_transition,
)
from clayquant.models import eg_smectite_layer

S = np.linspace(0.01, 0.7, 400)


def simple_layer(thickness: float, name: str = "test") -> LayerModel:
    return LayerModel.from_table(
        thickness,
        [(0.0, "Si", 1.0, 1.0), (thickness / 4.0, "O", 2.0, 1.5)],
        name=name,
    )


def laue(s: np.ndarray, d: float, n_layers: int) -> np.ndarray:
    """``(sin(pi N s d) / sin(pi s d))**2``, evaluated safely."""
    numerator = np.sin(np.pi * n_layers * s * d)
    denominator = np.sin(np.pi * s * d)
    return np.where(
        np.abs(denominator) < 1e-12,
        float(n_layers) ** 2,
        (numerator / np.where(np.abs(denominator) < 1e-12, 1.0, denominator)) ** 2,
    )


@pytest.mark.parametrize("n_layers", [1, 2, 5, 17])
def test_single_component_reproduces_laue_interference(n_layers):
    layer = simple_layer(10.0)
    stack = MixedLayerStack(layer, layer, fraction_a=1.0, csds=fixed_csds(n_layers))
    expected = np.abs(layer.structure_factor(S)) ** 2 * laue(S, 10.0, n_layers) / n_layers
    assert np.allclose(stack.intensity(S), expected, rtol=1e-9, atol=1e-9)


def test_two_identical_layer_types_behave_as_one():
    layer = simple_layer(10.0)
    reference = MixedLayerStack(layer, layer, 1.0, csds=fixed_csds(8)).intensity(S)
    for fraction in (0.0, 0.25, 0.5, 0.75):
        mixed = MixedLayerStack(layer, layer, fraction, csds=fixed_csds(8)).intensity(S)
        assert np.allclose(mixed, reference, rtol=1e-9, atol=1e-9)


def test_pure_end_member_ignores_the_absent_layer():
    a = simple_layer(10.0, "a")
    b = eg_smectite_layer()
    pure = MixedLayerStack(a, a, 1.0, csds=fixed_csds(6)).intensity(S)
    with_absent_b = MixedLayerStack(a, b, 1.0, csds=fixed_csds(6)).intensity(S)
    assert np.allclose(with_absent_b, pure, rtol=1e-9, atol=1e-9)


def test_single_layer_crystallites_have_no_interference():
    a = simple_layer(10.0, "a")
    b = eg_smectite_layer()
    stack = MixedLayerStack(a, b, 0.6, csds=fixed_csds(1))
    expected = 0.6 * np.abs(a.centered().structure_factor(S)) ** 2 + 0.4 * np.abs(
        b.centered().structure_factor(S)
    ) ** 2
    assert np.allclose(stack.intensity(S), expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("fraction", [0.05, 0.2, 0.5, 0.8, 0.95])
@pytest.mark.parametrize("ordering", ["random", "ordered", "segregated"])
def test_intensity_is_non_negative(fraction, ordering):
    a = simple_layer(10.0, "illite-like")
    b = eg_smectite_layer()
    if ordering == "random":
        transition = random_transition(fraction)
    elif ordering == "ordered":
        transition = markov_transition(fraction, min(0.99, (1.0 - fraction) / fraction * 0.99))
    else:
        transition = markov_transition(fraction, (1.0 - fraction) * 0.2)
    stack = MixedLayerStack(a, b, fraction, transition=transition, csds=lognormal_csds(12.0))
    intensity = stack.intensity(S)
    assert np.all(intensity > -1e-6 * intensity.max())


def test_csds_pair_weights():
    csds = fixed_csds(4)
    assert np.allclose(csds.pair_weights(), [1.0, 3 / 4, 2 / 4, 1 / 4])
    assert lognormal_csds(10.0).mean == pytest.approx(10.0, rel=0.05)


def test_random_transition_is_the_markov_special_case():
    assert np.allclose(random_transition(0.7), markov_transition(0.7, 0.3))


def test_markov_transition_rejects_impossible_ordering():
    with pytest.raises(ValueError, match="impossible"):
        markov_transition(0.9, 0.5)
