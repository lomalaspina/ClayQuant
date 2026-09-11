"""Diffraction by interstratified (mixed-layer) clay structures.

Theory
------
A crystallite is a stack of ``N`` layers along the ``c*`` axis.  Layer ``j`` is
of type ``t_j``, has complex layer structure factor ``F_m(s)`` and thickness
``d_m`` (``m`` being its type), with ``s = 2 sin(theta)/lambda = 1/d``.  The
scattered amplitude is

    A(s) = sum_j F_{t_j}(s) exp(2 pi i s z_j),    z_j = sum_{k<j} d_{t_k}

so the intensity, averaged over the stacking statistics, is

    I(s) = sum_j <|F|^2> + 2 Re sum_{j<k} <F_{t_k} F*_{t_j} exp(2 pi i s (z_k - z_j))>

Let the layer sequence be a stationary Markov chain with transition matrix
``P_mp`` (probability that a layer of type ``m`` is followed by one of type
``p``) and stationary layer proportions ``W_m``.  Define

    T_mp(s) = P_mp * exp(2 pi i s d_m)

The thickness in the phase factor is that of the *source* layer ``m``, because
the origin-to-origin distance between consecutive layers ``j`` and ``j+1`` is
the thickness of layer ``j``.  Summing over all paths of ``n`` steps then gives
``[T^n]_{mp}`` for the joint probability-and-phase of finding a layer of type
``p`` exactly ``n`` layers above one of type ``m``, so that

    I(s) = N sum_m W_m |F_m|^2
           + 2 Re sum_{n=1}^{N-1} (N - n) sum_{m,p} W_m F*_m [T^n]_{mp} F_p

Averaging over a distribution of crystallite thicknesses ``p(N)`` (the CSDS,
crystallite size distribution along ``c*``) and normalising per layer:

    I(s) / <N> = sum_m W_m |F_m|^2
                 + 2 Re sum_{n>=1} G(n) sum_{m,p} W_m F*_m [T^n]_{mp} F_p

    G(n) = <max(N - n, 0)> / <N>

This is the Markov/matrix ("Kakinoki-Komura") formalism that underlies
NEWMOD-style programs; it goes back to Hendricks & Teller (1942) and was applied
to clays by MacEwan (1958) and Reynolds (1967, 1980).  For one layer type it
reduces to ``|F|^2`` times the Laue interference function, which
``tests/test_mixed_layer.py`` checks explicitly.

Layer origin convention
-----------------------
A layer's structure factor must be referenced to the same point that its
thickness is measured from.  ClayQuant uses the centre of the 2:1 (or 1:1)
sheet as the layer origin, so a layer spans ``[-d/2, +d/2)`` and the interlayer
falls on the layer boundary; :meth:`clayquant.crystal.LayerModel.centered`
enforces this.  Junctions between unlike layers then join half an interlayer of
one type to half an interlayer of the other, which is exact for illite/smectite
(both are 2:1 layers with a cation/water interlayer) and approximate for
chlorite/smectite, where a chlorite layer contributes half of its brucite-like
hydroxide sheet to each boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crystal import LayerModel

__all__ = [
    "CSDS",
    "fixed_csds",
    "lognormal_csds",
    "random_transition",
    "markov_transition",
    "MixedLayerStack",
]


# --------------------------------------------------------------------------- #
# Crystallite size distribution along c*
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CSDS:
    """Distribution of the number of layers per crystallite.

    Attributes
    ----------
    n:
        Layer counts (integers >= 1).
    probability:
        Normalised probability of each layer count.
    """

    n: np.ndarray
    probability: np.ndarray

    def __post_init__(self) -> None:
        n = np.asarray(self.n, dtype=int)
        p = np.asarray(self.probability, dtype=float)
        if n.shape != p.shape:
            raise ValueError("n and probability must have the same shape")
        if np.any(n < 1):
            raise ValueError("crystallites must contain at least one layer")
        if np.any(p < 0) or not p.sum() > 0:
            raise ValueError("probabilities must be non-negative and sum to a positive value")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "probability", p / p.sum())

    @property
    def mean(self) -> float:
        """Mean number of layers per crystallite."""
        return float(np.sum(self.probability * self.n))

    @property
    def n_max(self) -> int:
        return int(self.n.max())

    def pair_weights(self) -> np.ndarray:
        """The weights ``G(n)`` for ``n = 0 .. n_max - 1``.

        ``G(n) = <max(N - n, 0)> / <N>`` counts, per layer, how many pairs of
        layers are separated by ``n`` positions.  ``G(0) = 1``.
        """
        separations = np.arange(self.n_max)
        counts = np.clip(self.n[:, None] - separations[None, :], 0, None)
        return (self.probability[:, None] * counts).sum(axis=0) / self.mean


def fixed_csds(n_layers: int) -> CSDS:
    """All crystallites contain exactly ``n_layers`` layers."""
    return CSDS(n=np.array([n_layers]), probability=np.array([1.0]))


def lognormal_csds(mean_n: float, beta: float = 0.35, n_max: int | None = None) -> CSDS:
    """Lognormal crystallite size distribution.

    Lognormal CSDS are the usual description of clay crystallite thickness
    distributions (Drits, Eberl & Srodon 1998).  ``beta`` is the standard
    deviation of ``ln(N)``; ``alpha = ln(mean_n) - beta**2 / 2`` is chosen so
    that the distribution has the requested mean.

    Parameters
    ----------
    mean_n:
        Mean number of layers per crystallite.
    beta:
        Width of the distribution in ``ln(N)``.  ``beta -> 0`` approaches a
        single crystallite thickness.
    n_max:
        Largest layer count retained; defaults to a value that captures the
        upper tail of the distribution.
    """
    if mean_n <= 0:
        raise ValueError("mean_n must be positive")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if beta == 0.0:
        return fixed_csds(max(1, int(round(mean_n))))
    if n_max is None:
        n_max = max(4, int(np.ceil(mean_n * np.exp(3.0 * beta))))
    alpha = np.log(mean_n) - beta**2 / 2.0
    n = np.arange(1, n_max + 1)
    p = np.exp(-((np.log(n) - alpha) ** 2) / (2.0 * beta**2)) / n
    return CSDS(n=n, probability=p)


# --------------------------------------------------------------------------- #
# Stacking statistics
# --------------------------------------------------------------------------- #


def random_transition(fraction_a: float) -> np.ndarray:
    """Transition matrix for fully random interstratification (Reichweite R = 0).

    Each layer's type is drawn independently, so every row of the matrix equals
    the layer proportions: ``P_AA = P_BA = W_A`` and ``P_AB = P_BB = W_B``.
    """
    if not 0.0 <= fraction_a <= 1.0:
        raise ValueError("fraction_a must lie in [0, 1]")
    w_b = 1.0 - fraction_a
    return np.array([[fraction_a, w_b], [fraction_a, w_b]])


def markov_transition(fraction_a: float, p_ab: float) -> np.ndarray:
    """Transition matrix from the layer proportion and one junction probability.

    ``p_ab`` is the probability that a layer of type A is followed by one of
    type B.  Stationarity requires ``W_A P_AB = W_B P_BA``, which fixes the
    remaining probabilities.  ``p_ab = W_B`` reproduces
    :func:`random_transition`; smaller values describe segregation (long runs of
    like layers) and larger values ordering (alternation, as in R1 ordered
    illite/smectite).
    """
    if not 0.0 < fraction_a < 1.0:
        raise ValueError("fraction_a must lie strictly between 0 and 1 for a Markov stack")
    w_a, w_b = fraction_a, 1.0 - fraction_a
    if not 0.0 <= p_ab <= 1.0:
        raise ValueError("p_ab must lie in [0, 1]")
    p_ba = w_a * p_ab / w_b
    if p_ba > 1.0 + 1e-12:
        raise ValueError(
            f"p_ab={p_ab} is impossible for fraction_a={fraction_a}: it implies P_BA={p_ba} > 1. "
            f"The maximum ordering is p_ab = {w_b / w_a:.4f}."
        )
    p_ba = min(p_ba, 1.0)
    return np.array([[1.0 - p_ab, p_ab], [p_ba, 1.0 - p_ba]])


# --------------------------------------------------------------------------- #
# The stack
# --------------------------------------------------------------------------- #


@dataclass
class MixedLayerStack:
    """A two-component interstratified stack.

    Parameters
    ----------
    layer_a, layer_b:
        The two layer types.  Their ``thickness`` attributes give ``d_A`` and
        ``d_B``; they are re-centred on the layer origin convention described in
        the module docstring.
    fraction_a:
        Proportion ``W_A`` of layer type A.
    transition:
        Stacking transition matrix.  Defaults to fully random interstratification
        for the given proportion.
    csds:
        Crystallite size distribution along ``c*``.
    name:
        Label carried through to generated patterns.
    """

    layer_a: LayerModel
    layer_b: LayerModel
    fraction_a: float
    transition: np.ndarray | None = None
    csds: CSDS | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction_a <= 1.0:
            raise ValueError("fraction_a must lie in [0, 1]")
        if self.transition is None:
            self.transition = random_transition(self.fraction_a)
        self.transition = np.asarray(self.transition, dtype=float)
        if self.transition.shape != (2, 2):
            raise ValueError("transition must be a 2x2 matrix")
        if not np.allclose(self.transition.sum(axis=1), 1.0):
            raise ValueError("rows of the transition matrix must sum to 1")
        if self.csds is None:
            self.csds = lognormal_csds(10.0)
        if not self.name:
            self.name = f"{self.layer_a.name}/{self.layer_b.name} {self.fraction_a:.2f}"

    @property
    def proportions(self) -> np.ndarray:
        return np.array([self.fraction_a, 1.0 - self.fraction_a])

    @property
    def mean_thickness(self) -> float:
        """Mean layer thickness in A, i.e. the mean origin-to-origin spacing."""
        return float(self.proportions @ np.array([self.layer_a.thickness, self.layer_b.thickness]))

    def intensity(self, s: np.ndarray) -> np.ndarray:
        """Diffracted intensity per layer at ``s = 2 sin(theta)/lambda`` [1/A].

        Returns the Markov-averaged ``I(s)/<N>`` of the module docstring, before
        any Lorentz-polarization, preferred-orientation or instrumental factors.
        """
        s = np.asarray(s, dtype=float)
        layers = (self.layer_a.centered(), self.layer_b.centered())
        weights = self.proportions

        f = np.stack([layer.structure_factor(s) for layer in layers])  # (2, M)
        thicknesses = np.array([layer.thickness for layer in layers])
        phases = np.exp(2j * np.pi * s[None, :] * thicknesses[:, None])  # (2, M)

        # T[m, p] = P[m, p] * exp(2 pi i s d_m)
        transition = self.transition[:, :, None] * phases[:, None, :]  # (2, 2, M)

        pair_weights = self.csds.pair_weights()
        intensity = pair_weights[0] * np.einsum("m...,m...->...", weights[:, None] * f.conj(), f).real

        vector = f.copy()  # u_0 = F
        for separation in range(1, len(pair_weights)):
            vector = np.einsum("mpi,pi->mi", transition, vector)
            term = np.einsum("mi,mi->i", weights[:, None] * f.conj(), vector)
            intensity = intensity + 2.0 * pair_weights[separation] * term.real
        return intensity
