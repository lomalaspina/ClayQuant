"""Pattern calculation: from structures to simulated diffractograms.

Three generators are provided:

:func:`powder_pattern`
    Full three-dimensional ``hkl`` pattern of a :class:`~clayquant.crystal.Crystal`,
    with a March-Dollase preferred-orientation correction about ``c*``.  Every
    reciprocal lattice point is enumerated individually, so each one receives
    the orientation factor appropriate to its own angle to the texture axis.

:func:`basal_pattern`
    Basal (00l) pattern of a :class:`~clayquant.mixed_layer.MixedLayerStack`,
    i.e. the continuous one-dimensional intensity of an interstratified stack.

:func:`mixed_layer_pattern`
    The two combined: the basal series from the interstratification model plus
    the non-basal reflections of the host structure.  The basal part is put on
    the same absolute scale as the ``hkl`` part by calibrating against the pure
    host, so that a vanishing expandable fraction reproduces the pure host
    pattern exactly (see :func:`basal_scale_factor`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .crystal import Crystal
from .emission import CU_KA_5LINE, EmissionProfile
from .mixed_layer import MixedLayerStack
from .optics import Divergence, lorentz_polarization, march_dollase
from .profile import PeakShape, accumulate_peaks, convolve_variable_fwhm

__all__ = [
    "Pattern",
    "Instrument",
    "Reflections",
    "reflections",
    "powder_pattern",
    "basal_pattern",
    "mixed_layer_pattern",
    "basal_scale_factor",
    "two_theta_grid",
]


def two_theta_grid(start: float = 2.0, stop: float = 40.0, step: float = 0.02) -> np.ndarray:
    """Angular grid in degrees, inclusive of ``stop`` where the step divides the range."""
    if step <= 0:
        raise ValueError("step must be positive")
    if stop <= start:
        raise ValueError("stop must exceed start")
    count = int(round((stop - start) / step)) + 1
    return start + step * np.arange(count)


@dataclass
class Pattern:
    """A diffraction pattern on a 2theta grid."""

    two_theta: np.ndarray
    intensity: np.ndarray
    name: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.two_theta = np.asarray(self.two_theta, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        if self.two_theta.shape != self.intensity.shape:
            raise ValueError("two_theta and intensity must have the same shape")

    def normalized(self, mode: str = "max") -> "Pattern":
        """Return a copy scaled to unit maximum (``"max"``) or unit area (``"area"``)."""
        if mode == "max":
            scale = float(np.max(self.intensity))
        elif mode == "area":
            scale = float(np.trapezoid(self.intensity, self.two_theta))
        else:
            raise ValueError("mode must be 'max' or 'area'")
        if scale <= 0:
            return replace(self, intensity=self.intensity.copy())
        return replace(self, intensity=self.intensity / scale)

    def interpolated(self, two_theta: np.ndarray) -> "Pattern":
        """Resample onto another 2theta grid."""
        two_theta = np.asarray(two_theta, dtype=float)
        return replace(
            self,
            two_theta=two_theta,
            intensity=np.interp(two_theta, self.two_theta, self.intensity, left=0.0, right=0.0),
        )

    def __add__(self, other: "Pattern") -> "Pattern":
        if not np.array_equal(self.two_theta, other.two_theta):
            other = other.interpolated(self.two_theta)
        return replace(self, intensity=self.intensity + other.intensity)

    def __mul__(self, factor: float) -> "Pattern":
        return replace(self, intensity=self.intensity * float(factor))

    __rmul__ = __mul__


@dataclass
class Instrument:
    """Radiation and instrument description."""

    emission: EmissionProfile = CU_KA_5LINE
    peak_shape: PeakShape = field(default_factory=lambda: PeakShape(w=0.01, eta=0.6))
    lp_mode: str = "powder"
    monochromator_two_theta: float | None = None
    lines_per_emission: int = 1
    divergence: Divergence | None = None
    width_source: str = ""
    """Where :attr:`peak_shape`'s size came from, when it was measured.

    ``"quartz"`` when it was fitted to the quartz K-alpha doublet, whose width
    in a clay separate is the instrument's; ``"isolated peaks"`` when no quartz
    could be fitted and it was scaled to the pattern's own peaks instead, which
    on an oriented mount are the clay basal reflections and are broader than the
    instrument by a factor that varies with the specimen.  Empty when nothing was
    measured and the default was kept.

    Carried on the instrument rather than only in a status line because the
    session has to be able to prefer one over the other: a width measured on
    quartz must not be replaced by one scaled to a clay peak
    (:meth:`clayquant.gui.state.SessionState.take_instrument_from`).
    """

    def sample_emission(self) -> tuple[np.ndarray, np.ndarray]:
        return self.emission.sample(self.lines_per_emission)

    def lp(self, two_theta: np.ndarray) -> np.ndarray:
        """Lorentz-polarization factor including the beam overflow correction.

        Without the overflow correction the Lorentz factor diverges as the angle
        goes to zero, which puts an unphysical ramp under the low-angle basal
        reflections that matter most for clays.
        """
        factor = lorentz_polarization(
            two_theta, mode=self.lp_mode, monochromator_two_theta=self.monochromator_two_theta
        )
        if self.divergence is not None:
            factor = factor * self.divergence.factor(two_theta)
        return factor


@dataclass
class Reflections:
    """A reflection list with everything needed to build a pattern."""

    hkl: np.ndarray
    d: np.ndarray
    f_squared: np.ndarray
    alpha: np.ndarray

    def select(self, mask: np.ndarray) -> "Reflections":
        return Reflections(self.hkl[mask], self.d[mask], self.f_squared[mask], self.alpha[mask])

    @property
    def is_basal(self) -> np.ndarray:
        return (self.hkl[:, 0] == 0) & (self.hkl[:, 1] == 0)


def reflections(
    crystal: Crystal, d_min: float, po_axis: Sequence[float] = (0.0, 0.0, 1.0)
) -> Reflections:
    """Enumerate all reflections of ``crystal`` with ``d >= d_min``.

    The full sphere of reciprocal lattice points is enumerated (excluding
    ``000``), each as a separate reflection.  Friedel pairs therefore appear
    twice, which is correct: both contribute their own diffraction ring, and
    each carries its own preferred-orientation factor.

    ``po_axis`` is the pole the stored ``alpha`` is measured from, and so the
    axis any March-Dollase factor built from this list will be about.  ``c*`` is
    the default because a clay platelet is flattened on 001; see
    :meth:`Crystal.angle_to`.
    """
    if d_min <= 0:
        raise ValueError("d_min must be positive")
    limits = [int(math.floor(length / d_min)) for length in (crystal.a, crystal.b, crystal.c)]
    ranges = [np.arange(-limit, limit + 1) for limit in limits]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]

    inv_d = crystal.inv_d(grid)
    keep = inv_d <= 1.0 / d_min
    grid, inv_d = grid[keep], inv_d[keep]

    sites = crystal.expanded_sites()
    f = crystal.structure_factor(grid, sites=sites)
    return Reflections(
        hkl=grid,
        d=1.0 / inv_d,
        f_squared=np.abs(f) ** 2,
        alpha=crystal.angle_to(grid, po_axis),
    )


def _build_from_reflections(
    reflection_list: Reflections,
    grid: np.ndarray,
    instrument: Instrument,
    r_march_dollase: float,
) -> np.ndarray:
    """Sum profiled reflections over the emission profile onto ``grid``."""
    wavelengths, weights = instrument.sample_emission()
    orientation = march_dollase(reflection_list.alpha, r_march_dollase)
    total = np.zeros_like(grid)
    for wavelength, weight in zip(wavelengths, weights):
        argument = wavelength / (2.0 * reflection_list.d)
        visible = argument < 1.0
        if not np.any(visible):
            continue
        two_theta = np.degrees(2.0 * np.arcsin(argument[visible]))
        in_range = (two_theta >= grid[0]) & (two_theta <= grid[-1])
        if not np.any(in_range):
            continue
        index = np.flatnonzero(visible)[in_range]
        two_theta = two_theta[in_range]
        intensity = (
            weight
            * reflection_list.f_squared[index]
            * orientation[index]
            * instrument.lp(two_theta)
        )
        fwhm = instrument.peak_shape.fwhm(two_theta, wavelength, reflection_list.alpha[index])
        total += accumulate_peaks(grid, two_theta, intensity, fwhm, instrument.peak_shape.eta)
    return total


def powder_pattern(
    crystal: Crystal,
    grid: np.ndarray | None = None,
    instrument: Instrument | None = None,
    r_march_dollase: float = 1.0,
    name: str = "",
    po_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> Pattern:
    """Simulate the powder pattern of a crystal structure.

    Parameters
    ----------
    r_march_dollase:
        March-Dollase preferred-orientation parameter about ``po_axis``.  ``1``
        is a random powder; values below 1 describe crystallites whose
        ``po_axis`` planes lie flat on the mount, and values above 1 the
        opposite - that axis lying *in* the mount plane, which is what a lath
        or a needle does when it settles.
    po_axis:
        The pole the orientation is about, in Miller indices.  ``c*`` describes
        a platelet flattened on 001, which is every layer silicate and so the
        default.  It is not every clay mineral: sepiolite and palygorskite are
        chain silicates whose crystallites are laths, and on the specimen
        measured here a sepiolite settles on its 110 face - about ``c*``,
        nothing describes it (Sec. A.37).
    """
    grid = two_theta_grid() if grid is None else np.asarray(grid, dtype=float)
    instrument = instrument or Instrument()
    wavelengths, _ = instrument.sample_emission()
    d_min = float(np.max(wavelengths)) / (2.0 * math.sin(math.radians(grid[-1] / 2.0)))
    found = reflections(crystal, d_min, po_axis=po_axis)
    intensity = _build_from_reflections(found, grid, instrument, r_march_dollase)
    return Pattern(
        two_theta=grid,
        intensity=intensity,
        name=name or f"{crystal.name} (PO r={r_march_dollase:g})",
        metadata={
            "kind": "powder",
            "phase": crystal.name,
            "source": crystal.source,
            "march_dollase": r_march_dollase,
            "n_reflections": len(found.d),
            "emission": instrument.emission.name,
            "lp_mode": instrument.lp_mode,
        },
    )


def peak_list(
    crystal: Crystal,
    two_theta_range: tuple[float, float],
    instrument: Instrument | None = None,
    r_march_dollase: float = 1.0,
    merge_within: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Merged reflection positions and relative intensities of a crystal.

    Reflections closer together than ``merge_within`` degrees are combined into
    one entry at their intensity-weighted position, which turns the raw
    reciprocal lattice enumeration into the peak list a diffractogram actually
    shows.  Intensities are scaled to a maximum of 1.

    Returns ``(two_theta, intensity)`` sorted by angle.
    """
    instrument = instrument or Instrument()
    low, high = float(two_theta_range[0]), float(two_theta_range[1])
    wavelength = instrument.emission.principal_wavelength
    d_min = wavelength / (2.0 * math.sin(math.radians(high / 2.0)))
    found = reflections(crystal, d_min)

    argument = wavelength / (2.0 * found.d)
    visible = argument < 1.0
    two_theta = np.degrees(2.0 * np.arcsin(argument[visible]))
    inside = (two_theta >= low) & (two_theta <= high)
    if not np.any(inside):
        return np.array([]), np.array([])
    index = np.flatnonzero(visible)[inside]
    two_theta = two_theta[inside]
    intensity = (
        found.f_squared[index]
        * march_dollase(found.alpha[index], r_march_dollase)
        * instrument.lp(two_theta)
    )

    order = np.argsort(two_theta)
    two_theta, intensity = two_theta[order], intensity[order]

    positions: list[float] = []
    heights: list[float] = []
    for angle, value in zip(two_theta, intensity):
        if positions and angle - positions[-1] <= merge_within:
            total = heights[-1] + value
            positions[-1] = (positions[-1] * heights[-1] + angle * value) / max(total, 1e-30)
            heights[-1] = total
        else:
            positions.append(float(angle))
            heights.append(float(value))

    peaks = np.asarray(positions)
    values = np.asarray(heights)
    if values.max() > 0:
        values = values / values.max()
    return peaks, values


def basal_pattern(
    stack: MixedLayerStack,
    grid: np.ndarray | None = None,
    instrument: Instrument | None = None,
    r_march_dollase: float = 1.0,
    name: str = "",
) -> Pattern:
    """Simulate the basal (00l) pattern of an interstratified stack.

    The continuous intensity ``I(s)`` of the Markov model is evaluated at
    ``s = 2 sin(theta)/lambda`` for every line of the emission profile, summed
    with the line weights, and then broadened by the instrumental profile.
    Basal reflections all lie along ``c*``, so the March-Dollase factor is the
    same constant for the whole series and only sets the overall scale.
    """
    grid = two_theta_grid() if grid is None else np.asarray(grid, dtype=float)
    instrument = instrument or Instrument()
    wavelengths, weights = instrument.sample_emission()

    total = np.zeros_like(grid)
    sin_theta = np.sin(np.radians(grid) / 2.0)
    for wavelength, weight in zip(wavelengths, weights):
        s = 2.0 * sin_theta / wavelength
        total += weight * stack.intensity(s)
    total *= instrument.lp(grid) * march_dollase(np.zeros_like(grid), r_march_dollase)

    shape = replace(instrument.peak_shape, size_c=None)
    fwhm = shape.fwhm(grid, instrument.emission.principal_wavelength)
    total = convolve_variable_fwhm(grid, total, fwhm, instrument.peak_shape.eta)

    return Pattern(
        two_theta=grid,
        intensity=total,
        name=name or stack.name,
        metadata={
            "kind": "basal",
            "phase": stack.name,
            "fraction_a": stack.fraction_a,
            "d_a": stack.layer_a.thickness,
            "d_b": stack.layer_b.thickness,
            "transition": stack.transition.tolist(),
            "csds_mean": stack.csds.mean,
            "march_dollase": r_march_dollase,
            "emission": instrument.emission.name,
            "lp_mode": instrument.lp_mode,
        },
    )


def basal_scale_factor(
    host: Crystal,
    layers_per_cell: int,
    grid: np.ndarray,
    instrument: Instrument,
    csds,
) -> float:
    """Scale that puts basal intensities of the stacking model onto the ``hkl`` scale.

    The interstratification model returns intensity per layer, while
    :func:`powder_pattern` returns intensity per unit cell of the host
    structure.  Rather than tracking the normalisation analytically - which is
    awkward for multi-layer polytypes, where the layers of a cell interfere -
    the two are compared for the pure host: the same structure, calculated both
    ways, must give the same basal series.  The ratio of integrated basal
    intensity is that scale.
    """
    layer = host.layer_model(layers_per_cell=layers_per_cell)
    pure = MixedLayerStack(layer, layer, fraction_a=1.0, csds=csds)
    model = basal_pattern(pure, grid, instrument, r_march_dollase=1.0)

    wavelengths, _ = instrument.sample_emission()
    d_min = float(np.max(wavelengths)) / (2.0 * math.sin(math.radians(grid[-1] / 2.0)))
    found = reflections(host, d_min)
    basal = found.select(found.is_basal)
    reference = _build_from_reflections(basal, grid, instrument, r_march_dollase=1.0)

    model_area = float(np.trapezoid(model.intensity, grid))
    reference_area = float(np.trapezoid(reference, grid))
    if model_area <= 0.0:
        return 0.0
    return reference_area / model_area


def mixed_layer_pattern(
    stack: MixedLayerStack,
    host: Crystal,
    layers_per_cell: int,
    grid: np.ndarray | None = None,
    instrument: Instrument | None = None,
    r_march_dollase: float = 1.0,
    include_non_basal: bool = True,
    name: str = "",
    basal_scale: float | None = None,
    host_reflections: Reflections | None = None,
) -> Pattern:
    """Pattern of an interstratified stack including the host's non-basal reflections.

    The basal series comes from the interstratification model; the non-basal
    reflections are those of the host structure, weighted by the host layer
    proportion.  This is the usual approximation for turbostratically stacked
    clays, whose ``hkl`` scattering is not coherent between layers and is
    therefore treated as a property of the individual layers rather than of the
    interstratified crystallite.  By construction the result converges to the
    pure host pattern as the expandable fraction goes to zero.

    ``basal_scale`` and ``host_reflections`` accept precomputed values.  Neither
    depends on the layer proportion or on the orientation parameter, so a series
    of compositions can reuse one pair instead of recomputing them per pattern.
    """
    grid = two_theta_grid() if grid is None else np.asarray(grid, dtype=float)
    instrument = instrument or Instrument()

    scale = (
        basal_scale_factor(host, layers_per_cell, grid, instrument, stack.csds)
        if basal_scale is None
        else float(basal_scale)
    )
    basal = basal_pattern(stack, grid, instrument, r_march_dollase)
    total = scale * basal.intensity

    n_non_basal = 0
    if include_non_basal:
        if host_reflections is None:
            wavelengths, _ = instrument.sample_emission()
            d_min = float(np.max(wavelengths)) / (2.0 * math.sin(math.radians(grid[-1] / 2.0)))
            host_reflections = reflections(host, d_min)
        found = host_reflections
        non_basal = found.select(~found.is_basal)
        n_non_basal = len(non_basal.d)
        total = total + stack.fraction_a * _build_from_reflections(
            non_basal, grid, instrument, r_march_dollase
        )

    metadata = dict(basal.metadata)
    metadata.update(
        {
            "kind": "mixed_layer",
            "host": host.name,
            "host_source": host.source,
            "basal_scale": scale,
            "n_non_basal": n_non_basal,
            "include_non_basal": include_non_basal,
        }
    )
    return Pattern(two_theta=grid, intensity=total, name=name or stack.name, metadata=metadata)
