"""Fitting a layer composition to a measured pattern of the pure mineral.

A published structure is of one specimen of a mineral.  For a chlorite that is a
weak foundation, because the octahedral sheets take iron for magnesium in
proportions that vary from one deposit to the next, iron scatters about twice as
strongly as the magnesium it replaces, and the two sheets sit at different
heights in the layer - so the composition acts directly on the basal intensities
that a quantification is read from.  One published clinochlore cannot describe
two chlorites of different iron content, and it does not: measured against two
pure chlorites, the same structure over-calculates the higher basal orders of
one and under-calculates the other's 003.

A measurement of the pure mineral fixes the composition.  Ratios of basal
intensities within one pattern are used rather than the intensities themselves,
because a ratio is free of the two things a pure mount does not reveal: how much
material is in the beam, and how well oriented it is.  Preferred orientation
multiplies a basal series by one factor (Sec. 2.9), so it divides out
completely; the amount of material does the same.

What remains in a ratio is the structure factors, the Lorentz-polarization
factor, the beam-overflow correction and the crystallite thickness distribution.
The last two are why the reference order matters: overflow is angle dependent
and depends on a specimen length that is not usually known to better than a few
millimetres, so the default reference is 002 rather than 001, 002 lying above
the angle where a clay mount is fully illuminated while 001 usually does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .background import BackgroundModel
from .crystal import LayerModel
from .mixed_layer import MixedLayerStack, lognormal_csds
from .models import CHLORITE_OCTAHEDRA, chlorite_layer
from .pattern import Instrument, basal_pattern
from .pattern import Pattern

__all__ = [
    "BasalSeries",
    "measure_basal_series",
    "ChloriteComposition",
    "fit_chlorite_iron",
]

CU_KA1 = 1.540596


@dataclass
class BasalSeries:
    """Integrated intensities of the 00l reflections of one measured pattern."""

    orders: tuple[int, ...]
    two_theta: tuple[float, ...]
    area: tuple[float, ...]
    reference: int
    spacing: float
    measurement: str = ""

    def normalised(self) -> dict[int, float]:
        """Areas divided by the reference order's."""
        index = self.orders.index(self.reference)
        base = self.area[index]
        if base <= 0.0:
            raise ValueError(
                f"the 00{self.reference} of {self.measurement or 'this pattern'} has no "
                "intensity to normalise on"
            )
        return {l: a / base for l, a in zip(self.orders, self.area)}


def two_theta_of(spacing: float, order: int, wavelength: float = CU_KA1) -> float:
    argument = wavelength * order / (2.0 * spacing)
    if not -1.0 < argument < 1.0:
        raise ValueError(f"00{order} of a {spacing:g} A spacing does not diffract")
    return float(np.degrees(2.0 * np.arcsin(argument)))


def measure_basal_series(
    pattern: Pattern,
    spacing: float,
    orders: tuple[int, ...] = (1, 2, 3, 4, 5),
    reference: int = 2,
    half_width: float = 1.0,
    background: "BackgroundModel | None" = None,
) -> BasalSeries:
    """Integrate the 00l reflections of a measured pattern of one pure phase.

    Each order is integrated over ``half_width`` degrees either side of where the
    spacing puts it.  Orders outside the measured range, and orders whose window
    would run off the end of it, are left out rather than reported as zero.

    ``half_width`` has to be generous, and 0.35 deg - a width that looks ample
    beside a basal reflection whose full width at half maximum is 0.1 to 0.2 deg
    - is not.  A basal reflection of a real clay carries a long diffuse tail from
    stacking disorder: on a measured chlorite such a window held between 36 %
    and 72 % of each order's area, and, worse for a ratio, a *different*
    fraction of each order.  At 1.0 deg the same pattern gives 85 to 93 %, and
    the obs/calc ratios of the higher orders settle: a chlorite 005 read 0.77
    at 0.35 deg, 1.22 at 0.6 deg and 1.34 at 1.0 deg, so a conclusion drawn at
    0.35 deg was measuring the window.  Check the sensitivity to this number
    before believing a result that depends on it.
    """
    model = background or BackgroundModel(chebyshev_degree=4, inverse=True, inverse_offset=1.0)
    fit = model.fit(pattern.two_theta, pattern.intensity, snip_window=4.0)
    stripped = np.clip(fit.subtract(pattern.two_theta, pattern.intensity), 0.0, None)

    kept, angles, areas = [], [], []
    for order in orders:
        try:
            centre = two_theta_of(spacing, order)
        except ValueError:
            continue
        if not (pattern.two_theta[0] + half_width < centre < pattern.two_theta[-1] - half_width):
            continue
        window = np.abs(pattern.two_theta - centre) < half_width
        if window.sum() < 3:
            continue
        kept.append(order)
        angles.append(centre)
        areas.append(float(np.trapezoid(stripped[window], pattern.two_theta[window])))
    if reference not in kept:
        raise ValueError(
            f"00{reference} is not inside the measured range, so there is nothing to "
            f"normalise on; orders found: {kept}"
        )
    return BasalSeries(
        orders=tuple(kept), two_theta=tuple(angles), area=tuple(areas),
        reference=reference, spacing=spacing, measurement=pattern.name,
    )


def calculated_basal_series(
    layer: LayerModel,
    series: BasalSeries,
    instrument: Instrument,
    mean_layers: float,
    csds_beta: float = 0.35,
    half_width: float = 1.0,
) -> dict[int, float]:
    """The same integrals, from a calculated pattern of ``layer``."""
    grid = np.arange(series.two_theta[0] - 2.0, series.two_theta[-1] + 2.0, 0.01)
    stack = MixedLayerStack(layer, layer, 1.0, csds=lognormal_csds(mean_layers, csds_beta))
    calculated = basal_pattern(stack, grid, instrument).intensity
    areas = {}
    for order, centre in zip(series.orders, series.two_theta):
        window = np.abs(grid - centre) < half_width
        areas[order] = float(np.trapezoid(calculated[window], grid[window]))
    base = areas[series.reference]
    if base <= 0.0:
        raise ValueError(f"the calculated 00{series.reference} is a node; choose another reference")
    return {order: value / base for order, value in areas.items()}


@dataclass
class ChloriteComposition:
    """Octahedral iron fitted to a measured pattern of one pure chlorite."""

    iron_2to1: float
    iron_hydroxide: float
    mean_layers: float
    residual: float
    """Root mean square difference in normalised basal intensity."""

    observed: dict[int, float] = field(default_factory=dict)
    calculated: dict[int, float] = field(default_factory=dict)
    published_residual: float = 0.0
    """The same residual for the published structure, for comparison."""

    measurement: str = ""

    @property
    def improvement(self) -> float:
        """How many times smaller the residual is than the published structure's."""
        return self.published_residual / self.residual if self.residual > 0 else float("inf")

    def describe(self) -> str:
        lines = [
            f"{self.measurement or 'chlorite'}: octahedral iron "
            f"{self.iron_2to1:.3f} in the 2:1 sheet, {self.iron_hydroxide:.3f} in the "
            f"hydroxide sheet, {self.mean_layers:.0f} layers per crystallite",
            f"  residual {self.residual:.4f} against {self.published_residual:.4f} "
            f"for the published structure ({self.improvement:.1f} times better)",
            "  00l  observed  calculated",
        ]
        for order in sorted(self.observed):
            lines.append("  %3d %9.3f %11.3f"
                         % (order, self.observed[order], self.calculated.get(order, float("nan"))))
        return "\n".join(lines)


def fit_chlorite_iron(
    pattern: Pattern,
    instrument: Instrument,
    spacing: float = 14.2782,
    irons: np.ndarray | None = None,
    mean_layers: tuple[float, ...] = (6.0, 8.0, 10.0, 15.0, 25.0, 40.0, 70.0),
    reference: int = 2,
    orders: tuple[int, ...] = (1, 2, 3, 4, 5),
    half_width: float = 1.0,
) -> ChloriteComposition:
    """Fit the octahedral iron of a chlorite to a measured pure-phase pattern.

    The two sheets and the crystallite thickness are searched on a grid.  A grid
    rather than a gradient search because the surface is cheap to evaluate, two
    parameters wide, and bounded on both by being occupancies; and because a
    grid cannot converge to a local minimum and report it as the answer.
    """
    series = measure_basal_series(
        pattern, spacing, orders=orders, reference=reference, half_width=half_width
    )
    observed = series.normalised()
    compared = [order for order in series.orders if order != reference]

    def residual_of(iron_a: float, iron_b: float, layers: float):
        calculated = calculated_basal_series(
            chlorite_layer(iron_a, iron_b), series, instrument, layers,
            half_width=half_width,
        )
        difference = [observed[order] - calculated[order] for order in compared]
        return float(np.sqrt(np.mean(np.square(difference)))), calculated

    def search(grid_a, grid_b, layer_set):
        best = None
        for layers in layer_set:
            for iron_a in grid_a:
                for iron_b in grid_b:
                    residual, calculated = residual_of(float(iron_a), float(iron_b), layers)
                    if best is None or residual < best[0]:
                        best = (residual, float(iron_a), float(iron_b), float(layers), calculated)
        return best

    if irons is not None:
        grid = np.asarray(irons, dtype=float)
        best = search(grid, grid, mean_layers)
    else:
        # Coarse over the whole square, then fine around what it found.  Both
        # stages are grids rather than gradient steps, so neither can settle
        # into a local minimum and report it as the answer, and the coarse
        # stage is what rules out a second minimum elsewhere.
        coarse = search(np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11), mean_layers)
        span = 0.1
        best = search(
            np.clip(np.arange(coarse[1] - span, coarse[1] + span + 1e-9, 0.02), 0.0, 1.0),
            np.clip(np.arange(coarse[2] - span, coarse[2] + span + 1e-9, 0.02), 0.0, 1.0),
            (coarse[3],),
        )
        if coarse[0] < best[0]:
            best = coarse

    from .models import load_crystal

    reference_crystal = load_crystal("chlorite")
    published_iron = {
        sheet: float(np.mean([site.occupancy for site in reference_crystal.sites
                              if site.label in labels and site.species.startswith("Fe")]))
        for sheet, labels in CHLORITE_OCTAHEDRA.items()
    }
    published = min(
        residual_of(published_iron["2:1"], published_iron["hydroxide"], layers)[0]
        for layers in mean_layers
    )
    residual, iron_a, iron_b, layers, calculated = best
    return ChloriteComposition(
        iron_2to1=iron_a,
        iron_hydroxide=iron_b,
        mean_layers=layers,
        residual=residual,
        observed=observed,
        calculated=calculated,
        published_residual=published,
        measurement=pattern.name,
    )
