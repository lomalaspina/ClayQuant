"""X-ray emission profiles.

An emission profile is a set of Lorentzian spectral lines, each with a relative
weight, a peak wavelength and a natural (Lorentzian) FWHM.  The default profile
is the five-line Cu Ka description of Holzer et al. (1997), in the form used by
TOPAS ``Lam_recs`` records: ``{weight  lambda[A]  fwhm[mA]}``.

The wavelength spread of the emission profile produces a 2theta broadening that
grows with angle.  Differentiating Bragg's law gives

    d(2theta) = 2 tan(theta) * dlambda / lambda ,

so instead of convolving with an angle-dependent kernel, a pattern calculator
can simply evaluate the monochromatic pattern at a set of discrete wavelengths
and sum the results with the corresponding weights.  :meth:`EmissionProfile.
sample` produces such a set: the delta-function limit (one wavelength per line)
or a quantile expansion of each Lorentzian (several wavelengths per line), which
reproduces the natural line widths exactly in the same summation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["EmissionLine", "EmissionProfile", "CU_KA_5LINE", "CU_KA1"]


@dataclass(frozen=True)
class EmissionLine:
    """One Lorentzian emission line.

    Attributes
    ----------
    weight:
        Relative integrated intensity.
    wavelength:
        Peak wavelength in Angstrom.
    fwhm:
        Natural Lorentzian FWHM in Angstrom (TOPAS ``Lam_recs`` quotes this in
        milli-Angstrom; :class:`EmissionProfile.from_topas` converts it).
    """

    weight: float
    wavelength: float
    fwhm: float = 0.0


class EmissionProfile:
    """A weighted set of :class:`EmissionLine` objects."""

    def __init__(self, lines: list[EmissionLine], name: str = "custom"):
        if not lines:
            raise ValueError("an emission profile needs at least one line")
        total = sum(line.weight for line in lines)
        if total <= 0:
            raise ValueError("emission line weights must sum to a positive value")
        self.lines = [
            EmissionLine(line.weight / total, line.wavelength, line.fwhm) for line in lines
        ]
        self.name = name

    @classmethod
    def from_topas(cls, records: list[tuple[float, float, float]], name: str = "custom") -> "EmissionProfile":
        """Build a profile from TOPAS ``Lam_recs`` rows ``(weight, lambda_A, fwhm_mA)``."""
        return cls(
            [EmissionLine(w, lam, fwhm_ma * 1e-3) for w, lam, fwhm_ma in records],
            name=name,
        )

    @property
    def mean_wavelength(self) -> float:
        """Weighted mean wavelength in Angstrom."""
        return float(sum(line.weight * line.wavelength for line in self.lines))

    @property
    def principal_wavelength(self) -> float:
        """Wavelength of the strongest line in Angstrom."""
        return max(self.lines, key=lambda line: line.weight).wavelength

    def sample(self, n_per_line: int = 1, tail: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
        """Discretise the profile into wavelengths and weights.

        Parameters
        ----------
        n_per_line:
            Number of sample wavelengths per emission line.  ``1`` ignores the
            natural line widths (delta lines).  Larger odd values expand each
            Lorentzian on an equal-probability (quantile) grid, which
            reproduces its natural width when the samples are summed.
        tail:
            Fraction of each Lorentzian's integrated intensity left outside the
            sampled range, split between the two tails.  Lorentzians have heavy
            tails, so this truncation is what keeps the sample grid finite.

        Returns
        -------
        wavelengths, weights:
            Arrays of equal length; ``weights`` sums to 1.
        """
        if n_per_line < 1:
            raise ValueError("n_per_line must be >= 1")
        if not 0.0 < tail < 1.0:
            raise ValueError("tail must lie in (0, 1)")

        wavelengths: list[float] = []
        weights: list[float] = []
        for line in self.lines:
            if n_per_line == 1 or line.fwhm <= 0.0:
                wavelengths.append(line.wavelength)
                weights.append(line.weight)
                continue
            # Inverse Lorentzian CDF on an equal-probability grid.
            quantiles = np.linspace(tail / 2.0, 1.0 - tail / 2.0, n_per_line)
            gamma = line.fwhm / 2.0
            offsets = gamma * np.tan(np.pi * (quantiles - 0.5))
            wavelengths.extend(line.wavelength + offsets)
            weights.extend([line.weight / n_per_line] * n_per_line)

        w = np.asarray(weights, dtype=float)
        return np.asarray(wavelengths, dtype=float), w / w.sum()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"EmissionProfile(name={self.name!r}, n_lines={len(self.lines)})"


# Cu Ka profile after Holzer, Fritsch, Deutsch, Hartwig & Forster,
# Phys. Rev. A 56 (1997) 4554: the Ka3 satellite plus the Ka11, Ka12, Ka21 and
# Ka22 components.  Rows are the TOPAS Lam_recs values {weight, lambda, fwhm/mA}.
CU_KA_5LINE = EmissionProfile.from_topas(
    [
        (0.0159, 1.534753, 3.6854),
        (0.5691, 1.540596, 0.4370),
        (0.0762, 1.541058, 0.6000),
        (0.2517, 1.544410, 0.5200),
        (0.0871, 1.544721, 0.6200),
    ],
    name="Cu Ka (5 lines, Holzer et al. 1997)",
)

CU_KA1 = EmissionProfile(
    [EmissionLine(1.0, 1.540596, 0.4370)],
    name="Cu Ka1",
)
