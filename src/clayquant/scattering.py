"""Atomic scattering factors.

Elastic X-ray scattering factors are evaluated with the analytical
parameterisation of Waasmaier & Kirfel, Acta Cryst. A51 (1995) 416-431:

    f0(q) = c + sum_i a_i * exp(-b_i * q**2),   q = sin(theta) / lambda  [1/A]

The coefficient table shipped in ``data/waasmaier_kirfel_f0.json`` covers the
neutral atoms and the ions tabulated by those authors.  It was extracted from
the open ``xraydb`` distribution of the same table.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources

import numpy as np

__all__ = ["f0", "debye_waller", "normalise_species", "available_species"]

_ION_PATTERN = re.compile(r"^([A-Z][a-z]?)(?:([0-9]*)([+-])|([+-])([0-9]*))?$")


@lru_cache(maxsize=1)
def _table() -> dict[str, dict]:
    with resources.files("clayquant.data").joinpath("waasmaier_kirfel_f0.json").open() as fh:
        return json.load(fh)


def available_species() -> list[str]:
    """Species names present in the Waasmaier & Kirfel table."""
    return sorted(_table())


def normalise_species(species: str) -> str:
    """Map a species label onto a key of the Waasmaier & Kirfel table.

    Accepts ``"Fe"``, ``"Fe3+"``, ``"Fe+3"``, ``"K+"`` (-> ``K1+``) and similar
    spellings.  Falls back to the neutral atom when the requested ion is not
    tabulated, so that e.g. ``"Ti3+"`` resolves even where only ``Ti`` exists.
    """
    table = _table()
    label = species.strip()
    if label in table:
        return label

    match = _ION_PATTERN.match(label)
    if match is None:
        raise KeyError(f"cannot parse scattering species {species!r}")
    element = match.group(1)
    digits = match.group(2) or match.group(5) or ""
    sign = match.group(3) or match.group(4) or ""
    if sign:
        charge = digits or "1"
        candidate = f"{element}{charge}{sign}"
        if candidate in table:
            return candidate
    if element in table:
        return element
    raise KeyError(f"no scattering factor tabulated for {species!r}")


def f0(species: str, q: np.ndarray | float) -> np.ndarray:
    """Elastic scattering factor of ``species`` at ``q = sin(theta)/lambda`` [1/A]."""
    entry = _table()[normalise_species(species)]
    q2 = np.asarray(q, dtype=float) ** 2
    total = np.full(q2.shape, entry["c"], dtype=float)
    for a, b in zip(entry["a"], entry["b"]):
        total += a * np.exp(-b * q2)
    return total


def debye_waller(b_iso: float, q: np.ndarray | float) -> np.ndarray:
    """Isotropic Debye-Waller factor ``exp(-B * q**2)`` with ``q = sin(theta)/lambda``.

    ``B`` follows the crystallographic convention ``exp(-B sin^2(theta)/lambda^2)``
    and is given in square Angstrom.
    """
    return np.exp(-b_iso * np.asarray(q, dtype=float) ** 2)
