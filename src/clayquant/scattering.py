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



_LABEL_PATTERN = re.compile(r"^([A-Z][a-z]?)[0-9]*$")


def _relax(label: str) -> str | None:
    """A species spelling real files use, rewritten to one this table knows.

    Four of them, each seen in a structure file rather than imagined:

    * a parenthesised charge, ``Ti(4+)`` for ``Ti4+``;
    * deuterium, which scatters x-rays as hydrogen and is written ``D``;
    * a site *label* left in the species field, ``Fe1`` or ``O3``, which is the
      commonest of the four and the one that makes a whole structure unusable
      rather than one site - note this is only reached after the strict parse
      has failed, so a genuine charge like ``Fe3+`` never comes here;
    * a symbol in the wrong case, ``FE`` or ``si``.

    Returns ``None`` where nothing sensible can be made of the label, which is
    the right answer for ``OH``: that is two elements, and guessing which one
    was meant would put a wrong mass into a weight percent.
    """
    text = label.replace("(", "").replace(")", "").strip()
    if not text:
        return None
    if text in ("D", "T"):      # deuterium and tritium scatter as hydrogen
        return "H"
    if _ION_PATTERN.match(text):
        return text
    bare = _LABEL_PATTERN.match(text)
    if bare is not None:
        return bare.group(1)
    if len(text) <= 2 and text.isalpha():
        cased = text[0].upper() + text[1:].lower()
        if _ION_PATTERN.match(cased):
            return cased
    return None


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
        # The strict spelling failed, so try the ones real files actually carry
        # before giving up.  These are fallbacks and run only here, so nothing
        # that already parsed can change meaning.
        relaxed = _relax(label)
        if relaxed is not None and relaxed in table:
            return relaxed
        if relaxed is not None:
            match = _ION_PATTERN.match(relaxed)
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
    # A label can pass the strict pattern and still not name a tabulated
    # element - "D" is a valid-looking symbol and is deuterium - so the
    # relaxations get a turn here too.
    relaxed = _relax(element)
    if relaxed is not None and relaxed in table:
        return relaxed
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
