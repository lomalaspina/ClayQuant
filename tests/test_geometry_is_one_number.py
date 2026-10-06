"""The goniometer radius must be one number, and every message must quote it.

It was two.  `clayquant.calibration` defined 240.0, `optics.Divergence` carried
its own 280.0 default, and six user-facing strings wrote 280 as a literal.  The
calculation used 240 everywhere, because `build_library` passes the constant
explicitly - so every library ever built was correct and every message about it
was wrong.

The desktop builder told an operator, in the window where they build it, that
their library had been calculated for a 280 mm instrument.  It had not.  They
reasonably believed the tool over the truth and were about to rebuild against a
geometry they do not own (Sec. A.71).

A number that appears in a message is a claim about the program's behaviour, and
nothing was checking that the claim was true.
"""
import pathlib
import re

import pytest

from clayquant.calibration import DEFAULT_GONIOMETER_RADIUS as FROM_CALIBRATION
from clayquant.library import (DEFAULT_DIVERGENCE_SLIT, DEFAULT_GONIOMETER_RADIUS,
                               DEFAULT_SPECIMEN_LENGTH, build_library)
from clayquant.optics import DEFAULT_GONIOMETER_RADIUS as FROM_OPTICS

SOURCE = pathlib.Path(__file__).resolve().parents[1]
SEARCHED = [SOURCE / "src" / "clayquant", SOURCE / "scripts"]


def test_there_is_exactly_one_definition():
    assert FROM_OPTICS is FROM_CALIBRATION
    assert DEFAULT_GONIOMETER_RADIUS == FROM_OPTICS
    text = (SOURCE / "src" / "clayquant" / "calibration.py").read_text()
    assert "DEFAULT_GONIOMETER_RADIUS = " not in text, "re-export it, do not redefine it"


def test_a_bare_divergence_uses_it():
    """The default that was 280 while the constant beside it said 240."""
    from clayquant.optics import Divergence
    assert Divergence(specimen_length=25.0).goniometer_radius == DEFAULT_GONIOMETER_RADIUS


def test_the_library_is_built_with_it():
    """What the message claims has to be what the calculation does."""
    import numpy as np
    library = build_library(grid=np.arange(8.0, 10.0, 0.05),
                            illite_smectite=(), chlorite_smectite=(),
                            fibrous_spacings={}, host_thicknesses={},
                            chlorite_iron=(), illite_composition=((1.0, 0.0),),
                            orientations=(1.0,), progress=False)
    geometry = library.metadata["geometry"]
    assert geometry["goniometer_radius"] == DEFAULT_GONIOMETER_RADIUS
    assert geometry["divergence"] == DEFAULT_DIVERGENCE_SLIT
    assert geometry["specimen_length"] == DEFAULT_SPECIMEN_LENGTH


# --------------------------------------------------------------------------- #
# no message may write a geometry number as a literal
# --------------------------------------------------------------------------- #

RADIUS_CLAIM = re.compile(r"(\d{3}(?:\.\d+)?)\s*mm\b[^\n]{0,40}"
                          r"(?:goniometer|radius)|(?:goniometer|radius)[^\n]{0,40}"
                          r"\b(\d{3}(?:\.\d+)?)\s*mm\b", re.IGNORECASE)


def python_sources():
    for folder in SEARCHED:
        for path in folder.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_no_source_states_a_radius_that_is_not_the_constant():
    """A literal radius beside the word 'goniometer' is a claim about behaviour.

    Lines naming another instrument's radius for contrast are the exception and
    must say so with the word 'instrument' or a make, which is why this looks
    only for claims about what *this* program does.
    """
    wrong = []
    for path in python_sources():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "DEFAULT_GONIOMETER_RADIUS" in line:
                continue
            for match in RADIUS_CLAIM.finditer(line):
                value = float(match.group(1) or match.group(2))
                if value != DEFAULT_GONIOMETER_RADIUS:
                    wrong.append(f"{path.relative_to(SOURCE)}:{number}: {line.strip()}")
    assert not wrong, (
        "these state a goniometer radius that is not the constant:\n  "
        + "\n  ".join(wrong))


def test_the_guard_would_have_caught_the_original_fault():
    """The exact string the desktop builder printed, which was false."""
    offending = "the library is calculated for a 280 mm goniometer radius,"
    assert RADIUS_CLAIM.search(offending)
    found = float(next(g for g in RADIUS_CLAIM.search(offending).groups() if g))
    assert found != DEFAULT_GONIOMETER_RADIUS
