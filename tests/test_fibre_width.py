"""The width of a channel clay's 110, which is the fibre's diameter.

Every one of these numbers came off the sepiolite standard's three mounts.  The
defect they pin was quiet: the pattern had the right lines in the right places
and the wrong width, so nothing looked broken and the fit simply could not use
it - see Sec. A.68.
"""
from dataclasses import replace

import numpy as np
import pytest

from clayquant.library import (
    DEFAULT_PEAK_SHAPE,
    FIBROUS_DIAMETERS,
    FIBROUS_SPACINGS,
    LibraryEntry,
    PatternLibrary,
    scaled_in_plane,
)
from clayquant.models import FIBRE_ORIENTATION, habit_axis, load_crystal
from clayquant.pattern import Instrument, powder_pattern, two_theta_grid
from clayquant.emission import CU_KA_5LINE
from clayquant.optics import Divergence

GRID = two_theta_grid(4.0, 34.0, 0.02)
MEASURED_FWHM = 0.618
"""FWHM in degrees of the sepiolite standard's 110 on the glycolated mount.

The air-dried mount gives 0.685 deg.  The heated one gives 2.49 deg, which is a
different mineral: sepiolite folds to the anhydride at 550 C.
"""


def an_instrument(size_ab):
    return Instrument(
        emission=CU_KA_5LINE,
        peak_shape=replace(DEFAULT_PEAK_SHAPE, size_ab=size_ab),
        lp_mode="powder",
        divergence=Divergence(specimen_length=25.0, goniometer_radius=240.0,
                              divergence=0.5, shape="round"),
    )


def fwhm(x, y, centre=7.15):
    j = int(np.argmin(np.abs(x - centre)))
    window = slice(max(0, j - 60), j + 61)
    j = window.start + int(np.argmax(y[window]))
    half = y[j] / 2.0
    lo, hi = j, j
    while lo > 0 and y[lo] > half:
        lo -= 1
    while hi < len(y) - 1 and y[hi] > half:
        hi += 1
    return float(x[hi] - x[lo])


@pytest.fixture(scope="module")
def sepiolite():
    return scaled_in_plane(load_crystal("sepiolite"), 12.35)


def a_pattern(sepiolite, size_ab):
    axis, space = habit_axis("sepiolite")
    return powder_pattern(sepiolite, GRID, an_instrument(size_ab),
                          r_march_dollase=FIBRE_ORIENTATION,
                          po_axis=axis, po_axis_space=space, name="s").intensity


# --------------------------------------------------------------------------- #
# the width the specimen has
# --------------------------------------------------------------------------- #

def test_a_platelet_width_makes_the_110_far_too_sharp(sepiolite):
    """The defect itself: 400 A is a clay platelet's lateral extent."""
    width = fwhm(GRID, a_pattern(sepiolite, 400.0))
    assert width < 0.3
    assert width < MEASURED_FWHM / 2


def test_a_fibre_width_reproduces_the_measured_one(sepiolite):
    """130 A is what both unheated mounts give, independently."""
    assert fwhm(GRID, a_pattern(sepiolite, 130.0)) == pytest.approx(0.635, abs=0.03)
    assert abs(fwhm(GRID, a_pattern(sepiolite, 130.0)) - MEASURED_FWHM) < 0.05


def test_the_width_falls_monotonically_with_the_diameter(sepiolite):
    widths = [fwhm(GRID, a_pattern(sepiolite, size)) for size in FIBROUS_DIAMETERS]
    assert widths == sorted(widths, reverse=True)


def test_the_span_brackets_the_measured_diameter():
    """Not at an edge: a fit landing on an end has been stopped, not determined."""
    assert min(FIBROUS_DIAMETERS) < 130.0 < max(FIBROUS_DIAMETERS)
    assert 130.0 in FIBROUS_DIAMETERS


def test_the_span_covers_the_reported_range_of_fibre_cross_sections():
    assert min(FIBROUS_DIAMETERS) == pytest.approx(90.0)
    assert max(FIBROUS_DIAMETERS) == pytest.approx(280.0)
    assert DEFAULT_PEAK_SHAPE.size_ab > max(FIBROUS_DIAMETERS)


# --------------------------------------------------------------------------- #
# it broadens across the fibre and not along it
# --------------------------------------------------------------------------- #

def test_the_diameter_leaves_the_reflections_along_the_fibre_alone(sepiolite):
    """size_ab enters as sin^2 of the angle from the axis the texture is about.

    For a channel clay that axis is ``c`` in direct space, so an ``hk0`` sits at
    90 degrees from it and takes the diameter in full, while a ``00l`` sits at 0
    and takes ``size_c``, which is None - a fibre is long along ``c`` and the
    measurement shows no size broadening there.
    """
    axis, space = habit_axis("sepiolite")
    from clayquant.pattern import reflections
    found = reflections(sepiolite, d_min=2.6, po_axis=axis, po_axis_space=space)
    basal = np.all(found.hkl[:, :2] == 0, axis=1)
    in_plane = found.hkl[:, 2] == 0
    assert np.degrees(found.alpha[in_plane]).round(6).tolist() == \
        [pytest.approx(90.0)] * int(in_plane.sum())
    assert basal.any()
    for a in np.degrees(found.alpha[basal]):
        assert a == pytest.approx(0.0) or a == pytest.approx(180.0)


def test_only_the_channel_clays_are_narrowed(sepiolite):
    """The platelets keep the default: their basal lines take size_c, not this."""
    kaolinite = load_crystal("kaolinite_1M")
    wide = powder_pattern(kaolinite, GRID, an_instrument(400.0), name="k").intensity
    narrow = powder_pattern(kaolinite, GRID, an_instrument(130.0), name="k").intensity
    basal = fwhm(GRID, wide, 12.4), fwhm(GRID, narrow, 12.4)
    assert basal[0] == pytest.approx(basal[1], abs=1e-9)


# --------------------------------------------------------------------------- #
# the library carries it
# --------------------------------------------------------------------------- #

def test_an_entry_remembers_its_diameter(tmp_path):
    grid = np.arange(4.0, 10.0, 0.02)
    lib = PatternLibrary(two_theta=grid)
    from clayquant.pattern import Pattern
    lib.add(Pattern(two_theta=grid, intensity=np.ones_like(grid), name="sepiolite D=130"),
            phase="sepiolite", thickness=12.35, domain_ab=130.0, strip_continuum=False)
    path = lib.save(tmp_path / "lib.npz")
    back = PatternLibrary.load(path)
    assert back.entries[0].domain_ab == pytest.approx(130.0)


def test_a_platelet_entry_has_no_diameter():
    """The two are different directions in different units, so not one field."""
    assert LibraryEntry(name="x", phase="illite",
                        intensity=np.zeros(3)).domain_ab is None


def test_the_diameter_is_an_axis_the_edge_detector_can_see():
    """A phase pinned at an end of this span has not been measured."""
    grid = np.arange(4.0, 10.0, 0.02)
    from clayquant.pattern import Pattern
    lib = PatternLibrary(two_theta=grid)
    for diameter in FIBROUS_DIAMETERS:
        lib.add(Pattern(two_theta=grid, intensity=np.ones_like(grid),
                        name=f"sepiolite D={diameter:g}"),
                phase="sepiolite", domain_ab=diameter, strip_continuum=False)
    assert lib.spanned()["sepiolite/domain_ab"] == sorted(FIBROUS_DIAMETERS)


def test_every_channel_clay_gets_the_span():
    """Palygorskite is a fibre too, and its diameter is not measured here.

    The span is applied to it on the structural argument alone - the reported
    cross-sections of the two minerals are the same range - and that is said
    plainly rather than being presented as a measurement.
    """
    assert set(FIBROUS_SPACINGS) == {"sepiolite", "palygorskite"}
