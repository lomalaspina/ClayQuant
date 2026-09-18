"""Finding accompanying minerals from what does not move between treatments.

The three mounts exist because the clays move and nothing else does: glycol
takes the smectite interlayer from about 15 to 17 angstrom and heating collapses
it to 10, while quartz, the feldspars, the carbonates and the oxides stay put.
Asking which peaks are common to all three is therefore a different measurement,
not a better statistic, and it is the only one of ClayQuant's three searches
that can find a minor phase whose strongest line is overlapped.

The case these tests are built around is real.  Rutile is 1.5 per cent of one
clay separate; its 110 falls on a feldspar reflection, so with albite in the fit
it explains 0.13 per cent of the pattern and thirty-five phases explain more.
Its two lines are at 27.47 and 36.07 degrees in the air-dried, glycolated and
heated scans alike, and on that test it comes second of nine with 100 per cent
coverage, while anatase, faujasite, graphite and calciolangbeinite - which the
single-scan fit ranked above it - have no stable line at all.
"""

import numpy as np
import pytest

from clayquant.crystal import AtomSite, Crystal
from clayquant.detection import stable_peaks, stable_phases
from clayquant.pattern import Instrument, Pattern
from clayquant.profile import PeakShape, pseudo_voigt

GRID = np.arange(4.0, 40.0, 0.0167)
INSTRUMENT = Instrument(peak_shape=PeakShape(u=0.0, v=0.0, w=0.09**2, eta=0.5))


def rutile():
    return Crystal(
        a=4.5937, b=4.5937, c=2.9587, alpha=90.0, beta=90.0, gamma=90.0,
        sites=[
            AtomSite(species="Ti", x=0.0, y=0.0, z=0.0, occupancy=1.0, b_iso=0.5, label="Ti"),
            AtomSite(species="O", x=0.3053, y=0.3053, z=0.0, occupancy=1.0, b_iso=0.6, label="O"),
        ],
        name="Rutile",
    )


def quartz():
    return Crystal(
        a=4.9137, b=4.9137, c=5.4047, alpha=90.0, beta=90.0, gamma=120.0,
        sites=[
            AtomSite(species="Si", x=0.4697, y=0.0, z=0.0, occupancy=1.0, b_iso=0.5, label="Si"),
            AtomSite(species="O", x=0.4135, y=0.2669, z=0.1191, occupancy=1.0,
                     b_iso=0.8, label="O"),
        ],
        name="Quartz",
    )


def a_mount(fixed, moving, seed=1):
    """A pattern with peaks at ``fixed`` and a clay peak at ``moving``."""
    y = np.zeros_like(GRID)
    for centre, height in fixed:
        profile = pseudo_voigt(GRID - centre, 0.10, 0.5)
        y += height * profile / profile.max()
    profile = pseudo_voigt(GRID - moving, 0.30, 0.5)
    y += 6000.0 * profile / profile.max()
    y = y + np.random.default_rng(seed).normal(0.0, 8.0, GRID.size)
    return Pattern(two_theta=GRID, intensity=np.clip(y, 0.0, None), name="mount")


def planted(crystal, scale, floor=0.10):
    """Every reflection of ``crystal`` above ``floor`` of its strongest."""
    from clayquant.pattern import peak_list

    positions, heights = peak_list(crystal, (4.0, 40.0), INSTRUMENT)
    keep = heights >= floor * heights.max()
    return list(zip(positions[keep], scale * heights[keep] / heights.max()))


# Quartz at full strength and rutile as a minor phase, every strong reflection
# of each present, so a coverage of 1 is the right answer for both.
FIXED = planted(quartz(), 8500.0) + planted(rutile(), 700.0)
MOUNTS = [a_mount(FIXED, 6.2, 1), a_mount(FIXED, 5.2, 2), a_mount(FIXED, 8.8, 3)]


def test_a_peak_in_every_mount_is_stable():
    peaks = stable_peaks(MOUNTS)
    angles = [peak.two_theta for peak in peaks]
    for expected in (26.64, 27.47):
        assert any(abs(angle - expected) < 0.06 for angle in angles), expected


def test_a_peak_that_moves_is_not_stable():
    peaks = stable_peaks(MOUNTS)
    for angle in (peak.two_theta for peak in peaks):
        assert not 5.0 < angle < 9.5, f"the clay peak at {angle} was called stable"


def test_one_mount_alone_makes_everything_stable():
    # Nothing to compare against, so the test degenerates; the caller is
    # expected to use the single-mount searches instead.
    peaks = stable_peaks([MOUNTS[0]])
    assert any(5.0 < peak.two_theta < 9.5 for peak in peaks)


def test_a_minor_overlapped_phase_is_found_with_full_coverage():
    database = {"Quartz": quartz(), "Rutile": rutile()}
    findings, _ = stable_phases(MOUNTS, database, two_theta_range=(4.0, 40.0),
                                instrument=INSTRUMENT)
    names = [evidence.name for evidence in findings]
    assert "Rutile" in names
    rutile_found = findings[names.index("Rutile")]
    assert rutile_found.score > 0.9
    assert rutile_found.n_matched >= 2


def test_coverage_does_not_depend_on_what_else_is_in_the_database():
    """The property that makes this work where a share of the pattern does not.

    A share is divided among whatever competes for the same intensity, so a
    phase drops when a candidate is added.  Coverage is a property of the phase
    and the stable peak list, so it does not move.
    """
    small = {"Rutile": rutile()}
    large = {"Rutile": rutile(), "Quartz": quartz()}
    alone, _ = stable_phases(MOUNTS, small, two_theta_range=(4.0, 40.0),
                             instrument=INSTRUMENT)
    together, _ = stable_phases(MOUNTS, large, two_theta_range=(4.0, 40.0),
                                instrument=INSTRUMENT)
    names = [evidence.name for evidence in together]
    assert alone[0].score == pytest.approx(together[names.index("Rutile")].score)


def test_a_phase_with_no_stable_line_is_not_reported():
    # Graphite's 002 is at 26.55, within a tenth of a degree of the quartz 101,
    # so it does match a stable peak - but its other line has nothing under it,
    # and one reflection is below min_matched.
    graphite = Crystal(
        a=2.464, b=2.464, c=6.711, alpha=90.0, beta=90.0, gamma=120.0,
        sites=[
            AtomSite(species="C", x=0.0, y=0.0, z=0.25, occupancy=1.0, b_iso=0.5, label="C1"),
            AtomSite(species="C", x=1 / 3, y=2 / 3, z=0.75, occupancy=1.0,
                     b_iso=0.5, label="C2"),
        ],
        name="Graphite",
    )
    findings, _ = stable_phases(MOUNTS, {"Graphite": graphite},
                                two_theta_range=(4.0, 40.0), instrument=INSTRUMENT)
    assert [evidence.name for evidence in findings] == []


def test_the_clays_are_left_to_the_clay_library():
    database = {"Quartz": quartz(), "illite": quartz()}
    findings, _ = stable_phases(MOUNTS, database, two_theta_range=(4.0, 40.0),
                                instrument=INSTRUMENT)
    assert "illite" not in [evidence.name for evidence in findings]


def test_the_restriction_is_honoured():
    database = {"Quartz": quartz(), "Rutile": rutile()}
    findings, _ = stable_phases(MOUNTS, database, two_theta_range=(4.0, 40.0),
                                instrument=INSTRUMENT, only={"Rutile"})
    assert [evidence.name for evidence in findings] == ["Rutile"]


def test_no_mounts_gives_nothing_rather_than_failing():
    assert stable_peaks([]) == []
    findings, peaks = stable_phases([], {"Quartz": quartz()}, instrument=INSTRUMENT)
    assert findings == [] and peaks == []


def test_the_coverage_threshold_shortens_the_list():
    database = {"Quartz": quartz(), "Rutile": rutile()}
    loose, _ = stable_phases(MOUNTS, database, two_theta_range=(4.0, 40.0),
                             instrument=INSTRUMENT, min_coverage=0.0)
    strict, _ = stable_phases(MOUNTS, database, two_theta_range=(4.0, 40.0),
                              instrument=INSTRUMENT, min_coverage=0.95)
    assert len(strict) <= len(loose)
    assert all(evidence.score >= 0.95 for evidence in strict)


def test_a_permissive_peak_finder_would_make_coverage_meaningless():
    """Why the thresholds are strict, pinned so they are not loosened casually.

    Taking every local maximum at three sigma gave 527 stable peaks on one real
    specimen, at which density a 0.05 degree window catches something near every
    calculated line and forty phases came out at 100 per cent coverage.  A peak
    list that matches everything distinguishes nothing.
    """
    strict = stable_peaks(MOUNTS)
    permissive = stable_peaks(MOUNTS, min_signal_to_noise=1.0, minimum_height=0.0,
                              separation=0.02, tolerance=0.3)
    assert len(permissive) > 3 * len(strict)
