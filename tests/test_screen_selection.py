"""Selecting accompanying minerals: what a phase must earn to be reported.

The screen used to put every candidate into one non-negative least squares
beside the whole clay library and rank by the share each took.  With a thousand
clay patterns and two hundred minerals over two thousand points that system is
not identifiable, and it failed in a way that looked like a tuning problem and
was not: on one real clay separate it ranked quartz first, and on the same
specimen with the zero error not yet applied it gave quartz nothing and credited
graphite - two reflections, one of which lands on the quartz 101 - with 8.9 per
cent of the pattern.

These tests pin the two properties that fix it: a phase is chosen for what it
explains that nothing already chosen explains, and a phase is allowed a small
bodily shift so that being sharp is not a disadvantage.
"""

import numpy as np
import pytest

from clayquant.crystal import AtomSite, Crystal
from clayquant.detection import screen_phases
from clayquant.library import PatternLibrary
from clayquant.pattern import Instrument, Pattern, powder_pattern
from clayquant.profile import PeakShape

GRID = np.arange(4.0, 40.0, 0.02)
INSTRUMENT = Instrument(peak_shape=PeakShape(u=0.0, v=0.0, w=0.09**2, eta=0.5))


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


def graphite():
    # Two reflections in range, 002 at 3.35 A, which is 0.09 deg from quartz 101.
    return Crystal(
        a=2.464, b=2.464, c=6.711, alpha=90.0, beta=90.0, gamma=120.0,
        sites=[
            AtomSite(species="C", x=0.0, y=0.0, z=0.25, occupancy=1.0, b_iso=0.5, label="C1"),
            AtomSite(species="C", x=1 / 3, y=2 / 3, z=0.75, occupancy=1.0, b_iso=0.5, label="C2"),
        ],
        name="Graphite",
    )


def rutile():
    return Crystal(
        a=4.5937, b=4.5937, c=2.9587, alpha=90.0, beta=90.0, gamma=90.0,
        sites=[
            AtomSite(species="Ti", x=0.0, y=0.0, z=0.0, occupancy=1.0, b_iso=0.5, label="Ti"),
            AtomSite(species="O", x=0.3053, y=0.3053, z=0.0, occupancy=1.0, b_iso=0.6, label="O"),
        ],
        name="Rutile",
    )


def a_clay_library():
    """A small stand-in: broad basal-looking humps the candidates compete against."""
    library = PatternLibrary(two_theta=GRID, entries=[], metadata={})
    for centre in (6.2, 8.9, 12.5, 17.8, 25.1):
        y = np.exp(-0.5 * ((GRID - centre) / 0.35) ** 2)
        library.add(Pattern(two_theta=GRID, intensity=y, name=f"clay{centre}"),
                    phase=f"clay{centre}", march_dollase=0.1)
    return library


def a_measurement(crystals, weights, shift=0.0, noise=0.0, seed=3):
    total = np.zeros_like(GRID)
    for crystal, weight in zip(crystals, weights):
        pattern = powder_pattern(crystal, GRID, INSTRUMENT, r_march_dollase=1.0)
        total = total + weight * pattern.intensity / pattern.intensity.max()
    for centre, height in ((8.9, 1.2), (12.5, 0.5)):
        total = total + height * np.exp(-0.5 * ((GRID - centre) / 0.35) ** 2)
    counts = 8000.0 * total + 60.0
    if noise:
        counts = counts + np.random.default_rng(seed).normal(0.0, noise, GRID.size)
    x = GRID + shift
    return Pattern(two_theta=x, intensity=np.clip(counts, 1.0, None), name="synthetic")


DATABASE = {"Quartz": quartz(), "Graphite": graphite(), "Rutile": rutile()}


def test_quartz_is_found_when_quartz_is_there():
    found = screen_phases(
        a_measurement([quartz()], [1.0]), DATABASE, a_clay_library(),
        two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, min_share=0.001, max_phases=3,
    )
    assert found
    assert found[0].name == "Quartz"


def test_a_two_line_phase_does_not_take_the_peak_that_belongs_to_quartz():
    # Graphite's 002 sits 0.09 deg from the quartz 101.  Under the old joint
    # solve it could be credited with that peak; chosen in rounds it cannot,
    # because quartz explains the same peak and five others.
    found = screen_phases(
        a_measurement([quartz()], [1.0]), DATABASE, a_clay_library(),
        two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, min_share=0.001, max_phases=3,
    )
    names = [evidence.name for evidence in found]
    assert names.index("Quartz") == 0
    if "Graphite" in names:
        graphite_share = found[names.index("Graphite")].score
        assert graphite_share < 0.25 * found[0].score


def test_a_short_line_list_is_no_bar_to_being_found():
    # The fix must not work by penalising phases with few reflections: rutile
    # has three in range and is a real mineral of these separates.
    found = screen_phases(
        a_measurement([rutile()], [1.0]), DATABASE, a_clay_library(),
        two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, min_share=0.001, max_phases=3,
    )
    assert found and found[0].name == "Rutile"


def test_a_phase_that_is_absent_is_not_reported_above_the_threshold():
    found = screen_phases(
        a_measurement([rutile()], [1.0]), DATABASE, a_clay_library(),
        two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, min_share=0.05, max_phases=3,
    )
    assert [evidence.name for evidence in found] == ["Rutile"]


def test_the_threshold_cuts_the_report_and_not_the_rounds():
    """The failure this prevents, from a run on a real clay separate.

    The score a phase is reported with is its share of the pattern, and the
    score selection proceeds on is the gain - what a phase adds after everything
    already chosen.  They are different numbers, and the second and third phase
    of a real assemblage have small gains however plainly they are present.
    Stopping the rounds when the gain fell below the reporting threshold cut the
    list to the first one or two phases, which reads as the minerals having gone
    missing and was the opposite of what the change was for.
    """
    findings = screen_phases(
        a_measurement([quartz(), rutile()], [1.0, 0.3]), DATABASE, a_clay_library(),
        two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, min_share=0.005, max_phases=3,
    )
    names = [evidence.name for evidence in findings]
    assert "Quartz" in names and "Rutile" in names


def test_an_unapplied_zero_error_does_not_lose_quartz():
    # This is the failure as it was reported: the search run before the zero
    # error step.  0.06 deg on a 0.09 deg peak is two thirds of a width.
    for shift in (-0.06, +0.06):
        found = screen_phases(
            a_measurement([quartz()], [1.0], shift=shift), DATABASE, a_clay_library(),
            two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, min_share=0.001, max_phases=3,
        )
        assert found, f"nothing found at shift {shift}"
        assert found[0].name == "Quartz", f"shift {shift} gave {[e.name for e in found]}"


def test_the_shared_zero_offset_is_fitted_and_reported():
    # A zero error moves every reflection of every phase by the same amount, so
    # it is fitted once and shared.  Reporting it matters: a large one says the
    # zero error is worth correcting in its own step, where it is measured
    # against a reference rather than inferred from the whole pattern.
    for shift in (-0.06, +0.06):
        found = screen_phases(
            a_measurement([quartz()], [1.0], shift=shift), DATABASE, a_clay_library(),
            two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, min_share=0.001,
            max_phases=3, align=0.15,
        )
        assert found[0].name == "Quartz"
        # The offset is the correction the screen applied to the measured
        # angles, so it comes back equal to the error in them.
        assert found[0].position_offset == pytest.approx(shift, abs=0.03)


def test_the_shared_offset_is_what_rescues_an_uncorrected_zero_error():
    uncorrected = a_measurement([quartz()], [1.0], shift=-0.06)
    with_offset = screen_phases(
        uncorrected, DATABASE, a_clay_library(), two_theta_range=(4.0, 40.0),
        instrument=INSTRUMENT, min_share=0.001, max_phases=3, align=0.15,
    )
    without = screen_phases(
        uncorrected, DATABASE, a_clay_library(), two_theta_range=(4.0, 40.0),
        instrument=INSTRUMENT, min_share=0.001, max_phases=3, align=0.0,
    )
    assert with_offset[0].score > 1.4 * without[0].score


def stretched_quartz(scale=1.012):
    """Quartz with every cell edge ``scale`` times larger."""
    cell = quartz()
    return Crystal(
        a=cell.a * scale, b=cell.b * scale, c=cell.c * scale,
        alpha=cell.alpha, beta=cell.beta, gamma=cell.gamma,
        sites=list(cell.sites), name="Quartz (stretched)",
    )


def test_a_cell_that_differs_from_the_database_is_still_matched():
    # This is what the unit cell allowance is for, and until now it reached the
    # position-matching search only: the competitive screen fixed every
    # candidate at its published cell.
    measurement = a_measurement([stretched_quartz()], [1.0])
    found = screen_phases(
        measurement, DATABASE, a_clay_library(), two_theta_range=(4.0, 40.0),
        instrument=INSTRUMENT, min_share=0.001, max_phases=3,
        cell_allowance=0.02, align=0.0,
    )
    assert found and found[0].name == "Quartz"
    assert found[0].cell_scale == pytest.approx(1.012, abs=0.006)


def test_the_cell_allowance_reaches_the_competitive_screen():
    measurement = a_measurement([stretched_quartz()], [1.0])
    loose = screen_phases(
        measurement, DATABASE, a_clay_library(), two_theta_range=(4.0, 40.0),
        instrument=INSTRUMENT, min_share=0.001, max_phases=3,
        cell_allowance=0.02, align=0.0,
    )
    tight = screen_phases(
        measurement, DATABASE, a_clay_library(), two_theta_range=(4.0, 40.0),
        instrument=INSTRUMENT, min_share=0.001, max_phases=3,
        cell_allowance=0.0, align=0.0,
    )
    assert loose[0].score > 1.3 * tight[0].score


def test_two_real_phases_are_both_found_and_ranked_by_what_they_explain():
    found = screen_phases(
        a_measurement([quartz(), rutile()], [1.0, 0.35]), DATABASE, a_clay_library(),
        two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, min_share=0.001, max_phases=3,
    )
    names = [evidence.name for evidence in found]
    assert "Quartz" in names and "Rutile" in names
    assert names.index("Quartz") < names.index("Rutile")


def test_the_scores_never_rise_down_the_list():
    found = screen_phases(
        a_measurement([quartz(), rutile()], [1.0, 0.35], noise=30.0), DATABASE,
        a_clay_library(), two_theta_range=(4.0, 40.0), instrument=INSTRUMENT,
        min_share=0.0005, max_phases=3,
    )
    scores = [evidence.score for evidence in found]
    assert scores == sorted(scores, reverse=True)


def test_an_empty_database_gives_nothing_rather_than_failing():
    assert screen_phases(a_measurement([quartz()], [1.0]), {}, a_clay_library(),
                         instrument=INSTRUMENT) == []


def test_a_phase_the_pattern_contradicts_is_not_considered_at_all():
    # min_agreement is a floor on entry, not a weight on the gain.  Weighting by
    # the agreement was tried and is wrong: it is measured against the pattern as
    # it stands, so a phase whose lines sit in an empty stretch scores higher
    # than one whose lines stand on clay peaks, and it demoted albite from second
    # to sixteenth on a real separate.  As a floor it removes only the flat
    # contradiction.
    found = screen_phases(
        a_measurement([quartz()], [1.0]), DATABASE, a_clay_library(),
        two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, min_share=0.0001,
        max_phases=3, min_agreement=0.5,
    )
    assert "Graphite" not in [evidence.name for evidence in found]


def test_restricting_the_search_finds_a_mineral_the_open_search_loses():
    """The rutile case, from a real clay separate.

    Rutile's three reflections in range are all present, at signal-to-noise 19,
    15 and 4, and their measured heights stand at 100 : 47 against a calculated
    100 : 44.  An unrestricted search over two hundred candidates does not
    select it, because phases the specimen cannot contain fit the same peaks and
    are taken first, having twelve lines to spread a claim over against rutile's
    three.  That is not a threshold to tune: on one scan the data does not
    distinguish them, and naming the phases the specimen can contain supplies
    what the data does not.
    """
    crowd = dict(DATABASE)
    measurement = a_measurement([quartz(), rutile()], [1.0, 0.25])
    restricted = screen_phases(
        measurement, crowd, a_clay_library(), two_theta_range=(4.0, 40.0),
        instrument=INSTRUMENT, min_share=0.0005, max_phases=4,
        only={"Quartz", "Rutile"},
    )
    names = [evidence.name for evidence in restricted]
    assert names == ["Quartz", "Rutile"]


def test_a_restriction_that_names_nothing_in_the_database_gives_nothing():
    found = screen_phases(
        a_measurement([quartz()], [1.0]), DATABASE, a_clay_library(),
        two_theta_range=(4.0, 40.0), instrument=INSTRUMENT, only={"Nepheline"},
    )
    assert found == []
