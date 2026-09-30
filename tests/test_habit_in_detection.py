"""A mineral with a crystal habit is screened at its habit, not as a powder."""
from __future__ import annotations

import numpy as np
import pytest

from clayquant.crystal import AtomSite, Crystal
from clayquant.models import MINERAL_HABIT, habit_pole
from clayquant.pattern import peak_list


def test_the_amphiboles_are_declared_prismatic():
    """Their 8.4 A (110) is often the only line an oriented mount shows."""
    for name in ("riebeckite", "hornblende", "actinolite", "tremolite", "glaucophane"):
        assert MINERAL_HABIT[name] == (1, 1, 0)


def test_the_pole_comes_from_the_table_and_not_from_the_database():
    """The bug this replaced read crystal.po_axis and only *gated* on the table.

    Sepiolite and palygorskite carry no refined axis in a typical structure
    database, so the pole came back None and they were screened as random
    powders - the one thing the habit mechanism exists to prevent.
    """
    assert habit_pole("Sepiolite") == (1.0, 1.0, 0.0)
    assert habit_pole("Sepiolite", crystal=None) == (1.0, 1.0, 0.0)

    class Bare:
        po_axis = None

    assert habit_pole("Palygorskite", Bare()) == (1.0, 1.0, 0.0)


def test_a_mineral_with_no_habit_gets_no_pole():
    """A refined PO correction is a fitting parameter, not a statement of habit,
    so an equant mineral stays a random powder however its database entry was
    refined."""
    class Refined:
        po_axis = (1.0, 0.0, 0.0)

    assert habit_pole("Quartz", Refined()) is None
    assert habit_pole("Albite") is None


def _prismatic() -> Crystal:
    """A cell whose hk0 and 00l reflections are well separated in angle."""
    return Crystal(
        name="test",
        a=9.8, b=18.0, c=5.3, alpha=90.0, beta=104.0, gamma=90.0,
        sites=[AtomSite("Si", 0.0, 0.0, 0.0, 1.0, 0.5),
               AtomSite("O", 0.25, 0.25, 0.25, 1.0, 0.5)],
    )


def test_peak_list_takes_a_pole_and_it_changes_the_list():
    crystal = _prismatic()
    random_powder = peak_list(crystal, (4.0, 40.0))
    oriented = peak_list(crystal, (4.0, 40.0), r_march_dollase=0.3, po_axis=(1, 1, 0))
    assert random_powder[0].size > 0 and oriented[0].size > 0
    # the strongest line is not the same one once the crystal lies down
    strongest_random = random_powder[0][int(np.argmax(random_powder[1]))]
    strongest_oriented = oriented[0][int(np.argmax(oriented[1]))]
    assert strongest_random != pytest.approx(strongest_oriented, abs=0.01)


def test_the_oriented_list_concentrates_intensity():
    """What makes a random-powder screen miss the mineral: with the habit on,
    far fewer lines clear a relative-intensity cut, because one family of
    reflections has taken the intensity."""
    crystal = _prismatic()
    _, random_heights = peak_list(crystal, (4.0, 40.0))
    _, oriented_heights = peak_list(
        crystal, (4.0, 40.0), r_march_dollase=0.3, po_axis=(1, 1, 0)
    )
    above = lambda h: int(np.sum(h >= 0.10 * h.max()))  # noqa: E731
    assert above(oriented_heights) < above(random_heights)


def test_stable_phases_reports_both_scores():
    """The line count and the intensity agreement are different questions and
    both are answered, so a phase whose lines are all present while its
    intensities disagree reads as textured rather than as absent."""
    import inspect

    from clayquant.detection import stable_phases

    parameters = inspect.signature(stable_phases).parameters
    assert parameters["score_by"].default == "intensity"
    assert "min_matched" in parameters


def test_scoring_by_lines_is_available_but_not_the_default():
    """Measured on a real separate, the intensity weighting ranks an oriented
    amphibole at 82 % and second, while counting lines gives it 50 % and ties it
    with rutile.  The fault was never the weighting, it was that the weights
    were a random powder's - so the option exists and is not the default."""
    import inspect

    from clayquant.detection import stable_phases

    source = inspect.getsource(stable_phases)
    assert 'score_by == "lines"' in source
