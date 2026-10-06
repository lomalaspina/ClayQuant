"""A restraint must still name the right entries after the library is cut down.

`screen_diagnostic_peaks` subsets the constraints it is given by *position* into
the library it is given.  So a constraint built against the whole library, passed
beside a one-entry-per-family subset, is cut to the first N columns of the whole
library rather than to the N entries actually in the fit - and then restrains
whichever entries happen to sit in those positions.

It is quiet twice over.  The sizes match, so nothing raises; and the mis-cut
constraint is only *used* when the screen drops an entry and refits, which is
precisely the case where the restraint still has work to do.

Found while chasing why the kaolinite share constraint appeared to saturate on
real samples.  That turned out to be a driver that never passed the constraints
to the screen at all - the fit satisfied the heated mount's bound exactly and the
screening step then refitted without it, at 19 points better R_wp and in
contradiction with the measurement (Sec. A.72).
"""
import numpy as np
import pytest

from clayquant.library import PatternLibrary
from clayquant.nnls import _library_subset, nnls_fit
from clayquant.pattern import Pattern
from clayquant.treatment import ExtraObservation

GRID = np.arange(4.0, 20.0, 0.05)


def a_peak(centre, height=1.0, width=0.25):
    return height * np.exp(-0.5 * ((GRID - centre) / width) ** 2)


@pytest.fixture
def library():
    lib = PatternLibrary(two_theta=GRID)
    for name, centre in (("alpha", 6.0), ("beta", 9.0), ("gamma", 12.0),
                         ("delta", 15.0)):
        lib.add(Pattern(two_theta=GRID, intensity=a_peak(centre), name=name),
                phase=name, strip_continuum=False)
    return lib


def a_block(n_entries, column, target=1.0, weight=10.0):
    """One row saying: the entry at `column` must contribute `target`."""
    design = np.zeros((1, n_entries))
    design[0, column] = 1.0
    return ExtraObservation(target=np.array([target]), design=design,
                            weight=weight, name=f"entry {column}")


def test_subset_keeps_the_column_with_its_entry(library):
    """The property the app relies on: subset by the same indices as the library."""
    block = a_block(len(library.entries), column=2)
    picked = np.array([1, 2, 3])
    cut = block.subset(picked)
    assert cut.design.shape == (1, 3)
    # gamma was column 2 of four; among entries 1,2,3 it is now column 1
    assert cut.design[0, 1] == 1.0
    assert cut.design.sum() == 1.0


def test_cutting_by_position_instead_restrains_the_wrong_entry(library):
    """What the bug did: arange(len(subset)) rather than the chosen indices."""
    block = a_block(len(library.entries), column=3)       # delta
    picked = np.array([1, 2, 3])                           # beta, gamma, delta
    wrong = block.subset(np.arange(len(picked)))           # columns 0,1,2
    assert wrong.design.sum() == 0.0, (
        "the restraint on delta has vanished entirely - it named a column that "
        "the mis-cut does not include, so the fit is restrained by nothing")
    right = block.subset(picked)
    assert right.design[0, 2] == 1.0


def test_a_mis_cut_constraint_changes_the_answer(library):
    """Not a bookkeeping nicety: the fitted coefficients differ."""
    measured = Pattern(two_theta=GRID, name="m",
                       intensity=600.0 + 3000.0 * a_peak(9.0) + 3000.0 * a_peak(15.0))
    picked = np.array([1, 2, 3])
    sub = _library_subset(library, picked)
    block = a_block(len(library.entries), column=3, target=2000.0, weight=50.0)

    right = nnls_fit(measured, sub, range_two_theta=(4.0, 20.0),
                     constraints=[block.subset(picked)])
    wrong = nnls_fit(measured, sub, range_two_theta=(4.0, 20.0),
                     constraints=[block.subset(np.arange(len(picked)))])
    delta_right = float(right.coefficients[2])
    delta_wrong = float(wrong.coefficients[2])
    assert delta_right != pytest.approx(delta_wrong, rel=1e-3)
    # the correctly cut one is pulled towards the target it names
    assert abs(delta_right - 2000.0) < abs(delta_wrong - 2000.0)


def test_the_size_check_does_not_catch_it(library):
    """Both cuts have the right number of columns, so nothing raises.

    This is why it had to be found by reasoning rather than by a crash.
    """
    block = a_block(len(library.entries), column=3)
    picked = np.array([1, 2, 3])
    assert block.subset(picked).n_entries == len(picked)
    assert block.subset(np.arange(len(picked))).n_entries == len(picked)
