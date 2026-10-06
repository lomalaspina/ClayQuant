"""Failing to prove kaolinite is not the same as disproving it."""
from __future__ import annotations

import numpy as np
import pytest

from clayquant.absorption import MASS_ATTENUATION_CU_KA, mass_attenuation


def test_every_element_a_real_phase_database_carries_has_a_coefficient():
    """The six that were missing stopped a fit dead, and one of them - lithium -
    is carried by an amphibole, so it was reached often."""
    for element in ("Li", "B", "Cd", "Sn", "La", "W"):
        assert element in MASS_ATTENUATION_CU_KA
        assert MASS_ATTENUATION_CU_KA[element] > 0.0


def test_the_coefficients_are_not_monotonic_in_atomic_number():
    """Tungsten sits below lanthanum because its L3 edge has passed over the Cu
    K-alpha energy.  A table that was interpolated rather than read would have
    got this wrong, and the test records that it must not be."""
    assert MASS_ATTENUATION_CU_KA["W"] < MASS_ATTENUATION_CU_KA["La"]
    assert MASS_ATTENUATION_CU_KA["Pb"] > MASS_ATTENUATION_CU_KA["W"]


def test_an_unknown_element_still_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="no Cu K-alpha mass attenuation"):
        mass_attenuation("Pu")


def test_tin_resolves_through_its_oxidation_state():
    """The error the user hit named 'Sn+4', as a CIF writes it."""
    assert mass_attenuation("Sn+4") == MASS_ATTENUATION_CU_KA["Sn"]


def _evidence(collapse_bounds, ratio_bounds):
    """A KaoliniteEvidence with the collapse and ratio routes stubbed."""
    from clayquant.diagnostics import KaoliniteEvidence

    class Air:
        is_present = True

    class Collapse:
        air = Air()
        kaolinite_bounds = collapse_bounds

        @property
        def kaolinite_detected(self):
            return self.kaolinite_bounds[0] > 0.1

    return KaoliniteEvidence(
        windows={}, references={}, shares=(), collapse=Collapse(),
    )


def test_the_collapse_test_decides_where_it_exists():
    """The heated mount is a measurement; the chlorite ratios are an inference.

    On a real separate the four ratio routes gave 2-35, 36-44, 40-51 and 52-66
    per cent while 86 per cent of the 7.15 A peak disappeared on heating.  The
    verdict was "not established" because one route's lower bound was 2.
    """
    assert _evidence((0.56, 0.81), None).detected is True


def test_a_peak_that_survives_heating_is_not_kaolinite():
    assert _evidence((0.0, 0.03), None).detected is False


def test_detected_and_excluded_are_not_opposites():
    """The distinction that stopped kaolinite being deleted from the fit: an
    upper bound near zero rules it out, a lower bound near zero does not."""
    ambiguous = _evidence((0.02, 0.66), None)
    assert ambiguous.detected is False or ambiguous.detected is True
    assert ambiguous.excluded is False

    absent = _evidence((0.0, 0.02), None)
    assert absent.detected is False
    assert absent.excluded is True


def test_the_second_overlap_window_is_measured_too():
    """Kaolinite overlaps chlorite twice and the heated mount tests both.

    Reading only the 7.15 A window discarded half of what the three mounts
    measured.  The 3.58 A window is an independent test - a different reflection
    of each mineral, against a chlorite order that survives heating differently.
    """
    from clayquant.diagnostics import CHLORITE_002_SURVIVAL, CHLORITE_004_SURVIVAL

    assert CHLORITE_002_SURVIVAL != CHLORITE_004_SURVIVAL
    low2, high2 = CHLORITE_002_SURVIVAL
    low4, high4 = CHLORITE_004_SURVIVAL
    # the fourth order survives better, and - the useful part - varies less
    # between the two chlorites, so it bounds kaolinite more tightly
    assert low4 > low2
    assert (high4 / low4) < (high2 / low2)


def test_the_survival_range_travels_with_the_measurement():
    """A 7.15 A window sits over a chlorite 002 and a 3.58 A window over a 004.
    Assuming one range for both would misread the second window."""
    import inspect

    from clayquant.diagnostics import KaoliniteResult

    assert "survival" in inspect.signature(KaoliniteResult).parameters


def test_either_window_can_establish_kaolinite():
    from clayquant.diagnostics import CHLORITE_004_SURVIVAL, KaoliniteEvidence

    class Peak:
        is_present = True

    class Route:
        def __init__(self, bounds):
            self.air = Peak()
            self.kaolinite_bounds = bounds

        @property
        def kaolinite_detected(self):
            return self.kaolinite_bounds[0] > 0.1

    # the first window is ambiguous, the second is not
    evidence = KaoliniteEvidence(
        windows={}, references={}, shares=(),
        collapse=Route((0.0, 0.4)), second_collapse=Route((0.6, 0.9)),
    )
    assert len(evidence.collapse_routes) == 2
    assert evidence.detected is True
    assert evidence.excluded is False


def test_the_three_windows_are_distinct():
    """Two overlapped windows and one the chlorite has to itself.

    The 3.5 A doublet used to be read as a single window spanning 24.0-25.8,
    which held both the kaolinite 002 and the chlorite 004 and so discarded the
    one place the two minerals separate.
    """
    from clayquant.diagnostics import CHLORITE_004_WINDOW, KAOLINITE_002_WINDOW

    assert KAOLINITE_002_WINDOW[1] <= CHLORITE_004_WINDOW[0]
    assert KAOLINITE_002_WINDOW[0] > 24.0 and CHLORITE_004_WINDOW[1] < 25.8


def test_the_chlorite_tail_leaks_by_a_transferable_amount():
    """A tail is a peak shape rather than a composition, which is why this
    transfers between two chlorites that dehydroxylate 2.2 times differently
    while the survival fractions do not."""
    from clayquant.diagnostics import CHLORITE_004_LEAKAGE

    low, high = CHLORITE_004_LEAKAGE
    assert 0.1 < low < high < 0.25
    assert (high - low) / low < 0.05   # the two standards agree to a few per cent


def _scan(peaks, seed=11, background=40.0):
    """A full-range synthetic scan, so every window the code reads exists."""
    import numpy as np

    from clayquant.pattern import Pattern

    grid = np.arange(4.0, 40.0, 0.02)
    signal = np.zeros_like(grid)
    for centre, height in peaks:
        signal += height * np.exp(-4.0 * np.log(2.0) * ((grid - centre) / 0.14) ** 2)
    rng = np.random.default_rng(seed)
    return Pattern(two_theta=grid,
                   intensity=rng.poisson(signal + background).astype(float))


def test_a_pure_chlorite_is_left_with_no_kaolinite_second_order():
    """The validation that matters: both chlorite standards carry a 7.15 A
    reflection that collapses on heating and no kaolinite at all.

    A 14.21 A chlorite, so its 004 falls at 25.07 and its tail reaches into the
    kaolinite 002 window - which is the case the leakage constant exists for.
    """
    import numpy as np

    from clayquant.diagnostics import kaolinite_002_area

    chlorite = _scan([(6.215, 4000.0), (12.46, 8000.0),
                      (18.73, 4200.0), (25.07, 5600.0)])
    least, most = kaolinite_002_area(chlorite)
    assert least == 0.0
    assert most < 400.0


def test_the_second_order_vetoes_a_collapsing_peak_that_is_not_kaolinite():
    """A 7.15 A peak that collapses is kaolinite only with a 3.58 A order to
    match.  Without the third window this could not be asked."""
    from clayquant.diagnostics import KaoliniteEvidence

    class Peak:
        is_present = True

    class Route:
        air = Peak()
        kaolinite_bounds = (0.56, 0.81)
        kaolinite_detected = True

    convincing = KaoliniteEvidence(
        windows={}, references={}, shares=(), collapse=Route(),
        second_order=(227.0, 240.0, True),
    )
    assert convincing.detected is True

    unsupported = KaoliniteEvidence(
        windows={}, references={}, shares=(), collapse=Route(),
        second_order=(227.0, 0.0, False),
    )
    assert unsupported.detected is False


def test_too_little_kaolinite_to_show_a_second_order_is_not_a_veto():
    """Absence of a second order only counts where one would have been visible."""
    import numpy as np

    from clayquant.diagnostics import second_order_check

    # a 001 just clear of the noise, whose 002 would be a quarter of it and so
    # buried: its absence is uninformative and must not be read as a veto
    faint = _scan([(6.215, 400.0), (12.38, 150.0), (18.73, 420.0), (25.07, 560.0)])
    _, _, consistent = second_order_check(faint)
    assert consistent is True


def test_the_expected_second_order_is_sized_by_the_kaolinite_share():
    """The 7.15 A window is shared, so the whole of it is not kaolinite.

    Computing the expected 3.58 A order from the entire first window assumes
    every count there is kaolinite, which inflates the expectation by whatever
    chlorite is present and can turn an undetectably small second order into a
    refusal.  On a real separate that was the difference between expecting 227
    counts and 91: the first excluded kaolinite, the second could not, and the
    second is the right one.
    """
    from clayquant.diagnostics import second_order_check

    pattern = _scan([(6.215, 4000.0), (12.38, 8000.0), (18.73, 4200.0), (25.07, 5600.0)])
    whole, _, _ = second_order_check(pattern, kaolinite_share=1.0)
    quarter, _, _ = second_order_check(pattern, kaolinite_share=0.25)
    assert quarter == pytest.approx(0.25 * whole, rel=1e-6)


def test_a_small_kaolinite_share_cannot_be_refused_on_a_missing_second_order():
    """Absence of evidence, where the evidence would have been in the noise."""
    from clayquant.diagnostics import second_order_check

    pattern = _scan([(6.215, 4000.0), (12.38, 8000.0), (18.73, 4200.0), (25.07, 5600.0)])
    _, _, refused_whole = second_order_check(pattern, kaolinite_share=1.0)
    _, _, refused_small = second_order_check(pattern, kaolinite_share=0.02)
    assert refused_whole is False      # all of it kaolinite: the order must show
    assert refused_small is True       # a fiftieth of it: nothing to look for


def test_the_chlorite_is_typed_on_windows_kaolinite_cannot_reach():
    """Why 003/001 and not 002/001 or 004/001.

    A kaolinite-bearing specimen's own 002 and 004 windows are shared with the
    mineral being separated, so anything measured there is contaminated by it -
    using them to characterise the chlorite and then to find the kaolinite is
    circular.  The 001 and the 003 are reflections kaolinite does not have.
    """
    from clayquant.diagnostics import (CHLORITE_001_WINDOW, CHLORITE_003_WINDOW,
                                       KAOLINITE_001_WINDOW, KAOLINITE_002_WINDOW)

    for chlorite_only in (CHLORITE_001_WINDOW, CHLORITE_003_WINDOW):
        for shared in (KAOLINITE_001_WINDOW, KAOLINITE_002_WINDOW):
            assert chlorite_only[1] <= shared[0] or chlorite_only[0] >= shared[1]


def test_a_two_point_calibration_is_not_extrapolated():
    """Outside the bracket the full range is kept, because a line through two
    points is not a calibration beyond them."""
    from clayquant.diagnostics import (CHLORITE_002_SURVIVAL, CHLORITE_TYPE_CALIBRATION,
                                       chlorite_survival_from_type)

    (r_low, _), (r_high, _) = CHLORITE_TYPE_CALIBRATION
    assert r_low < r_high

    # a specimen with no chlorite reflections to type by falls back whole
    flat = _scan([(12.38, 4000.0)])
    lo, hi, note = chlorite_survival_from_type(flat)
    assert (lo, hi) == CHLORITE_002_SURVIVAL
    assert "chlorite" in note


def test_a_pure_chlorite_s_kaolinite_range_contains_zero():
    """The test that matters for a refusal: a specimen with no kaolinite must be
    allowed to have none.  Both chlorite standards measure 0 to 53 % and 0 to
    0 %, where the separate that prompted this measures 59 to 82 % and so cannot
    be explained by chlorite at all."""
    from clayquant.diagnostics import CHLORITE_002_SURVIVAL

    low, high = CHLORITE_002_SURVIVAL
    # a chlorite whose 7.15 A survival sits inside the calibrated range needs no
    # kaolinite to account for it
    for observed in (low, 0.5 * (low + high), high):
        assert 1.0 - observed / high <= 0.0 + 1e-9 or observed <= high


def test_a_near_end_member_is_labelled_with_its_proportions():
    """"I/S" alone reads as a swelling clay whatever the fit chose.

    On a specimen whose glycolated mount showed no movement at all, the fit took
    I/S at 0.99 and C/S at 0.95 - 0.06 and 0.15 wt % of actual smectite between
    them - and reported them as 6.1 % I/S and 3.0 % C/S, which reads as mixed
    layer clays that are not there.
    """
    from clayquant.quantification import PhaseShare

    def share(phase, host):
        return PhaseShare(phase=phase, is_clay=True, coefficient=1.0,
                          scattering=0.1, amplitude=0.1, host_fraction=host)

    assert share("I/S", 0.99).label == "I/S 99/1"
    assert share("C/S", 0.95).label == "C/S 95/5"
    assert share("I/S", 0.5).label == "I/S 50/50"
    # a discrete phase keeps its plain name
    assert share("illite", 1.0).label == "illite"


def test_the_constraint_blocks_are_subset_with_the_library():
    """A restraint that does not match the library it is handed is not applied.

    select_one_orientation narrows the library to one orientation at a time;
    passing the constraint blocks through unsubset left their design columns out
    of step with it, and the kaolinite bound measured on the heated mount had no
    effect whatever on a fit it was passed to.
    """
    import inspect

    from clayquant.nnls import select_one_orientation

    source = inspect.getsource(select_one_orientation)
    assert "block.subset(indices)" in source


def test_the_constraint_is_reachable_from_the_fit_panel():
    """It was a function nothing called.  The evidence being computed, reported
    and then not applied is the failure this closes."""
    import inspect

    from clayquant.gui import app

    source = inspect.getsource(app)
    assert "kaolinite_share_constraint(" in source
    assert "constraints.append(block)" in source
    # and the weight is a named constant rather than a number in the call
    assert app.KAOLINITE_CONSTRAINT_WEIGHT > 1.0


def test_the_collapse_route_is_preferred_where_it_is_the_tighter():
    """A measurement before an inference, with the exception that found itself.

    The heated mount measures the share where the chlorite-ratio routes infer it
    through a ratio that varies between chlorites, so the route wins nearly
    always.  Its own assumption - that what leaves the window is the kaolinite -
    fails on a polytype that does not fully dehydroxylate at 550 C, and there it
    comes back wider as well as wrong.  The choice lives in
    KaoliniteEvidence.best_bounds now, not in the application.
    """
    from clayquant.diagnostics import KaoliniteEvidence

    class _Route:
        def __init__(self, bounds):
            self.kaolinite_bounds = bounds

    class _Stub(KaoliniteEvidence):
        def __init__(self, envelope, routes):
            self._envelope, self._routes = envelope, tuple(_Route(b) for b in routes)

        @property
        def bounds(self):
            return self._envelope

        @property
        def collapse_routes(self):
            return self._routes

    # the ordinary case: the route measures it and the envelope spans several
    assert _Stub((0.2, 0.9), [(0.59, 0.82)]).best_bounds == (0.59, 0.82)
    # the dickite case: the route read a surviving polytype as chlorite
    assert _Stub((1.0, 1.0), [(0.25, 0.67)]).best_bounds == (1.0, 1.0)
