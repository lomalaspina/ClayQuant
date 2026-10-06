"""Which kaolinite bound to apply, and when not to apply one at all.

Both were found by running the nine standards through the whole workflow.  The
restraint is strong - weight 50 against one pattern - so a bound that is wrong
or that says nothing is not a neutral thing to pass it.
"""
import numpy as np
import pytest

from clayquant.diagnostics import BOUND_WIDTH_LIMIT, KaoliniteEvidence
from clayquant.pattern import Pattern
from clayquant.treatment import WINDOW_REFLECTION_SHARE, kaolinite_share_constraint


class _Route:
    """A collapse route with the one property best_bounds reads."""

    def __init__(self, bounds):
        self.kaolinite_bounds = bounds


class _Evidence(KaoliniteEvidence):
    """KaoliniteEvidence with its two inputs stubbed, so the choice is testable."""

    def __init__(self, envelope, routes):
        self._envelope = envelope
        self._routes = tuple(_Route(b) for b in routes)

    @property
    def bounds(self):
        return self._envelope

    @property
    def collapse_routes(self):
        return self._routes


def test_the_collapse_route_is_taken_when_it_is_tighter():
    """Which it usually is, being a measurement rather than a span of several."""
    evidence = _Evidence((0.2, 0.9), [(0.59, 0.82)])
    assert evidence.best_bounds == (0.59, 0.82)


def test_the_envelope_is_kept_when_the_route_is_wider():
    """The dickite standard: the route says 25-67 % of a mineral that is all of it.

    Applying the route there, at the weight this restraint carries, drove R_wp
    to 582 per cent and halved the kaolinite.  Taking the narrower returns the
    envelope's 100-100 and 99 per cent kaolinite.
    """
    evidence = _Evidence((1.0, 1.0), [(0.25, 0.67)])
    assert evidence.best_bounds == (1.0, 1.0)


def test_a_route_that_measured_nothing_is_passed_over():
    evidence = _Evidence((0.3, 0.6), [(float("nan"), float("nan")), (0.4, 0.5)])
    assert evidence.best_bounds == (0.4, 0.5)


def test_the_narrowest_of_several_routes_wins():
    evidence = _Evidence((0.0, 1.0), [(0.2, 0.8), (0.45, 0.55)])
    assert evidence.best_bounds == (0.45, 0.55)


def test_a_bound_spanning_half_the_range_says_nothing():
    """The montmorillonite case: no reflection in the window, so 0 to 1."""
    assert not _Evidence((0.0, 1.0), []).informative
    assert not _Evidence((0.2, 0.9), []).informative
    assert _Evidence((0.59, 0.82), []).informative
    assert _Evidence((1.0, 1.0), []).informative


def test_the_width_limit_is_half_the_range():
    assert BOUND_WIDTH_LIMIT == pytest.approx(0.5)
    assert _Evidence((0.0, BOUND_WIDTH_LIMIT - 0.01), []).informative
    assert not _Evidence((0.0, BOUND_WIDTH_LIMIT + 0.01), []).informative


def test_bounds_that_could_not_be_measured_are_not_informative():
    assert not _Evidence((float("nan"), float("nan")), []).informative


# --------------------------------------------------------------------------- #
# the window has to hold a reflection
# --------------------------------------------------------------------------- #

class _Library:
    def __init__(self, grid, entries):
        self.two_theta = grid
        self.entries = entries
        self.names = [e.name for e in entries]


class _Entry:
    def __init__(self, name, phase, intensity):
        self.name, self.phase, self.intensity = name, phase, intensity
        self.air_intensity = None


def _library(grid):
    """One kaolinite and one chlorite entry, both with a line in the window."""
    line = np.exp(-0.5 * ((grid - 12.4) / 0.1) ** 2)
    return _Library(grid, [_Entry("kaolinite_1M PO=0.3", "kaolinite_1M", line),
                           _Entry("chlorite PO=0.3", "chlorite", line)])


def test_a_window_with_a_reflection_gives_a_constraint():
    grid = np.arange(4.0, 34.0, 0.02)
    tall = 1000.0 * np.exp(-0.5 * ((grid - 12.4) / 0.1) ** 2)
    block = kaolinite_share_constraint(
        Pattern(grid, tall, name="m"), _library(grid), (0.59, 0.82), weight=50.0)
    assert block is not None


def test_a_window_holding_only_noise_gives_none():
    """On a montmorillonite it carries 80 counts against a strongest line of
    ten thousand, and the share measured from it is noise over noise."""
    grid = np.arange(4.0, 34.0, 0.02)
    elsewhere = 10000.0 * np.exp(-0.5 * ((grid - 8.8) / 0.1) ** 2)
    trace = 80.0 * np.exp(-0.5 * ((grid - 12.4) / 0.1) ** 2)
    block = kaolinite_share_constraint(
        Pattern(grid, elsewhere + trace, name="m"), _library(grid), (0.0, 1.0), weight=50.0)
    assert block is None


def test_the_threshold_sits_where_the_standards_put_it():
    """100 per cent on the kaolinites and chlorites, 0.8 and 5.6 on the others."""
    assert WINDOW_REFLECTION_SHARE == pytest.approx(0.10)
    grid = np.arange(4.0, 34.0, 0.02)
    tallest = 10000.0 * np.exp(-0.5 * ((grid - 8.8) / 0.1) ** 2)
    for share, wanted in ((0.056, None), (0.008, None), (1.0, "block"), (0.5, "block")):
        window = share * 10000.0 * np.exp(-0.5 * ((grid - 12.4) / 0.1) ** 2)
        block = kaolinite_share_constraint(
            Pattern(grid, tallest + window, name="m"), _library(grid),
            (0.59, 0.82), weight=50.0)
        assert (block is None) == (wanted is None), share


def test_a_narrow_bound_is_not_the_same_as_a_sound_one():
    """The two guards are independent and neither subsumes the other.

    On the montmorillonite standard the envelope says 0 to 1 while a collapse
    route measured from the same empty window says 1 to 1.  The narrower of
    those is narrow, so the width test passes it; what rejects it is the window
    test, which sees the pattern the evidence does not.
    """
    evidence = _Evidence((0.0, 1.0), [(1.0, 1.0)])
    assert evidence.best_bounds == (1.0, 1.0)
    assert evidence.informative           # the bound is narrow ...

    grid = np.arange(4.0, 34.0, 0.02)
    tallest = 10000.0 * np.exp(-0.5 * ((grid - 8.8) / 0.1) ** 2)
    trace = 80.0 * np.exp(-0.5 * ((grid - 12.4) / 0.1) ** 2)
    block = kaolinite_share_constraint(
        Pattern(grid, tallest + trace, name="m"), _library(grid),
        evidence.best_bounds, weight=50.0)
    assert block is None                  # ... and the window rejects it anyway
