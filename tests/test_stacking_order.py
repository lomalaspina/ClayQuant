"""The stacking-order axis: what it spans, and that it survives a round trip.

Ordering is the one axis that is *not* a free cross-product with composition,
because the two interstratified series do not order alike and neither orders at
every composition.  These tests pin that down, since the whole value of the
axis is that it adds the stacks nature makes and not the ones it does not.
"""
import numpy as np
import pytest

from clayquant.library import (
    ILLITE_SMECTITE_ORDERING_ONSET,
    ORDERING_DEGREES,
    LibraryEntry,
    PatternLibrary,
    ordered_transition,
    orderings_for,
)
from clayquant.mixed_layer import MixedLayerStack, lognormal_csds
from clayquant.models import CIF_SOURCES, eg_smectite_layer, load_crystal
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, basal_pattern
from clayquant.profile import PeakShape


def test_illite_smectite_orders_only_once_it_has_illitised():
    """Below the onset the series is random; above it both stacks exist."""
    for fraction in (0.20, 0.40, 0.50):
        assert orderings_for("I/S", fraction) == (0.0,), fraction
    for fraction in (ILLITE_SMECTITE_ORDERING_ONSET, 0.65, 0.90, 0.99):
        assert orderings_for("I/S", fraction) == ORDERING_DEGREES, fraction


def test_chlorite_smectite_is_random_only():
    """Its ordered member is corrensite, a phase of its own, not a degree here."""
    for fraction in (0.85, 0.90, 0.95):
        assert orderings_for("C/S", fraction) == (0.0,)


def test_an_end_member_has_no_junctions_to_order():
    assert orderings_for("I/S", 1.0) == (0.0,)
    assert orderings_for("I/S", 0.0) == (0.0,)
    assert ordered_transition(1.0, 1.0) is None
    assert ordered_transition(0.0, 1.0) is None


def test_random_is_no_transition_at_all():
    """So the stack builds its own and the two paths cannot drift apart."""
    assert ordered_transition(0.7, 0.0) is None


def test_maximum_ordering_is_the_reichweite_one_ideal():
    """Every guest layer followed by a host layer, which is what R1 means."""
    for fraction in (0.60, 0.70, 0.85, 0.95):
        transition = ordered_transition(fraction, 1.0)
        assert transition[1][0] == pytest.approx(1.0), fraction
        # and stationarity still holds
        assert fraction * transition[0][1] == pytest.approx(
            (1.0 - fraction) * transition[1][0])


def _pattern(fraction, ordering):
    instrument = Instrument(
        peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0),
        divergence=Divergence(specimen_length=25.0, goniometer_radius=240.0,
                              divergence=0.5, shape="round", beam_width=14.0),
    )
    grid = np.arange(2.0, 32.0, 0.02)
    host = load_crystal("illite")
    stack = MixedLayerStack(
        host.layer_model(layers_per_cell=CIF_SOURCES["illite"].layers_per_cell),
        eg_smectite_layer(),
        fraction_a=fraction,
        transition=ordered_transition(fraction, ordering),
        csds=lognormal_csds(15.0, 0.35),
    )
    return grid, np.asarray(basal_pattern(stack, grid, instrument,
                                          r_march_dollase=1.0).intensity)


def test_ordering_puts_a_superlattice_reflection_where_random_has_none():
    """The R1 signature, and the reason the axis is worth its entries.

    An ordered illite/smectite repeats on the *pair* of layers, so it reflects
    at their sum where the random stack reflects at neither.  Measured here at
    70/30, where a first-order chain has the most room to order.
    """
    grid, random = _pattern(0.70, 0.0)
    _, ordered = _pattern(0.70, 1.0)

    def maxima(y):
        """Local maxima between 2.4 and 4 deg, i.e. about 22 to 37 A.

        Amplitude alone will not do: both patterns climb steeply towards the
        low-angle limit, so the window is bright either way.  What the ordering
        adds is a *maximum* where the random stack only has a shoulder.
        """
        inside = (grid > 2.4) & (grid < 4.0)
        peak = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
        return [float(grid[1:-1][i]) for i in np.where(peak & inside[1:-1])[0]]

    assert maxima(random) == []
    found = maxima(ordered)
    assert len(found) == 1, found
    spacing = 1.5406 / (2.0 * np.sin(np.radians(found[0] / 2.0)))
    # The pair of layers, near the 10.0 + 16.9 A sum that defines R1 ordering.
    assert 25.0 < spacing < 35.0, spacing


def test_an_illite_rich_stack_has_little_room_left_to_order():
    """Which is why R2 and R3 exist, and is stated rather than hidden."""
    _, random = _pattern(0.85, 0.0)
    _, ordered = _pattern(0.85, 1.0)
    difference = np.abs(ordered / ordered.max() - random / random.max()).max()
    assert difference < 0.15


def test_the_ordering_survives_a_save_and_a_load(tmp_path):
    grid = np.linspace(2.0, 30.0, 64)
    library = PatternLibrary(two_theta=grid, entries=[])
    from clayquant.pattern import Pattern
    for ordering in (0.0, 1.0):
        library.add(Pattern(grid, np.linspace(1.0, 2.0, 64), name=f"I/S o={ordering}"),
                    phase="I/S", fraction=0.7, ordering=ordering)
    path = tmp_path / "ordered.npz"
    library.save(path)
    back = PatternLibrary.load(path)
    assert [entry.ordering for entry in back.entries] == [0.0, 1.0]


def test_a_library_from_before_the_axis_loads_as_random(tmp_path):
    """Back compatibility: no column means random stacks, not a crash."""
    grid = np.linspace(2.0, 30.0, 64)
    library = PatternLibrary(two_theta=grid, entries=[])
    from clayquant.pattern import Pattern
    library.add(Pattern(grid, np.linspace(1.0, 2.0, 64), name="I/S"),
                phase="I/S", fraction=0.7)
    path = tmp_path / "old.npz"
    library.save(path)
    with np.load(path, allow_pickle=True) as data:
        kept = {key: data[key] for key in data.files if key != "ordering"}
    np.savez_compressed(path, **kept)
    back = PatternLibrary.load(path)
    assert back.entries[0].ordering == 0.0


def test_the_entry_default_is_random():
    assert LibraryEntry(name="x", phase="I/S", intensity=np.zeros(4)).ordering == 0.0
