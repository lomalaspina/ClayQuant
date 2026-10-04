"""An interstratified series must not reach its own host as an end member.

An I/S at a host fraction of 1.00 is illite; a C/S at 1.00 is chlorite.  Carrying
one mineral under two names does not add anything the library did not have - it
splits that mineral arbitrarily between two near-identical columns, and the split
is set by whichever is marginally cheaper rather than by the specimen.  Measured
cosines between the pairs were 0.963 for I/S 1.00 against illite and 0.980 for
C/S 1.00 against chlorite, and reporting the two together instead of dropping
one proved thirteen times less reproducible across orientations.

The width freedom those end members used to supply is not a reason to bring them
back.  It belongs to the discrete phases themselves, through
:data:`~clayquant.library.DISCRETE_THICKNESSES`.
"""

import pytest

from clayquant.library import (
    CHLORITE_SMECTITE_FRACTIONS,
    DISCRETE_THICKNESSES,
    ILLITE_SMECTITE_FRACTIONS,
)


@pytest.mark.parametrize(
    "fractions,series,host",
    [(ILLITE_SMECTITE_FRACTIONS, "I/S", "illite"),
     (CHLORITE_SMECTITE_FRACTIONS, "C/S", "chlorite")],
)
def test_the_series_stops_short_of_its_host(fractions, series, host):
    assert 1.0 not in fractions, (
        f"{series} at a host fraction of 1.00 is {host}, and carrying one mineral "
        f"under two names splits it between two columns that differ by nothing a "
        f"measurement can see"
    )
    assert max(fractions) < 1.0


@pytest.mark.parametrize("host", ["illite", "chlorite"])
def test_the_host_carries_its_own_width_axis_instead(host):
    """What the end members had been supplying, put where it belongs.

    Without this the only entries in the library wide enough to be a real illite
    are mixed-layer ones - the discrete illite spans 0.080 to 0.100 deg at its
    10 A reflection where a pure illite standard measures 0.418 - and the fit
    reports a pure illite as 99 % I/S because nothing else can be that wide.
    """
    assert host in DISCRETE_THICKNESSES, (
        f"{host} has no crystallite-size axis, so its basal width is fixed at "
        f"the instrumental value and a real specimen cannot be fitted by it"
    )
    sizes = DISCRETE_THICKNESSES[host]
    assert len(sizes) >= 2
    assert None in sizes, "the unbroadened entry must lead, so it keeps the plain name"
    finite = [s for s in sizes if s is not None]
    assert min(finite) <= 200.0, (
        "the axis must reach the ~190 A that a 0.418 deg basal reflection implies"
    )


def test_the_builder_applies_the_axis_without_being_asked():
    """The default has to be wired, not merely defined.

    A constant that no build path passes is a constant that does nothing, and the
    operator rebuilding from the GUI does not pass one.
    """
    import inspect

    from clayquant.library import DISCRETE_THICKNESSES, build_library

    parameter = inspect.signature(build_library).parameters["domain_sizes"]
    assert parameter.default is None, "the default is resolved inside, not in the signature"
    source = inspect.getsource(build_library)
    assert "DISCRETE_THICKNESSES if domain_sizes is None" in source, (
        "build_library does not fall back to DISCRETE_THICKNESSES, so a caller "
        "that passes nothing - which is every caller in the GUI and the CLI - "
        "gets no width axis on the discrete phases"
    )
