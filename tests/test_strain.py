"""Microstrain broadening of the discrete clay patterns.

A clay peak on an oriented mount is much wider than the instrument alone makes
it - near 12 deg a well collimated scan gives about 0.05 deg where the kaolinite
and illite 001 measure 0.084 deg - and a discrete library entry carries no
crystallite thickness distribution to explain that with.  These tests pin the
term that does: its angular dependence, that it survives a round trip through a
stored library, and that the fit is made to choose one value of it per
composition rather than mix several.
"""

import numpy as np
import pytest

from clayquant.library import LibraryEntry, PatternLibrary
from clayquant.nnls import orientation_families
from clayquant.profile import PeakShape

WAVELENGTH = 1.540596


def test_no_strain_is_the_instrument_alone():
    angles = np.array([6.2, 12.4, 25.0, 35.0])
    plain = PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6)
    strained = PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, strain=0.0)
    assert np.allclose(plain.fwhm(angles, WAVELENGTH), strained.fwhm(angles, WAVELENGTH))


def test_strain_widens_every_reflection():
    angles = np.array([6.2, 12.4, 25.0, 35.0])
    narrow = PeakShape(w=0.0025, strain=0.0).fwhm(angles, WAVELENGTH)
    wide = PeakShape(w=0.0025, strain=0.5).fwhm(angles, WAVELENGTH)
    assert np.all(wide > narrow)


def test_strain_grows_as_tan_theta():
    """The signature that separates strain from a finite crystallite size."""
    angles = np.array([10.0, 20.0, 40.0, 60.0])
    theta = np.radians(angles) / 2.0
    # With no instrumental width left the strain term stands alone, up to the
    # floor the model keeps on the variance.
    widths = PeakShape(u=0.0, v=0.0, w=0.0, strain=0.4).fwhm(angles, WAVELENGTH)
    assert np.allclose(widths, 0.4 * np.tan(theta), rtol=1e-4)


def test_the_stated_value_is_the_width_at_tan_theta_one():
    """The TOPAS convention, so a refined Strain_L can be carried over."""
    two_theta = np.degrees(2.0 * np.arctan(1.0))
    width = PeakShape(u=0.0, v=0.0, w=0.0, strain=0.37).fwhm(np.array([two_theta]), WAVELENGTH)
    assert width[0] == pytest.approx(0.37, rel=1e-6)


def test_size_and_strain_are_told_apart_by_their_angular_dependence():
    low, high = np.array([12.0]), np.array([36.0])
    size = PeakShape(u=0.0, v=0.0, w=0.0, size_c=1000.0)
    strain = PeakShape(u=0.0, v=0.0, w=0.0, strain=0.5)
    # Matched at the low angle, they must not agree at the high one.
    ratio_size = size.fwhm(high, WAVELENGTH)[0] / size.fwhm(low, WAVELENGTH)[0]
    ratio_strain = strain.fwhm(high, WAVELENGTH)[0] / strain.fwhm(low, WAVELENGTH)[0]
    assert ratio_strain > 2.0 * ratio_size


def _library():
    grid = np.linspace(4.0, 30.0, 64)
    library = PatternLibrary(two_theta=grid)
    for strain in (0.0, 0.5):
        for r in (0.3, 1.0):
            intensity = np.exp(-0.5 * ((grid - 12.4) / (0.05 + strain)) ** 2)
            library.entries.append(
                LibraryEntry(
                    name=f"kaolinite_2M PO={r:g} e={strain:g}",
                    phase="kaolinite_2M",
                    intensity=intensity,
                    march_dollase=r,
                    thickness=7.16,
                    strain=strain,
                )
            )
    return library


def test_strain_round_trips_through_a_saved_library(tmp_path):
    saved = _library().save(tmp_path / "library.npz")
    loaded = PatternLibrary.load(saved)
    assert [entry.strain for entry in loaded.entries] == [0.0, 0.0, 0.5, 0.5]


def test_a_library_without_the_column_loads_at_zero_strain(tmp_path):
    """Libraries built before the axis existed carry the instrumental width."""
    path = tmp_path / "old.npz"
    library = _library()
    with np.load(library.save(path), allow_pickle=True) as data:
        fields = {key: data[key] for key in data.files if key != "strain"}
    np.savez_compressed(path, **fields)
    loaded = PatternLibrary.load(path)
    assert all(entry.strain == 0.0 for entry in loaded.entries)


def test_one_strain_per_composition():
    """Orientation and strain are both chosen, not mixed: one family, four members."""
    labels = orientation_families(_library())
    assert len(set(labels)) == 1
    assert labels[0]


def test_the_kaolinites_carry_the_axis_and_the_others_do_not():
    """Illite and chlorite take their width from the interstratified series."""
    from clayquant.library import DISCRETE_STRAINS

    assert set(DISCRETE_STRAINS) == {"kaolinite_1M", "kaolinite_2M"}
    for values in DISCRETE_STRAINS.values():
        assert values[0] == 0.0
        assert values == tuple(sorted(values))


def test_the_command_line_reads_a_strain_per_phase():
    from clayquant.library import _parse_strains

    assert _parse_strains(None) is None
    assert _parse_strains(["none"]) == {}
    assert _parse_strains(["kaolinite_2M=0,0.5,1"]) == {"kaolinite_2M": (0.0, 0.5, 1.0)}
    with pytest.raises(SystemExit):
        _parse_strains(["kaolinite_2M"])
    with pytest.raises(SystemExit):
        _parse_strains(["quartz=0,1"])


def test_a_built_library_widens_only_the_phases_given_the_axis():
    from clayquant.library import build_library
    from clayquant.optics import Divergence
    from clayquant.pattern import Instrument

    built = build_library(
        grid=np.arange(8.0, 20.0, 0.01),
        instrument=Instrument(
            peak_shape=PeakShape(u=0.004, v=-0.001, w=0.002, eta=0.5, size_ab=600.0),
            divergence=Divergence(),
        ),
        orientations=(0.3,),
        illite_smectite=(1.00,),
        chlorite_smectite=(1.00,),
        csds_means=(15.0,),
        host_thicknesses={},
        strains={"kaolinite_2M": (0.0, 1.0)},
        air_dried_thickness=None,
    )
    strains = {entry.name: entry.strain for entry in built.entries}
    assert strains["kaolinite_2M PO=0.3"] == 0.0
    assert strains["kaolinite_2M PO=0.3 e=1"] == 1.0
    assert all(value == 0.0 for name, value in strains.items() if "kaolinite_2M" not in name)

    def breadth(name):
        """Integral breadth: area over height, which widening raises and scaling does not."""
        row = built.entries[built.names.index(name)].intensity
        return float(np.sum(row) / np.max(row))

    assert breadth("kaolinite_2M PO=0.3 e=1") > 1.5 * breadth("kaolinite_2M PO=0.3")
