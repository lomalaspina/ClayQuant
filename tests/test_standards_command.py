"""The command that fits weighed standards and calibrates on them.

The arithmetic is tested in tests/test_weighed_mount.py; what is tested here is
that the command reads its table, fits with one orientation per mount, keeps an
impure standard out of the calibration, and says what it did.
"""

import csv

import numpy as np
import pytest

from clayquant.library import build_library
from clayquant.pattern import Pattern, two_theta_grid
from clayquant.standards import (
    Standard,
    fit_standards,
    main,
    read_standards,
)


@pytest.fixture(scope="module")
def library():
    return build_library(
        grid=two_theta_grid(4.0, 30.0, 0.05), orientations=(0.2, 0.5),
        illite_smectite=(1.0,), chlorite_smectite=(), chlorite_iron=(),
        illite_composition=(), csds_means=(15.0,), host_thicknesses={},
        air_dried_thickness=None, strains={}, progress=False)


def a_mount(library, phase_name, scale, path):
    """Write a synthetic measurement of one library entry as a two-column file."""
    entry = next(e for e in library.entries if e.name.startswith(phase_name))
    intensity = scale * (entry.normalization or 1.0) * np.asarray(entry.intensity)
    path.write_text("\n".join(
        "%.4f %.6f" % (x, y)
        for x, y in zip(library.two_theta, intensity + 20.0)))
    return entry


def test_the_table_is_read_with_its_columns_in_any_spelling(tmp_path):
    path = tmp_path / "standards.csv"
    path.write_text(
        "Name,File,Mass (mg),Phase,Area (cm2)\n"
        "Kaolinite 12,k12.xrdml,2.6,kaolinite_1M,4.91\n"
        "Dickite 7,d7.xrdml,2.3,,\n"
    )
    standards = read_standards(path)
    assert [s.name for s in standards] == ["Kaolinite 12", "Dickite 7"]
    assert standards[0].mass_mg == pytest.approx(2.6)
    assert standards[0].phase == "kaolinite_1M"
    assert standards[0].area_cm2 == pytest.approx(4.91)
    # left empty on purpose: not a single-phase standard
    assert standards[1].phase == ""
    assert standards[1].area_cm2 is None
    # paths resolve against the table's own directory
    assert standards[0].path == tmp_path / "k12.xrdml"


def test_a_mount_with_no_mass_is_refused(tmp_path):
    path = tmp_path / "standards.csv"
    path.write_text("name,file,mass,phase\nKaolinite,k.xrdml,,kaolinite_1M\n")
    with pytest.raises(ValueError, match="no mass"):
        read_standards(path)


def test_an_empty_table_is_refused(tmp_path):
    path = tmp_path / "standards.csv"
    path.write_text("name,file,mass,phase\n")
    with pytest.raises(ValueError, match="lists no mounts"):
        read_standards(path)


def test_a_missing_measurement_is_reported_not_raised(tmp_path, library):
    standards = [Standard(name="gone", path=tmp_path / "nothing.xy", mass_mg=2.0,
                          phase="illite")]
    result, = fit_standards(standards, library)
    assert result.mount is None
    assert "not found" in result.note


def test_a_mount_is_fitted_at_one_orientation(tmp_path, library):
    a_mount(library, "kaolinite_1M PO=0.2", 1.0, tmp_path / "k.xy")
    standards = [Standard(name="k", path=tmp_path / "k.xy", mass_mg=2.6,
                          phase="kaolinite_1M", area_cm2=4.91)]
    result, = fit_standards(standards, library)
    assert result.mount is not None
    assert result.orientation == pytest.approx(0.2)
    assert result.identified
    assert result.mount.comparable_constant == pytest.approx(
        result.mount.constant / 0.2**3)


def test_the_command_runs_and_writes_a_calibration(tmp_path, library, capsys):
    path = tmp_path / "library.npz"
    library.save(path)
    a_mount(library, "kaolinite_1M PO=0.2", 1.0, tmp_path / "k.xy")
    a_mount(library, "illite PO=0.2", 1.0, tmp_path / "i.xy")
    a_mount(library, "chlorite PO=0.2", 1.0, tmp_path / "c.xy")
    table = tmp_path / "standards.csv"
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "file", "mass", "phase", "area"])
        writer.writerow(["kao", "k.xy", 2.6, "kaolinite_1M", 4.91])
        writer.writerow(["ill", "i.xy", 2.4, "illite", 4.91])
        writer.writerow(["chl", "c.xy", 2.5, "", 4.91])

    out = tmp_path / "calibration.json"
    assert main([str(table), "-l", str(path), "-o", str(out), "--range", "5", "28"]) == 0
    said = capsys.readouterr().out
    assert "on one r" in said
    assert "k(" in said

    import json
    factors = json.loads(out.read_text())["factors"]
    # the chlorite mount had no phase given, so it is fitted but not calibrated on
    assert set(factors) == {"kaolinite_1M", "illite"}


def test_a_library_that_is_not_there_says_how_to_build_one(tmp_path, capsys):
    table = tmp_path / "standards.csv"
    table.write_text("name,file,mass,phase\nk,k.xy,2.6,kaolinite_1M\n")
    with pytest.raises(SystemExit):
        main([str(table), "-l", str(tmp_path / "absent.npz")])
    assert "clayquant-build-library" in capsys.readouterr().err
