"""A structure library's preferred-orientation axis is not a crystal habit.

Two things went wrong together when the orientation pole was made settable.  A
TOPAS library writes ``0 0 0`` for a phase whose PO macro carries no axis - 72
of the 205 phases in the one this was found on - and a zero vector is not a
direction, so it reached the geometry and stopped a fit with "is not a
direction".  And where the axis is real, it records which axis a refinement
corrected about; taking it as a habit would span four orientations for every
accompanying mineral and let an equant quartz grain be fitted at r = 0.3.
"""

import json

import pytest

from clayquant.bern import load_phase_database
from clayquant.crystal import Crystal
from clayquant.gui.app import habit_of
from clayquant.models import FIBRE_AXES, MINERAL_HABIT


def a_database(tmp_path, poles: dict[str, list[int] | None]):
    records = []
    for name, pole in poles.items():
        records.append({
            "name": name, "icsd": None, "space_group": "P 1",
            "symops": ["x, y, z"],
            "cell": {"a": 5.0, "b": 5.0, "c": 5.0,
                     "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "po_hkl": pole, "is_clay": False,
            "sites": [{"species": "Si", "x": 0.0, "y": 0.0, "z": 0.0,
                       "occupancy": 1.0, "b_iso": 1.0, "label": "Si1"}],
        })
    path = tmp_path / "phases.json"
    path.write_text(json.dumps({"source": "test", "n_phases": len(records),
                                "phases": records}))
    return load_phase_database(path)


def test_a_zero_pole_becomes_no_pole(tmp_path):
    """The crash: 'is not a direction', in the middle of a fit."""
    database = a_database(tmp_path, {"Aluminium": [0, 0, 0]})
    assert database["Aluminium"].po_axis is None


def test_a_real_pole_survives(tmp_path):
    database = a_database(tmp_path, {"Rutile": [1, 1, 0]})
    assert database["Rutile"].po_axis == (1.0, 1.0, 0.0)


def test_a_missing_pole_is_no_pole(tmp_path):
    database = a_database(tmp_path, {"Quartz": None})
    assert database["Quartz"].po_axis is None


def test_a_zero_pole_given_straight_to_a_crystal_is_refused_the_same_way():
    crystal = Crystal(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0,
                      sites=[], po_axis=(0.0, 0.0, 0.0))
    assert crystal.po_axis is None


def test_a_pole_that_is_not_three_indices_is_refused():
    with pytest.raises(ValueError, match="three indices"):
        Crystal(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0,
                sites=[], po_axis=(1.0, 1.0))


def test_a_refined_axis_is_not_a_habit(tmp_path):
    """Quartz and the feldspars carry one, and are equant all the same."""
    database = a_database(tmp_path, {"Quartz": [1, 0, 1], "Albite": [0, 0, 1]})
    for name in ("Quartz", "Albite"):
        assert database[name].po_axis is not None
        assert habit_of(name, database[name]) is None


def test_a_mineral_with_a_habit_takes_the_database_axis(tmp_path):
    database = a_database(tmp_path, {"Hornblende": [1, 1, 0]})
    assert habit_of("Hornblende", database["Hornblende"]) == (1.0, 1.0, 0.0)


def test_a_mineral_with_a_habit_and_no_axis_falls_back_on_the_table(tmp_path):
    database = a_database(tmp_path, {"Riebeckite": None})
    assert database["Riebeckite"].po_axis is None
    assert habit_of("Riebeckite", database["Riebeckite"]) == MINERAL_HABIT["riebeckite"]


def test_a_needle_is_not_offered_the_plate_path(tmp_path):
    """Its axis is a direct-space direction in FIBRE_AXES (Sec. A.65)."""
    database = a_database(tmp_path, {"Palygorskite": None})
    assert habit_of("Palygorskite", database["Palygorskite"]) is None
    assert FIBRE_AXES["palygorskite"] == (0, 0, 1)


def test_the_name_is_matched_however_it_is_capitalised(tmp_path):
    database = a_database(tmp_path, {"HORNBLENDE": [1, 1, 0]})
    assert habit_of("HORNBLENDE", database["HORNBLENDE"]) is not None


def test_a_zero_pole_no_longer_stops_a_pattern(tmp_path):
    """End to end, on the shape of the failure: a phase out of a database with
    a zero axis is calculated without complaint."""
    from clayquant.pattern import powder_pattern, two_theta_grid

    database = a_database(tmp_path, {"Aluminium": [0, 0, 0]})
    crystal = database["Aluminium"]
    pole = habit_of("Aluminium", crystal)
    pattern = powder_pattern(crystal, two_theta_grid(5.0, 40.0, 0.1),
                             r_march_dollase=1.0,
                             po_axis=(0.0, 0.0, 1.0) if pole is None else pole)
    assert float(max(pattern.intensity)) > 0.0
