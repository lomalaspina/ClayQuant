"""Importing an accompanying mineral from a CIF rather than from TOPAS.

The structure library is the right source when it has the mineral.  It does not
have everything - a specimen can contain something nobody in the lab has ever
refined - and then a published CIF is what there is.
"""

import json

import pytest

from clayquant.bern import load_phase_database, parse_cif_structure, write_phase_database

CIF = """\
data_test
_chemical_name_mineral 'Sepiolite'
_symmetry_space_group_name_H-M 'P n c n'
_cell_length_a 13.3067
_cell_length_b 26.9720
_cell_length_c 5.2664
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
x,y,z
-x,-y,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_U_iso_or_equiv
Si1 Si 0.20000 0.05000 0.50000 1.000 0.010
Mg1 Mg 0.00000 0.00000 0.00000 1.000 0.010
O1 O 0.10000 0.10000 0.25000 1.000 0.015
"""


@pytest.fixture
def cif(tmp_path):
    path = tmp_path / "sepiolite_COD_9014723.cif"
    path.write_text(CIF)
    return path


def test_the_mineral_name_comes_from_the_file(cif):
    phase, = parse_cif_structure(cif)
    assert phase.name == "Sepiolite"
    assert phase.space_group == "P n c n"
    assert len(phase.sites) == 3


def test_the_symmetry_is_taken_as_stated_not_expanded(cif):
    """Which is why this needs neither gemmi nor a symbol gemmi recognises."""
    phase, = parse_cif_structure(cif)
    assert phase.symops == ["x,y,z", "-x,-y,z"]


def test_a_file_without_a_mineral_name_is_named_after_itself(tmp_path):
    path = tmp_path / "palygorskite_COD_1533366.cif"
    path.write_text(CIF.replace("_chemical_name_mineral 'Sepiolite'\n", ""))
    phase, = parse_cif_structure(path)
    assert phase.name == "Palygorskite"


def test_something_that_is_not_a_cif_is_reported_not_raised(tmp_path):
    path = tmp_path / "notes.cif"
    path.write_text("this is not a structure\n")
    skipped: list[str] = []
    assert parse_cif_structure(path, skipped=skipped) == []
    assert skipped and "notes.cif" in skipped[0]


def test_a_cif_reaches_the_phase_database_and_reads_back(cif, tmp_path):
    output = tmp_path / "phases.json"
    counts = write_phase_database([cif], output)
    assert counts["written"] == 1
    assert counts["failed"] == 0
    assert json.loads(output.read_text())["phases"][0]["name"] == "Sepiolite"

    crystal = load_phase_database(output)["Sepiolite"]
    assert crystal.a == pytest.approx(13.3067)
    assert crystal.c == pytest.approx(5.2664)
    assert crystal.symops == ["x,y,z", "-x,-y,z"]
    assert crystal.cell_mass > 0.0


def test_a_cif_and_a_topas_library_can_be_imported_together(cif, tmp_path):
    """The point of the feature: the library for what it has, a CIF for the
    mineral it does not."""
    topas = tmp_path / "library.xml"
    topas.write_text(
        '<?xml version="1.0"?><MENU><MENU_ITEM LABEL="Quartz"><CODE>\n'
        "str\n"
        "  phase_name Quartz\n"
        "  space_group P3221\n"
        "  a 4.9137 b 4.9137 c 5.4047 al 90 be 90 ga 120\n"
        "  site Si1 num_posns 0 x 0.4697 y 0 z 0 occ Si+4 1 beq 0.5\n"
        "  site O1  num_posns 0 x 0.4135 y 0.2669 z 0.1191 occ O-2 1 beq 0.8\n"
        "</CODE></MENU_ITEM></MENU>\n"
    )
    output = tmp_path / "phases.json"
    counts = write_phase_database([topas, cif], output)
    assert counts["written"] == 2
    assert set(load_phase_database(output)) == {"Quartz", "Sepiolite"}
