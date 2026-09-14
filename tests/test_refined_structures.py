"""Taking structures from a refinement instead of from a publication.

A published structure is of somebody else's specimen.  Where a refinement of
*this* specimen exists, its cell and its refined occupancies describe what was
in the beam, and for a clay that is where it matters most: the octahedral
Fe-for-Mg and Fe-for-Al substitutions are refinable against the basal
intensities, they differ between deposits, and iron scatters about twice as
strongly as the magnesium it replaces.

TOPAS writes a refinement's result back in the same syntax it read, so a ``.out``
file is plain TOPAS input and the same parser serves.  What is new, and what is
tested here, is reading it without the macro escaping, the spellings a
refinement uses that a hand-written library does not, and the substitution of
one structure for another inside the loader the pattern library goes through.
"""

from __future__ import annotations

import textwrap

import pytest

from clayquant import models
from clayquant.bern import (
    clay_key_for,
    looks_like_a_macro_library,
    parse_refinement,
    refined_clay_structures,
    refined_crystals,
)

QUARTZ = """\
\tstr
\t\tphase_name "Quartz"
\t\ta a_Quartz  4.91969`
\t\tc c_Quartz  5.41066`
\t\tspace_group "P3121"
\t\tsite Si1 x 0.4701 y 0 z 0 occ Si+4 1. beq 0.6
\t\tsite O1  x 0.4139 y 0.2674 z 0.1188 occ O-2 1. beq 1.0
\t\tscale scale_Quartz  0.00194104013`
\t\tcell_volume vol_Quartz  113.410`
\t\tcell_mass mass_Quartz  180.253
\t\tweight_percent wp_Quartz  51.773`
"""


def write(tmp_path, text, name="refinement.out"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_a_refinement_output_is_read_without_macro_unescaping(tmp_path):
    """The jEdit path would mangle plain TOPAS; the two are told apart by looking."""
    assert not looks_like_a_macro_library(QUARTZ)
    assert looks_like_a_macro_library('<item name="Quartz" type="macro">')

    phases = parse_refinement(write(tmp_path, QUARTZ))
    assert [p.name for p in phases] == ["Quartz"]
    phase = phases[0]
    assert phase.cell["a"] == pytest.approx(4.91969)
    assert phase.cell["c"] == pytest.approx(5.41066)
    assert phase.stated_mass == pytest.approx(180.253)
    assert phase.stated_volume == pytest.approx(113.410)
    assert [s.species for s in phase.sites] == ["Si+4", "O-2"]


def test_the_rest_of_the_refinement_is_passed_over(tmp_path):
    """A .out is mostly instrument, background and agreement factors."""
    text = (
        "\tr_wp rval_rwp  6.70995676\n"
        "\tgof rval_gof  1.7980035\n"
        "\tbkg @  152.5 -558.2  290.3\n"
        "\tZero_Error(@, 0.06422517`)\n"
        "\tLP_Factor(0)\n"
        + QUARTZ
    )
    skipped: list[str] = []
    assert [p.name for p in parse_refinement(write(tmp_path, text), skipped=skipped)] == ["Quartz"]
    assert skipped == []


def test_a_site_may_state_its_own_multiplicity(tmp_path):
    """"num_posns 4" - what a refinement of a low-symmetry phase writes.

    Before this was accepted, a whole phase was lost: the site lines did not
    match, so the block had no atoms and was rejected for having none.
    """
    text = """\
    \tstr
    \t\tphase_name "Clinochlore"
    \t\tspace_group "C -1"
    \t\ta @  5.492495`
    \t\tb @  9.275810`
    \t\tc @  14.280135`
    \t\tal @  89.34883`
    \t\tbe @  97.04046`
    \t\tga @  88.05500`
    \t\tprm Mg1Fe1_occ  0.699776445` min 0 max 1
    \t\tsite\tMg1 \tnum_posns\t2\tx\t0\ty\t0\tz\t0\tocc\tMg+2 \t= Mg1Fe1_occ;:  0.69978`\tbeq\t0.62
    \t\tsite\tFe1 \tnum_posns\t2\tx\t0\ty\t0\tz\t0\tocc\tFe+3 \t= 1-Mg1Fe1_occ;:  0.30022`\tbeq\t0.62
    """
    phases = parse_refinement(write(tmp_path, text))
    assert len(phases) == 1
    occupancies = {s.label: s.occupancy for s in phases[0].sites}
    assert occupancies == {"Mg1": pytest.approx(0.69978), "Fe1": pytest.approx(0.30022)}
    assert sum(occupancies.values()) == pytest.approx(1.0, abs=1e-4)


def test_a_refined_occupancy_survives_into_the_crystal(tmp_path):
    text = """\
    \tstr
    \t\tphase_name "Illite"
    \t\ta a_Illite 5.203089`
    \t\tb b_Illite 9.040800`
    \t\tc c_Illite 20.087833`
    \t\tbe be_Illite 97.10957`
    \t\tspace_group "C12/c1"
    \t\tprm Al2Fe2_Illite  0.950351347` min 0.75 max 1
    \t\tsite Al2 x 0.2586 y 0.0828 z 0.0068 occ Al+3 = Al2Fe2_Illite;:  0.95035` beq 5.8
    \t\tsite Fe2 x 0.2586 y 0.0828 z 0.0068 occ Fe+3 = 1 - Al2Fe2_Illite;:  0.04965` beq 5.8
    \t\tsite O1  x 0.4623 y 0.9194 z 0.0505 occ O-2 1. beq 1.18
    """
    crystals = refined_crystals(write(tmp_path, text))
    assert set(crystals) == {"Illite"}
    sites = {s.label: s for s in crystals["Illite"].sites}
    assert sites["Fe2"].occupancy == pytest.approx(0.04965)
    assert sites["Al2"].occupancy == pytest.approx(0.95035)
    # The iron is on the aluminium's position, which is what a substitution is.
    assert (sites["Fe2"].x, sites["Fe2"].z) == (sites["Al2"].x, sites["Al2"].z)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Illite", "illite"),
        ("Clinochlore", "chlorite"),
        ("Chlorite", "chlorite"),
        ("Kaolinite2M", "kaolinite_2M"),
        ("Kaolinite 2M", "kaolinite_2M"),
        ("Kaolinite_1M", "kaolinite_1M"),
        ("Kaolinite", "kaolinite_1M"),
        ("Illite-HiCryst", "illite"),
        ("Quartz", None),
        ("Albite", None),
        ("Rutile", None),
    ],
)
def test_a_refinements_phase_name_maps_to_a_library_key(name, expected):
    """The refinement calls it what the person refining it typed."""
    assert clay_key_for(name) == expected


def test_a_refinement_whose_clays_are_peaks_phases_supplies_none(tmp_path):
    """The commonest case, and it must not look like success.

    Modelling a clay as hkl_Is is how an oriented mount is usually refined; such
    a refinement fits the clay's intensity and says nothing about its structure.
    """
    text = """\
    \thkl_Is
    \t\tphase_name "Illite-HKL"
    \t\tspace_group "P1"
    \t\ta 5.2 b 9.0 c 20.1
    \t\tload hkl_m_d_th2 I
    """ + QUARTZ
    skipped: list[str] = []
    assert refined_clay_structures(write(tmp_path, text), skipped=skipped) == {}
    assert any("peaks phase" in note for note in skipped)


def test_a_refined_structure_replaces_the_published_one(tmp_path):
    """And the loader the pattern library goes through returns it."""
    text = """\
    \tstr
    \t\tphase_name "Clinochlore"
    \t\tspace_group "C -1"
    \t\ta 5.492495
    \t\tb 9.275810
    \t\tc 14.280135
    \t\tal 89.34883
    \t\tbe 97.04046
    \t\tga 88.05500
    \t\tsite Mg1 num_posns 2 x 0 y 0 z 0 occ Mg+2 0.7 beq 0.62
    \t\tsite Fe1 num_posns 2 x 0 y 0 z 0 occ Fe+3 0.3 beq 0.62
    """
    structures = refined_clay_structures(write(tmp_path, text))
    assert set(structures) == {"chlorite"}

    try:
        models.use_refined_structures(structures)
        assert set(models.refined_structures()) == {"chlorite"}
        loaded = models.load_crystal("chlorite")
        assert loaded.c == pytest.approx(14.280135)
        assert {s.label for s in loaded.sites} == {"Mg1", "Fe1"}
        # A refined structure needs no file, so it counts as available.
        assert models.available_phases()["chlorite"]
        # And the layer model, which is what the clay library actually uses,
        # is derived from it rather than from a structure cached earlier.
        assert models.load_layer("chlorite").thickness == pytest.approx(loaded.d001)
    finally:
        models.clear_refined_structures()

    assert models.refined_structures() == {}


def test_going_back_to_the_published_structures_is_not_left_to_a_stale_cache(tmp_path):
    """load_crystal is memoised; an override that outlived its welcome is a trap."""
    text = QUARTZ.replace('"Quartz"', '"Clinochlore"').replace(
        '\t\ta a_Quartz  4.91969`\n\t\tc c_Quartz  5.41066`\n\t\tspace_group "P3121"',
        '\t\ta 5.4\n\t\tb 9.2\n\t\tc 14.2\n\t\tspace_group "P1"',
    )
    structures = refined_clay_structures(write(tmp_path, text))
    assert set(structures) == {"chlorite"}
    models.use_refined_structures(structures)
    assert models.load_crystal("chlorite").c == pytest.approx(14.2)
    models.clear_refined_structures()
    if not models.available_phases()["chlorite"]:
        with pytest.raises(FileNotFoundError):
            models.load_crystal("chlorite")
    else:
        assert models.load_crystal("chlorite").c != pytest.approx(14.2)


def test_a_structure_the_library_does_not_ask_for_is_refused():
    with pytest.raises(KeyError, match="not structures the library asks for"):
        models.use_refined_structures({"quartz": object()})
    assert models.refined_structures() == {}


def test_a_refinement_laid_over_a_library_replaces_what_it_names(tmp_path):
    """The library supplies breadth; the refinement supplies authority.

    A laboratory library has hundreds of phases and is the same for every
    specimen.  A refinement has a handful and is of *this* specimen.  Importing
    both, in that order, keeps the breadth and prefers the refined structure
    wherever there is one.
    """
    from clayquant.bern import load_phase_database, write_phase_database

    library = tmp_path / "library.out"
    library.write_text(
        textwrap.dedent(QUARTZ + QUARTZ.replace('"Quartz"', '"Rutile"').replace("_Quartz", "_R")),
        encoding="utf-8",
    )
    refinement = tmp_path / "refined.out"
    refinement.write_text(
        textwrap.dedent(QUARTZ.replace("4.91969", "4.88000")), encoding="utf-8"
    )

    counts = write_phase_database([library, refinement], tmp_path / "phases.json")
    assert counts["written"] == 2, "two distinct phases, not three"
    assert counts["replaced"] == 1
    crystals = load_phase_database(tmp_path / "phases.json")
    assert set(crystals) == {"Quartz", "Rutile"}
    assert crystals["Quartz"].a == pytest.approx(4.88000), "the refinement's cell won"
    assert crystals["Rutile"].a == pytest.approx(4.91969), "and the library kept the rest"


def test_one_source_still_works_as_before(tmp_path):
    """write_phase_database took a single path and must keep taking one."""
    from clayquant.bern import write_phase_database

    path = tmp_path / "one.out"
    path.write_text(textwrap.dedent(QUARTZ), encoding="utf-8")
    counts = write_phase_database(path, tmp_path / "phases.json")
    assert counts["written"] == 1 and counts["replaced"] == 0
