"""Reading a TOPAS structure library written by TOPAS rather than by hand.

A library maintained in a laboratory accumulates the spellings TOPAS itself
produces, and every one of them that the reader does not know costs a phase.
The cases below are the ones that actually cost phases in a 205-entry library:
each is the smallest item that reproduces it.

The other half of this is that a phase which cannot be read must say so.  Before
these tests, an unreadable item was dropped before it became a record, so it was
invisible to every count taken afterwards - adding a phase to the library and
re-importing gave the same number of phases as before, with no indication why.
"""

from __future__ import annotations

import pytest

from clayquant.bern import (
    decode_macro_item,
    parse_macro_library,
    parse_topas_structures,
)


def macro_item(name: str, lines: list[str]) -> str:
    """A jEdit macro item that would insert ``lines`` into TOPAS."""
    inserts = "\n".join(
        f'\t\tbuffer.insert(textArea.getCaretPosition(), "\\n{line}");' for line in lines
    )
    return f'<item name="{name}" type="macro">\n{inserts}\n\t\t</item>\n'


QUARTZ = [
    "\\tstr",
    '\\t\\tphase_name \\&quot;Quartz\\&quot;',
    '\\t\\tspace_group \\&quot;P3221\\&quot;',
    "\\t\\ta a_Quartz 4.9134",
    "\\t\\tc c_Quartz 5.4052",
    "\\t\\tsite Si1 x 0.4701 y 0 z 0 occ Si+4 1. beq 0.6",
    "\\t\\tsite O1  x 0.4139 y 0.2674 z 0.1188 occ O-2 1. beq 1.0",
]


def only_phase(text: str, skipped: list[str] | None = None):
    phases = parse_topas_structures(decode_macro_item(text), skipped=skipped)
    assert len(phases) == 1, [p.name for p in phases]
    return phases[0]


def test_a_hand_written_phase_still_reads():
    phase = only_phase(macro_item("Quartz", QUARTZ))
    assert phase.name == "Quartz"
    assert phase.space_group == "P3221"
    assert phase.cell["a"] == pytest.approx(4.9134)
    assert len(phase.sites) == 2


def test_a_refined_value_carries_a_backtick():
    """TOPAS marks a refined parameter with a backtick, and welds limits onto it."""
    lines = list(QUARTZ)
    lines[3] = "\\t\\ta a_Quartz  4.9134` "
    lines[4] = "\\t\\tc c_Quartz  5.4052`_LIMIT_MIN_5.3 min 5.3 max 5.5"
    phase = only_phase(macro_item("Quartz", lines))
    assert phase.cell["a"] == pytest.approx(4.9134)
    assert phase.cell["c"] == pytest.approx(5.4052)


def test_an_occupancy_expression_keeps_its_evaluated_value():
    """occ = 1-FeMgOCC; :0.00004` - the value after the colon is what TOPAS had."""
    lines = list(QUARTZ) + [
        "\\t\\tsite Mg1 x 0.16 y 0.5 z 0.25 occ Mg+2 =1-FeMgOCC;:0.00004`\\tbeq 1",
        "\\t\\tsite Fe1 x 0.16 y 0.5 z 0.25 occ Fe+2 =FeMgOCC;:0.99996`\\tbeq 1",
    ]
    phase = only_phase(macro_item("Sekaninaite-like", lines))
    occupancies = {site.label: site.occupancy for site in phase.sites}
    assert occupancies["Mg1"] == pytest.approx(0.00004)
    assert occupancies["Fe1"] == pytest.approx(0.99996)
    assert occupancies["Mg1"] + occupancies["Fe1"] == pytest.approx(1.0)


def test_a_space_group_written_with_spaces():
    """"C c c m" and "Cccm" are the same group; TOPAS writes either."""
    lines = list(QUARTZ)
    lines[2] = '\\t\\tspace_group \\&quot;C c c m\\&quot;'
    assert only_phase(macro_item("Spaced", lines)).space_group == "C c c m"


def test_a_comment_after_str():
    """The NIST LaB6 entry notes on that line that its cell is certified."""
    lines = list(QUARTZ)
    lines[0] = "\\tstr ' Fixed LP's from the NIST SRM660a certificate"
    assert only_phase(macro_item("LaB6-like", lines)).name == "Quartz"


def test_a_second_block_may_share_the_cell_of_the_first():
    """Two crystallinities of one mineral: the second refers to the first's cell.

    This is how a TOPAS library models a crystallinity distribution, and both
    blocks are wanted - they are fitted with separate scale factors.
    """
    lines = QUARTZ + [
        "\\tstr",
        '\\t\\tphase_name \\&quot;Quartz-HiCryst\\&quot;',
        '\\t\\tspace_group \\&quot;P3221\\&quot;',
        "\\t\\ta =a_Quartz;",
        "\\t\\tc =c_Quartz;",
        "\\t\\tsite Si1 x 0.4701 y 0 z 0 occ Si+4 1. beq 0.6",
    ]
    phases = parse_topas_structures(decode_macro_item(macro_item("Quartz-2", lines)))
    assert [p.name for p in phases] == ["Quartz", "Quartz-HiCryst"]
    assert phases[1].cell["a"] == pytest.approx(phases[0].cell["a"])
    assert phases[1].cell["c"] == pytest.approx(phases[0].cell["c"])


def test_a_reference_to_a_parameter_that_is_not_set_is_recorded():
    lines = list(QUARTZ)
    lines[3] = "\\t\\ta =a_Somewhere_Else;"
    skipped: list[str] = []
    phases = parse_topas_structures(decode_macro_item(macro_item("Dangling", lines)),
                                    skipped=skipped)
    assert phases == [] and skipped
    assert "no cell parameters" in skipped[0]


def test_an_unreadable_phase_is_reported_rather_than_dropped(tmp_path):
    """The fault that made a new phase seem not to import at all."""
    good = macro_item("Quartz", QUARTZ)
    broken = macro_item("Mystery", [line for line in QUARTZ if not line.startswith("\\t\\ta ")
                                    and not line.startswith("\\t\\tc ")])
    library = tmp_path / "library.xml"
    library.write_text(good + broken, encoding="utf-8")

    skipped: list[str] = []
    phases = parse_macro_library(library, skipped=skipped)
    assert [p.name for p in phases] == ["Quartz"]
    assert len(skipped) == 1, "one item failed, so one explanation"
    assert "cell" in skipped[0]


def test_a_peaks_phase_says_what_it_is(tmp_path):
    """hkl_Is lists reflections instead of atoms; it cannot be calculated from."""
    item = macro_item("Amorphous", [
        "\\thkl_Is",
        '\\t\\tphase_name \\&quot;amorphous\\&quot;',
        '\\t\\tspace_group \\&quot;Pmmm\\&quot;',
        "\\t\\ta 34.4 b 0.1 c 0.1",
        "\\t\\tload hkl_m_d_th2 I",
    ])
    library = tmp_path / "library.xml"
    library.write_text(item, encoding="utf-8")
    skipped: list[str] = []
    assert parse_macro_library(library, skipped=skipped) == []
    assert len(skipped) == 1
    assert "peaks phase" in skipped[0] and "hkl_Is" in skipped[0]


def test_an_item_that_is_not_a_structure_at_all_is_passed_over(tmp_path):
    """A menu is full of headings and helpers; they are not failures."""
    library = tmp_path / "library.xml"
    library.write_text(
        macro_item("A heading", ["'-------------------", "' just a note"])
        + macro_item("Quartz", QUARTZ),
        encoding="utf-8",
    )
    skipped: list[str] = []
    phases = parse_macro_library(library, skipped=skipped)
    assert [p.name for p in phases] == ["Quartz"]
    assert skipped == []
