"""Patterns built from measured peak lists, as a powder diffraction file gives them.

The point of the module is what it *cannot* do as much as what it can, so that
is tested too: an entry with no indices cannot be oriented, and an entry with no
structure has no mass and so no weight percent.
"""

from pathlib import Path

import numpy as np
import pytest

from clayquant import peaklist

from clayquant.pattern import Instrument
from clayquant.peaklist import (
    PeakList,
    compare_with_calculated,
    build_peaklist_library,
    peaklist_pattern,
    peaklist_reflections,
    read_peak_lists,
)

QUARTZ = """
# Quartz, as it would be copied out of a powder diffraction file.
PHASE   Quartz
SOURCE  ICDD PDF 01-070-7344
CELL    4.9134 4.9134 5.4052 90 90 120
  4.2550    22    1  0  0
  3.3435   100    1  0  1
  2.4573     8    1  1  0
  2.2815     8    1  0  2

PHASE   Montmorillonite-18A
SOURCE  ICDD PDF 00-012-0219
  15.000  100
   4.490   80
   2.570   60
"""


@pytest.fixture
def entries(tmp_path):
    path = tmp_path / "peaks.txt"
    path.write_text(QUARTZ, encoding="utf-8")
    return read_peak_lists(path)


def grid():
    return np.arange(4.0, 60.0, 0.005)


# --- reading --------------------------------------------------------------


def test_the_blocks_are_read_with_their_source(entries):
    quartz, montmorillonite = entries
    assert quartz.name == "Quartz"
    assert quartz.source == "ICDD PDF 01-070-7344"
    assert quartz.d.size == 4
    assert quartz.hkl is not None and quartz.cell is not None
    assert quartz.can_orient is True
    assert montmorillonite.name == "Montmorillonite-18A"
    assert montmorillonite.hkl is None and montmorillonite.cell is None
    assert montmorillonite.can_orient is False


@pytest.mark.parametrize(
    "token, expected",
    [("001", (0, 0, 1)), ("110", (1, 1, 0)), ("-1-11", (-1, -1, 1)), ("11-2", (1, 1, -2))],
)
def test_indices_joined_into_one_token_are_read(tmp_path, token, expected):
    path = tmp_path / "one.txt"
    path.write_text(f"PHASE X\nCELL 5 5 5 90 90 90\n 3.0 100 {token}\n", encoding="utf-8")
    entry = read_peak_lists(path)[0]
    assert tuple(entry.hkl[0]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, message",
    [
        ("3.0 100\n", "before any PHASE"),
        ("PHASE X\n", "no peaks"),
        ("PHASE X\nCELL 5 5 5 90 90\n 3.0 100\n", "six parameters"),
        ("PHASE X\n nonsense here\n", "cannot read a peak"),
        ("# nothing\n", "no PHASE blocks"),
        ("PHASE\n 3.0 100\n", "PHASE with no name"),
    ],
)
def test_a_file_it_cannot_read_raises_with_the_line(tmp_path, text, message):
    """A peak list one row short is a pattern wrong in a way nothing notices."""
    path = tmp_path / "bad.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_peak_lists(path)


def test_a_mismatched_peak_list_is_refused():
    with pytest.raises(ValueError, match="spacings"):
        PeakList(name="X", d=[1.0, 2.0], intensity=[100.0])
    with pytest.raises(ValueError, match="no peaks"):
        PeakList(name="X", d=[], intensity=[])
    with pytest.raises(ValueError, match="zero or negative"):
        PeakList(name="X", d=[0.0], intensity=[100.0])
    with pytest.raises(ValueError, match="index triples"):
        PeakList(name="X", d=[1.0, 2.0], intensity=[100.0, 50.0], hkl=[[0, 0, 1]])


# --- the pattern ----------------------------------------------------------


def test_the_peaks_land_where_the_spacings_say(entries):
    quartz = entries[0]
    pattern = peaklist_pattern(quartz, grid(), Instrument())
    strongest = grid()[int(np.argmax(pattern.intensity))]
    # 3.3435 A at Cu K-alpha is 26.64 deg, and it is the 100 % line.
    assert strongest == pytest.approx(26.64, abs=0.05)


def test_the_integrated_intensities_reproduce_the_entry(entries):
    """The Lorentz-polarization factor must be taken off and put back once.

    An entry's intensities are observed, so they already carry that factor,
    while the pattern assembler applies it itself.  Handing them over unchanged
    would apply it twice - a factor of thirty across a 4 to 40 degree scan, and
    a pattern tilted from end to end.  Integrating each peak of the result and
    comparing with the entry is what catches it.
    """
    quartz = entries[0]
    x = grid()
    pattern = peaklist_pattern(quartz, x, Instrument())
    angles = quartz.two_theta(Instrument().emission.principal_wavelength)
    areas = []
    for angle in angles:
        window = (x > angle - 0.35) & (x < angle + 0.35)
        areas.append(float(np.trapezoid(pattern.intensity[window], x[window])))
    areas = np.array(areas)
    assert np.allclose(100.0 * areas / areas.max(), quartz.intensity, atol=0.5)


def test_orientation_is_refused_without_the_indices(entries):
    """A March-Dollase factor needs the angle to c*, which needs hkl and a cell.

    Guessing would enhance every peak in the entry alike, which is not what an
    oriented mount does to anything.
    """
    montmorillonite = entries[1]
    with pytest.raises(ValueError, match="indices"):
        peaklist_pattern(montmorillonite, grid(), Instrument(), r_march_dollase=0.3)
    # r = 1 is fine: the factor is 1 at every angle, so the unknown does not enter.
    assert np.all(np.isfinite(peaklist_pattern(montmorillonite, grid(), Instrument()).intensity))


def test_orientation_enhances_the_basal_reflections_when_it_can(tmp_path):
    path = tmp_path / "clay.txt"
    path.write_text(
        "PHASE Kaolinite\nCELL 5.15 8.94 7.39 91.7 104.9 89.8\n"
        " 7.150 100 0 0 1\n 3.573  80 0 0 2\n 4.460  40 0 2 0\n",
        encoding="utf-8",
    )
    entry = read_peak_lists(path)[0]
    x = grid()
    random = peaklist_pattern(entry, x, Instrument(), r_march_dollase=1.0)
    oriented = peaklist_pattern(entry, x, Instrument(), r_march_dollase=0.3)

    def height(angle, pattern):
        window = (x > angle - 0.4) & (x < angle + 0.4)
        return float(np.max(pattern.intensity[window]))

    basal = height(12.38, oriented) / height(12.38, random)     # 001 at 7.15 A
    prism = height(19.90, oriented) / height(19.90, random)     # 020 at 4.46 A
    assert basal > 3.0 * prism


def test_the_metadata_carries_the_entry_it_came_from(entries):
    pattern = peaklist_pattern(entries[0], grid(), Instrument())
    assert pattern.metadata["kind"] == "peak list"
    assert pattern.metadata["source"] == "ICDD PDF 01-070-7344"
    assert pattern.metadata["indexed"] is True
    assert peaklist_pattern(entries[1], grid(), Instrument()).metadata["indexed"] is False


def test_reflections_beyond_the_wavelength_are_dropped(tmp_path):
    """A spacing below half the wavelength diffracts at no angle at all."""
    path = tmp_path / "small.txt"
    path.write_text("PHASE X\n 3.00 100\n 0.50 50\n", encoding="utf-8")
    entry = read_peak_lists(path)[0]
    found = peaklist_reflections(entry, Instrument())
    assert found.d.size == 1


# --- the separate library -------------------------------------------------


def test_the_library_is_separate_and_says_what_it_is(entries):
    library = build_peaklist_library(entries, grid(), Instrument())
    assert library.metadata["kind"] == "peak list"
    assert "weight percent" in library.metadata["note"]
    assert "licensed" in library.metadata["note"]
    assert library.metadata["n_phases"] == 2


def test_an_unindexed_entry_contributes_only_a_random_powder(entries):
    library = build_peaklist_library(entries, grid(), Instrument(),
                                     orientations=(0.3, 1.0))
    by_phase = {}
    for entry in library.entries:
        by_phase.setdefault(entry.phase, []).append(entry.march_dollase)
    assert sorted(by_phase["Quartz"]) == [0.3, 1.0]
    assert by_phase["Montmorillonite-18A"] == [1.0]
    assert library.metadata["unindexed"] == ["Montmorillonite-18A"]


def test_the_entries_carry_no_mass_so_no_weight_percent_is_implied(entries):
    """The reason it is a separate file from the clay library.

    A weight percent needs the mass of what a pattern was calculated from
    (Sec. 2.13).  These were not calculated from anything with a mass, so an
    entry that claimed one would be inventing it.
    """
    library = build_peaklist_library(entries, grid(), Instrument())
    for entry in library.entries:
        assert getattr(entry, "unit_mass", None) in (None, 0.0)


# --- scoring an entry against a structure ---------------------------------


def calculated_kaolinite():
    from clayquant.models import load_crystal
    from clayquant.pattern import peak_list

    return load_crystal("kaolinite_1M"), peak_list(
        load_crystal("kaolinite_1M"), (3.0, 40.0), Instrument(), r_march_dollase=1.0
    )


@pytest.mark.skipif(
    not Path(__file__).resolve().parent.parent.joinpath("structures").is_dir(),
    reason="needs the licensed clay structures",
)
def test_an_entry_taken_from_the_structure_scores_perfectly():
    """A comparison with a known right answer, so the scoring can be trusted.

    The stand-in entry is the structure's own strongest lines, rounded as a file
    would round them.  Anything other than a near-exact match would mean the
    comparison is measuring something else.
    """
    _, calculated = calculated_kaolinite()
    wavelength = Instrument().emission.principal_wavelength
    spacings = wavelength / (2.0 * np.sin(np.radians(calculated[0] / 2.0)))
    order = np.argsort(-calculated[1])[:8]
    entry = PeakList(
        name="Kaolinite (stand-in)",
        d=np.round(spacings[order], 3),
        intensity=np.round(100.0 * calculated[1][order]),
    )
    result = compare_with_calculated(entry, calculated, Instrument())
    assert result.matched == result.total == 8
    assert result.worst_deviation < 0.02
    assert result.intensity_agreement > 0.98
    # The other half of the comparison: what the entry has no line for.
    assert result.unexplained_calculated, "the structure has more lines than the entry"


def test_a_line_too_far_off_is_left_unmatched():
    entry = PeakList(name="X", d=[7.15, 5.00], intensity=[100.0, 50.0])
    # A calculated list holding only the first of them.
    angles = np.array([12.36])
    result = compare_with_calculated(entry, (angles, np.array([1.0])), Instrument())
    assert result.matched == 1
    assert result.total == 2
    unmatched = [line for line in result.lines if line[2] is None]
    assert len(unmatched) == 1 and unmatched[0][0] == 5.00


def test_an_entry_with_the_wrong_spacing_is_scored_as_such():
    """The case that matters: an 18 A entry against a 16.9 A model.

    A montmorillonite entry for a different solvation state sits over a degree
    away at low angle, which is what the deviation is there to show.
    """
    ours = PeakList(name="glycol smectite", d=[16.86], intensity=[100.0])
    theirs = PeakList(name="Montmorillonite-18A", d=[18.0], intensity=[100.0])
    wavelength = Instrument().emission.principal_wavelength
    calculated = (ours.two_theta(wavelength), np.array([1.0]))
    result = compare_with_calculated(theirs, calculated, Instrument(), tolerance=2.0)
    assert result.matched == 1
    assert abs(result.lines[0][4]) > 0.3, "over a third of a degree apart at 5 deg"


def test_the_agreement_falls_when_the_intensities_disagree():
    wavelength = Instrument().emission.principal_wavelength
    entry = PeakList(name="X", d=[7.15, 3.58], intensity=[100.0, 60.0])
    angles = entry.two_theta(wavelength)
    same = compare_with_calculated(entry, (angles, np.array([1.0, 0.6])), Instrument())
    swapped = compare_with_calculated(entry, (angles, np.array([0.6, 1.0])), Instrument())
    assert same.intensity_agreement > 0.98
    assert swapped.intensity_agreement < same.intensity_agreement
    assert "matched" in same.summary()


def test_an_empty_calculated_list_matches_nothing():
    entry = PeakList(name="X", d=[7.15], intensity=[100.0])
    result = compare_with_calculated(entry, (np.array([]), np.array([])), Instrument())
    assert result.matched == 0
    assert result.intensity_agreement == 0.0


# --- Reference cards as a powder-file front end exports them -----------------

CARD = """Name and formula

Reference code:\t 00-052-1044

Mineral name:\t Chlorite-serpentine
Compound name:\t Magnesium Aluminum Silicate Hydroxide

Crystallographic parameters

Crystal system:\t Hexagonal

a (Å):\t 5.3400
b (Å):\t 5.3400
c (Å):\t 14.1090
Alpha (°):\t 90.0000
Beta (°):\t 90.0000
Gamma (°):\t 120.0000

Measured density (g/cm3):\t -1.00

Subfiles and quality

Quality:\t Indexed (I)

Comments

Sample Source or Locality:\t Only 00l reflections are recorded.

Peak list
No.hkld [Å]2θ [°]I [%]100114.150006.24132.020027.0400012.563100.030034.7000018.86620.0
Stick Pattern
"""


def _card(tmp_path, text=CARD, name="card.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_a_plain_text_card(tmp_path):
    entry = peaklist.read_highscore_card(_card(tmp_path))
    assert entry.name == "Chlorite-serpentine"
    assert entry.source == "ICDD PDF 00-052-1044"
    assert entry.d.size == 3
    assert entry.d[0] == pytest.approx(14.15)
    assert entry.intensity.tolist() == [32.0, 100.0, 20.0]


def test_a_card_gives_its_cell_and_can_be_oriented(tmp_path):
    entry = peaklist.read_highscore_card(_card(tmp_path))
    assert entry.cell == pytest.approx((5.34, 5.34, 14.109, 90.0, 90.0, 120.0))
    assert entry.can_orient


def test_a_card_keeps_the_header_fields_that_decide_what_it_means(tmp_path):
    entry = peaklist.read_highscore_card(_card(tmp_path))
    # The quality mark and the locality note are what say this entry is basal
    # only and indexed rather than refined; a reader of a fit needs both.
    assert entry.metadata["quality"] == "Indexed (I)"
    assert "00l" in entry.metadata["sample source or locality"]


def test_a_cards_negative_density_is_not_read_as_a_cell_parameter(tmp_path):
    # A card writes an unknown quantity as -1.00, which must not become a cell.
    entry = peaklist.read_highscore_card(_card(tmp_path))
    assert entry.cell is not None and all(value > 0.0 for value in entry.cell)


def test_indices_come_back_as_the_spacing_says_not_as_the_digits_suggest(tmp_path):
    # 0010 is 0 0 10 and is also 0 1 0 with a leading zero; only the cell says
    # which, and this table's own spacing of 1.41200 A settles it at 0 0 10.
    text = CARD.replace(
        "100114.150006.24132.020027.0400012.563100.030034.7000018.86620.0",
        "100114.150006.24132.0200101.4120066.123100.0",
    )
    entry = peaklist.read_highscore_card(_card(tmp_path, text))
    assert entry.hkl is not None
    assert entry.hkl[1].tolist() == [0.0, 0.0, 10.0]


def test_the_index_column_does_not_lose_a_zero_to_the_spacing(tmp_path):
    # 020 and d = 4.45298 read equally well as 02 and d = 04.45298 on the
    # digits alone.  Only the rule that a formatter writes no leading zero
    # keeps three indices in the index column.
    text = CARD.replace(
        "100114.150006.24132.020027.0400012.563100.030034.7000018.86620.0",
        "10204.4529819.923100.0",
    ).replace("c (Å):\t 14.1090", "c (Å):\t 7.2600").replace(
        "a (Å):\t 5.3400", "a (Å):\t 5.1400"
    ).replace("b (Å):\t 5.3400", "b (Å):\t 8.9100").replace(
        "Gamma (°):\t 120.0000", "Gamma (°):\t 90.0000"
    )
    entry = peaklist.read_highscore_card(_card(tmp_path, text))
    assert entry.hkl is not None
    assert entry.hkl[0].tolist() == [0.0, 2.0, 0.0]
    assert entry.d[0] == pytest.approx(4.45298)


def test_the_intensity_column_is_not_robbed_of_its_leading_digit(tmp_path):
    # Reading 12.5991 degrees and 00.0 % instead of 12.599 and 100.0 % agrees
    # with Bragg's law quite as well, and better; the strongest line being
    # 100 % is what rules it out.
    entry = peaklist.read_highscore_card(_card(tmp_path))
    assert entry.intensity.max() == pytest.approx(100.0)


def test_a_card_whose_table_has_no_hundred_per_cent_line_is_refused(tmp_path):
    text = CARD.replace("12.563100.0", "12.56390.0")
    with pytest.raises(ValueError, match="100 %"):
        peaklist.read_highscore_card(_card(tmp_path, text))


def test_a_card_with_no_peak_table_says_so(tmp_path):
    text = CARD.split("Peak list")[0]
    with pytest.raises(ValueError, match="no peak table"):
        peaklist.read_highscore_card(_card(tmp_path, text))


def test_a_card_whose_table_breaks_names_the_line(tmp_path):
    text = CARD.replace("30034.7000018.86620.0", "30034.7000018.866")
    with pytest.raises(ValueError, match="line 3"):
        peaklist.read_highscore_card(_card(tmp_path, text))


def test_rtf_is_flattened_to_the_text_a_reader_sees(tmp_path):
    body = CARD.replace("\n", r"\par ").replace("\t", r"\tab ")
    rtf = r"{\rtf1\deff0{\fonttbl{\f0 Calibri;}}\pard\plain\fs22 " + body + "}"
    entry = peaklist.read_highscore_card(_card(tmp_path, rtf, "card.rtf"))
    assert entry.name == "Chlorite-serpentine"
    assert entry.d.size == 3


def test_repeated_card_labels_are_joined_rather_than_overwritten(tmp_path):
    text = CARD.replace(
        "Quality:\t Indexed (I)",
        "Color:\t White\nQuality:\t Indexed (I)\nColor:\t green",
    )
    entry = peaklist.read_highscore_card(_card(tmp_path, text))
    assert entry.metadata["color"] == "White; green"


def test_a_card_round_trips_through_the_plain_text_format(tmp_path):
    entry = peaklist.read_highscore_card(_card(tmp_path))
    written = peaklist.write_peak_lists([entry], tmp_path / "peaks.txt")
    again = peaklist.read_peak_lists(written)
    assert len(again) == 1
    assert again[0].name == entry.name
    assert again[0].source == entry.source
    assert again[0].d == pytest.approx(entry.d)
    assert again[0].intensity == pytest.approx(entry.intensity)
    assert again[0].cell == pytest.approx(entry.cell)
    assert again[0].hkl == pytest.approx(entry.hkl)


def test_the_written_file_warns_that_it_is_licensed_data(tmp_path):
    entry = peaklist.read_highscore_card(_card(tmp_path))
    written = peaklist.write_peak_lists([entry], tmp_path / "peaks.txt")
    assert "licensed" in written.read_text()


def test_a_card_reports_how_well_its_own_indexing_holds(tmp_path):
    entry = peaklist.read_highscore_card(_card(tmp_path))
    # 14.150, 7.040 and 4.700 A against c = 14.1090: a rational series to a few
    # parts per thousand, which is what a basal-only entry should look like.
    assert entry.metadata["indexing_worst_error"] < 0.01
