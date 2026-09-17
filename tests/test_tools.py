"""The two double-clickable setup windows, and the shortcuts that open them.

The window itself needs tkinter and the work it does needs numpy, and one
interpreter with both is not guaranteed - which is why the jobs are module-level
functions rather than closures inside the window, and why they are what is
tested here.
"""

import sys
import time
from pathlib import Path

import pytest

from clayquant.desktop import ALL_SHORTCUTS, MAIN, TOOLS, Shortcut, desktop_entry
from clayquant.tools import JOBS, TITLES, Runner, build_library_job, import_structures_job

ROOT = Path(__file__).resolve().parent.parent


# --- the worker -----------------------------------------------------------


def wait_for(runner, limit=10.0):
    started = time.monotonic()
    while runner.running and time.monotonic() - started < limit:
        time.sleep(0.02)
    assert not runner.running, "the worker did not finish"


def test_what_the_job_prints_reaches_the_window():
    """A double-clicked program has no console, so print has to go somewhere."""
    runner = Runner()
    runner.start(lambda: print("halfway through"))
    wait_for(runner)
    assert "halfway through" in runner.drain()
    assert runner.failed is False


def test_a_job_that_raises_is_reported_rather_than_lost():
    runner = Runner()
    runner.start(lambda: (_ for _ in ()).throw(ValueError("no such file")))
    wait_for(runner)
    output = runner.drain()
    assert runner.failed is True
    assert "ValueError: no such file" in output


def test_the_worker_leaves_stdout_as_it_found_it():
    """It redirects the real stdout, so failing to restore it would be silent."""
    before = sys.stdout
    runner = Runner()
    runner.start(lambda: print("x"))
    wait_for(runner)
    assert sys.stdout is before


def test_both_steps_have_a_window_and_a_job():
    assert sorted(TITLES) == sorted(JOBS) == ["build-library", "import-structures"]


# --- the jobs -------------------------------------------------------------


@pytest.mark.parametrize(
    "values, message",
    [
        ({}, "Choose a structure library"),
        ({"source": "somewhere.xml"}, "Choose where to write"),
    ],
)
def test_the_import_job_says_what_is_missing(values, message):
    with pytest.raises(ValueError, match=message):
        import_structures_job(values, say=lambda text: None)


def test_the_library_job_says_what_is_missing():
    with pytest.raises(ValueError, match="Choose where to write"):
        build_library_job({}, say=lambda text: None)


def test_the_library_job_refuses_a_refinement_with_no_structures(tmp_path):
    """A refinement modelling its clays as peaks phases has nothing to take."""
    refinement = tmp_path / "peaks_only.out"
    refinement.write_text("xdd nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no clay structure"):
        build_library_job({"output": str(tmp_path / "lib.npz"),
                           "refinement": str(refinement)}, say=lambda text: None)


def test_the_import_job_writes_a_database_and_reports_what_it_found(tmp_path):
    # The same shape of jEdit macro library the importer is tested against in
    # test_bern_import, so this exercises the job and not a format guess.
    lines = [
        "\\tstr",
        '\\t\\tphase_name \\&quot;Quartz\\&quot;',
        '\\t\\tspace_group \\&quot;P3221\\&quot;',
        "\\t\\ta a_Quartz 4.9134",
        "\\t\\tc c_Quartz 5.4052",
        "\\t\\tsite Si1 x 0.4701 y 0 z 0 occ Si+4 1. beq 0.6",
        "\\t\\tsite O1  x 0.4139 y 0.2674 z 0.1188 occ O-2 1. beq 1.0",
    ]
    inserts = "\n".join(
        f'\t\tbuffer.insert(textArea.getCaretPosition(), "\\n{line}");' for line in lines
    )
    library = tmp_path / "structures.xml"
    library.write_text(
        '<?xml version="1.0"?>\n<MODE>\n'
        f'<item name="Quartz" type="macro">\n{inserts}\n\t\t</item>\n</MODE>\n',
        encoding="utf-8",
    )
    output = tmp_path / "out" / "phases.json"
    said: list[str] = []
    report = import_structures_job(
        {"source": str(library), "output": str(output)}, say=said.append
    )
    assert output.is_file()
    assert report["written"] == 1
    assert "Wrote 1 phases" in "".join(said)


# --- the shortcuts --------------------------------------------------------


def test_there_is_an_icon_for_each_step_beside_the_main_one():
    assert MAIN in ALL_SHORTCUTS
    assert len(TOOLS) == 2
    arguments = {shortcut.argument for shortcut in TOOLS}
    assert arguments == {"import-structures", "build-library"}
    assert len({shortcut.slug for shortcut in ALL_SHORTCUTS}) == 3


def test_the_tool_icons_run_a_module_and_not_a_console_script():
    """Console scripts are written at install time and at no other.

    A shortcut naming one would break for anybody who pulled a version that
    added it without reinstalling - which is the failure the installer already
    has to warn about.  A module is there as soon as the source is.
    """
    for shortcut in TOOLS:
        command = shortcut.command()
        assert command[1:3] == ["-m", "clayquant.tools"]
        assert command[0] == str(Path(sys.executable))


def test_each_shortcut_describes_itself(tmp_path):
    for shortcut in ALL_SHORTCUTS:
        text = desktop_entry(shortcut.command(), None, shortcut)
        assert f"Name={shortcut.name}" in text
        assert f"Comment={shortcut.comment}" in text
        assert "Terminal=false" in text


def test_a_shortcut_with_no_module_falls_back_to_the_interface():
    plain = Shortcut(name="X", slug="x", comment="y")
    assert "clayquant" in " ".join(plain.command()).lower()


# --- which found phases arrive ticked -------------------------------------


class Finding:
    """Just enough of a PhaseEvidence for the ticking rule."""

    def __init__(self, name, score, intensity_agreement=0.8):
        self.name = name
        self.score = score
        self.intensity_agreement = intensity_agreement


def test_the_competitive_screen_ticks_by_the_share_asked_for():
    from clayquant.gui.app import phases_to_tick

    findings = [Finding("Albite", 0.045), Finding("Quartz", 0.015), Finding("Rutile", 0.004)]
    assert phases_to_tick(findings, screened=True, tick_above=3.0) == ["Albite"]
    assert phases_to_tick(findings, screened=True, tick_above=1.0) == ["Albite", "Quartz"]
    assert phases_to_tick(findings, screened=True, tick_above=10.0) == []


def test_a_share_without_intensity_agreement_is_not_ticked():
    """The graphite case, from a real run on a clay separate.

    Graphite has two reflections in range and its 002 sits within a tenth of a
    degree of the quartz 101.  It took 8.9 % of the pattern, reported two of two
    lines found and a signal-to-noise of 153, and its intensity agreement was
    zero: the pattern holds nothing where its other line belongs.  Every figure
    on that row reads as confirmation except the one that can contradict them.
    """
    from clayquant.gui.app import phases_to_tick

    findings = [
        Finding("Quartz", 0.05, intensity_agreement=0.54),
        Finding("Graphite", 0.089, intensity_agreement=0.0),
    ]
    assert phases_to_tick(findings, screened=True, tick_above=1.0) == ["Quartz"]


def test_an_overlapped_mineral_is_still_ticked():
    # The threshold has to admit a real mineral whose lines have company; it is
    # only the flat contradiction that is excluded.
    from clayquant.gui.app import phases_to_tick

    findings = [Finding("Rutile", 0.02, intensity_agreement=0.30)]
    assert phases_to_tick(findings, screened=True, tick_above=1.0) == ["Rutile"]


def test_position_matching_ticks_nothing_however_high_it_scores():
    """The failure this prevents, from a real run on a clay separate.

    With no clay library the search falls back to matching peak positions, where
    galena scored 95 %, otavite 99 % and cassiterite 96 % - none of them credible
    in a clay separate, and all of them were arriving pre-ticked, one click from
    entering the fit.  A position-matching score is not a share of the pattern,
    so there is no threshold that would make pre-ticking honest.
    """
    from clayquant.gui.app import phases_to_tick

    findings = [Finding("Galena", 0.95), Finding("Otavite", 0.99), Finding("Corundum", 1.0)]
    for threshold in (0.5, 1.0, 3.0, 10.0):
        assert phases_to_tick(findings, screened=False, tick_above=threshold) == []
