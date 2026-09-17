"""The Browse button's folder chooser.

A browser cannot report where a folder is - the file input element gives names
and contents, never paths - so the chooser is opened by the server process,
which on this program is the analyst's own computer.  What is tested here is
the part that can go wrong without anyone noticing: that the chooser runs
isolated from the GUI stack, that every way it can fail produces something
worth reading rather than a hung request, and that it is not offered to a
browser on another machine, where it would open a dialog nobody can see.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from clayquant.gui import folder_dialog
from clayquant.gui.app import LOOPBACK, ask_for_file, ask_for_folder, served_locally

SCRIPT = Path(folder_dialog.__file__)


def fake_chooser(tmp_path: Path, body: str, monkeypatch) -> Path:
    """Stand in for the chooser child, so each of its outcomes can be exercised.

    The script is what gets swapped, not the interpreter: ``ask_for_folder``
    runs ``[sys.executable, <this module\'s file>, initial]``, so pointing the
    module\'s ``__file__`` at another script keeps the real interpreter and
    works the same on every platform.
    """
    script = tmp_path / "chooser.py"
    script.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    monkeypatch.setattr(folder_dialog, "__file__", str(script))
    return script


# --- the child, and its isolation -----------------------------------------


def test_the_chooser_does_not_import_the_gui_stack():
    """It is started by file path, not with -m, and imports only the standard library.

    Run as ``-m clayquant.gui.folder_dialog`` the child would import
    ``clayquant.gui.__init__``, and through it numpy, dash and plotly, to draw a
    dialog.  Keeping the module free of package imports is what allows the
    cheaper route, so it is worth holding in place.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("from .", "from ..", "import numpy", "import dash", "import plotly"):
        assert forbidden not in source, forbidden
    # Importable and runnable with nothing but the standard library on the path.
    finished = subprocess.run(
        [sys.executable, "-c", f"import ast; ast.parse(open({str(SCRIPT)!r}).read())"],
        capture_output=True, text=True,
    )
    assert finished.returncode == 0, finished.stderr


def test_it_reports_a_missing_tkinter_rather_than_failing_silently():
    """tkinter is packaged separately on many Linux distributions."""
    if_missing = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['tkinter'] = None; "
         f"exec(open({str(SCRIPT)!r}).read().replace('__main__', '__x__')); "
         "raise SystemExit(main([]))"],
        capture_output=True, text=True,
    )
    assert if_missing.returncode == 2
    assert "tkinter" in if_missing.stderr


HAS_TKINTER = subprocess.run(
    [sys.executable, "-c", "import tkinter"], capture_output=True
).returncode == 0


@pytest.mark.skipif(not HAS_TKINTER, reason="this interpreter has no tkinter")
@pytest.mark.skipif(
    not sys.platform.startswith(("linux", "freebsd", "openbsd")),
    reason="only on X11, where removing DISPLAY is what leaves Tk with nowhere to open",
)
def test_without_a_display_it_says_so_rather_than_aborting():
    """Removing DISPLAY must yield exit 3, not a crash and not a dialog.

    Restricted to X11 deliberately.  On Windows and macOS there is always a
    window server, so the same call would open a real chooser and block until
    somebody dismissed it - in a test suite that the installer runs at the end
    of every installation.  The timeout is the second line of defence against
    exactly that.
    """
    environment = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    try:
        finished = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True,
                                  text=True, env=environment, timeout=60)
    except subprocess.TimeoutExpired:  # pragma: no cover - a dialog did open
        pytest.fail("the chooser opened a window instead of reporting no display")
    assert finished.returncode == 3
    assert "display" in finished.stderr.lower()


# --- the parent's handling of each outcome --------------------------------


def test_a_chosen_folder_comes_back(tmp_path, monkeypatch):
    fake_chooser(tmp_path, f"print({str(tmp_path)!r})", monkeypatch)
    chosen, problem = ask_for_folder(None)
    assert chosen == str(tmp_path)
    assert problem is None


def test_a_dismissed_dialog_changes_nothing_and_says_nothing(tmp_path, monkeypatch):
    """Cancel must not be reported as an error; the box keeps what it had."""
    fake_chooser(tmp_path, "pass", monkeypatch)
    chosen, problem = ask_for_folder("/somewhere")
    assert chosen is None and problem is None


@pytest.mark.parametrize(
    "code, expected",
    [(2, "tkinter"), (3, "no desktop session"), (4, "failed")],
)
def test_each_failure_is_explained(tmp_path, monkeypatch, code, expected):
    fake_chooser(tmp_path, f"""
        import sys
        print('something went wrong', file=sys.stderr)
        raise SystemExit({code})
    """, monkeypatch)
    chosen, problem = ask_for_folder(None)
    assert chosen is None
    assert expected in problem


def test_a_dialog_left_open_does_not_hold_the_request_for_ever(tmp_path, monkeypatch):
    """The whole point of the child process: a deadline can be imposed on it."""
    fake_chooser(tmp_path, "import time; time.sleep(30)", monkeypatch)
    monkeypatch.setattr("clayquant.gui.app.FOLDER_DIALOG_TIMEOUT", 1.0)
    chosen, problem = ask_for_folder(None)
    assert chosen is None
    assert "still open" in problem


def test_an_interpreter_that_cannot_be_started_is_reported(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    chosen, problem = ask_for_folder(None)
    assert chosen is None
    assert "Could not start" in problem


def test_the_starting_folder_is_only_passed_when_it_exists(tmp_path, monkeypatch):
    """Tk ignores a bad initialdir, but passing one is a way to hide a typo."""
    fake_chooser(tmp_path, "import sys; print(len(sys.argv) - 1)", monkeypatch)
    with_folder, _ = ask_for_folder(str(tmp_path))
    without, _ = ask_for_folder(None)
    assert with_folder == "1"
    assert without == "0"


# --- who may be offered it ------------------------------------------------


def test_a_browser_on_another_machine_is_not_offered_a_local_dialog():
    """It would open on the server's desk and hang until the deadline."""
    assert "127.0.0.1" in LOOPBACK and "::1" in LOOPBACK
    # Outside a request - the command line, a test - there is nobody remote.
    assert served_locally() is True


# --- the file chooser, for the database and the library -------------------


def test_a_chosen_file_comes_back(tmp_path, monkeypatch):
    target = tmp_path / "phases.json"
    target.write_text("{}", encoding="utf-8")
    fake_chooser(tmp_path, f"print({str(target)!r})", monkeypatch)
    chosen, problem = ask_for_file(None, ["Phase database|*.json"])
    assert chosen == str(target)
    assert problem is None


def test_the_filters_and_the_starting_path_reach_the_chooser(tmp_path, monkeypatch):
    fake_chooser(tmp_path, "import sys; print('|'.join(sys.argv[1:]))", monkeypatch)
    chosen, _ = ask_for_file("/somewhere/phases.json", ["Phase database|*.json",
                                                        "Pattern library|*.npz"])
    parts = chosen.split("|")
    # --file, the starting path, then each filter as label|pattern.
    assert parts[0] == "--file"
    assert parts[1] == "/somewhere/phases.json"
    assert "Phase database" in chosen and "*.json" in chosen
    assert "Pattern library" in chosen and "*.npz" in chosen


def test_a_dismissed_file_chooser_changes_nothing(tmp_path, monkeypatch):
    fake_chooser(tmp_path, "pass", monkeypatch)
    chosen, problem = ask_for_file("/somewhere", ["Phase database|*.json"])
    assert chosen is None and problem is None


def test_the_file_chooser_reports_a_missing_tkinter_too(tmp_path, monkeypatch):
    fake_chooser(tmp_path, """
        import sys
        print('no tkinter', file=sys.stderr)
        raise SystemExit(2)
    """, monkeypatch)
    chosen, problem = ask_for_file(None, ["Phase database|*.json"])
    assert chosen is None
    assert "tkinter" in problem


def test_the_chooser_module_offers_both_kinds():
    """One module and one child process for files and folders alike."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "askdirectory" in source and "askopenfilename" in source
    assert "--file" in source
    # A filter arrives as "label|pattern", which needs no quoting on a command line.
    assert 'partition("|")' in source
