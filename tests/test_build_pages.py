"""The GitHub Pages entry point, and that it stays in step with the manual.

The failure this guards against is quiet: a mis-split manual publishes with its
stylesheet in the body and renders as an unstyled wall of text, which nobody
sees until someone else opens the site.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_pages.py"


def load():
    spec = importlib.util.spec_from_file_location("build_pages", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_pages():
    return load()


def test_the_published_page_is_current(build_pages):
    """``docs/index.html`` is generated and committed, so it can fall behind."""
    assert build_pages.TARGET.exists(), "run python scripts/build_pages.py"
    assert build_pages.TARGET.read_text(encoding="utf-8") == build_pages.build(), (
        "docs/index.html is out of date; run python scripts/build_pages.py"
    )


def test_it_is_a_standalone_document(build_pages):
    """Without a doctype a browser renders in quirks mode and the layout shifts."""
    page = build_pages.build()
    assert page.startswith("<!doctype html>")
    for required in ('<html lang="en">', "<head>", "</head>", "<body>", "</body>", "</html>",
                     '<meta charset="utf-8">', 'name="viewport"'):
        assert required in page, required


def test_the_styles_and_title_are_in_the_head(build_pages):
    page = build_pages.build()
    head = page[: page.index("<body>")]
    body = page[page.index("<body>"):]
    assert "<style>" in head and "<title>" in head
    assert "<style>" not in body
    assert "<nav id=" in body


def test_nothing_of_the_manual_is_dropped(build_pages):
    source = build_pages.SOURCE.read_text(encoding="utf-8")
    page = build_pages.build()
    for tag in ("h2", "h3", "h4", "table", "tr", "td", "div", "p"):
        pattern = r"<%s\b" % tag
        assert len(re.findall(pattern, page)) == len(re.findall(pattern, source)), tag


def test_no_internal_link_dangles(build_pages):
    page = build_pages.build()
    anchors = set(re.findall(r'id="([^"]+)"', page))
    assert not {h for h in re.findall(r'href="#([^"]+)"', page) if h not in anchors}


def test_a_manual_that_carries_its_own_skeleton_is_refused(build_pages, tmp_path, monkeypatch):
    """The Artifact service adds a skeleton too; two would be a broken document."""
    source = tmp_path / "manual.html"
    source.write_text("<!doctype html><html><head></head><body>x</body></html>", encoding="utf-8")
    monkeypatch.setattr(build_pages, "SOURCE", source)
    with pytest.raises(SystemExit, match="must"):
        build_pages.build()


def test_a_manual_it_cannot_split_is_refused(build_pages, tmp_path, monkeypatch):
    source = tmp_path / "manual.html"
    source.write_text("<title>t</title><style>x</style><p>no navigation rail</p>", encoding="utf-8")
    monkeypatch.setattr(build_pages, "SOURCE", source)
    with pytest.raises(SystemExit, match="head and body"):
        build_pages.build()


def test_a_split_that_loses_the_styles_is_refused(build_pages, tmp_path, monkeypatch):
    source = tmp_path / "manual.html"
    source.write_text("<nav id='rail'></nav><style>x</style>", encoding="utf-8")
    monkeypatch.setattr(build_pages, "SOURCE", source)
    with pytest.raises(SystemExit, match="no title or styles"):
        build_pages.build()


def test_pages_serves_an_index_and_keeps_jekyll_away():
    assert (ROOT / "docs" / "index.html").is_file(), (
        "GitHub Pages serves a directory's index.html and nothing else"
    )
    assert (ROOT / "docs" / ".nojekyll").exists()


def test_the_windows_entry_points_exist_and_bypass_the_execution_policy():
    """Windows refuses a .ps1 by default; a .cmd is exempt, so that is the door in.

    Checked here rather than left to the documentation because the failure is
    total - the installer does not run at all - and because the CRLF endings a
    batch file needs are the kind of thing a later edit silently undoes.
    """
    for name, script in (("install.cmd", "install.ps1"),
                         ("clayquant.cmd", "clayquant.ps1")):
        path = ROOT / name
        assert path.is_file(), name
        raw = path.read_bytes()
        assert raw.count(b"\r\n") > 0 and raw.count(b"\n") == raw.count(b"\r\n"), (
            f"{name} must use CRLF endings: cmd.exe can mis-parse a label, and so "
            "fail a goto, in a batch file that ends its lines with LF alone"
        )
        text = raw.decode("ascii")
        assert "-ExecutionPolicy Bypass" in text
        assert f'"%~dp0{script}"' in text
        assert "%*" in text, f"{name} must pass its arguments through"


def test_no_powershell_probe_passes_double_quotes_through_the_and_operator():
    """The bug that told a machine with Python 3.12 it had no Python 3.10.

    Windows PowerShell hands arguments to a native program through a legacy
    quoting path that strips embedded double quotes, so

        & $exe -c 'import sys; print("%d.%d" % sys.version_info[:2])'

    reaches the interpreter as print(%d.%d % ...) and dies with a SyntaxError.
    PowerShell 7 passes it correctly, so the failure appears only where most
    people run the installer, and it appeared as an installer that could not
    find an interpreter that was on PATH.

    The rule this fixes in place: no argument passed to a native command
    through the call operator may contain a double quote.  Where one is
    unavoidable, drive the process directly and build the command line, which
    is what install.ps1's Invoke-Probe does.
    """
    for name in ("install.ps1", "clayquant.ps1"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if not re.search(r"&\s*\$\w+", code):
                continue
            # Arguments the call operator passes, as single-quoted literals.
            for literal in re.findall(r"'([^']*)'", code):
                assert '"' not in literal, (
                    f"{name}:{number} passes a double quote through the call "
                    f"operator: {literal!r} - Windows PowerShell will strip it"
                )


def test_the_installer_bounds_and_explains_a_failed_probe():
    """A probe that hangs must time out, and a rejection must say why.

    Both were absent, and between them they turned a one-line quoting bug into
    an installer that stopped with the single most misleading message it could
    have produced.
    """
    text = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert "WaitForExit" in text, "a probe must not be able to hang the installation"
    assert "ReadToEndAsync" in text, "read the child's output while waiting, or it deadlocks"
    assert "why each was rejected" in text
    assert "$script:LastProbe" in text
    # PowerShell 7 would otherwise throw past every $LASTEXITCODE check below.
    assert "PSNativeCommandUseErrorActionPreference" in text
