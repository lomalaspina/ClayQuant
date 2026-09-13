"""Desktop shortcuts: the parts that can be checked on any platform.

Writing a Windows shortcut needs Windows and an application bundle needs macOS,
so what is tested here is everything up to that point - which command the
shortcut runs, what the freedesktop entry contains, that an icon file is turned
into something each platform can read, and that the Linux path writes and
removes the files it says it does.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

from clayquant import desktop


def test_the_command_uses_this_environment():
    """A shortcut must not depend on anything being on PATH or activated."""
    command = desktop.launch_command()
    assert command
    first = Path(command[0])
    assert first.is_absolute()
    if len(command) == 1:
        assert first.name.startswith("clayquant-gui")
        assert first.parent == Path(sys.executable).parent
    else:
        assert first == Path(sys.executable)
        assert command[1:] == ["-m", "clayquant.gui.app"]


def test_the_entry_is_a_valid_desktop_file():
    text = desktop.desktop_entry(["/opt/env/bin/clayquant-gui"], None)
    lines = text.splitlines()
    assert lines[0] == "[Desktop Entry]"
    fields = dict(line.split("=", 1) for line in lines[1:] if "=" in line)
    assert fields["Type"] == "Application"
    assert fields["Name"] == "ClayQuant"
    assert fields["Exec"] == "/opt/env/bin/clayquant-gui"
    # Without this the desktop opens a terminal window and leaves it open.
    assert fields["Terminal"] == "false"
    assert "Icon" not in fields
    assert text.endswith("\n")


def test_a_path_with_spaces_is_quoted():
    text = desktop.desktop_entry(["/home/a b/env/bin/clayquant-gui"], None)
    assert 'Exec="/home/a b/env/bin/clayquant-gui"' in text


def test_the_icon_is_named_when_there_is_one(tmp_path):
    text = desktop.desktop_entry(["/x/clayquant-gui"], tmp_path / "clayquant.png")
    assert f"Icon={tmp_path / 'clayquant.png'}" in text


def sample_png(path: Path, size: int = 64) -> Path:
    pytest.importorskip("PIL", reason="Pillow is not installed")
    from PIL import Image

    Image.new("RGBA", (size, size), (150, 105, 66, 255)).save(path)
    return path


def test_an_ico_is_written_that_windows_can_read(tmp_path):
    png = sample_png(tmp_path / "icon.png", size=256)
    ico = desktop.png_to_ico(png, tmp_path / "icon.ico")
    data = ico.read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind) == (0, 1)   # 0 reserved, 1 = icon rather than cursor
    assert count >= 1


def test_the_ico_fallback_needs_no_pillow(tmp_path, monkeypatch):
    """Without Pillow the PNG is wrapped by hand rather than skipped."""
    png = sample_png(tmp_path / "icon.png", size=256)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def without_pillow(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no Pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_pillow)
    ico = desktop.png_to_ico(png, tmp_path / "hand.ico")
    data = ico.read_bytes()
    assert struct.unpack("<HHH", data[:6]) == (0, 1, 1)
    width, height, _, _, planes, bits, length, offset = struct.unpack("<BBBBHHII", data[6:22])
    assert (width, height) == (0, 0)    # 0 encodes 256
    assert (planes, bits) == (1, 32)
    assert data[offset:offset + length] == png.read_bytes()


def test_a_file_that_is_not_a_png_is_refused(tmp_path):
    fake = tmp_path / "not.png"
    fake.write_bytes(b"GIF89a" + b"\x00" * 40)
    with pytest.raises(ValueError, match="not a PNG"):
        desktop._png_size(fake.read_bytes())


@pytest.mark.skipif(sys.platform != "linux", reason="freedesktop paths are Linux")
def test_install_and_remove_on_linux(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("XDG_DESKTOP_DIR", raising=False)

    written = desktop.install_shortcuts()
    names = {path.name for path in written}
    assert names == {"clayquant.desktop"}
    assert len(written) == 2, "one for the menu and one for the desktop"
    for path in written:
        assert path.is_file()
        assert path.read_text().startswith("[Desktop Entry]")
        assert path.stat().st_mode & 0o111, "must be executable to be trusted"

    removed = desktop.remove_shortcuts()
    assert set(removed) == set(written)
    assert not any(path.exists() for path in written)


@pytest.mark.skipif(sys.platform != "linux", reason="freedesktop paths are Linux")
def test_the_desktop_folder_can_be_named_in_another_language(tmp_path, monkeypatch):
    """A German desktop is ~/Schreibtisch, and user-dirs.dirs says so."""
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    (home / "Schreibtisch").mkdir()
    (home / ".config" / "user-dirs.dirs").write_text(
        'XDG_DESKTOP_DIR="$HOME/Schreibtisch"\n', encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("XDG_DESKTOP_DIR", raising=False)
    assert desktop._desktop_directory() == home / "Schreibtisch"
