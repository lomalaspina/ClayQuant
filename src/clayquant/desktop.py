"""Desktop shortcuts, so ClayQuant can be started without a terminal.

Someone analysing clay separates should not have to remember to activate a
virtual environment first.  This writes the thing each desktop expects - a
``.desktop`` file on Linux, an application bundle on macOS, a shortcut on
Windows - pointing at the interpreter of the environment ClayQuant is installed
in, so double-clicking it starts the server and opens the default browser.

One implementation rather than three: the installers call this, and so can the
user, with ``./clayquant shortcut``.  The platform differences are confined to
three functions below and the rest - which command to run, which icon to use,
what to report - is shared and testable anywhere.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import struct
import subprocess
import sys
from pathlib import Path

__all__ = [
    "launch_command",
    "icon_source",
    "install_shortcuts",
    "remove_shortcuts",
    "shortcut_locations",
]

APP_NAME = "ClayQuant"
COMMENT = "Quantification of clay mineral assemblages from oriented mounts"


def project_root() -> Path:
    """The checkout, when running from one; otherwise the package directory."""
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], here.parents[1]):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parent


def launch_command() -> list[str]:
    """The command a shortcut should run.

    The installed console script of *this* environment is preferred, because it
    needs nothing on PATH and no activation.  Where it is missing - an
    environment built before the command existed - the module is run through the
    same interpreter, which always works.
    """
    interpreter = Path(sys.executable)
    for name in ("clayquant-gui", "clayquant-gui.exe"):
        candidate = interpreter.with_name(name)
        if candidate.is_file():
            return [str(candidate)]
    return [str(interpreter), "-m", "clayquant.gui.app"]


ICON_SUFFIXES = (".png", ".jpg", ".jpeg", ".ico", ".icns", ".bmp", ".gif", ".webp")
PLACEHOLDER_STEM = "clayquant-placeholder"


def icon_source() -> Path | None:
    """The icon to use, or ``None`` if there is not one.

    Drop an image called ``clayquant`` into ``assets/`` and it is used - any
    common raster format, and any spelling of the name.  That last part matters
    more than it sounds: Linux filesystems are case-sensitive, so a file saved
    as ``ClayQuant.PNG`` is a different file from ``clayquant.png`` and an exact
    match would ignore it and go on using the placeholder without saying so.

    The placeholder shipped with ClayQuant is named apart, so a supplied icon
    always wins and is never overwritten by an update.
    """
    assets = project_root() / "assets"
    if not assets.is_dir():
        return None
    supplied, placeholder = [], []
    for entry in sorted(assets.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in ICON_SUFFIXES:
            continue
        stem = entry.stem.lower()
        if stem == PLACEHOLDER_STEM:
            placeholder.append(entry)
        elif stem == "clayquant":
            supplied.append(entry)
    for group in (supplied, placeholder):
        # A real raster format first, whatever order the directory listed them.
        for suffix in ICON_SUFFIXES:
            for entry in group:
                if entry.suffix.lower() == suffix:
                    return entry
    return None


# --------------------------------------------------------------------------- #
# Icon conversion
# --------------------------------------------------------------------------- #

def png_to_ico(png: Path, ico: Path) -> Path:
    """Wrap a PNG in an ICO container for Windows.

    Windows has read PNG-compressed icon entries since Vista, so the PNG can be
    stored as it is with a header in front of it.  Pillow is used when it is
    installed, because it can write the several smaller sizes Explorer prefers;
    without it the single PNG is wrapped by hand rather than leaving the
    shortcut without an icon.
    """
    ico.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        with Image.open(png) as image:
            image.convert("RGBA").save(
                ico, format="ICO",
                sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
            )
        return ico

    data = png.read_bytes()
    width, height = _png_size(data)
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack(
        "<BBBBHHII",
        0 if width >= 256 else width,    # 0 means 256
        0 if height >= 256 else height,
        0, 0, 1, 32, len(data), len(header) + 16,
    )
    ico.write_bytes(header + entry + data)
    return ico


def _png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def png_to_icns(png: Path, icns: Path) -> Path | None:
    """Build a macOS icon set, or ``None`` if the tools are not there."""
    iconutil = shutil.which("iconutil")
    if iconutil is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    iconset = icns.with_suffix(".iconset")
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    with Image.open(png) as image:
        square = image.convert("RGBA")
        for size in (16, 32, 128, 256, 512):
            square.resize((size, size)).save(iconset / f"icon_{size}x{size}.png")
            square.resize((size * 2, size * 2)).save(iconset / f"icon_{size}x{size}@2x.png")
    subprocess.run([iconutil, "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    shutil.rmtree(iconset)
    return icns


# --------------------------------------------------------------------------- #
# The three platforms
# --------------------------------------------------------------------------- #

def desktop_entry(command: list[str], icon: Path | None) -> str:
    """The contents of a freedesktop ``.desktop`` file."""
    executable = " ".join(_quote(part) for part in command)
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        f"Name={APP_NAME}",
        f"Comment={COMMENT}",
        f"Exec={executable}",
        "Terminal=false",
        "Categories=Science;Education;Physics;",
        "Keywords=XRD;diffraction;clay;mineralogy;",
    ]
    if icon is not None:
        lines.append(f"Icon={icon}")
    return "\n".join(lines) + "\n"


def _quote(part: str) -> str:
    return f'"{part}"' if " " in part else part


def _desktop_directory() -> Path | None:
    """The user's Desktop, whatever it is called in their language."""
    configured = os.environ.get("XDG_DESKTOP_DIR")
    if configured:
        return Path(configured)
    user_dirs = Path.home() / ".config" / "user-dirs.dirs"
    if user_dirs.is_file():
        for line in user_dirs.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("XDG_DESKTOP_DIR="):
                value = line.split("=", 1)[1].strip().strip('"')
                return Path(value.replace("$HOME", str(Path.home())))
    fallback = Path.home() / "Desktop"
    return fallback if fallback.is_dir() else None


def as_png(icon: Path, target: Path) -> Path:
    """Copy or convert ``icon`` to a PNG at ``target``.

    A supplied icon may be a JPEG or a BMP; the desktops want a PNG.  Without
    Pillow it can only be copied, which is right when it is already a PNG and
    the best that can be done when it is not.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if icon.suffix.lower() == ".png":
        shutil.copyfile(icon, target)
        return target
    try:
        from PIL import Image
    except ImportError:
        shutil.copyfile(icon, target)
        return target
    with Image.open(icon) as image:
        image.convert("RGBA").save(target, format="PNG")
    return target


def _install_linux(desktop: bool, menu: bool) -> list[Path]:
    command = launch_command()
    icon = icon_source()
    installed_icon = None
    if icon is not None:
        installed_icon = as_png(
            icon, Path.home() / ".local/share/icons/hicolor/512x512/apps/clayquant.png"
        )

    written: list[Path] = []
    text = desktop_entry(command, installed_icon)
    if menu:
        entry = Path.home() / ".local/share/applications" / "clayquant.desktop"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(text, encoding="utf-8")
        entry.chmod(0o755)
        written.append(entry)
    if desktop:
        directory = _desktop_directory()
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            entry = directory / "clayquant.desktop"
            entry.write_text(text, encoding="utf-8")
            # GNOME will not launch an entry it does not consider trusted; the
            # executable bit is what marks it so.
            entry.chmod(0o755)
            _mark_trusted(entry)
            written.append(entry)
    _refresh_desktop_database()
    return written


def _mark_trusted(entry: Path) -> None:
    gio = shutil.which("gio")
    if gio is None:
        return
    subprocess.run([gio, "set", str(entry), "metadata::trusted", "true"],
                   check=False, capture_output=True)


def _refresh_desktop_database() -> None:
    update = shutil.which("update-desktop-database")
    if update is not None:
        subprocess.run([update, str(Path.home() / ".local/share/applications")],
                       check=False, capture_output=True)


def _install_macos(desktop: bool, menu: bool) -> list[Path]:
    """Write an application bundle; ``menu`` means ~/Applications."""
    command = launch_command()
    bundle = (Path.home() / "Applications" / f"{APP_NAME}.app")
    macos = bundle / "Contents" / "MacOS"
    resources = bundle / "Contents" / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    runner = macos / APP_NAME
    runner.write_text(
        "#!/bin/sh\n"
        f"exec {' '.join(_quote(part) for part in command)}\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": "org.clayquant.app",
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1.0",
        "LSBackgroundOnly": False,
        "NSHighResolutionCapable": True,
    }
    icon = icon_source()
    if icon is not None:
        icns = png_to_icns(icon, resources / "ClayQuant.icns")
        if icns is not None:
            info["CFBundleIconFile"] = icns.name
        else:
            shutil.copyfile(icon, resources / "ClayQuant.png")
    (bundle / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))

    written = [bundle]
    if desktop:
        link = Path.home() / "Desktop" / f"{APP_NAME}.app"
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
        if not link.exists():
            link.symlink_to(bundle)
            written.append(link)
    return written


def _install_windows(desktop: bool, menu: bool) -> list[Path]:
    command = launch_command()
    icon = icon_source()
    icon_file = None
    if icon is not None:
        icon_file = png_to_ico(icon, project_root() / "assets" / "clayquant.ico")

    targets = []
    if desktop:
        targets.append(Path(os.path.expanduser("~")) / "Desktop" / f"{APP_NAME}.lnk")
    if menu:
        targets.append(
            Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
            / "Microsoft/Windows/Start Menu/Programs" / f"{APP_NAME}.lnk"
        )

    written: list[Path] = []
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        arguments = " ".join(_quote(part) for part in command[1:])
        script = [
            "$shell = New-Object -ComObject WScript.Shell",
            f"$link = $shell.CreateShortcut('{target}')",
            f"$link.TargetPath = '{command[0]}'",
            f"$link.Arguments = '{arguments}'",
            f"$link.WorkingDirectory = '{project_root()}'",
            f"$link.Description = '{COMMENT}'",
        ]
        if icon_file is not None:
            script.append(f"$link.IconLocation = '{icon_file}'")
        script.append("$link.Save()")
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "; ".join(script)],
            check=True, capture_output=True,
        )
        written.append(target)
    return written


def shortcut_locations() -> list[Path]:
    """Where a shortcut would be written, without writing one."""
    if sys.platform.startswith("win"):
        return [
            Path(os.path.expanduser("~")) / "Desktop" / f"{APP_NAME}.lnk",
            Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
            / "Microsoft/Windows/Start Menu/Programs" / f"{APP_NAME}.lnk",
        ]
    if sys.platform == "darwin":
        return [Path.home() / "Applications" / f"{APP_NAME}.app",
                Path.home() / "Desktop" / f"{APP_NAME}.app"]
    places = [Path.home() / ".local/share/applications" / "clayquant.desktop"]
    directory = _desktop_directory()
    if directory is not None:
        places.append(directory / "clayquant.desktop")
    return places


def install_shortcuts(desktop: bool = True, menu: bool = True) -> list[Path]:
    """Create the shortcuts, returning what was written."""
    if sys.platform.startswith("win"):
        return _install_windows(desktop, menu)
    if sys.platform == "darwin":
        return _install_macos(desktop, menu)
    return _install_linux(desktop, menu)


def remove_shortcuts() -> list[Path]:
    """Delete any shortcut this module would have created."""
    removed = []
    for path in shortcut_locations():
        if path.is_symlink() or path.is_file():
            path.unlink()
            removed.append(path)
        elif path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``clayquant shortcut``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="clayquant shortcut",
        description="Create or remove desktop shortcuts for ClayQuant.",
    )
    parser.add_argument("--no-desktop", dest="desktop", action="store_false",
                        help="do not put one on the desktop")
    parser.add_argument("--no-menu", dest="menu", action="store_false",
                        help="do not put one in the applications menu")
    parser.add_argument("--remove", action="store_true", help="delete the shortcuts instead")
    parser.add_argument("--where", action="store_true",
                        help="print where they would go and exit")
    arguments = parser.parse_args(argv)

    if arguments.where:
        for path in shortcut_locations():
            print(path)
        return 0
    if arguments.remove:
        removed = remove_shortcuts()
        print(f"removed {len(removed)} shortcut(s)")
        for path in removed:
            print(f"  {path}")
        return 0

    icon = icon_source()
    if icon is None:
        print(f"note: no icon found in {project_root() / 'assets'}; "
              f"the shortcut will use the desktop's default. Put one there named "
              f"clayquant.png and run this again.")
    elif icon.stem.lower() == PLACEHOLDER_STEM:
        print(f"using the placeholder icon ({icon.name}). To use your own, put it in "
              f"{icon.parent} named clayquant.png and run this again.")
    else:
        print(f"icon: {icon}")
    written = install_shortcuts(desktop=arguments.desktop, menu=arguments.menu)
    print(f"created {len(written)} shortcut(s)")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
