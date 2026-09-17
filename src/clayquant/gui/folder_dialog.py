"""A native folder chooser, run in a process of its own.

A browser cannot hand a web page the path of a folder on disk.  The file input
element reports names and contents, never locations, and that is a deliberate
part of the sandbox rather than an oversight; no amount of JavaScript will get
``C:\\Users\\you\\Documents\\HM`` out of it.  What makes a chooser possible here
is that ClayQuant's server is the analyst's own computer: the process answering
the page can open the operating system's own dialog and read the answer
directly.

It runs in a child process, for three reasons that are all about where a GUI
toolkit may live:

* Tk expects to own the main thread.  A Dash callback runs on whichever worker
  thread the request landed on, and on macOS opening a window off the main
  thread is not merely unreliable but forbidden by Cocoa.
* A toolkit that cannot reach a display aborts the process.  In a child, that
  is a non-zero exit code to report; in the server, it is the end of the
  session.
* A child can be given a deadline, so a dialog left open on someone's desk
  cannot hold a request open for ever.

Nothing is passed on the command line but an optional starting folder, and the
answer comes back on stdout - one line, empty if the dialog was dismissed.
"""

from __future__ import annotations

import sys
from pathlib import Path

TITLE = "Select the folder holding the diffraction scans"


def choose_folder(initial: str | None = None) -> str | None:
    """Open the platform's folder chooser and return the chosen path.

    Returns ``None`` if the dialog was cancelled.  Raises ``ImportError`` where
    tkinter is not installed, which is usual on Linux distributions that package
    it separately, and ``RuntimeError`` where there is no display to open it on.
    """
    import tkinter
    from tkinter import filedialog

    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:  # no display, or none this process may use
        raise RuntimeError(str(exc)) from exc

    try:
        # The chooser is opened by the server process, which owns no window the
        # analyst is looking at, so without this it can appear behind the
        # browser and look as though the button did nothing.
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        try:
            root.focus_force()
        except tkinter.TclError:
            pass

        start = initial if initial and Path(initial).is_dir() else None
        chosen = filedialog.askdirectory(title=TITLE, initialdir=start, mustexist=True)
    finally:
        try:
            root.destroy()
        except tkinter.TclError:
            pass

    # Tk returns an empty string on cancel, and forward slashes on every
    # platform including Windows.  Path normalises the separators back.
    if not chosen:
        return None
    return str(Path(chosen))


def main(argv: list[str]) -> int:
    initial = argv[0] if argv else None
    try:
        chosen = choose_folder(initial)
    except ImportError as exc:
        print(f"tkinter is not available: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"no display for a dialog: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 - the parent reports whatever this was
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    if chosen is None:
        return 0
    print(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
