"""Double-clickable windows for the two setup steps.

Importing a structure library and building the reference library are the two
things that have to happen before the browser interface is any use, and both
were command lines with paths in them.  Typing a Windows path into a terminal is
where this program loses people, so each step gets a window with file pickers
and a Run button, and an icon that opens it.

    python -m clayquant.tools import-structures
    python -m clayquant.tools build-library

or, from a checkout, ``.\\clayquant.cmd import-structures-gui`` and
``.\\clayquant.cmd build-library-gui``; ``clayquant shortcut`` puts both on the
desktop beside the main one.

Why tkinter and not the browser interface
-----------------------------------------
These windows exist to be double-clicked, and a Dash application needs a server
and a browser tab before it can show a file picker - which is the thing being
avoided.  tkinter is in the standard library, opens a window in one call, and is
already a dependency of the folder chooser the Load tab uses
(:mod:`clayquant.gui.folder_dialog`), so nothing new is required.

The work runs in a worker thread and the window polls it.  Building a library
takes minutes, and a window that stops repainting for minutes is a window
people force-quit.
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path

TITLES = {
    "import-structures": "ClayQuant - import a structure library",
    "build-library": "ClayQuant - build the reference library",
}


def _require_tkinter():
    try:
        import tkinter
        from tkinter import filedialog, ttk
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise SystemExit(
            "This Python has no tkinter, so these windows cannot open.\n"
            "Use the command line instead:\n"
            "    clayquant import-structures <file.xml> -o structures/phases.json\n"
            "    clayquant build-library -o library/clays.npz\n"
            f"({exc})"
        ) from exc
    return tkinter, filedialog, ttk


class Runner:
    """A worker thread whose output the window can drain without blocking."""

    def __init__(self) -> None:
        self.lines: queue.Queue[str] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.failed = False
        self.done = False

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, work) -> None:
        self.failed = False
        self.done = False

        def wrapper() -> None:
            # Everything the job prints goes to the window rather than to a
            # console that a double-clicked program does not have.
            class Tee:
                def write(inner, text: str) -> int:
                    if text:
                        self.lines.put(text)
                    return len(text)

                def flush(inner) -> None:
                    return None

            previous_out, previous_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = Tee()
            try:
                work()
            except Exception:  # noqa: BLE001 - reported in the window
                self.failed = True
                self.lines.put("\n" + traceback.format_exc())
            finally:
                sys.stdout, sys.stderr = previous_out, previous_err
                self.done = True

        self.thread = threading.Thread(target=wrapper, daemon=True)
        self.thread.start()

    def drain(self) -> str:
        chunks = []
        while True:
            try:
                chunks.append(self.lines.get_nowait())
            except queue.Empty:
                break
        return "".join(chunks)



def import_structures_job(values: dict[str, str], say=print) -> dict:
    """Import one or two structure libraries into a phase database.

    Separated from the window so that it can be tested without a display, which
    matters here: the window needs tkinter and the work needs numpy, and an
    interpreter with both is not guaranteed.
    """
    from .bern import write_phase_database

    sources = [values.get("source", "").strip()]
    extra = values.get("source2", "").strip()
    if extra:
        sources.append(extra)
    output = values.get("output", "").strip()
    if not sources[0]:
        raise ValueError("Choose a structure library to import.")
    if not output:
        raise ValueError("Choose where to write the database.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    report = write_phase_database(sources, output, verbose=True)
    say(f"\nWrote {report['written']} phases to {output}"
        f" ({report['clay']} phyllosilicates, {report['failed']} could not be read).")
    if report["disagreeing"]:
        say(f"{report['disagreeing']} of them do not match the mass TOPAS states for them, "
            "listed above - worth looking at before relying on them.")
    return report


def build_library_job(values: dict[str, str], say=print):
    """Calculate the reference library, optionally from a refinement's structures."""
    from .bern import refined_clay_structures
    from .library import build_library
    from .models import use_refined_structures

    output = values.get("output", "").strip()
    if not output:
        raise ValueError("Choose where to write the library.")
    refinement = values.get("refinement", "").strip()
    if refinement:
        skipped: list[str] = []
        structures = refined_clay_structures(refinement, skipped=skipped)
        if not structures:
            raise ValueError(
                f"{refinement} holds no clay structure the library uses. A refinement that "
                "models its clays as hkl_Is peaks phases has no structure to take."
                + ("\n" + "\n".join(f"  {note}" for note in skipped) if skipped else "")
            )
        use_refined_structures(structures)
        say("Clay structures taken from the refinement:")
        for key, crystal in sorted(structures.items()):
            say(f"  {key}: {crystal.name}")
        for note in skipped:
            say(f"  skipped: {note}")
    instrument = None
    measurement = values.get("measurement", "").strip()
    if measurement:
        from .background import BackgroundModel
        from .io import read_pattern, resolve_user_path
        from .library import instrument_from_measurement

        pattern = read_pattern(resolve_user_path(measurement))
        # The width is measured at the half maximum of the peaks, so it is read
        # off the pattern with its background gone; taken with the background
        # still there the half maximum sits too high up the peak and the width
        # comes out too narrow.
        fit = BackgroundModel(chebyshev_degree=4, inverse=True, inverse_offset=1.0).fit(
            pattern.two_theta, pattern.intensity, snip_window=4.0)
        instrument, note = instrument_from_measurement(pattern, background=fit)
        say(f"Instrument taken from {Path(measurement).name}:")
        say(f"  {note}")
    else:
        say("No scan given, so the library is calculated for a 280 mm goniometer radius,")
        say("a 0.5 deg divergence slit and a generic peak width. If that is not your")
        say("instrument, run this again with a scan from it: the reference peaks will")
        say("otherwise be the wrong width for every fit made against this library.")
    say("Calculating; this takes a few minutes.")
    library = build_library(instrument=instrument)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    library.save(output)
    say(library.describe())
    say(f"\nWrote {len(library.entries)} patterns to {output}")
    return library


JOBS = {
    "import-structures": import_structures_job,
    "build-library": build_library_job,
}


def _panel(step: str):
    """Build the window for one step and return it."""
    tkinter, filedialog, ttk = _require_tkinter()
    from .desktop import project_root

    root = tkinter.Tk()
    root.title(TITLES[step])
    root.geometry("760x520")

    fields: dict[str, tkinter.StringVar] = {}

    def row(parent, label: str, key: str, default: str, kind: str, patterns=None) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=12, pady=4)
        ttk.Label(frame, text=label, width=22).pack(side="left")
        variable = tkinter.StringVar(value=default)
        fields[key] = variable
        ttk.Entry(frame, textvariable=variable).pack(side="left", fill="x", expand=True)

        def browse() -> None:
            if kind == "open":
                chosen = filedialog.askopenfilename(
                    title=label, filetypes=(patterns or []) + [("All files", "*")]
                )
            elif kind == "save":
                chosen = filedialog.asksaveasfilename(
                    title=label, filetypes=(patterns or []) + [("All files", "*")]
                )
            else:
                chosen = filedialog.askdirectory(title=label)
            if chosen:
                variable.set(str(Path(chosen)))

        ttk.Button(frame, text="Browse…", command=browse, width=10).pack(side="left", padx=6)

    body = ttk.Frame(root)
    body.pack(fill="x", pady=(10, 0))

    if step == "import-structures":
        ttk.Label(
            body,
            text=("Convert a TOPAS or jEdit structure library into the phase database\n"
                  "the main mineral search uses."),
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 6))
        row(body, "Structure library", "source", "",
            "open", [("Structure library", "*.xml"), ("TOPAS include", "*.inc")])
        row(body, "Second library (optional)", "source2", "", "open",
            [("Structure library", "*.xml")])
        row(body, "Write the database to", "output",
            str(project_root() / "structures" / "phases.json"),
            "save", [("Phase database", "*.json")])
    else:
        ttk.Label(
            body,
            text=("Calculate the reference pattern library the fit chooses from.\n"
                  "This takes a few minutes and only has to be done once per instrument\n"
                  "setting - not once per sample."),
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 6))
        row(body, "Write the library to", "output",
            str(project_root() / "library" / "clays.npz"),
            "save", [("Pattern library", "*.npz")])
        row(body, "A scan from your instrument", "measurement", "", "open",
            [("Diffraction data", "*.xrdml *.xy *.raw *.dat *.txt"),
             ("XRDML", "*.xrdml"), ("All files", "*.*")])
        ttk.Label(
            body,
            text=("Any scan measured on the instrument and setting the samples will be.\n"
                  "It is read for the goniometer radius and the divergence slit, which the\n"
                  "file records, and its peaks are measured for the width - and nothing\n"
                  "else: no data from it goes into the library. Without it the library is\n"
                  "calculated for a 280 mm radius, a 0.5 deg slit and a generic width, and\n"
                  "if that is not your instrument the reference peaks are the wrong width\n"
                  "for every fit you do with it."),
            justify="left",
            foreground="#555555",
        ).pack(anchor="w", padx=12, pady=(0, 6))
        row(body, "Refinement (optional)", "refinement", "", "open",
            [("TOPAS output", "*.out")])

    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=12, pady=8)
    run_button = ttk.Button(controls, text="Run")
    run_button.pack(side="left")
    status = ttk.Label(controls, text="")
    status.pack(side="left", padx=10)

    log = tkinter.Text(root, wrap="word", height=18)
    log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def say(text: str) -> None:
        log.insert("end", text)
        log.see("end")

    runner = Runner()

    def start() -> None:
        if runner.running:
            return
        log.delete("1.0", "end")
        run_button.state(["disabled"])
        status.configure(text="working…")
        values = {key: variable.get() for key, variable in fields.items()}
        runner.start(lambda: JOBS[step](values))
        root.after(100, poll)

    def poll() -> None:
        say(runner.drain())
        if runner.running:
            root.after(150, poll)
            return
        say(runner.drain())
        run_button.state(["!disabled"])
        status.configure(text="failed - see below" if runner.failed else "done")

    run_button.configure(command=start)
    return root


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    step = argv[0] if argv else ""
    if step not in TITLES:
        print("usage: python -m clayquant.tools {import-structures|build-library}")
        return 2
    _panel(step).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
