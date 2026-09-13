"""Session state for the ClayQuant GUI.

The GUI is a local single-user application, so the loaded patterns and the
corrections applied to them live in one server-side object rather than being
serialised into the browser.  That keeps full-resolution scans and a 200-pattern
library out of every callback round trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..background import BackgroundFit, BackgroundModel
from ..io import SUPPORTED_SUFFIXES, read_pattern
from ..library import PatternLibrary
from ..pattern import Pattern

__all__ = ["MountState", "SessionState", "STATE", "MOUNTS"]

MOUNTS = ("air", "glycol", "heated")
"""The three oriented mounts: air-dried, glycol-solvated and heated."""

MOUNT_LABELS = {
    "air": "Air-dried",
    "glycol": "Ethylene glycol",
    "heated": "Heated (500 C)",
}


@dataclass
class MountState:
    """One loaded mount and the corrections applied to it."""

    raw: Pattern | None = None
    zero_error: float = 0.0
    background_model: BackgroundModel | None = None
    background_fit: BackgroundFit | None = None

    @property
    def is_loaded(self) -> bool:
        return self.raw is not None

    def corrected(self) -> Pattern | None:
        """The pattern with the zero error removed."""
        if self.raw is None:
            return None
        if self.zero_error == 0.0:
            return self.raw
        from ..calibration import apply_zero_error

        return apply_zero_error(self.raw, self.zero_error)

    def subtracted(self) -> Pattern | None:
        """The pattern with the zero error removed and the background subtracted."""
        pattern = self.corrected()
        if pattern is None:
            return None
        if self.background_fit is None:
            return pattern
        return Pattern(
            two_theta=pattern.two_theta,
            intensity=self.background_fit.subtract(pattern.two_theta, pattern.intensity),
            name=f"{pattern.name} (background subtracted)",
            metadata=pattern.metadata,
        )


@dataclass
class SessionState:
    """Everything the GUI is working on."""

    mounts: dict[str, MountState] = field(
        default_factory=lambda: {name: MountState() for name in MOUNTS}
    )
    directory: Path | None = None
    library: PatternLibrary | None = None
    library_path: Path | None = None
    phase_database: dict = field(default_factory=dict)
    phase_database_path: Path | None = None
    detected: list = field(default_factory=list)
    selected_main: list[str] = field(default_factory=list)
    fit_result: object | None = None
    quantification: object | None = None

    def load_phase_database(self, path: str | Path) -> int:
        """Load the accompanying-mineral phase database, returning its size."""
        from ..bern import load_phase_database

        self.phase_database = load_phase_database(path)
        self.phase_database_path = Path(path)
        return len(self.phase_database)

    def list_files(self, directory: str | Path) -> list[str]:
        """Supported data files in ``directory``, sorted by name.

        The path is interpreted forgivingly - quotes are stripped and a Windows
        path is translated when running on Linux - and a failure explains what
        was tried rather than echoing the path back.
        """
        from ..io import describe_path_problem, resolve_user_path

        path = resolve_user_path(directory)
        if not path.is_dir():
            raise NotADirectoryError(describe_path_problem(directory, path))
        self.directory = path

        present: list[str] = []
        other: set[str] = set()
        for entry in path.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix.lower() in SUPPORTED_SUFFIXES:
                present.append(entry.name)
            elif entry.suffix:
                other.add(entry.suffix.lower())

        if not present:
            detail = (
                f" It holds {len(other)} other kind(s) of file ({', '.join(sorted(other))})."
                if other
                else " It holds no files with an extension at all."
            )
            raise FileNotFoundError(
                f"No readable diffraction files in {path}.{detail} "
                f"ClayQuant reads {', '.join(SUPPORTED_SUFFIXES)}. "
                f"If your data is in another format, export it as two-column .xy "
                f"from the instrument software."
            )
        return sorted(present)

    def load(self, mount: str, filename: str) -> Pattern:
        """Load ``filename`` from the current directory into ``mount``."""
        if mount not in self.mounts:
            raise KeyError(f"unknown mount {mount!r}; expected one of {MOUNTS}")
        if self.directory is None:
            raise ValueError("no data directory has been selected")
        pattern = read_pattern(self.directory / filename)
        self.mounts[mount] = MountState(raw=pattern)
        return pattern

    def loaded_mounts(self) -> list[str]:
        return [name for name in MOUNTS if self.mounts[name].is_loaded]

    def reset(self) -> None:
        self.mounts = {name: MountState() for name in MOUNTS}


STATE = SessionState()
"""The single session, shared by all callbacks."""
