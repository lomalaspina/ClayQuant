"""Reading measured diffractograms.

Supported without extra dependencies:

* Two- or three-column text (``.xy``, ``.xye``, ``.txt``, ``.dat``, ``.csv``,
  ``.asc``, ``.prn``), with or without header lines, whitespace/comma/semicolon
  separated and with either decimal separator convention.
* Bruker ``.uxd`` / ``.dql`` ASCII exports, including multi-``_RANGE_`` files.
* Bruker ``.brml`` (a ZIP container of XML), the DIFFRAC.EVA/SUITE format.
* PANalytical ``.xrdml``.

Bruker's binary ``.raw`` is read on a best-effort basis for the RAW1.01 and
RAW4 layouts.  If a file is not recognised, export it as ``.xy`` from the vendor
software; the error message says so.
"""

from __future__ import annotations

import os
import re
import struct
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path

import numpy as np

from .pattern import Pattern

__all__ = [
    "read_pattern",
    "read_text_pattern",
    "resolve_user_path",
    "describe_path_problem",
    "running_under_wsl",
    "SUPPORTED_SUFFIXES",
]

# --------------------------------------------------------------------------- #
# Paths typed by a user
#
# ClayQuant is very often run inside WSL while the measurements sit on the
# Windows side, and a path copied from Explorer - C:\Users\...\samples\TM -
# means nothing to Python running under Linux, where that drive is mounted at
# /mnt/c. Rejecting such a path with "no such directory" is technically true and
# completely unhelpful, so it is translated instead, and where translation
# cannot help, the reason is explained.
# --------------------------------------------------------------------------- #

_WINDOWS_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$", re.S)
_WSL_UNC = re.compile(r"^\\\\wsl(?:\$|\.localhost)\\[^\\]+\\(.*)$", re.S)


def running_under_wsl() -> bool:
    """Whether this process is running in the Windows Subsystem for Linux."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def resolve_user_path(text: str | Path) -> Path:
    r"""Interpret a path a user typed, as forgivingly as is safe.

    Handles surrounding quotes (Explorer's "Copy as path" supplies them), a
    leading ``~``, and Windows paths given to a program running on Linux: a
    drive letter is mapped onto its mount point, so ``C:\Users\x`` becomes
    ``/mnt/c/Users/x``, and a ``\\wsl$\Distro\home\x`` share becomes
    ``/home/x``. On Windows the path is returned unchanged.
    """
    if text is None:
        raise ValueError("no path given")
    cleaned = str(text).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        raise ValueError("no path given")

    if os.name != "nt":
        unc = _WSL_UNC.match(cleaned)
        if unc:
            return Path("/" + unc.group(1).replace("\\", "/")).expanduser()
        drive = _WINDOWS_DRIVE.match(cleaned)
        if drive:
            letter, rest = drive.group(1).lower(), drive.group(2).replace("\\", "/")
            for root in (Path("/mnt"), Path("/media"), Path("/")):
                candidate = root / letter / rest
                if candidate.exists():
                    return candidate
            # Nothing matched; return the usual WSL location so that the error
            # message names the path the user most likely meant.
            return Path("/mnt") / letter / rest
    return Path(cleaned).expanduser()


def describe_path_problem(text: str | Path, resolved: Path) -> str:
    """A sentence explaining why ``resolved`` could not be used, for a user.

    Written for someone who typed a path into a box, not for a developer: it
    names what was tried and what to type instead.
    """
    original = str(text).strip().strip("\"'")
    parts = [f"{original!r} is not a folder I can open."]

    if os.name != "nt" and _WINDOWS_DRIVE.match(original):
        parts.append(
            "That is a Windows path, and ClayQuant is running on Linux"
            + (" (inside WSL)" if running_under_wsl() else "")
            + f", where it would be {resolved}."
        )
        mount = Path("/mnt") / _WINDOWS_DRIVE.match(original).group(1).lower()
        if not mount.exists():
            parts.append(
                f"{mount} does not exist, so that drive is not mounted here. "
                f"In WSL the Windows drives normally appear under /mnt; check with 'ls /mnt'."
            )
        else:
            parts.append(
                f"'{mount}' exists, so the drive is mounted, but the rest of the path does not - "
                f"check the folder names."
            )
        parts.append(
            "Windows paths are translated automatically, so either form may be typed; "
            "the Linux form is the one being looked for."
        )
    elif not resolved.exists():
        parent = resolved.parent
        if parent.exists():
            try:
                nearby = sorted(p.name for p in parent.iterdir() if p.is_dir())[:8]
            except OSError:
                nearby = []
            if nearby:
                parts.append(f"{parent} exists and contains: {', '.join(nearby)}")
        else:
            parts.append(f"Neither it nor its parent {parent} exists.")
    elif resolved.is_file():
        parts.append("That is a file, not a folder. Give the folder that contains it.")
    return " ".join(parts)


SUPPORTED_SUFFIXES = (
    ".xy",
    ".xye",
    ".txt",
    ".dat",
    ".csv",
    ".asc",
    ".prn",
    ".uxd",
    ".dql",
    ".brml",
    ".xrdml",
    ".raw",
)

_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def read_pattern(path: str | Path, name: str = "") -> Pattern:
    """Read a measured pattern, dispatching on the file extension."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".brml":
        two_theta, intensity, metadata = _read_brml(path)
    elif suffix == ".xrdml":
        two_theta, intensity, metadata = _read_xrdml(path)
    elif suffix == ".raw":
        two_theta, intensity, metadata = _read_bruker_raw(path)
    else:
        two_theta, intensity, metadata = _read_text(path)

    order = np.argsort(two_theta)
    return Pattern(
        two_theta=two_theta[order],
        intensity=intensity[order],
        name=name or path.stem,
        metadata={**metadata, "path": str(path), "format": suffix.lstrip(".")},
    )


def read_text_pattern(path: str | Path, name: str = "") -> Pattern:
    """Read a text diffractogram (also used as the fallback for unknown formats)."""
    return read_pattern(path, name=name)


# --------------------------------------------------------------------------- #
# Text formats
# --------------------------------------------------------------------------- #


def _numbers(line: str) -> list[float]:
    """Numbers on a line, tolerating comma decimal separators."""
    candidate = line.strip()
    if not candidate:
        return []
    # Comma as decimal separator (e.g. "5,020  1234") but not as a column
    # separator ("5.020,1234"): treat commas surrounded by digits as decimal
    # points only when no dot is present anywhere on the line.
    if "." not in candidate and re.search(r"\d,\d", candidate):
        candidate = re.sub(r"(?<=\d),(?=\d)", ".", candidate)
    candidate = candidate.replace(";", " ").replace(",", " ").replace("\t", " ")
    return [float(token) for token in _NUMBER_RE.findall(candidate)]


def _read_text(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    if re.search(r"^\s*_(RANGE|2THETA|START|STEPSIZE|COUNTS)", text, re.MULTILINE | re.IGNORECASE):
        return _read_uxd(lines)

    rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped[0] in "#!*'/":
            continue
        values = _numbers(stripped)
        if len(values) >= 2:
            rows.append(values)
        elif len(values) == 1 and rows:
            # A single trailing column of counts after a header block.
            rows.append(values)

    if not rows:
        raise ValueError(
            f"{path.name}: no numeric data found. Export the scan as a two-column "
            f"'.xy' file (2theta and intensity) and try again."
        )

    widths = {len(row) for row in rows}
    if widths == {1}:
        raise ValueError(
            f"{path.name}: only one column of numbers was found, so the 2theta scale is "
            f"unknown. Export as two columns (2theta and intensity)."
        )
    common = max(widths, key=lambda width: sum(1 for row in rows if len(row) == width))
    table = np.array([row[:common] for row in rows if len(row) >= common], dtype=float)
    return table[:, 0], table[:, 1], {}


def _read_uxd(lines: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bruker UXD/DQL ASCII: keyword header blocks followed by counts."""
    two_theta: list[float] = []
    intensity: list[float] = []
    metadata: dict = {}
    start = step = None
    position = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("_") or stripped.startswith(";"):
            key, _, value = stripped.lstrip("_;").partition("=")
            key = key.strip().upper()
            value = value.strip()
            if key in {"START", "2THETA"} and value:
                start = float(_numbers(value)[0])
                position = start
            elif key in {"STEPSIZE", "STEP_SIZE"} and value:
                step = float(_numbers(value)[0])
            elif key == "RANGE":
                position = start
            elif value:
                metadata.setdefault(key.lower(), value)
            continue
        values = _numbers(stripped)
        if not values:
            continue
        if len(values) >= 2 and step is None:
            two_theta.extend(values[0::2])
            intensity.extend(values[1::2])
            continue
        if start is None or step is None:
            raise ValueError("UXD file gives counts but no _START/_STEPSIZE header")
        if position is None:
            position = start
        for value in values:
            two_theta.append(position)
            intensity.append(value)
            position += step

    if not two_theta:
        raise ValueError("UXD file contained no data points")
    return np.asarray(two_theta), np.asarray(intensity), metadata


# --------------------------------------------------------------------------- #
# XML formats
# --------------------------------------------------------------------------- #


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_brml(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            entry
            for entry in archive.namelist()
            if entry.lower().endswith(".xml") and "rawdata" in entry.lower()
        ]
        if not candidates:
            candidates = [entry for entry in archive.namelist() if entry.lower().endswith(".xml")]
        if not candidates:
            raise ValueError(f"{path.name}: BRML archive contains no XML")
        for entry in sorted(candidates):
            root = ElementTree.fromstring(archive.read(entry))
            result = _parse_brml_root(root)
            if result is not None:
                return result
    raise ValueError(
        f"{path.name}: could not locate a scan inside the BRML archive. "
        f"Export the scan as '.xy' instead."
    )


def _parse_brml_root(root) -> tuple[np.ndarray, np.ndarray, dict] | None:
    counts: list[float] = []
    for element in root.iter():
        if _localname(element.tag) == "Datum" and element.text:
            values = _numbers(element.text)
            if values:
                counts.append(values[-1])
    if not counts:
        return None

    start = stop = None
    for element in root.iter():
        if _localname(element.tag) != "ScanAxisInfo":
            continue
        if (element.get("AxisName") or "").lower() not in {"twotheta", "two theta", "2theta"}:
            continue
        for child in element:
            tag = _localname(child.tag)
            value = child.get("Value")
            if value is None:
                continue
            if tag == "Start":
                start = float(value)
            elif tag == "Stop":
                stop = float(value)
    if start is None or stop is None:
        for element in root.iter():
            if _localname(element.tag) == "Start" and element.get("Value"):
                start = float(element.get("Value"))
            elif _localname(element.tag) == "Stop" and element.get("Value"):
                stop = float(element.get("Value"))
    if start is None or stop is None:
        return None

    two_theta = np.linspace(start, stop, len(counts))
    wavelength = None
    for element in root.iter():
        if _localname(element.tag) in {"WaveLengthAverage", "WaveLengthKalpha1"}:
            value = element.get("Value")
            if value:
                wavelength = float(value)
                break
    metadata = {"wavelength": wavelength} if wavelength else {}
    return two_theta, np.asarray(counts, dtype=float), metadata


def _read_xrdml(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a PANalytical XRDML scan.

    Intensities appear as ``<counts>`` or ``<intensities>`` depending on the
    software version, and a measurement may hold several appended ``<scan>``
    elements, which are concatenated.  The instrument configuration carried in
    the file (wavelengths, goniometer radius, divergence slit) is returned as
    metadata, so a calculation can be set up to match the measurement.
    """
    root = ElementTree.fromstring(path.read_text(encoding="utf-8-sig", errors="replace"))

    two_theta_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray] = []
    for scan in (element for element in root.iter() if _localname(element.tag) == "scan"):
        counts: list[float] = []
        start = stop = None
        positions: list[float] = []
        for element in scan.iter():
            tag = _localname(element.tag)
            if tag in {"counts", "intensities"} and element.text:
                counts = _numbers(element.text)
            elif tag == "positions" and (element.get("axis") or "").lower() == "2theta":
                for child in element:
                    child_tag = _localname(child.tag)
                    if child_tag == "startPosition" and child.text:
                        start = float(child.text)
                    elif child_tag == "endPosition" and child.text:
                        stop = float(child.text)
                    elif child_tag == "listPositions" and child.text:
                        positions = _numbers(child.text)
        if not counts:
            continue
        if positions and len(positions) == len(counts):
            two_theta_parts.append(np.asarray(positions, dtype=float))
        elif start is not None and stop is not None:
            two_theta_parts.append(np.linspace(start, stop, len(counts)))
        else:
            continue
        intensity_parts.append(np.asarray(counts, dtype=float))

    if not two_theta_parts:
        raise ValueError(
            f"{path.name}: could not read a 2theta scan from the XRDML file. "
            f"Expected a <scan> containing <dataPoints> with <counts> or <intensities> "
            f"and a 2Theta <positions> range."
        )

    metadata: dict = {}
    for element in root.iter():
        tag = _localname(element.tag)
        text = (element.text or "").strip()
        if tag == "kAlpha1" and text:
            metadata["wavelength"] = float(text)
            metadata["k_alpha1"] = float(text)
        elif tag == "kAlpha2" and text:
            metadata["k_alpha2"] = float(text)
        elif tag == "ratioKAlpha2KAlpha1" and text:
            metadata["k_alpha2_ratio"] = float(text)
        elif tag == "radius" and text:
            metadata.setdefault("goniometer_radius", float(text))
        elif tag == "anodeMaterial" and text:
            metadata["anode"] = text
        elif tag == "commonCountingTime" and text:
            metadata["counting_time"] = float(text)
        elif tag == "id" and text and "sample_id" not in metadata:
            metadata["sample_id"] = text
    for element in root.iter():
        if _localname(element.tag) != "divergenceSlit":
            continue
        for child in element:
            if _localname(child.tag) == "angle" and (child.text or "").strip():
                metadata["divergence_slit"] = float(child.text)
                break
        if "divergence_slit" in metadata:
            break

    return (
        np.concatenate(two_theta_parts),
        np.concatenate(intensity_parts),
        metadata,
    )


# --------------------------------------------------------------------------- #
# Bruker binary RAW (best effort)
# --------------------------------------------------------------------------- #


def _read_bruker_raw(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    blob = path.read_bytes()
    if blob[:7] == b"RAW1.01":
        return _read_raw1(blob)
    if blob[:4] == b"RAW4":
        return _read_raw4(blob)
    raise ValueError(
        f"{path.name}: unrecognised Bruker RAW variant (header {blob[:8]!r}). "
        f"Binary RAW layouts are version specific; please export the scan as '.xy' "
        f"(two columns: 2theta and intensity) from DIFFRAC.EVA and load that."
    )


def _read_raw1(blob: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    n_points = struct.unpack_from("<I", blob, 0x00B4)[0]
    start = struct.unpack_from("<d", blob, 0x00DC)[0]
    step = struct.unpack_from("<d", blob, 0x00E4)[0]
    offset = 0x02C0
    if n_points == 0 or step == 0.0:
        raise ValueError("RAW1.01 header did not contain a usable scan range")
    counts = np.frombuffer(blob, dtype="<f4", count=n_points, offset=offset).astype(float)
    return start + step * np.arange(n_points), counts, {}


def _read_raw4(blob: bytes) -> tuple[np.ndarray, np.ndarray, dict]:
    raise ValueError(
        "Bruker RAW4 files are not supported. Export the scan as '.xy' "
        "(two columns: 2theta and intensity) from DIFFRAC.EVA and load that."
    )
