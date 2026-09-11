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

import re
import struct
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path

import numpy as np

from .pattern import Pattern

__all__ = ["read_pattern", "read_text_pattern", "SUPPORTED_SUFFIXES"]

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
    root = ElementTree.fromstring(path.read_text(errors="replace"))
    counts: list[float] = []
    start = stop = None
    for element in root.iter():
        tag = _localname(element.tag)
        if tag == "intensities" and element.text:
            counts = _numbers(element.text)
        elif tag == "positions" and (element.get("axis") or "").lower() == "2theta":
            for child in element:
                if _localname(child.tag) == "startPosition" and child.text:
                    start = float(child.text)
                elif _localname(child.tag) == "endPosition" and child.text:
                    stop = float(child.text)
    if not counts or start is None or stop is None:
        raise ValueError(f"{path.name}: could not read a 2theta scan from the XRDML file")
    return (
        np.linspace(start, stop, len(counts)),
        np.asarray(counts, dtype=float),
        {},
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
