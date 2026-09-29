"""Fit mounts of known minerals and known mass, and calibrate on them.

A relative analysis cannot check itself.  It says how the phases of one mount
divide that mount between them, and it says it whatever the calculated patterns
get wrong, because an error common to the numerator and the denominator
cancels.  Mounts of known single minerals, weighed, break that open twice: the
fit has to name the right mineral, and it has to account for a known number of
milligrams with one constant that serves every mount measured through the same
optics.  Where a phase's mounts sit consistently above or below the rest, that
ratio is the factor by which its calculated pattern misdescribes the real
mineral - the ``k`` of Sec. 2.13, texture and microabsorption and structural
error together.

Two things have to be right for the constants to be comparable at all, and both
are done here rather than left to the caller.

Every clay of one mount is fitted at one orientation.  They settled out of one
suspension onto one plate, so they share a texture, and a fit free to put one
clay at ``r = 0.1`` and another at 1 differs by a thousand in the mass behind
the same scattering.

The constants are then compared with that orientation divided out.  A basal
series is enhanced by ``r`` to the power -3, so the mass behind a given measured
intensity goes as ``r`` cubed, and mounts fitted at 0.1 and 0.5 differ by 125
before any mineralogy is considered.  On nine standards this took the spread in
the constant from 393-fold to 15 (Sec. A.43).

The mounts should carry comparable masses.  A film of a few milligrams on a
25 mm disc is already optically thick at low angle, so counts stop rising with
mass - four times the material for seven per cent more signal, on the two
montmorillonite mounts this was written from - and a heavy mount is not
comparable with a light one however well each is fitted.  Matching them to
within about a factor of two is what makes the saturation divide out; 2 to 3 mg
is the band these standards used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


from .background import clayfit_background
from .io import read_pattern, resolve_user_path
from .library import PatternLibrary
from .nnls import orientation_families, select_one_orientation
from .quantification import (
    Calibration,
    WeighedMount,
    agreement_between,
    calibration_from_weighed_mounts,
    quantify,
)

__all__ = [
    "Standard",
    "StandardResult",
    "read_standards",
    "fit_standards",
    "main",
]

DEFAULT_RANGE = (4.0, 38.0)


@dataclass
class Standard:
    """One weighed mount of a known mineral."""

    name: str
    path: Path
    mass_mg: float
    phase: str = ""
    """The library phase this is a standard of, or empty where it is not single
    phase.  A mount that carries something else - a dickite with a sulfate in
    it, a montmorillonite with an illite - puts the impurity into its own factor
    and must not be used to calibrate, which is what leaving this empty says."""

    area_cm2: float | None = None


@dataclass
class StandardResult:
    """What a standard's mount fitted to."""

    standard: Standard
    r_wp: float
    orientation: float | None
    shares: list[tuple[str, float]]
    mount: WeighedMount | None
    note: str = ""

    @property
    def found(self) -> str:
        return self.shares[0][0] if self.shares else ""

    @property
    def identified(self) -> bool:
        """Whether the largest share is the mineral the mount is a standard of."""
        return bool(self.standard.phase) and self.found == self.standard.phase


def read_standards(path: str | Path, root: str | Path | None = None) -> list[Standard]:
    """Read the table of mounts: name, file, mass in mg, phase, area in cm^2.

    A header row is expected and its column names are matched case-insensitively
    on their first word, so ``mass_mg``, ``Mass (mg)`` and ``MASS`` all serve.
    ``phase`` may be left empty for a mount that is not single phase: it is
    still fitted and reported, and only left out of the calibration.

    Blank lines and lines beginning with ``#`` are skipped, because this is a
    table somebody keeps by hand beside a balance and the notes belong with it.
    """
    path = Path(path)
    base = Path(root) if root is not None else path.parent
    out: list[Standard] = []
    with path.open(newline="") as handle:
        rows = [line for line in handle
                if line.strip() and not line.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        fields = {
            (key or "").strip().lower().split("(")[0].strip().rstrip("_"): (value or "").strip()
            for key, value in row.items()
        }
        name = fields.get("name") or fields.get("mount") or fields.get("standard") or ""
        measurement = fields.get("file") or fields.get("path") or ""
        if not name and not measurement:
            continue
        mass = fields.get("mass") or fields.get("mass_mg") or ""
        if not mass:
            raise ValueError(f"{name or measurement!r} has no mass; it cannot be calibrated on")
        measurement = measurement or f"{name}.xrdml"
        candidate = Path(measurement)
        area = fields.get("area") or ""
        out.append(Standard(
            name=name or candidate.stem,
            path=candidate if candidate.is_absolute() else base / candidate,
            mass_mg=float(mass),
            phase=fields.get("phase", ""),
            area_cm2=float(area) if area else None,
        ))
    if not out:
        raise ValueError(f"{path} lists no mounts")
    return out


def fit_standards(
    standards: list[Standard],
    library: PatternLibrary,
    range_two_theta: tuple[float, float] = DEFAULT_RANGE,
    area_cm2: float | None = None,
    minimum_share: float = 3.0,
) -> list[StandardResult]:
    """Fit each mount with one orientation shared by all its clays."""
    families = orientation_families(library)
    results: list[StandardResult] = []
    for standard in standards:
        try:
            pattern = read_pattern(standard.path)
        except FileNotFoundError:
            results.append(StandardResult(standard, float("nan"), None, [], None,
                                          note=f"{standard.path} not found"))
            continue
        background = clayfit_background(pattern.two_theta, pattern.intensity)
        try:
            selection = select_one_orientation(
                pattern, library, families, background=background,
                range_two_theta=range_two_theta)
        except Exception as exc:                      # noqa: BLE001 - reported, not hidden
            results.append(StandardResult(standard, float("nan"), None, [], None,
                                          note=f"the fit failed: {exc}"))
            continue
        result = selection.result
        quantification = quantify(result)
        shares = sorted(
            ((share.phase, float(share.clay_weight or share.weight))
             for share in quantification.shares
             if (share.clay_weight or share.weight) > minimum_share),
            key=lambda row: -row[1],
        )
        area = standard.area_cm2 if standard.area_cm2 is not None else area_cm2
        try:
            mount = WeighedMount.from_fit(
                result, standard.mass_mg * 1e-3, area,
                orientation=selection.orientation)
            mount.measurement = standard.name
        except ValueError as exc:
            mount, note = None, str(exc)
        else:
            note = ""
        results.append(StandardResult(standard, float(result.r_wp),
                                      selection.orientation, shares, mount, note))
    return results


def _report(results: list[StandardResult], calibration: Calibration | None) -> str:
    lines = ["%-22s %6s %6s %5s %13s %13s  %s"
             % ("mount", "mg", "Rwp", "r", "constant", "on one r", "largest shares")]
    for row in results:
        if row.mount is None:
            lines.append("%-22s %6.1f  %s" % (row.standard.name, row.standard.mass_mg,
                                              row.note or "not fitted"))
            continue
        shares = ", ".join("%s %.0f%%" % (phase, weight) for phase, weight in row.shares[:3])
        mark = ""
        if row.standard.phase:
            mark = " ok" if row.identified else " <- expected " + row.standard.phase
        lines.append("%-22s %6.1f %6.3f %5s %13.4e %13.4e  %s%s"
                     % (row.standard.name, row.standard.mass_mg, row.r_wp,
                        ("%.2f" % row.orientation) if row.orientation else "-",
                        row.mount.constant, row.mount.comparable_constant, shares, mark))

    mounts = [row.mount for row in results if row.mount is not None]
    if len(mounts) > 1:
        report = agreement_between(mounts)
        lines.append("")
        lines.append("One constant has to serve every mount.  On one orientation they "
                     "spread %.1f-fold" % report["ratio_max_to_min"])
        lines.append("(before dividing the orientation out, %.1f-fold)."
                     % _raw_spread(mounts))
        for who, factor in sorted(report["departures"].items(), key=lambda kv: -kv[1]):
            lines.append("   %-22s x%.2f" % (who, factor))

    if calibration is not None:
        lines.append("")
        lines.append(calibration.source)
        lines.append("Read these as ratios: a factor above 1 is a phase whose calculated")
        lines.append("pattern delivers less intensity per gram than the others', so its")
        lines.append("weight percent needs raising by that much.")
        for phase, factor in sorted(calibration.factors.items(), key=lambda kv: -kv[1]):
            lines.append("   k(%-14s) = %.3f" % (phase, factor))
    return "\n".join(lines)


def _raw_spread(mounts: list[WeighedMount]) -> float:
    values = [mount.constant for mount in mounts if mount.constant > 0.0]
    return max(values) / min(values) if values and min(values) > 0.0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    """Command line entry point for ``clayquant-standards``."""
    parser = argparse.ArgumentParser(
        prog="clayquant-standards",
        description=(
            "Fit weighed mounts of known minerals and measure the per-phase "
            "calibration factors they imply."
        ),
        epilog=(
            "The table is a CSV with columns name, file, mass (mg), phase and "
            "optionally area (cm2).  Leave phase empty for a mount that is not "
            "single phase: it is fitted and reported but left out of the "
            "calibration, because an impure standard puts its impurity into its "
            "own factor."
        ),
    )
    parser.add_argument("standards", type=Path,
                        help="CSV table of the weighed mounts")
    parser.add_argument("-l", "--library", type=Path,
                        default=Path("library/clayquant_library.npz"))
    parser.add_argument("--root", type=Path, default=None,
                        help="directory the table's file paths are relative to "
                             "(default: the table's own directory)")
    parser.add_argument("--area", type=float, default=None,
                        help="deposited area in cm2 for mounts that do not give "
                             "their own; a 25 mm disc is 4.91")
    parser.add_argument("--range", type=float, nargs=2, default=list(DEFAULT_RANGE),
                        metavar=("START", "STOP"), help="2theta range to fit over")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="write the calibration factors here as JSON")
    arguments = parser.parse_args(argv)

    try:
        standards = read_standards(arguments.standards, arguments.root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    library_path = resolve_user_path(arguments.library)
    if not Path(library_path).is_file():
        parser.error(
            f"{arguments.library} not found. Build one first:\n"
            "  clayquant-build-library --measurement <a scan from your instrument> "
            "--specimen-length 25 --specimen-shape round"
        )
    library = PatternLibrary.load(library_path)

    results = fit_standards(standards, library, tuple(arguments.range), arguments.area)
    labelled = [(row.mount, row.standard.phase) for row in results
                if row.mount is not None and row.standard.phase]
    calibration = None
    if labelled:
        try:
            calibration = calibration_from_weighed_mounts(labelled)
        except ValueError as exc:
            print(f"no calibration: {exc}")
    print(_report(results, calibration))

    if arguments.out and calibration is not None:
        import json

        from .quantification import InstrumentConstant

        # The per-phase counts per gram go out alongside the k factors, because
        # they are what lets an unweighed mount be corrected at all: a fit gives
        # a relative mass, this turns it into grams, and grams are what the film
        # absorption needs.  Only mounts that were weighed contribute one.
        weighed = [
            (item.mount, item.standard.phase)
            for item in results
            if item.mount is not None and getattr(item.standard, "phase", "")
        ]
        constants = InstrumentConstant.from_weighed_mounts(
            weighed, source=calibration.source
        ) if weighed else None

        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source": calibration.source, "factors": calibration.factors}
        if constants is not None:
            payload["counts_per_gram"] = constants.per_phase
        with arguments.out.open("w") as handle:
            json.dump(payload, handle, indent=1)
        print(f"\nwritten to {arguments.out}")
        if constants is not None:
            print("  counts per gram (at r = 1), for inferring an unweighed mount's mass:")
            for phase, value in sorted(constants.per_phase.items()):
                print(f"    {phase:<16s} {value:.4g}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
