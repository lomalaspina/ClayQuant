"""Turning a fit into a quantification table.

The fit returns a scale factor per library pattern.  This module groups those
into phases, splits them into clay minerals and accompanying ("main") minerals,
and reports two bases:

*all phases* - every fitted phase's share of the pattern;
*clay basis* - the same clay phases renormalised to 100% with the accompanying
minerals taken out, which is what a clay mineral assemblage is normally quoted
on.  Quartz is an accompanying mineral here even though it is also the
calibration standard.

Two measures are given for each phase, because they answer different questions
and neither is a weight percent:

``scattering`` - the phase's share of the calculated integrated intensity.  This
is what the diffractogram actually measures.

``amplitude`` - its share of the fitted scale factors.  Broad interstratified
patterns carry far more integrated area per unit peak height than a sharp
discrete phase, so the two differ, sometimes by a factor of several.

Converting either into weight percent needs a reference intensity ratio per
phase - the calculated intensity per unit mass in the same geometry - and for an
oriented mount it also depends on how the mount was prepared.  Pass calibrated
ratios to :meth:`Quantification.weight_percent` if you have them; otherwise read
the numbers as relative.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .bern import is_clay_phase
from .nnls import FitResult

__all__ = ["PhaseShare", "Quantification", "quantify", "CLAY_LIBRARY_PHASES"]

CLAY_LIBRARY_PHASES = frozenset(
    {"illite", "chlorite", "kaolinite_1M", "kaolinite_2M", "smectite_EG", "I/S", "C/S"}
)
"""Library phase keys that are clay minerals by construction."""


def _is_clay(phase: str) -> bool:
    return phase in CLAY_LIBRARY_PHASES or is_clay_phase(phase)


@dataclass
class PhaseShare:
    """One phase's contribution to a fit."""

    phase: str
    is_clay: bool
    coefficient: float
    scattering: float
    amplitude: float
    clay_scattering: float = 0.0
    clay_amplitude: float = 0.0
    entries: list[str] = field(default_factory=list)

    @property
    def group(self) -> str:
        return "clay" if self.is_clay else "main"


@dataclass
class Quantification:
    """Grouped, normalised result of a fit."""

    shares: list[PhaseShare]
    r_wp: float
    r_p: float
    measurement: str = ""

    @property
    def clays(self) -> list[PhaseShare]:
        return [share for share in self.shares if share.is_clay]

    @property
    def main_minerals(self) -> list[PhaseShare]:
        return [share for share in self.shares if not share.is_clay]

    @property
    def clay_total_scattering(self) -> float:
        """Share of the whole pattern that the clay minerals account for."""
        return sum(share.scattering for share in self.clays)

    def weight_percent(self, reference_intensity_ratios: dict[str, float],
                       clay_basis: bool = False) -> dict[str, float]:
        """Weight fractions in percent from per-phase reference intensity ratios."""
        selected = self.clays if clay_basis else self.shares
        missing = {share.phase for share in selected} - set(reference_intensity_ratios)
        if missing:
            raise KeyError(f"no reference intensity ratio given for {sorted(missing)}")
        scaled = {
            share.phase: share.scattering / reference_intensity_ratios[share.phase]
            for share in selected
            if share.scattering > 0
        }
        total = sum(scaled.values())
        if total <= 0:
            return {phase: 0.0 for phase in scaled}
        return {phase: 100.0 * value / total for phase, value in scaled.items()}

    def table(self, clay_basis: bool = False) -> list[dict]:
        """Rows ready for display or CSV, largest share first."""
        selected = self.clays if clay_basis else self.shares
        rows = []
        for share in selected:
            row = {
                "phase": share.phase,
                "group": share.group,
                "coefficient": share.coefficient,
                "scattering_percent": 100.0
                * (share.clay_scattering if clay_basis else share.scattering),
                "amplitude_percent": 100.0
                * (share.clay_amplitude if clay_basis else share.amplitude),
                "entries": "; ".join(share.entries),
            }
            rows.append(row)
        return sorted(rows, key=lambda row: row["scattering_percent"], reverse=True)

    def to_csv(self, path: str | Path, clay_basis: bool = False) -> Path:
        """Write the table to ``path``.

        The header records what the numbers are and what they are not, so the
        file cannot be mistaken for weight percent later.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.table(clay_basis=clay_basis)
        basis = "clay minerals only, renormalised to 100%" if clay_basis else "all fitted phases"
        with path.open("w", newline="") as handle:
            handle.write(f"# ClayQuant quantification: {basis}\n")
            handle.write(f"# measurement: {self.measurement}\n")
            handle.write(f"# Rwp = {100.0 * self.r_wp:.2f}%, Rp = {100.0 * self.r_p:.2f}%\n")
            if clay_basis:
                handle.write(
                    f"# clay minerals account for "
                    f"{100.0 * self.clay_total_scattering:.2f}% of the whole pattern\n"
                )
            handle.write(
                "# scattering_percent is each phase's share of the calculated integrated "
                "intensity; amplitude_percent is its share of the fitted scale factors. "
                "Neither is a weight percent without a reference intensity ratio per phase.\n"
            )
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "phase",
                    "group",
                    "coefficient",
                    "scattering_percent",
                    "amplitude_percent",
                    "entries",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "coefficient": f"{row['coefficient']:.6g}",
                        "scattering_percent": f"{row['scattering_percent']:.3f}",
                        "amplitude_percent": f"{row['amplitude_percent']:.3f}",
                    }
                )
        return path

    def summary(self) -> str:
        lines = [
            f"Rwp = {100.0 * self.r_wp:.2f}%   Rp = {100.0 * self.r_p:.2f}%",
            "",
            "Clay minerals (renormalised to 100%, accompanying minerals removed):",
        ]
        for row in self.table(clay_basis=True):
            lines.append(
                f"  {row['phase']:<16s} {row['scattering_percent']:6.2f}%"
                f"   (amplitude {row['amplitude_percent']:6.2f}%)"
            )
        lines.append(
            f"  clay minerals are {100.0 * self.clay_total_scattering:.1f}% of the whole pattern"
        )
        if self.main_minerals:
            lines += ["", "Accompanying minerals (of the whole pattern):"]
            for share in sorted(self.main_minerals, key=lambda s: s.scattering, reverse=True):
                lines.append(f"  {share.phase:<16s} {100.0 * share.scattering:6.2f}%")
        lines += [
            "",
            "These are shares of diffracted intensity, not weight percent.",
        ]
        return "\n".join(lines)


def quantify(result: FitResult, clay_phases: set[str] | None = None) -> Quantification:
    """Group a fit by phase and normalise it on both bases.

    Parameters
    ----------
    clay_phases:
        Phase names to treat as clay minerals, overriding the default
        classification.  Use it when a coarse mica should count as an
        accompanying mineral instead, or the other way round.
    """
    totals: dict[str, PhaseShare] = {}
    for name, phase, coefficient, scattering, amplitude in zip(
        result.names,
        result.phases,
        result.coefficients,
        result.scattering_fraction,
        result.amplitude_fraction,
    ):
        if coefficient <= 0.0:
            continue
        clay = phase in clay_phases if clay_phases is not None else _is_clay(phase)
        share = totals.get(phase)
        if share is None:
            share = PhaseShare(
                phase=phase, is_clay=clay, coefficient=0.0, scattering=0.0, amplitude=0.0
            )
            totals[phase] = share
        share.coefficient += float(coefficient)
        share.scattering += float(scattering)
        share.amplitude += float(amplitude)
        share.entries.append(name)

    shares = sorted(totals.values(), key=lambda share: share.scattering, reverse=True)
    clay_scattering = sum(share.scattering for share in shares if share.is_clay)
    clay_amplitude = sum(share.amplitude for share in shares if share.is_clay)
    for share in shares:
        if share.is_clay:
            share.clay_scattering = share.scattering / clay_scattering if clay_scattering else 0.0
            share.clay_amplitude = share.amplitude / clay_amplitude if clay_amplitude else 0.0

    return Quantification(
        shares=shares,
        r_wp=result.r_wp,
        r_p=result.r_p,
        measurement=str(result.metadata.get("measurement", "")),
    )
