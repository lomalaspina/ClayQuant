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

``weight`` - an estimate in weight percent, computed rather than calibrated.
The fitted coefficient of a library pattern, divided by the normalisation that
pattern was stored at, is proportional to how many of its scattering units are
in the beam; multiplying by the mass of one unit gives a mass, and the constant
of proportionality is the instrument and the specimen, which every phase in one
measurement shares.  This is the relation Rietveld analysis writes as ``W``
proportional to ``S(ZMV)``.

What that assumes is stated where it is reported and in the manual, and one
assumption is worth repeating here: it takes the calculated pattern to describe
what the phase actually contributes, texture included.  In an oriented mount
texture differs between phases and between preparations, and the fit cannot tell
a phase that is more strongly oriented from one that is more abundant.  A
per-phase calibration factor is provided for when that has been measured -
:class:`Calibration` - and is 1 for every phase until it has been.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .bern import is_clay_phase
from .nnls import FitResult

__all__ = [
    "PhaseShare",
    "Quantification",
    "Calibration",
    "quantify",
    "CLAY_LIBRARY_PHASES",
]

CLAY_LIBRARY_PHASES = frozenset(
    {"illite", "chlorite", "kaolinite_1M", "kaolinite_2M", "smectite_EG", "I/S", "C/S"}
)
"""Library phase keys that are clay minerals by construction."""


def _is_clay(phase: str) -> bool:
    return phase in CLAY_LIBRARY_PHASES or is_clay_phase(phase)


@dataclass
class Calibration:
    """Per-phase factors correcting a computed weight percent.

    The computed value assumes the calculated pattern accounts for everything a
    phase contributes.  What it cannot account for in an oriented mount is how
    strongly that particular phase orients in that particular preparation, and
    microabsorption, and any error in the structural model.  All of it lands in
    one factor per phase, ``k``, by which the computed mass is multiplied.

    Every ``k`` is 1 until measured, which is the honest starting point rather
    than a correction invented for the purpose.  To measure them, analyse
    specimens of known composition through the same preparation and solve for
    the ``k`` that reproduces it: :meth:`from_known_composition` does the
    arithmetic for one specimen.
    """

    factors: dict[str, float] = field(default_factory=dict)
    source: str = "uncalibrated (every factor 1)"

    def factor(self, phase: str) -> float:
        return float(self.factors.get(phase, 1.0))

    @classmethod
    def from_known_composition(
        cls, computed: dict[str, float], known: dict[str, float], source: str = ""
    ) -> "Calibration":
        """Solve for the factors that turn ``computed`` percentages into ``known``.

        Both are weight percent by phase.  The result is normalised so the
        factors average 1, since only their ratios matter: a common factor
        cancels when the percentages are renormalised to 100.
        """
        factors = {
            phase: known[phase] / computed[phase]
            for phase in known
            if computed.get(phase, 0.0) > 0.0 and known[phase] > 0.0
        }
        if not factors:
            raise ValueError("no phase appears in both the computed and the known composition")
        mean = sum(factors.values()) / len(factors)
        return cls(
            factors={phase: value / mean for phase, value in factors.items()},
            source=source or "measured against a known composition",
        )

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"source": self.source, "factors": self.factors}, indent=1),
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(factors={str(k): float(v) for k, v in data.get("factors", {}).items()},
                   source=str(data.get("source", str(path))))


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
    mass: float = 0.0
    """Mass of this phase in the specimen, in the arbitrary units of the fit."""

    orientation: float = 0.0
    """The March-Dollase parameter the fit chose, averaged over the phase's entries.

    Worth reading next to the weight percent, because it enters it steeply.  For
    a basal series every reflection is enhanced by ``r`` to the power -3, so a
    phase fitted at ``r = 0.2`` is calculated to scatter 125 times more per gram
    than the same phase at ``r = 1``.  The fit chooses ``r`` to match the shape
    of the pattern, not to measure the texture, so a phase whose ``r`` is far
    from its neighbours' is the first place to look when a weight percent seems
    wrong.
    """

    weight: float = 0.0
    """Weight percent over all fitted phases; 0 when it could not be computed."""

    clay_weight: float = 0.0
    """Weight percent over the clay minerals alone."""

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
    calibration: "Calibration" = field(default_factory=lambda: Calibration())
    weights_available: bool = False
    """Whether a weight percent could be computed at all.

    It cannot when any phase that took a share of the pattern lacks the mass or
    the volume of the unit it was calculated for - a library written before
    these were recorded, or a phase added to the fit without them.  It is all or
    nothing on purpose: weighing some of the phases and renormalising those to
    100% would read as a complete analysis with a mineral missing from it.
    """

    unweighable: list[str] = field(default_factory=list)
    """Entries that took a share of the pattern but could not be weighed."""

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

    def computed_weight_percent(self, clay_basis: bool = False) -> dict[str, float]:
        """Weight percent as calculated, without any measured calibration."""
        selected = self.clays if clay_basis else self.shares
        key = "clay_weight" if clay_basis else "weight"
        return {share.phase: getattr(share, key) for share in selected if share.mass > 0.0}

    def weight_percent(self, reference_intensity_ratios: dict[str, float],
                       clay_basis: bool = False) -> dict[str, float]:
        """Weight fractions in percent from per-phase reference intensity ratios.

        This is the older route, kept for a laboratory that has measured RIRs of
        its own: each phase's share of diffracted intensity is divided by its
        ratio.  :meth:`computed_weight_percent` needs no ratios at all, because
        the calculated patterns carry the mass of what they were computed from.
        """
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
                "weight_percent": (share.clay_weight if clay_basis else share.weight)
                if self.weights_available else "",
                "march_dollase": share.orientation,
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
            if self.weights_available:
                handle.write(
                    "# weight_percent is computed from the fitted scale factors and the mass of "
                    "the unit each pattern was calculated for, as W proportional to S(ZMV). It "
                    "assumes the calculated pattern accounts for everything the phase "
                    "contributes, which in an oriented mount includes how strongly that phase "
                    "orients: the fit cannot tell a well-oriented phase from an abundant one.\n"
                )
                handle.write(f"# calibration: {self.calibration.source}\n")
            else:
                handle.write(
                    "# weight_percent is empty: this library does not record the mass of the "
                    "units its patterns were calculated for. Rebuild it to get weight percent.\n"
                )
            handle.write(
                "# scattering_percent is each phase's share of the calculated integrated "
                "intensity; amplitude_percent is its share of the fitted scale factors. "
                "Neither is a weight percent.\n"
            )
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "phase",
                    "group",
                    "coefficient",
                    "weight_percent",
                    "march_dollase",
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


def quantify(
    result: FitResult,
    clay_phases: set[str] | None = None,
    calibration: "Calibration | None" = None,
) -> Quantification:
    """Group a fit by phase and normalise it on both bases.

    Parameters
    ----------
    clay_phases:
        Phase names to treat as clay minerals, overriding the default
        classification.  Use it when a coarse mica should count as an
        accompanying mineral instead, or the other way round.
    calibration:
        Per-phase factors for the weight percent; see :class:`Calibration`.
        Without one every factor is 1 and the weight percent is as calculated.
    """
    calibration = calibration or Calibration()
    masses = (
        result.relative_mass
        if len(result.relative_mass) == len(result.coefficients)
        else [0.0] * len(result.coefficients)
    )
    orientations = (
        result.march_dollase
        if len(result.march_dollase) == len(result.coefficients)
        else [0.0] * len(result.coefficients)
    )
    weighted_orientation: dict[str, float] = {}
    totals: dict[str, PhaseShare] = {}
    for name, phase, coefficient, scattering, amplitude, mass, orientation in zip(
        result.names,
        result.phases,
        result.coefficients,
        result.scattering_fraction,
        result.amplitude_fraction,
        masses,
        orientations,
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
        share.mass += float(mass) * calibration.factor(phase)
        weighted_orientation[phase] = (
            weighted_orientation.get(phase, 0.0) + float(coefficient) * float(orientation)
        )
        share.entries.append(name)

    shares = sorted(totals.values(), key=lambda share: share.scattering, reverse=True)
    clay_scattering = sum(share.scattering for share in shares if share.is_clay)
    clay_amplitude = sum(share.amplitude for share in shares if share.is_clay)
    total_mass = sum(share.mass for share in shares)
    clay_mass = sum(share.mass for share in shares if share.is_clay)
    for share in shares:
        if share.coefficient > 0.0:
            share.orientation = weighted_orientation.get(share.phase, 0.0) / share.coefficient
        if total_mass > 0.0:
            share.weight = 100.0 * share.mass / total_mass
        if share.is_clay:
            share.clay_scattering = share.scattering / clay_scattering if clay_scattering else 0.0
            share.clay_amplitude = share.amplitude / clay_amplitude if clay_amplitude else 0.0
            if clay_mass > 0.0:
                share.clay_weight = 100.0 * share.mass / clay_mass

    return Quantification(
        shares=shares,
        r_wp=result.r_wp,
        r_p=result.r_p,
        measurement=str(result.metadata.get("measurement", "")),
        calibration=calibration,
        weights_available=total_mass > 0.0,
        unweighable=list(result.metadata.get("unweighable_entries", [])),
    )
