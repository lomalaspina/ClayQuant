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
import math
from dataclasses import dataclass, field
from pathlib import Path

from .bern import is_clay_phase
from .nnls import FitResult

__all__ = [
    "PhaseShare",
    "ExternalStandard",
    "calibration_factor",
    "Quantification",
    "Calibration",
    "quantify",
    "CLAY_LIBRARY_PHASES",
    "INTERSTRATIFIED_HOSTS",
    "reported_phase",
    "CLAYFIT_ANCHOR_ORIENTATION",
    "FIXED_ORIENTATION_PHASES",
    "rebase_fixed_orientation",
    "CLAYFIT_SCALE_FACTORS",
    "clayfit_weight_fractions",
]

CLAY_LIBRARY_PHASES = frozenset(
    {"illite", "chlorite", "kaolinite_1M", "kaolinite_2M", "smectite_EG", "I/S", "C/S"}
)
"""Library phase keys that are clay minerals by construction."""


INTERSTRATIFIED_HOSTS = {"I/S": "illite", "C/S": "chlorite"}
"""The discrete mineral each interstratified series becomes at fraction 1."""


def reported_phase(phase: str, fraction: float | None) -> str:
    """The mineral an entry *is*, which is not always the series it was built in.

    ClayQuant's library spans each interstratified series to its own endmember,
    so ``I/S 1.00/0.00`` and ``C/S 1.00/0.00`` exist - and they are 100 % host
    layers and 0 % smectite, which is to say they are an illite and a chlorite.
    On this library that is 90 entries named ``I/S`` and 60 named ``C/S`` that
    contain no expandable layer at all.

    Grouping the fitted entries by the label they were built under therefore
    reports a specimen with no swelling clay in it as majority mixed-layer:
    on one real mount the table read ``C/S 25%`` and ``I/S 14%``, both with an
    expandable content of 0 %, where the honest reading is chlorite 25 % and
    illite 14 %.  A reader who knows that an I/S at 3 % expandable layers is an
    illite still cannot be expected to know that an I/S at *zero* is one, and
    nothing in the table said so.

    So a series entry at fraction 1 is reported as its host mineral, and
    everything else keeps its series name.  This changes only how fitted
    entries are grouped for reporting; the library, the fit and the
    coefficients are untouched.
    """
    host = INTERSTRATIFIED_HOSTS.get(phase)
    if host is None or fraction is None:
        return phase
    value = float(fraction)
    if not math.isfinite(value):
        # Not recorded, which is not the same as recorded as 1.  An older
        # library's I/S 0.60/0.40 arrives this way and must keep its name.
        return phase
    return host if value >= 1.0 - 1e-9 else phase


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

    @classmethod
    def from_clayfit(
        cls, library, scale_factors: dict[str, float] | None = None
    ) -> "Calibration":
        """Clayfit's family calibration, as factors on ClayQuant's computed mass.

        See :data:`CLAYFIT_SCALE_FACTORS` for where the numbers come from and
        what carrying them over does and does not buy.  The arithmetic is the
        same as :func:`clayfit_weight_fractions`, expressed as one factor per
        phase so it goes through :func:`quantify` with everything else:

            k = (anchor / K) / (unit mass x unit volume)

        The right-hand denominator is what ClayQuant would compute for that
        phase from first principles; the numerator is what Clayfit's calibration
        says the amount is instead.  So ``k`` is the ratio between the two, and a
        ``k`` far from 1 is a statement that the calculation and the calibrated
        measurement disagree about that phase by that much - worth reading as a
        result and not only as a correction.

        The factors depend on the library, because both the anchor and the unit
        mass do, so this is computed and not tabulated.  A phase the library
        holds and Clayfit does not calibrate keeps ``k = 1``, which leaves it on
        the computed basis; that is the honest treatment for the
        interstratified series, which Clayfit calibrates per composition rather
        than per family and so has no single factor for.
        """
        factors = CLAYFIT_SCALE_FACTORS if scale_factors is None else scale_factors
        most_oriented: dict[str, float] = {}
        for entry in library.entries:
            r = float(entry.march_dollase)
            if entry.phase not in most_oriented or r < most_oriented[entry.phase]:
                most_oriented[entry.phase] = r
        result: dict[str, float] = {}
        skipped: list[str] = []
        for entry in library.entries:
            phase = entry.phase
            if phase in result or phase in skipped or phase not in factors:
                continue
            if abs(float(entry.march_dollase) - most_oriented[phase]) > 1e-9:
                continue
            # Clayfit anchors every factor on its PO_01 profile, so a family
            # whose most oriented pattern here is at some other r is not
            # comparable and must not be calibrated as though it were.  A pure
            # basal series scales as r**-3, so accepting a mismatch would import
            # the factor multiplied by a thousand: the glycol smectite, which
            # this library holds only as a random powder, came out at k = 0.0003
            # before this check, and that number is the orientation mismatch and
            # nothing else.
            if abs(float(entry.march_dollase) - CLAYFIT_ANCHOR_ORIENTATION) > 1e-9:
                skipped.append(phase)
                continue
            anchor = float(entry.normalization or 0.0)
            mass = float(entry.unit_mass or 0.0)
            volume = float(entry.unit_volume or 0.0)
            if anchor <= 0.0 or mass <= 0.0 or volume <= 0.0:
                continue
            result[phase] = (anchor / factors[phase]) / (mass * volume)
        if not result:
            raise ValueError(
                "none of the phases Clayfit calibrates is in this library with an "
                "absolute scale, a unit mass and a unit volume, so its factors cannot "
                "be re-anchored"
            )
        mean = sum(result.values()) / len(result)
        note = (
            "Clayfit family calibration (Clayfit4 measurements), re-anchored on this "
            "library; the interstratified series stay on the computed basis"
        )
        if skipped:
            note += (
                f"; {', '.join(sorted(skipped))} left uncalibrated because this library "
                f"holds no pattern for it at r = {CLAYFIT_ANCHOR_ORIENTATION:g}, which is "
                f"where Clayfit's factors are anchored"
            )
        return cls(
            factors={phase: value / mean for phase, value in result.items()},
            source=note,
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
class ExternalStandard:
    """The instrument constant, from a measurement of a pure standard.

    A weight percent from a fitted scale factor is relative: it divides one
    mount between its phases and says nothing about whether the calculated
    pattern accounted for everything a phase contributes.  A standard measured
    through the same optics fixes the absolute scale, through

        W(p) = S(p) (ZMV)(p) mu_m / K

    (O'Connor & Raven 1988), and ``K`` is what this holds.  For a pure standard
    of known weight fraction it is simply the fitted mass times the standard's
    mass attenuation coefficient.

    Everything that differs between the standard's measurement and the
    specimen's has to be either identical or corrected for, and the list is
    longer than it looks: the tube, the slits, the mask, the counting time per
    step, the step size and the detector settings.  A divergence slit of 0.25
    deg against 0.5 deg is not a constant factor but an angle-dependent one,
    because a fixed slit overflows the specimen at low angle and not at high.
    :meth:`comparable_with` checks what can be checked from the files.
    """

    constant: float
    phase: str
    mass_attenuation: float
    weight_fraction: float = 1.0
    measurement: str = ""
    conditions: dict = field(default_factory=dict)

    @classmethod
    def from_fit(
        cls,
        result: FitResult,
        phase: str,
        mass_attenuation: float,
        weight_fraction: float = 1.0,
        conditions: dict | None = None,
    ) -> "ExternalStandard":
        """``K`` from a fit of a measurement of the standard."""
        if not 0.0 < weight_fraction <= 1.0:
            raise ValueError(
                f"the standard's weight fraction must lie in (0, 1], not {weight_fraction}"
            )
        mass = _fitted_mass(result, phase)
        if mass <= 0.0:
            raise ValueError(
                f"the fit of {result.metadata.get('measurement', 'the standard')} gives "
                f"{phase!r} no mass, so there is no constant to take from it"
            )
        return cls(
            constant=mass * mass_attenuation / weight_fraction,
            phase=phase,
            mass_attenuation=mass_attenuation,
            weight_fraction=weight_fraction,
            measurement=str(result.metadata.get("measurement", "")),
            conditions=dict(conditions or {}),
        )

    def comparable_with(self, conditions: dict) -> list[str]:
        """What differs between the standard's conditions and these, if anything."""
        if not self.conditions:
            return ["the standard's measurement conditions were not recorded, so nothing "
                    "could be checked against them"]
        differences = []
        for key, mine in sorted(self.conditions.items()):
            theirs = conditions.get(key)
            if theirs is None:
                differences.append(f"{key} is not recorded for the specimen")
            elif isinstance(mine, float) and isinstance(theirs, (int, float)):
                if abs(mine - float(theirs)) > 1e-6 * max(1.0, abs(mine)):
                    differences.append(f"{key}: standard {mine:g}, specimen {theirs:g}")
            elif mine != theirs:
                differences.append(f"{key}: standard {mine}, specimen {theirs}")
        return differences


def _fitted_mass(result: FitResult, phase: str) -> float:
    """The relative mass the fit gives one phase, summed over its entries."""
    masses = (
        result.relative_mass
        if len(result.relative_mass) == len(result.coefficients)
        else [0.0] * len(result.coefficients)
    )
    return float(sum(
        mass for name, mass, coefficient in zip(result.phases, masses, result.coefficients)
        if name == phase and coefficient > 0.0
    ))


def calibration_factor(
    result: FitResult,
    phase: str,
    mass_attenuation: float,
    standard: ExternalStandard,
    weight_fraction: float = 1.0,
) -> float:
    """The factor ``k`` of Sec. 2.13 for one phase, from a pure-phase mount.

    A mount of the pure mineral has a known weight fraction, so the relation can
    be read the other way round: whatever it takes to make the fitted mass agree
    with the weight actually there is the factor by which the calculated pattern
    fails to describe the phase.  That is texture, microabsorption and any error
    in the structural model, together, which is what ``k`` is defined to hold.

    ``weight_fraction`` is the phase's weight fraction in the mount, which is 1
    only for a genuinely pure one; a standard carrying a few per cent of quartz
    should say so, or its ``k`` absorbs the difference.
    """
    if not 0.0 < weight_fraction <= 1.0:
        raise ValueError(f"a weight fraction must lie in (0, 1], not {weight_fraction}")
    if standard.constant <= 0.0:
        raise ValueError("the standard's constant is not positive, so it cannot be divided by")
    mass = _fitted_mass(result, phase)
    if mass <= 0.0:
        raise ValueError(
            f"the fit gives {phase!r} no mass, so there is no factor to be had from it"
        )
    return (mass * mass_attenuation / weight_fraction) / standard.constant


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

    host_fraction: float = 1.0
    """Host-layer fraction of this phase, averaged over its entries.

    1 for a discrete mineral.  For an interstratified phase it is the
    proportion of host layers the fit chose - 0.8 for an 80/20 illite/smectite.
    It is reported because the phase name does not carry it: an "I/S" at 0.99 is
    a stack of essentially pure illite, and a table that calls it I/S without
    the number reads as if smectite had been found.
    """

    @property
    def expandable_fraction(self) -> float:
        """Proportion of expandable layers in this phase, 0 for a discrete one."""
        return 0.0 if self.host_fraction >= 1.0 else 1.0 - self.host_fraction

    absolute_weight: float = 0.0
    """Weight percent of the whole specimen, when an internal standard fixes it.

    ``weight`` is a share of what was fitted and sums to 100 whatever was left
    out; this does not.  A specimen containing amorphous material, or a phase
    absent from the library, has a ``weight`` that overstates every phase in it
    by the same factor, and nothing in the fit reveals the factor.  A weighed
    addition of a standard does: see :attr:`Quantification.unaccounted`.
    """

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

    internal_standard: str = ""
    """The phase added to the specimen in a known amount, if there was one."""

    standard_weight: float = 0.0
    """How much of it was added, as a weight percent of the spiked specimen."""

    absolute_scale: float = 0.0
    """Factor from a fitted weight percent to a weight percent of the specimen.

    The ratio of what was weighed out to what the fit recovers.  It is 1 for a
    specimen that is wholly crystalline and wholly in the library; above 1 when
    something is present that the fit does not account for, since the standard
    is then over-stated against phases that are all under-stated together.
    """

    rebased_orientation: dict = field(default_factory=dict)
    """Phases put on the other clays' texture, as ``{phase: (from_r, to_r)}``.

    Empty when nothing needed it.  Worth reporting rather than doing silently:
    the correction is cubic in ``r``, so between a library built at 1 and clays
    fitted at 0.1 it moves a weight percent by a factor of a thousand, and a
    reader comparing two runs needs to know which basis each is on
    (:func:`rebase_fixed_orientation`).
    """

    unaccounted: float = 0.0
    """Weight percent of the *spiked* specimen the fitted phases do not account for.

    Amorphous material, and any crystalline phase missing from the fit, together:
    the measurement separates them from the phases that were fitted but not from
    each other.  See :meth:`original_basis` for the same quantity as a fraction
    of the material that was actually sampled, which is what is normally quoted.
    """

    @property
    def absolute(self) -> bool:
        """Whether weights are of the specimen rather than of what was fitted."""
        return self.absolute_scale > 0.0

    def original_basis(self) -> dict[str, float]:
        """Weight percent of the specimen as it was before the standard was added.

        :attr:`PhaseShare.absolute_weight` and :attr:`unaccounted` are fractions
        of the *spiked* powder, which is the mixture the diffractometer saw.
        What a result refers to, though, is the material that was sampled, and
        the standard is not part of it: every phase is larger by
        1/(1 - W_std) once the standard is taken back out, a factor of 1.25 for
        the usual 20 % spike.  That is the basis an amorphous content is
        conventionally quoted on, and the difference is too large to leave to
        the reader.

        The standard itself is not in the returned mapping, since it is not part
        of the specimen; the key ``"unaccounted"`` carries the amorphous and
        unfitted remainder.  Empty when there was no usable standard.
        """
        if not self.absolute:
            return {}
        keep = 1.0 - self.standard_weight / 100.0
        if keep <= 0.0:
            return {}
        basis = {
            share.phase: share.absolute_weight / keep
            for share in self.shares
            if share.phase != self.internal_standard
        }
        basis["unaccounted"] = self.unaccounted / keep
        return basis

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
                # Only on the whole-specimen basis: a weight percent of the clay
                # minerals renormalised to 100 is already not a weight percent of
                # the specimen, so scaling it by the standard would mean nothing.
                "absolute_weight_percent": share.absolute_weight
                if self.absolute and not clay_basis else "",
                "march_dollase": share.orientation,
                "host_fraction": share.host_fraction,
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
            if self.absolute and not clay_basis:
                handle.write(
                    f"# absolute_weight_percent is of the whole specimen, fixed by "
                    f"{self.standard_weight:g}% of {self.internal_standard} weighed into it; "
                    f"the fitted phases account for {100.0 - self.unaccounted:.2f}% of it and "
                    f"{self.unaccounted:.2f}% is amorphous or missing from the fit\n"
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
                    "absolute_weight_percent",
                    "march_dollase",
                    "host_fraction",
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
    internal_standard: tuple[str, float] | None = None,
    rebase_orientation: bool = True,
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
    internal_standard:
        The name of a phase weighed into the specimen, and how much of it was
        added as a weight percent of the spiked specimen - ``("Corundum", 20.0)``
        for the usual 20 % of corundum.  The fit's weight percent is a share of
        what was fitted and sums to 100 however much of the specimen was left
        out of it; a standard of known weight converts that to a weight percent
        of the specimen, and the difference from 100 is what the fit does not
        account for.  The phase must be one the fit found: a standard that was
        added and not fitted leaves the analysis relative, and says so through
        :attr:`Quantification.absolute` rather than by raising.
    rebase_orientation:
        Put a phase stored at one fixed orientation - the glycolated smectite -
        on the texture the fit measured for the other platy clays, rather than
        on whatever the library was built with.  See
        :func:`rebase_fixed_orientation`; the factor is cubic, so this is worth
        several orders of magnitude and not a refinement.  Pass ``False`` to see
        the library's own basis.
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
    fractions = (
        result.fraction
        if len(result.fraction) == len(result.coefficients)
        else [1.0] * len(result.coefficients)
    )
    weighted_orientation: dict[str, float] = {}
    weighted_fraction: dict[str, float] = {}
    totals: dict[str, PhaseShare] = {}
    for name, phase, coefficient, scattering, amplitude, mass, orientation, fraction in zip(
        result.names,
        result.phases,
        result.coefficients,
        result.scattering_fraction,
        result.amplitude_fraction,
        masses,
        orientations,
        fractions,
    ):
        if coefficient <= 0.0:
            continue
        # An endmember of a series is the discrete mineral, whatever it was
        # built under; see reported_phase.
        phase = reported_phase(phase, fraction)
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
        # An entry with no fraction of its own is a discrete mineral, which is
        # all host layers; the *report* says 1 while the array says nothing.
        known = (1.0 if fraction is None or not math.isfinite(float(fraction))
                 else float(fraction))
        weighted_fraction[phase] = (
            weighted_fraction.get(phase, 0.0) + float(coefficient) * known
        )
        share.entries.append(name)

    shares = sorted(totals.values(), key=lambda share: share.scattering, reverse=True)
    for share in shares:
        if share.coefficient > 0.0:
            share.orientation = weighted_orientation.get(share.phase, 0.0) / share.coefficient
            share.host_fraction = weighted_fraction.get(share.phase, 0.0) / share.coefficient
    # Before any share is taken: the rebasing changes a mass, and every weight
    # percent below is a share of the total mass.
    rebased = rebase_fixed_orientation(shares) if rebase_orientation else {}
    clay_scattering = sum(share.scattering for share in shares if share.is_clay)
    clay_amplitude = sum(share.amplitude for share in shares if share.is_clay)
    total_mass = sum(share.mass for share in shares)
    clay_mass = sum(share.mass for share in shares if share.is_clay)
    for share in shares:
        if total_mass > 0.0:
            share.weight = 100.0 * share.mass / total_mass
        if share.is_clay:
            share.clay_scattering = share.scattering / clay_scattering if clay_scattering else 0.0
            share.clay_amplitude = share.amplitude / clay_amplitude if clay_amplitude else 0.0
            if clay_mass > 0.0:
                share.clay_weight = 100.0 * share.mass / clay_mass

    standard_name, standard_weight, scale = "", 0.0, 0.0
    if internal_standard is not None:
        standard_name, standard_weight = internal_standard[0], float(internal_standard[1])
        if not 0.0 < standard_weight < 100.0:
            raise ValueError(
                "the internal standard's weight percent must lie between 0 and 100, "
                f"not {standard_weight}"
            )
        fitted = next((share for share in shares if share.phase == standard_name), None)
        if fitted is not None and fitted.weight > 0.0 and total_mass > 0.0:
            scale = standard_weight / fitted.weight
            for share in shares:
                share.absolute_weight = share.weight * scale

    return Quantification(
        shares=shares,
        r_wp=result.r_wp,
        r_p=result.r_p,
        measurement=str(result.metadata.get("measurement", "")),
        calibration=calibration,
        weights_available=total_mass > 0.0,
        unweighable=list(result.metadata.get("unweighable_entries", [])),
        internal_standard=standard_name,
        standard_weight=standard_weight,
        absolute_scale=scale,
        unaccounted=(100.0 - sum(share.absolute_weight for share in shares)) if scale else 0.0,
        rebased_orientation=rebased,
    )


# --- Clayfit's family calibration, carried over ------------------------------

CLAYFIT_ANCHOR_ORIENTATION = 0.1

FIXED_ORIENTATION_PHASES = ("smectite_EG",)
"""Phases whose library pattern is one entry at one assumed orientation.

A pure basal series has the same shape at every March-Dollase ``r`` - every
reflection is enhanced by the same ``r**-3`` - so storing it at ten orientations
would give ten identical columns and the fit could not choose between them.  The
glycolated smectite is the only such phase here, and it is stored once.

The consequence is that its ``r`` is an *assumption*, not a measurement, and the
assumption goes straight into its weight: the pattern at ``r = 0.1`` is exactly
1000 times as intense as the same pattern at ``r = 1``, so the mass behind one
fitted coefficient differs by that factor.  Left at the library's value beside an
illite the fit put at 0.1, a smectite stored at 1 is reported with a thousand
times too much mass relative to it, which on one real mount turned 0.95 % of the
scattering into 22 % of the clay weight.  :func:`rebase_fixed_orientation` is
what puts it back on the same basis.
"""


def rebase_fixed_orientation(
    shares: "list[PhaseShare]",
    phases: "tuple[str, ...]" = FIXED_ORIENTATION_PHASES,
) -> dict[str, tuple[float, float]]:
    """Put a fixed-orientation phase on the texture the fit found for the others.

    The fit measures ``r`` for illite, chlorite and the kaolinites, from their
    non-basal reflections; it cannot measure it for a phase whose every
    reflection is basal, because the orientation scales such a pattern without
    changing its shape.  Reporting that phase at whatever ``r`` the library
    happened to be built with therefore puts it on a different basis from the
    clays it is being compared with, and the difference is ``(r_library /
    r_fitted)**3`` - a factor of 1000 between 1 and 0.1.

    The assumption made here instead is that a platy clay in one oriented mount
    has the texture the other platy clays in that mount have, and that is
    measured: the mass is rescaled to the coefficient-weighted mean ``r`` of the
    clay phases whose orientation the fit did choose.  It is still an
    assumption, and it is the one worth making, because the alternative is an
    assumption made at library-build time by someone who has not seen the
    specimen.

    Modifies the shares in place and returns ``{phase: (r_from, r_to)}`` for
    those it rebased, so the caller can say what it did.
    """
    measured = [
        share for share in shares
        if share.is_clay and share.phase not in phases
        and share.coefficient > 0.0 and share.orientation > 0.0
    ]
    if not measured:
        return {}
    weight = sum(share.coefficient for share in measured)
    if weight <= 0.0:
        return {}
    target = sum(share.coefficient * share.orientation for share in measured) / weight
    if not 0.0 < target <= 1.0:
        return {}

    rebased: dict[str, tuple[float, float]] = {}
    for share in shares:
        if share.phase not in phases or share.mass <= 0.0:
            continue
        stored = float(share.orientation)
        if stored <= 0.0 or abs(stored - target) < 1e-9:
            continue
        # The pattern is I(r) = r**-3 I(1), stored at unit maximum, so the same
        # fitted coefficient implies a mass smaller by (target / stored)**3.
        share.mass *= (target / stored) ** 3
        rebased[share.phase] = (stored, target)
        share.orientation = target
    return rebased
"""March-Dollase parameter Clayfit's scale factors are anchored on (its PO_01)."""

CLAYFIT_SCALE_FACTORS: dict[str, float] = {
    "chlorite": 358.0343993649871,
    "illite": 113.15656786986817,
    "kaolinite_1M": 180.223981959315,
    "kaolinite_2M": 104.678273593158,
    "smectite_EG": 754.5132206968542,
}
"""Per-family intensity calibration taken from Clayfit5, by ClayQuant phase name.

These are the ``calibration_scale_factor`` values of Clayfit5's
``oriented_clay.py``, determined offline against measurements held in the older
Clayfit4 database.  In Clayfit each family's calculated profile is multiplied by
its factor before the fit, and the fitted coefficient is then read as
proportional to the amount of that family - so the factors are what put the five
families on one scale of counts per unit amount.

What carries over and what does not.  The *ratios* between these numbers are a
calibration of real mixtures and are worth having; the numbers themselves are
tied to Clayfit's own profiles, whose absolute scale comes from TOPAS
conventions ClayQuant does not share.  So they cannot be used as they stand and
are re-anchored in :func:`clayfit_weight_fractions` against ClayQuant's own
calculated pattern for the same family at the same orientation, which is the one
place the two can be made to agree.

The assumption this rests on, stated plainly because it is not a small one:
that ClayQuant's calculated pattern for a family stands in the same relation to
the amount of that family as Clayfit's TOPAS profile does, up to the single
constant being re-anchored.  Where the two calculations differ in more than
scale - and they do; Clayfit's illite and chlorite profiles come from structures
refined against one specimen, ClayQuant's from the published entries - the
calibration carries that difference with it.  It is a family-to-family
correction, and it does not make the absolute weight percent right.

C/S and I/S are absent on purpose.  Clayfit calibrates its mixed-layer profiles
individually, one factor per composition, not one per family, so there is no
single number to carry over for an interstratified series.
"""


def clayfit_weight_fractions(
    result,
    library,
    scale_factors: dict[str, float] | None = None,
) -> dict[str, float]:
    """Weight fractions on Clayfit's family calibration.

    Clayfit's arithmetic is that a family's fitted coefficient is proportional to
    its amount once its profile has been multiplied by ``K``, with the profile
    anchored at its own maximum for the most strongly oriented variant it holds
    (``PO_01``, ``r = 0.1``).  Reproducing it here needs both halves of that:

    * the amount-proportional quantity per entry, which is
      ``coefficient / normalization`` - the library stores its patterns at unit
      maximum, so the coefficient alone has the orientation dependence divided
      out of it and is not proportional to anything;
    * the same anchor, which is ClayQuant's own raw maximum for that family's
      most strongly oriented pattern, standing in for Clayfit's 10 000.

    "Most strongly oriented" is how Clayfit's own anchor is reproduced without
    naming a number that is only right for some of the families.  Clayfit
    anchors its chlorite, illite and two kaolinite families on ``PO_01``, which
    is ``r = 0.1`` and is the smallest orientation parameter they span; its
    smectite family it does not orient at all, spanning basal spacing and width
    instead, so there is no ``PO_01`` to anchor on and its factor belongs to the
    first profile in that family.  Taking each family's smallest
    ``march_dollase`` gives the first in both cases.

    So each family's amount comes out as
    ``sum(c / normalization) * anchor / K``, and the results are renormalised.
    A family with no factor and no entries in the fit is simply absent from the
    answer; a family that was fitted and has no factor raises, because silently
    dropping a mineral that is present renormalises the rest to 100 per cent
    between them and the answer then looks complete while something is missing
    from it.

    Returns a mapping of phase to weight fraction, summing to 1 over the phases
    it covers.
    """
    factors = CLAYFIT_SCALE_FACTORS if scale_factors is None else scale_factors
    most_oriented: dict[str, float] = {}
    for entry in library.entries:
        r = float(entry.march_dollase)
        if entry.phase not in most_oriented or r < most_oriented[entry.phase]:
            most_oriented[entry.phase] = r
    anchors: dict[str, float] = {}
    for entry in library.entries:
        if abs(float(entry.march_dollase) - most_oriented[entry.phase]) > 1e-9:
            continue
        divisor = float(entry.normalization or 0.0)
        if divisor > anchors.get(entry.phase, 0.0):
            anchors[entry.phase] = divisor

    units: dict[str, float] = {}
    for entry, coefficient in zip(library.entries, result.coefficients):
        if coefficient <= 0.0:
            continue
        divisor = float(entry.normalization or 1.0)
        if divisor <= 0.0:
            continue
        units[entry.phase] = units.get(entry.phase, 0.0) + float(coefficient) / divisor

    fitted = {phase for phase, value in units.items() if value > 0.0}
    missing = sorted(phase for phase in fitted if phase not in factors)
    if missing:
        raise KeyError(
            f"no Clayfit scale factor for {missing}; give one in scale_factors or "
            f"quantify without this calibration, because leaving a fitted phase out "
            f"renormalises the rest to 100 % between them"
        )
    amounts: dict[str, float] = {}
    for phase in fitted:
        anchor = anchors.get(phase)
        if not anchor:
            raise KeyError(
                f"the library records no absolute scale for {phase}, so Clayfit's factor "
                f"for it cannot be re-anchored; rebuild the library so its entries carry "
                f"their normalization"
            )
        amounts[phase] = units[phase] * anchor / factors[phase]
    total = sum(amounts.values())
    if total <= 0.0:
        return {phase: 0.0 for phase in amounts}
    return {
        phase: value / total
        for phase, value in sorted(amounts.items(), key=lambda item: -item[1])
    }
