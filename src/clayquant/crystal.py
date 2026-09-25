"""Crystal structures, CIF input and structure factors.

Two representations are used:

:class:`Crystal`
    A full three-dimensional structure (cell, symmetry operations, atom sites),
    read from a CIF.  Used to compute ``hkl`` powder patterns of the discrete
    phases (illite, chlorite, kaolinite).

:class:`LayerModel`
    A one-dimensional projection of a single silicate layer onto the stacking
    axis: atoms at absolute heights ``z`` [A] within a layer of repeat distance
    ``thickness`` [A].  This is the representation used for basal (00l)
    calculations and for the interstratification model, where only the electron
    density projected onto c* matters.

Structure factors follow the usual convention

    F(hkl) = sum_j  occ_j * f_j(q) * exp(-B_j q**2) * exp(2 pi i (h x_j + k y_j + l z_j))

with ``q = sin(theta)/lambda``.  For a basal reflection (00l) the phase reduces
to ``2 pi i l z_j``, i.e. only the fractional coordinate along c enters, whatever
the cell angles; equivalently, in the :class:`LayerModel` form, the phase is
``2 pi i s z_j`` with ``s = 2 sin(theta)/lambda = 1/d`` and ``z_j`` in Angstrom.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .scattering import debye_waller, f0

__all__ = ["AtomSite", "Crystal", "LayerModel", "read_cif", "U_TO_B"]

U_TO_B = 8.0 * math.pi**2


@dataclass(frozen=True)
class AtomSite:
    species: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_iso: float = 1.0
    label: str = ""


# --------------------------------------------------------------------------- #
# CIF reading
# --------------------------------------------------------------------------- #

_NUMBER = re.compile(
    r"^([-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)(?:\(\d+\))?$"
)


def _as_float(token: str, default: float | None = None) -> float:
    """Parse a CIF numeric token, discarding any ``(esd)`` suffix."""
    token = token.strip()
    if token in {".", "?", ""}:
        if default is None:
            raise ValueError("missing numeric CIF value with no default")
        return default
    match = _NUMBER.match(token)
    if match is None:
        raise ValueError(f"cannot parse CIF number {token!r}")
    return float(match.group(1))


def _tokenize(line: str) -> list[str]:
    return re.findall(r"'[^']*'|\"[^\"]*\"|\S+", line)


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


_TERM = re.compile(r"([+-]?)\s*(?:(\d+)\s*/\s*(\d+)|(\d*\.?\d+))?\s*\*?\s*([xyz])?")


def _parse_symop_component(text: str) -> tuple[np.ndarray, float]:
    """Parse one component of a symmetry operation, e.g. ``-x+1/2``.

    Returns the coefficients of (x, y, z) and the translation part.  Written out
    explicitly rather than with :func:`eval`, which must never be used on file
    content.
    """
    coeffs = np.zeros(3)
    shift = 0.0
    text = text.strip().lower().replace(" ", "")
    if not text:
        raise ValueError("empty symmetry operation component")
    position = 0
    seen = False
    while position < len(text):
        match = _TERM.match(text, position)
        if match is None or match.end() == match.start():
            raise ValueError(f"cannot parse symmetry component {text!r}")
        sign_text, num, den, dec, var = match.groups()
        sign = -1.0 if sign_text == "-" else 1.0
        if num is not None and den is not None:
            value = float(num) / float(den)
        elif dec is not None:
            value = float(dec)
        else:
            value = 1.0
        if var is not None:
            coeffs["xyz".index(var)] += sign * value
        else:
            shift += sign * value
        seen = True
        position = match.end()
    if not seen:
        raise ValueError(f"cannot parse symmetry component {text!r}")
    return coeffs, shift


def parse_symop(operation: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a CIF ``x, y, z`` style symmetry operation into (rotation, translation)."""
    parts = operation.strip().strip("'\"").split(",")
    if len(parts) != 3:
        raise ValueError(f"symmetry operation must have three components: {operation!r}")
    rotation = np.zeros((3, 3))
    translation = np.zeros(3)
    for row, part in enumerate(parts):
        coeffs, shift = _parse_symop_component(part)
        rotation[row] = coeffs
        translation[row] = shift
    return rotation, translation


def read_cif(path: str | Path) -> "Crystal":
    """Read a single-block CIF file into a :class:`Crystal`.

    Supports the subset of CIF produced by ICSD, COD and AMCSD exports: cell
    parameters, an explicit ``_space_group_symop_operation_xyz`` loop (or the
    ``_symmetry_equiv_pos_as_xyz`` alias) and an atom site loop with fractional
    coordinates, occupancies and ``B_iso`` or ``U_iso`` displacement parameters.
    """
    path = Path(path)
    text = path.read_text(errors="replace")

    cell: dict[str, float] = {}
    symops: list[str] = []
    sites: list[AtomSite] = []
    name = path.stem
    mineral = ""

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index].strip()
        index += 1
        if not raw or raw.startswith("#"):
            continue

        if raw.startswith("_cell_length_") or raw.startswith("_cell_angle_"):
            tokens = _tokenize(raw)
            if len(tokens) >= 2:
                cell[tokens[0].replace("_cell_", "")] = _as_float(tokens[1])
            continue
        if raw.startswith("_chemical_name_mineral"):
            tokens = _tokenize(raw)
            if len(tokens) >= 2:
                # TOPAS writes a name out as ?Clinochlore? - it uses "?" where
                # CIF wants a quote - and a bare "?" is CIF's own marker for a
                # value that is not known, which is not a name either.
                mineral = _unquote(tokens[1]).strip("'\"? ")
            continue
        if raw.startswith("_database_code_ICSD"):
            tokens = _tokenize(raw)
            if len(tokens) >= 2:
                name = f"ICSD {tokens[1]}"
            continue

        if raw != "loop_":
            continue

        # Collect the loop header, then its rows.
        tags: list[str] = []
        while index < len(lines):
            candidate = lines[index].strip()
            if candidate.startswith("_"):
                tags.append(candidate.split()[0])
                index += 1
            else:
                break
        rows: list[list[str]] = []
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or candidate.startswith("_") or candidate.startswith("#"):
                break
            if candidate == "loop_" or candidate.startswith("data_"):
                break
            tokens = _tokenize(candidate)
            index += 1
            if len(tokens) == len(tags):
                rows.append(tokens)
            elif rows and len(tokens) < len(tags):
                rows[-1].extend(tokens)

        symop_tags = {"_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"}
        matching = symop_tags.intersection(tags)
        if matching:
            column = tags.index(next(iter(matching)))
            symops.extend(_unquote(row[column]) for row in rows)
            continue

        if "_atom_site_fract_x" in tags:
            def column(tag: str) -> int | None:
                return tags.index(tag) if tag in tags else None

            idx_species = column("_atom_site_type_symbol")
            idx_label = column("_atom_site_label")
            idx_x = tags.index("_atom_site_fract_x")
            idx_y = tags.index("_atom_site_fract_y")
            idx_z = tags.index("_atom_site_fract_z")
            idx_occ = column("_atom_site_occupancy")
            idx_b = column("_atom_site_B_iso_or_equiv")
            idx_u = column("_atom_site_U_iso_or_equiv")
            for row in rows:
                label = _unquote(row[idx_label]) if idx_label is not None else ""
                species = _unquote(row[idx_species]) if idx_species is not None else label
                if idx_b is not None:
                    b_iso = _as_float(row[idx_b], default=float("nan"))
                elif idx_u is not None:
                    b_iso = _as_float(row[idx_u], default=float("nan")) * U_TO_B
                else:
                    b_iso = float("nan")
                sites.append(
                    AtomSite(
                        species=re.sub(r"\s+", "", species),
                        x=_as_float(row[idx_x]),
                        y=_as_float(row[idx_y]),
                        z=_as_float(row[idx_z]),
                        occupancy=_as_float(row[idx_occ], default=1.0) if idx_occ is not None else 1.0,
                        b_iso=b_iso,
                        label=label,
                    )
                )

    required = {"length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"}
    missing = required.difference(cell)
    if missing:
        raise ValueError(f"{path.name}: CIF is missing cell parameters {sorted(missing)}")
    if not sites:
        raise ValueError(f"{path.name}: CIF contains no atom sites")
    if not symops:
        symops = ["x, y, z"]

    return Crystal(
        a=cell["length_a"],
        b=cell["length_b"],
        c=cell["length_c"],
        alpha=cell["angle_alpha"],
        beta=cell["angle_beta"],
        gamma=cell["angle_gamma"],
        sites=sites,
        symops=symops,
        name=mineral or name,
        # "Clinochlore (ICSD 164234)" is worth saying; "Clinochlore
        # (Clinochlore)", which is what a file carrying only a mineral name
        # gives, is not.
        source=f"{mineral} ({name})" if mineral and name and name != mineral else (mineral or name),
    )


# --------------------------------------------------------------------------- #
# Crystal
# --------------------------------------------------------------------------- #


@dataclass
class Crystal:
    """A three-dimensional crystal structure."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    sites: list[AtomSite]
    symops: list[str] = field(default_factory=lambda: ["x, y, z"])
    name: str = ""
    source: str = ""
    default_b_iso: float = 1.0
    po_axis: tuple[float, float, float] | None = None
    """The pole of this mineral's preferred orientation, in Miller indices.

    A habit rather than a structural parameter: it says which face the
    crystallites settle on, which is a property of how the mineral grows and not
    of its cell.  ``None`` means an equant crystallite with no orientation worth
    describing, which is most accompanying minerals.  Every layer silicate is
    flattened on 001 and so has ``(0, 0, 1)``, which is also what the clay
    library assumes throughout; a lath-shaped mineral does not, and saying so is
    the only way its pattern can be described at all (Sec. A.37).
    """

    def __post_init__(self) -> None:
        self.sites = [
            replace(site, b_iso=self.default_b_iso) if math.isnan(site.b_iso) else site
            for site in self.sites
        ]

    # -- metric ---------------------------------------------------------------
    @property
    def _angles_rad(self) -> tuple[float, float, float]:
        return (math.radians(self.alpha), math.radians(self.beta), math.radians(self.gamma))

    @property
    def volume(self) -> float:
        """Unit cell volume in A**3."""
        ca, cb, cg = (math.cos(x) for x in self._angles_rad)
        return self.a * self.b * self.c * math.sqrt(
            1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
        )

    @property
    def metric_reciprocal(self) -> np.ndarray:
        """Reciprocal metric tensor ``G*`` such that ``1/d**2 = h G* h``."""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self._angles_rad
        ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
        sa, sb, sg = math.sin(alpha), math.sin(beta), math.sin(gamma)
        v = self.volume
        a_star, b_star, c_star = b * c * sa / v, a * c * sb / v, a * b * sg / v
        cos_alpha_star = (cb * cg - ca) / (sb * sg)
        cos_beta_star = (ca * cg - cb) / (sa * sg)
        cos_gamma_star = (ca * cb - cg) / (sa * sb)
        return np.array(
            [
                [a_star**2, a_star * b_star * cos_gamma_star, a_star * c_star * cos_beta_star],
                [a_star * b_star * cos_gamma_star, b_star**2, b_star * c_star * cos_alpha_star],
                [a_star * c_star * cos_beta_star, b_star * c_star * cos_alpha_star, c_star**2],
            ]
        )

    def inv_d(self, hkl: np.ndarray) -> np.ndarray:
        """``1/d`` in 1/A for an (N, 3) array of Miller indices."""
        hkl = np.atleast_2d(np.asarray(hkl, dtype=float))
        gstar = self.metric_reciprocal
        return np.sqrt(np.einsum("ni,ij,nj->n", hkl, gstar, hkl))

    def d_spacing(self, hkl: np.ndarray) -> np.ndarray:
        """``d`` in A for an (N, 3) array of Miller indices."""
        return 1.0 / self.inv_d(hkl)

    @property
    def d001(self) -> float:
        """Basal spacing ``d(001) = 1/|c*|`` in A."""
        return float(self.d_spacing([[0, 0, 1]])[0])

    def angle_to(self, hkl: np.ndarray, pole: Sequence[float]) -> np.ndarray:
        """Angle in radians between each reflection vector and a reciprocal direction.

        ``pole`` is given in Miller indices, so ``(0, 0, 1)`` is ``c*`` and
        ``(1, 1, 0)`` the normal of the 110 planes.  This is the angle a
        preferred-orientation factor needs: the March-Dollase distribution is
        about a *pole*, and which pole depends on the crystallite's shape.  A
        clay platelet is flattened on 001, so its pole is ``c*`` and that is the
        default everywhere; a lath-shaped crystallite is not, and a sepiolite
        lying on its 110 face needs the 110 pole to be described at all.
        """
        hkl = np.atleast_2d(np.asarray(hkl, dtype=float))
        pole = np.asarray(pole, dtype=float)
        gstar = self.metric_reciprocal
        length = float(pole @ gstar @ pole)
        if length <= 0.0:
            raise ValueError(f"{tuple(pole)} is not a direction")
        numerator = np.einsum("ni,ij,j->n", hkl, gstar, pole)
        denominator = self.inv_d(hkl) * math.sqrt(length)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.clip(numerator / denominator, -1.0, 1.0)
        return np.arccos(cosine)

    def angle_to_cstar(self, hkl: np.ndarray) -> np.ndarray:
        """Angle in radians between each reflection vector and the ``c*`` axis."""
        return self.angle_to(hkl, (0.0, 0.0, 1.0))

    # -- content --------------------------------------------------------------
    @property
    def cell_mass(self) -> float:
        """Mass of the contents of one unit cell, in g/mol.

        Summed over the symmetry-expanded sites with their occupancies, so it is
        the mass of exactly what the structure factor was computed from.  That
        correspondence is the whole point of it: the fitted scale factor of a
        phase is proportional to how many of these cells are in the beam, so
        multiplying by this converts it to something proportional to mass
        (:mod:`clayquant.masses`).
        """
        from .masses import atomic_weight

        return float(sum(site.occupancy * atomic_weight(site.species)
                         for site in self.expanded_sites()))

    def expanded_sites(self, tolerance: float = 1e-4) -> list[AtomSite]:
        """Apply the symmetry operations and return the full cell content."""
        operations = [parse_symop(op) for op in self.symops]
        expanded: list[AtomSite] = []
        seen: list[tuple[str, np.ndarray]] = []
        for site in self.sites:
            position = np.array([site.x, site.y, site.z])
            for rotation, translation in operations:
                new = np.mod(rotation @ position + translation, 1.0)
                duplicate = False
                for species, other in seen:
                    if species != site.species:
                        continue
                    delta = np.abs(new - other)
                    delta = np.minimum(delta, 1.0 - delta)
                    if np.all(delta < tolerance):
                        duplicate = True
                        break
                if duplicate:
                    continue
                seen.append((site.species, new))
                expanded.append(
                    replace(site, x=float(new[0]), y=float(new[1]), z=float(new[2]))
                )
        return expanded

    def structure_factor(self, hkl: np.ndarray, sites: list[AtomSite] | None = None) -> np.ndarray:
        """Complex structure factors for an (N, 3) array of Miller indices."""
        hkl = np.atleast_2d(np.asarray(hkl, dtype=float))
        if sites is None:
            sites = self.expanded_sites()
        q = self.inv_d(hkl) / 2.0
        total = np.zeros(len(hkl), dtype=complex)
        cache: dict[str, np.ndarray] = {}
        for site in sites:
            if site.species not in cache:
                cache[site.species] = f0(site.species, q)
            amplitude = site.occupancy * cache[site.species] * debye_waller(site.b_iso, q)
            phase = 2.0 * math.pi * (hkl @ np.array([site.x, site.y, site.z]))
            total += amplitude * np.exp(1j * phase)
        return total

    def layer_model(self, layers_per_cell: int = 1, name: str = "") -> "LayerModel":
        """Project the structure onto the stacking axis as a single layer.

        The projected electron density of a ``2M`` or ``IIb`` polytype repeats
        every ``d(001)/layers_per_cell``, because successive layers differ only
        by rotations and in-plane shifts, which leave the projection on c*
        unchanged.  Atom heights are therefore folded modulo the single-layer
        thickness, which both yields the correct one-layer model and averages
        the duplicated layers of a multi-layer cell.
        """
        if layers_per_cell < 1:
            raise ValueError("layers_per_cell must be >= 1")
        thickness = self.d001 / layers_per_cell
        heights: list[float] = []
        species: list[str] = []
        occupancies: list[float] = []
        b_isos: list[float] = []
        for site in self.expanded_sites():
            heights.append(math.fmod(site.z * self.d001, thickness))
            species.append(site.species)
            occupancies.append(site.occupancy / layers_per_cell)
            b_isos.append(site.b_iso)
        return LayerModel(
            thickness=thickness,
            z=np.asarray(heights),
            species=list(species),
            occupancy=np.asarray(occupancies),
            b_iso=np.asarray(b_isos),
            name=name or self.name,
            source=self.source,
        )


# --------------------------------------------------------------------------- #
# One-dimensional layer model
# --------------------------------------------------------------------------- #


@dataclass
class LayerModel:
    """Electron density of one silicate layer projected onto the stacking axis.

    Attributes
    ----------
    thickness:
        Layer repeat distance along the stacking axis in A (e.g. 10.0 for
        illite, 16.86 for the ethylene-glycol smectite complex).
    z:
        Absolute atom heights within the layer in A.
    species, occupancy, b_iso:
        Scattering species, site occupancies and isotropic displacement
        parameters, parallel to ``z``.
    """

    thickness: float
    z: np.ndarray
    species: list[str]
    occupancy: np.ndarray
    b_iso: np.ndarray
    name: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, dtype=float)
        self.occupancy = np.asarray(self.occupancy, dtype=float)
        self.b_iso = np.asarray(self.b_iso, dtype=float)
        sizes = {len(self.z), len(self.species), len(self.occupancy), len(self.b_iso)}
        if len(sizes) != 1:
            raise ValueError("z, species, occupancy and b_iso must have equal length")
        if self.thickness <= 0:
            raise ValueError("layer thickness must be positive")

    @classmethod
    def from_table(
        cls,
        thickness: float,
        rows: list[tuple[float, str, float, float]],
        name: str = "",
        source: str = "",
        mirror: bool = False,
    ) -> "LayerModel":
        """Build a layer from ``(z_in_A, species, occupancy, B_iso)`` rows.

        With ``mirror=True`` the rows are taken to describe half of a
        centrosymmetric layer whose mirror plane sits at ``z = 0``; every site
        with ``z != 0`` is duplicated at ``-z``.  This is the form in which
        one-dimensional clay layer models are usually published.
        """
        heights: list[float] = []
        species: list[str] = []
        occupancies: list[float] = []
        b_isos: list[float] = []
        for z, sp, occ, b_iso in rows:
            heights.append(float(z))
            species.append(sp)
            occupancies.append(float(occ))
            b_isos.append(float(b_iso))
            if mirror and abs(z) > 1e-9:
                heights.append(-float(z))
                species.append(sp)
                occupancies.append(float(occ))
                b_isos.append(float(b_iso))
        return cls(
            thickness=thickness,
            z=np.asarray(heights),
            species=species,
            occupancy=np.asarray(occupancies),
            b_iso=np.asarray(b_isos),
            name=name,
            source=source,
        )

    @property
    def electrons(self) -> float:
        """Total number of electrons per layer (the ``s -> 0`` structure factor)."""
        return float(abs(self.structure_factor(np.array([0.0]))[0]))

    @property
    def mass(self) -> float:
        """Mass of one layer, in g/mol.

        The counterpart of :attr:`electrons` for quantification: the
        interstratification model computes intensity per layer, so a fitted
        scale factor is proportional to a number of layers, and this converts
        that to mass.
        """
        from .masses import atomic_weight

        return float(sum(float(occupancy) * atomic_weight(species)
                         for species, occupancy in zip(self.species, self.occupancy)))

    def structure_factor(self, s: np.ndarray) -> np.ndarray:
        """Complex layer structure factor at ``s = 2 sin(theta)/lambda = 1/d`` [1/A].

        ``s`` is continuous: for a periodic stack of identical layers the basal
        reflections fall at ``s = l / thickness``, but interstratified stacks
        scatter at every ``s``, which is what the mixed-layer model integrates.
        """
        s = np.asarray(s, dtype=float)
        q = s / 2.0
        total = np.zeros(s.shape, dtype=complex)
        cache: dict[str, np.ndarray] = {}
        for z, species, occupancy, b_iso in zip(self.z, self.species, self.occupancy, self.b_iso):
            if species not in cache:
                cache[species] = f0(species, q)
            amplitude = occupancy * cache[species] * debye_waller(b_iso, q)
            total += amplitude * np.exp(2j * math.pi * s * z)
        return total

    def centered(self) -> "LayerModel":
        """Wrap atom heights into ``[-thickness/2, +thickness/2)``.

        This is the layer origin convention used for stacking (see
        :mod:`clayquant.mixed_layer`): the silicate sheet sits at the centre of
        the layer and the interlayer falls on the layer boundary.  Structures
        from CIFs of micas, chlorites and kaolinites already place the
        octahedral sheet at ``z = 0``, so wrapping is all that is needed.
        """
        half = self.thickness / 2.0
        wrapped = np.mod(self.z + half, self.thickness) - half
        return replace(self, z=wrapped)

    def shifted(self, delta_z: float) -> "LayerModel":
        """Return a copy with all atom heights shifted by ``delta_z`` [A]."""
        return replace(self, z=self.z + delta_z)

    def with_thickness(self, thickness: float, scale_z: bool = False) -> "LayerModel":
        """Return a copy with a different layer repeat distance.

        With ``scale_z=True`` the atom heights are scaled by the same ratio,
        which keeps the density profile similar in shape; otherwise only the
        repeat distance changes (appropriate when an interlayer expands while
        the 2:1 layer itself is unchanged).
        """
        if scale_z:
            return replace(self, thickness=thickness, z=self.z * (thickness / self.thickness))
        return replace(self, thickness=thickness)
