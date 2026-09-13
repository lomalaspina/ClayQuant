# ClayQuant

Calculation of X-ray powder diffraction patterns for mica (illite), illite/smectite
and chlorite/smectite with rotational disorder, preferred orientation and random
interstratification — and quantification of clay mineral assemblages from oriented
mounts of a clay separate.

Two things in one program, because the second needs the first:

1. **A pattern calculator.** Basal (00l) and full `hkl` patterns from crystal
   structures, March–Dollase preferred orientation, and randomly interstratified
   two-component stacks by the Markov/matrix method, for any emission profile.
2. **A quantification workflow.** Load the air-dried, heated and glycol-solvated
   mounts, set the 2θ zero error on quartz, choose a background, identify
   kaolinite by its collapse and expandable clay by its swelling, find the
   accompanying minerals, then fit the whole pattern by non-negative least
   squares and report the clay assemblage.

```bash
./install.sh                     # Linux, macOS, WSL   (install.ps1 on Windows)
source .venv/bin/activate
clayquant-gui                    # the workflow, in a browser
clayquant-build-library -o library/clays.npz
clayquant-import-structures your_structures.xml -o structures/phases.json
```

The installer picks an interpreter, builds the virtual environment, installs
everything and checks the result. It exists because `python -m venv` uses
whichever interpreter `python` happens to be — on Ubuntu and WSL usually an
older one than you installed — and each Python version carries its own `venv`
and `ensurepip`. That is why `python3.12-venv` does not satisfy a `python` that
is really 3.10, and why the error names 3.10. The installer tries every
interpreter it can find, newest first, actually testing each one, and if none
can build an environment it names the exact package to install.

## The physics

### Interstratification

For a crystallite of `N` layers of types A and B stacked along `c*`, with complex
one-dimensional layer structure factors `F_A(s)`, `F_B(s)`, thicknesses `d_A`,
`d_B` and a stationary Markov chain of junction probabilities `P_mn`, the
diffracted intensity per layer is

```
I(s)/<N> = Σ_m W_m |F_m|² + 2 Re Σ_{n≥1} G(n) · fᴴ Tⁿ F
```

with `T_mn = P_mn exp(2πi s d_m)`, `f_m = W_m F*_m`, and `G(n) =
⟨max(N-n,0)⟩/⟨N⟩` averaging the pair separations over the crystallite size
distribution. In the infinite-crystal limit this collapses to the
Kakinoki–Komura form `Re{fᴴ[2(I−T)⁻¹ − I]F}`. The derivation is written out in
`mixed_layer.py`; `R = 0` (random) and Reichweite `R = 1` ordering both follow
from the transition matrix, so an ordered rectorite-like stack is just a
different `P`.

Crystallite thickness enters as a lognormal distribution of `N`, which is what
sets basal peak width.

### Layer models

The 2:1 and 1:1 layers come from CIFs; for basal reflections only the electron
density projected on `c*` matters, so a multi-layer polytype cell is folded into
one layer — exact for `2M₁` and `IIb`, whose layers differ by rotations and
in-plane shifts that leave the projection unchanged.

The ethylene glycol smectite layer is the two-layer glycol complex of
**Reynolds (1965)**, *Am. Mineral.* **50**, 990–1001, transcribed from his
Table 1 (d(001) = 16.86 Å). Reproducing his Table 2 from those parameters gives
**R(|Fo|) = 0.039** over the 13 observed orders, against the R = 0.042 of the
paper's own best model — `tests/test_reynolds_1965.py` asserts it.

### Lorentz–polarization

The **random-powder** factor `(1 + cos²2θ)/(2 sin²θ cosθ)` is the default, even
for oriented mounts. That is not an approximation made for convenience: Bradley
(1954) and Schoen (1962) found it to work, and Reynolds (1965, Tables 2 and 4)
showed for both a glycol–montmorillonite complex and a dry Na-montmorillonite
that it is *required* to reproduce observed basal intensities, the single-crystal
factor giving large deviations. The two conventions differ by a factor `l` across
a basal series, so mixing them corrupts relative basal intensities. Both are
available.

A beam-overflow correction is applied, since the Lorentz factor diverges as
`1/sin²θ` but a real measurement does not: below the angle where the irradiated
length exceeds the specimen, part of the beam misses it.

## Structure data

ClayQuant ships **no** structure data. ICSD content is licensed and must not be
redistributed, so export it yourself:

| Phase | ICSD | File name in the structure directory |
| --- | --- | --- |
| Illite | 90144 | `illite_ICSD_90144.cif` |
| Chlorite (clinochlore IIb-4) | 164234 | `chlorite_ICSD_164234.cif` |
| Kaolinite 1M | 63192 | `kaolinite_1M_ICSD_63192.cif` |
| Kaolinite 2M | 30285 | `kaolinite_2M_ICSD_30285.cif` |

Put them in `structures/` or set `$CLAYQUANT_STRUCTURE_DIR`.

For the accompanying minerals, `clayquant-import-structures` converts a
TOPAS/jEdit macro structure library into a phase database. All 205 phases of the
library this was built against convert, including the origin-choice settings
TOPAS spells `Fd-3mS`/`Fd-3mZ`. Strongest-line positions reproduce reference
values to 0.03° for quartz, calcite, dolomite, magnetite, the feldspars, halite,
hematite, rutile, siderite, anhydrite and fluorite.

## The library

`clayquant-build-library` calculates, on your instrument's grid:

* illite, chlorite, kaolinite 1M and 2M at ten March–Dollase parameters, 0.1 to 1.0;
* the Reynolds glycol smectite layer;
* illite/smectite at fifteen compositions from 0.20/0.80 to 0.99/0.01;
* chlorite/smectite at 0.95/0.05, 0.90/0.10 and 0.85/0.15;

each interstratified composition at several **crystallite thicknesses** and
**layer spacings**, because both control where and how wide the basal peaks are
and neither can be fixed in advance:

* 10 layers per crystallite give a 001 width near 0.8°, while a well-crystallised
  illite measures nearer 0.12° and needs about 70;
* ICSD 90144 gives d(001) = 10.022 Å, but a real sample measured here puts it at
  9.95 Å from its 001 *and* 002 — 0.066° at the 001, half a peak width, which for
  a fixed library is the difference between fitting that peak and missing it.

Spanning both lets the fit choose. On a real measurement this took Rwp from 49%
to 34%.

## The workflow

**Zero error.** Quartz 100 at 4.2551 Å (20.86°) clears every clay basal
reflection, with a slider stepping 0.01° — half the usual measurement step. The
search window is deliberately narrow: quartz 100 sits only 0.4° from the
kaolinite 020 band at 4.36 Å, and in a clay separate quartz is often weak while
kaolinite is strong. Measured across eleven samples, the detected shifts are
−0.02° to −0.06° and agree to 0.02° between the three mounts of a sample; where
quartz is genuinely absent the tool says so rather than reporting a number.

**Background.** Polynomial, Chebyshev and exponential components, with the `1/x`
term accumulable on top of any of them, plus a non-parametric peak-stripped
option — the direct-beam tail of an oriented mount is not a low-order polynomial.
Models are fitted to a peak-stripped estimate, not to the raw counts: a clay
pattern has no peak-free points below about 8°, and fitting the raw intensities
there drags the background up into the basal reflections, removing the very
signal being quantified.

That makes the stripping width, not the choice of component, the control that
decides where the background lies — the components then follow the estimate to a
few counts. Narrow settings lift the background into the peaks; wide ones strip
the direct-beam tail as well. Across eleven mounts the fit improves steeply up to
about 4° and is flat beyond, so 4° is the default, and the estimate is drawn on
the plot so the choice is visible rather than implicit.

**Kaolinite** is identified by the collapse of the 7.15 Å reflection on heating,
scaled on a survivor reflection so the two mounts are comparable; what survives
is chlorite 002. **Expandable clay** is identified by the shift of the low-angle
reflection under glycol — 13.8 → 16.1 Å on one of the test samples.

**Accompanying minerals** are found by fitting every candidate in competition
with the clay library in one non-negative least-squares solve, ranking each by
the share of the pattern it takes. A dialog lists what was found with its
evidence and you confirm what to include. Position matching alone is much weaker
— it ranked quartz 56th of 205 on a real separate, behind ilmenite and cementite,
while the competitive fit put quartz and albite first and second with a clean gap.

**Quantification** is reported on two bases — all fitted phases, and the clay
minerals renormalised to 100% with the accompanying minerals removed — and
exported as two CSVs plus four plots: the three mounts overlaid, the fit, every
fitted phase separated, and the clay phases alone.

## What the numbers are, and are not

The fit returns scale factors. Each phase's share of *diffracted intensity* is
reported; **that is not a weight percent.** Converting needs a reference
intensity ratio per phase — the calculated intensity per unit mass in the same
geometry — and for an oriented mount it also depends on how the mount was
prepared. Pass calibrated ratios to `Quantification.weight_percent()` if you
have them; otherwise read the numbers as relative.

Rwp is computed with counting-statistics weights from the *raw* counts, not from
the background-subtracted intensity — weighting by the latter puts the largest
weight on the flat regions between peaks, where only noise remains.

## Status

Verified against 33 real PANalytical measurements of eleven clay separates (all
read, and agreeing bit-for-bit with the vendor's own `.xy` exports). Typical
Rwp on those is 33–36%. One kaolinite-dominated sample fits poorly (76%) and is
not yet understood — a fixed library cannot adjust peak positions or widths
continuously, so a sample whose spacings fall outside the spanned range will fit
badly. Read the plots, not just the table.

## Layout

| Module | |
| --- | --- |
| `crystal.py` | CIF input, symmetry, structure factors, 1-D layer models |
| `scattering.py` | atomic scattering factors |
| `emission.py` | emission profiles, including the 5-line Cu Kα used here |
| `optics.py` | Lorentz–polarization, March–Dollase, beam overflow |
| `profile.py` | peak shapes |
| `mixed_layer.py` | the interstratification model |
| `pattern.py` | pattern assembly |
| `library.py` | the reference library and its builder |
| `bern.py` | TOPAS structure library import |
| `io.py` | XRDML, `.xy`, `.raw`, two-column text |
| `calibration.py`, `background.py` | zero error, backgrounds |
| `diagnostics.py` | kaolinite collapse, expandable swelling |
| `detection.py` | finding the accompanying minerals |
| `nnls.py`, `quantification.py`, `plots.py` | the fit, the tables, the figures |
| `gui/` | the Dash application |
