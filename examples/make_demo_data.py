"""Generate a synthetic three-mount demo dataset.

Writes ``demo_air.xy``, ``demo_glycol.xy`` and ``demo_heated.xy``: the three
oriented mounts of one imaginary clay separate, containing illite, randomly
interstratified illite/smectite, kaolinite, chlorite and a little quartz, with
a 2theta zero error, a realistic background and counting noise.

The mounts behave as real ones do: kaolinite's 7.15 A reflection is present in
the air-dried mount and gone from the heated one, while chlorite's 002 survives
in both; the expandable interlayers sit near 15 A air-dried and near 16.9 A
after glycol solvation.

Use it to try the GUI without having to get your own vendor files read first:

    python examples/make_demo_data.py --out demo_data
    clayquant-gui          # then point the folder box at demo_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from clayquant.calibration import QUARTZ_100_D, QUARTZ_101_D, reference_two_theta
from clayquant.mixed_layer import MixedLayerStack, lognormal_csds
from clayquant.models import eg_smectite_layer, load_crystal, load_layer
from clayquant.optics import Divergence
from clayquant.pattern import (
    Instrument,
    basal_pattern,
    mixed_layer_pattern,
    powder_pattern,
    two_theta_grid,
)
from clayquant.profile import PeakShape, pseudo_voigt

ZERO_ERRORS = {"air": 0.06, "glycol": -0.04, "heated": 0.09}
MOUNT_SCALE = {"air": 1.0, "glycol": 0.88, "heated": 1.15}
ORIENTATION = 0.2


def quartz(two_theta: np.ndarray) -> np.ndarray:
    total = np.zeros_like(two_theta)
    for d, height in ((QUARTZ_100_D, 900.0), (QUARTZ_101_D, 4200.0)):
        center = reference_two_theta(d)
        total += height * 0.11 * pseudo_voigt(two_theta - center, 0.11, 0.5)
    return total


def background(two_theta: np.ndarray) -> np.ndarray:
    return 35.0 + 1100.0 / two_theta + 0.02 * (two_theta - 20.0) ** 2


def build(out: Path, seed: int = 20260911) -> list[Path]:
    grid = two_theta_grid(2.0, 40.0, 0.02)
    instrument = Instrument(
        peak_shape=PeakShape(u=0.02, v=-0.005, w=0.012, eta=0.6, size_ab=400.0),
        divergence=Divergence(),
    )
    csds = lognormal_csds(9.0)

    illite = load_crystal("illite")
    illite_layer = load_layer("illite")
    chlorite = load_crystal("chlorite")
    kaolinite = load_crystal("kaolinite_1M")

    def normalized(pattern):
        usable = pattern.two_theta >= 4.0
        return pattern.intensity / float(np.max(pattern.intensity[usable]))

    illite_pattern = normalized(powder_pattern(illite, grid, instrument, ORIENTATION))
    chlorite_pattern = normalized(powder_pattern(chlorite, grid, instrument, ORIENTATION))
    kaolinite_pattern = normalized(powder_pattern(kaolinite, grid, instrument, ORIENTATION))

    glycol_smectite = eg_smectite_layer()
    glycolated_is = normalized(
        mixed_layer_pattern(
            MixedLayerStack(illite_layer, glycol_smectite, 0.75, csds=csds),
            illite,
            2,
            grid,
            instrument,
            r_march_dollase=ORIENTATION,
        )
    )
    # Air-dried: the same interstratification with a two-water-layer smectite
    # interlayer at 15.0 A instead of the 16.86 A glycol complex.
    air_smectite = glycol_smectite.with_thickness(15.0, scale_z=True)
    air_is = normalized(
        mixed_layer_pattern(
            MixedLayerStack(illite_layer, air_smectite, 0.75, csds=csds),
            illite,
            2,
            grid,
            instrument,
            r_march_dollase=ORIENTATION,
        )
    )
    # Heated: expandable interlayers collapse onto 10 A, kaolinite is destroyed.
    collapsed_is = normalized(
        basal_pattern(
            MixedLayerStack(illite_layer, illite_layer, 1.0, csds=csds), grid, instrument,
            r_march_dollase=ORIENTATION,
        )
    )

    mixtures = {
        "air": 0.40 * illite_pattern + 0.30 * air_is + 0.20 * kaolinite_pattern
        + 0.10 * chlorite_pattern,
        "glycol": 0.40 * illite_pattern + 0.30 * glycolated_is + 0.20 * kaolinite_pattern
        + 0.10 * chlorite_pattern,
        "heated": 0.40 * illite_pattern + 0.30 * collapsed_is + 0.10 * chlorite_pattern,
    }

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for mount, mixture in mixtures.items():
        clean = MOUNT_SCALE[mount] * (9000.0 * mixture + quartz(grid)) + background(grid)
        counts = rng.poisson(np.clip(clean, 0.0, None)).astype(float)
        angles = grid + ZERO_ERRORS[mount]
        path = out / f"demo_{mount}.xy"
        with path.open("w") as handle:
            handle.write(f"# ClayQuant synthetic demo, {mount} mount\n")
            handle.write(f"# zero error {ZERO_ERRORS[mount]:+.2f} deg 2theta (to be recovered)\n")
            for angle, count in zip(angles, counts):
                handle.write(f"{angle:.4f}  {count:.0f}\n")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo_data"))
    parser.add_argument("--seed", type=int, default=20260911)
    arguments = parser.parse_args(argv)
    for path in build(arguments.out, arguments.seed):
        print(f"wrote {path}")
    print(
        "\nTrue composition (amplitude fractions of unit-maximum patterns):\n"
        "  illite            0.40\n"
        "  I/S 0.75/0.25     0.30\n"
        "  kaolinite         0.20  (absent from the heated mount)\n"
        "  chlorite          0.10\n"
        f"Zero errors: {ZERO_ERRORS}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
