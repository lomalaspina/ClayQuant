"""Print what a saved clay library was calculated with.

Run it on a ``.npz`` to see whether it matches the instrument a measurement was
made on::

    python scripts/inspect_library.py library/clays.npz

The three things worth checking are the layer spacings spanned, the peak width
and the geometry.  A library built before any measurement was loaded carries the
defaults - 280 mm and a generic width - and its reference peaks are then the
wrong width for the data, which a fit can only absorb as intensity missing at
every strong peak.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from clayquant.library import PatternLibrary
from clayquant.profile import PeakShape


def describe(path: str) -> int:
    library = PatternLibrary.load(path)
    counts: dict[str, int] = {}
    orientations: dict[str, set] = {}
    for entry in library.entries:
        counts[entry.phase] = counts.get(entry.phase, 0) + 1
        orientations.setdefault(entry.phase, set()).add(round(float(entry.march_dollase), 3))
    print(f"{path}: {len(library.entries)} patterns on "
          f"{library.two_theta[0]:.2f}-{library.two_theta[-1]:.2f} deg")
    for phase in sorted(counts):
        print(f"   {phase:14s} {counts[phase]:5d} patterns, "
              f"r in {sorted(orientations[phase])}")

    spacings = library.metadata.get("host_thicknesses")
    print(f"\n  layer spacings spanned : {spacings if spacings else 'not recorded'}")
    smectite = library.metadata.get("smectite_orientation")
    print(f"  smectite orientation   : "
          f"{smectite if smectite is not None else 'not recorded'}")

    shape = library.metadata.get("peak_shape")
    if shape:
        model = PeakShape(
            u=float(shape.get("u", 0.0)), v=float(shape.get("v", 0.0)),
            w=float(shape.get("w", 0.01)), eta=float(shape.get("eta", 0.5)),
            size_c=shape.get("size_c"), size_ab=shape.get("size_ab"),
        )
        widths = " ".join(
            f"{angle:.0f} deg: {float(model.fwhm(np.array([angle]), 1.540596)[0]):.3f}"
            for angle in (8.0, 20.0, 26.0, 36.0)
        )
        print(f"  peak width (FWHM)      : {widths}")
    else:
        print("  peak width             : not recorded (built before this was stored)")

    geometry = library.metadata.get("geometry")
    if geometry:
        print(f"  geometry               : "
              f"{geometry['goniometer_radius']:.0f} mm radius, "
              f"{geometry['divergence']:g} deg slit, "
              f"{geometry['specimen_length']:g} mm specimen")
        if abs(float(geometry["goniometer_radius"]) - 280.0) < 1e-6 and \
                abs(float(geometry["divergence"]) - 0.5) < 1e-6:
            print("\n  This is the default geometry, which means the library was built "
                  "before\n  a measurement was loaded. If your instrument is not a 280 mm "
                  "one with a\n  0.5 deg slit, load a mount first and build again.")
    else:
        print("  geometry               : not recorded (built before this was stored)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", help="path to a clay library .npz")
    arguments = parser.parse_args(argv)
    return describe(arguments.library)


if __name__ == "__main__":
    sys.exit(main())
