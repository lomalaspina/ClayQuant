# Structure data

ClayQuant does not ship structure data, and none of the files it needs here are
in this repository. ICSD content is licensed by FIZ Karlsruhe and may not be
redistributed, so export it from your own subscription and drop it in this
folder. The `.gitignore` keeps `*.cif` and `phases.json` out of git so that a
local copy is never committed by accident.

ClayQuant looks for this folder in the following order:

1. `$CLAYQUANT_STRUCTURE_DIR`, if set;
2. a `structures` folder in the directory you run from — this one;
3. the package data directory.

## The four clay structures

Export each from ICSD as CIF and save it here under exactly this name:

| Phase | ICSD | File name | Reference |
| --- | --- | --- | --- |
| Illite | 90144 | `illite_ICSD_90144.cif` | Gualtieri (2000) *J. Appl. Cryst.* **33**, 267–278 |
| Chlorite (clinochlore IIb-4) | 164234 | `chlorite_ICSD_164234.cif` | Zanazzi, Comodi, Nazzareni & Andreozzi (2009) *Eur. J. Mineral.* **21**, 581–589 |
| Kaolinite 1M | 63192 | `kaolinite_1M_ICSD_63192.cif` | Bish & Von Dreele (1989) *Clays Clay Miner.* **37**, 289–296 |
| Kaolinite 2M | 30285 | `kaolinite_2M_ICSD_30285.cif` | Gruner (1932) *Z. Kristallogr.* **83**, 75–88 |

Check what is found with:

```python
from clayquant.models import available_phases, structure_directory
print(structure_directory(), available_phases())
```

The ethylene glycol smectite layer needs no file: it is the published
one-dimensional model of Reynolds (1965), transcribed in `clayquant/models.py`.

## The accompanying minerals

Quartz, feldspars, carbonates and the rest come from your own TOPAS structure
library, converted once:

```bash
clayquant-import-structures your_structures.xml -o structures/phases.json
```

This needs `gemmi` (`pip install -e ".[import]"`), which is used only to expand
space group symbols; the generated `phases.json` stores the symmetry operations
explicitly, so nothing else is needed to read it afterwards. That file is
derived from licensed structure data too, and is likewise kept out of git.
