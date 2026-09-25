# Structure data

ClayQuant does not ship structure data, and none of the files it needs here are
in this repository. ICSD content is licensed by FIZ Karlsruhe and may not be
redistributed, so export it from your own subscription and drop it in this
folder. The `.gitignore` keeps `*.cif` and `phases.json` out of git so that a
local copy is never committed by accident.

ClayQuant searches all of these, in this order:

1. `$CLAYQUANT_STRUCTURE_DIR`, if set;
2. a `structures` folder in the directory you run from — this one;
3. that directory itself;
4. the `structures` folder of a source checkout;
5. the package data directory.

## The four clay structures

Export each from ICSD as CIF and put it here. The name below is what ClayQuant
looks for first, but it is not required: a file is also accepted if its name
carries the ICSD code, or the phase name with its polytype, in any letter case.
`Kaolinite_1M_63192.cif` and `Illite_ICSD_90144.cif` are both found.

| Phase | ICSD | File name | Reference |
| --- | --- | --- | --- |
| Illite | 90144 | `illite_ICSD_90144.cif` | Gualtieri (2000) *J. Appl. Cryst.* **33**, 267–278 |
| Chlorite (clinochlore IIb-4) | 164234 | `chlorite_ICSD_164234.cif` | Zanazzi, Comodi, Nazzareni & Andreozzi (2009) *Eur. J. Mineral.* **21**, 581–589 |
| Kaolinite 1M | 63192 | `kaolinite_1M_ICSD_63192.cif` | Bish & Von Dreele (1989) *Clays Clay Miner.* **37**, 289–296 |
| Kaolinite 2M | 30285 | `kaolinite_2M_ICSD_30285.cif` | Gruner (1932) *Z. Kristallogr.* **83**, 75–88 |

Check what is found with:

```python
from clayquant.models import describe_structure_search
print(describe_structure_search())
```

It prints every directory searched and the file matched to each phase, and the
same report appears in the error when a structure cannot be loaded.

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

### A mineral the library does not have

A specimen can contain something nobody in the lab has refined. Give the same
command a CIF and it becomes a phase alongside the rest — the file states its
symmetry operations, so this path needs no `gemmi` and no space group symbol:

```bash
clayquant-import-structures your_structures.xml sepiolite_COD_9014723.cif \
    -o structures/phases.json
```

The phase is named after the file's `_chemical_name_mineral`, or after the file
itself if it does not state one, and that name is what appears in the results
table. Sources are read in increasing order of authority, so a CIF given last
replaces a phase of the same name from the library.

The CIF has to carry `_atom_site_type_symbol`. Some AMCSD exports give only
`_atom_site_label`, whose spellings — `Wat10`, `O-H2`, `AlMg1` — do not name an
element unambiguously; take the same structure from COD instead, which does.

### The fibrous clays

Sepiolite and palygorskite are clay minerals but not layer silicates: they are
chain silicates, their crystallites are laths rather than plates, and their
strongest reflection is a `110` — 12.1 Å for sepiolite, 10.4 Å for palygorskite
— and not a basal series. They are therefore accompanying minerals here, imported
as CIFs and fitted like quartz, rather than members of the oriented clay library.
Both are in COD, public domain:

| Mineral | COD | Reference |
| --- | --- | --- |
| Sepiolite | 9014723 | Sanchez *et al.* (2011) *Am. Mineral.* **96**, 1443–1454 |
| Palygorskite | 1533365 (C2/m) or 1533366 (Pbmn) | Chiari, Giustetto & Ricchiardi (2003) *Eur. J. Mineral.* **15**, 21–33 |

The two palygorskite entries are the same structure in two settings; import one,
since the second would replace the first under the same mineral name.
