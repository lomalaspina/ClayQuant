"""Print the command names declared in ``[project.scripts]``, space separated.

Used by the installers to check that the environment really has every command
the source declares.  It reads the file rather than the installed metadata on
purpose: the question being asked is whether the environment has fallen behind
the checkout, so the checkout has to be the authority.

``tomllib`` arrived in Python 3.11 and the installers accept 3.10, so there is a
plain-text fallback.  The table is a flat list of ``name = "module:function"``
lines, which needs no general TOML parser to read.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def command_names(pyproject: str | Path) -> list[str]:
    text = Path(pyproject).read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:
        pass
    else:
        data = tomllib.loads(text)
        return list(data.get("project", {}).get("scripts", {}))

    block = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if block is None:
        return []
    return re.findall(r"^\s*([A-Za-z0-9_.-]+)\s*=", block.group(1), re.M)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    pyproject = arguments[0] if arguments else "pyproject.toml"
    print(" ".join(command_names(pyproject)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
