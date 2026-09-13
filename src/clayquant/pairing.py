"""Recognising the three mounts of one sample from their file names.

A sample measured as three oriented mounts is almost always written to three
files whose names differ in one token: ``DBB_17_PW_air.xrdml``,
``DBB_17_PW_eg.xrdml``, ``DBB_17_PW_heat.xrdml``.  Selecting one of them in the
interface should therefore be enough to find the other two, whatever the
extension.

The matching is deliberately conservative.  The token has to stand as a word of
its own - delimited by an underscore, hyphen, dot or space, or sitting at the
end of the name - because a substring test would read the ``ad`` of
``Bad_Segeberg`` as an air-dried mount.  The rest of the name has to agree
exactly, so ``DBB_17_air`` finds ``DBB_17_eg`` and not ``DBB_17_PW_eg``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

__all__ = ["MOUNT_TOKENS", "mount_of", "sample_key", "find_siblings"]

MOUNT_TOKENS: dict[str, tuple[str, ...]] = {
    "air": ("air", "airdried", "airdry", "ad", "lutro"),
    "glycol": ("eg", "gly", "glyc", "glycol", "egl", "ethylenglykol"),
    "heated": ("heat", "heated", "ht", "350", "500", "550"),
}
"""Name tokens taken as evidence of each mount.

``lutro`` (*lufttrocken*) and ``ethylenglykol`` are here because the laboratory
this was written for labels some of its scans in German.  The temperatures are
tokens because a heated mount is often named by the temperature it was held at.
"""

_SPLIT = re.compile(r"([^A-Za-z0-9]+)")


def _token_owner(token: str) -> str | None:
    lowered = token.lower()
    for mount, tokens in MOUNT_TOKENS.items():
        if lowered in tokens:
            return mount
    return None


def _locate(stem: str) -> tuple[int, str, list[str]] | None:
    """Find the mount token in ``stem``.

    Returns the index of the token within the split parts, the mount it names,
    and the parts themselves.  The *last* qualifying token wins: the mount is
    conventionally the final element of the name, and an earlier part may
    coincide with a token by accident.
    """
    parts = _SPLIT.split(stem)
    for index in range(len(parts) - 1, -1, -1):
        if index % 2:  # a separator
            continue
        mount = _token_owner(parts[index])
        if mount is not None:
            return index, mount, parts
    return None


def mount_of(filename: str | Path) -> str | None:
    """Which mount ``filename`` names, or ``None`` if it names none."""
    located = _locate(Path(filename).stem)
    return None if located is None else located[1]


def sample_key(filename: str | Path) -> str | None:
    """The name with the mount token removed, lowercased.

    Two files share a key when they are two mounts of the same sample.  It is
    ``None`` for a name that carries no recognisable mount token.
    """
    stem = Path(filename).stem
    located = _locate(stem)
    if located is None:
        return None
    index, _, parts = located
    before = "".join(parts[:index]).rstrip("_-. ")
    after = "".join(parts[index + 1:]).lstrip("_-. ")
    return f"{before}|{after}".lower()


def find_siblings(filename: str | Path, candidates: Iterable[str | Path]) -> dict[str, str]:
    """The files among ``candidates`` that are mounts of the same sample.

    The result maps mount name to file name and includes ``filename`` itself.
    Files of the same extension are preferred - a laboratory that exports both
    ``.xrdml`` and ``.xy`` should not end up with one of each - but a mount
    present only in another supported format is still found.
    """
    name = Path(filename).name
    key = sample_key(name)
    if key is None:
        return {}

    suffix = Path(name).suffix.lower()
    found: dict[str, tuple[int, str]] = {}
    for candidate in candidates:
        candidate_name = Path(candidate).name
        if sample_key(candidate_name) != key:
            continue
        mount = mount_of(candidate_name)
        if mount is None:
            continue
        rank = 0 if Path(candidate_name).suffix.lower() == suffix else 1
        if mount not in found or rank < found[mount][0]:
            found[mount] = (rank, candidate_name)

    siblings = {mount: value for mount, (_, value) in found.items()}
    siblings[mount_of(name) or ""] = name
    siblings.pop("", None)
    return siblings
