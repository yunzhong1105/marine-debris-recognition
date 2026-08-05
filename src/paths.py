"""Project and dataset roots, resolved in one place.

The dataset does not have to live inside the repository. Keeping the 15k images
as a sibling of the checkout lets several branches or worktrees share one copy,
which matters when the images are several GB:

    workspace/
      data/                     <- dataset here
      marine-debris-recognition/
        src/

Both layouts work without editing anything; see `data_root()` for the order.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Locate `data/`, whether it sits inside the project or beside it.

    Resolution order: $MDR_DATA, then <project>/data, then <project>/../data.
    Falls back to <project>/data so error messages name the conventional place
    rather than something the user never mentioned.
    """
    override = os.environ.get("MDR_DATA")
    if override:
        return Path(override).expanduser().resolve()
    inside = PROJECT / "data"
    if inside.exists():
        return inside
    sibling = PROJECT.parent / "data"
    if sibling.exists():
        return sibling
    return inside


def to_windows(path: Path) -> str:
    """Render an absolute path the way the Windows interpreter needs it.

    Training runs under Windows Python while these scripts may be invoked from
    WSL, so `/mnt/c/...` has to come out as `C:/...`. Derived from the actual
    path rather than hardcoded: a hardcoded root silently invalidates every
    generated path the moment a directory is renamed.
    """
    resolved = path.resolve()
    parts = resolved.parts
    if len(parts) > 3 and parts[1] == "mnt" and len(parts[2]) == 1:
        return f"{parts[2].upper()}:/" + "/".join(parts[3:])
    return resolved.as_posix()


DATA = data_root()
