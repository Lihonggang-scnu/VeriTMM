"""Canonical UTF-8 JSON digest helpers shared by the TMM engine and harness.

Every deterministic content hash in this package uses one convention:
``ensure_ascii=False`` with sorted keys and compact separators.  ASCII-escaped
serialization would silently diverge for tasks containing characters such as
micrometre symbols, Greek letters, or Chinese text, so it is deliberately never
used for content digests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON content with the stable UTF-8 canonical convention."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_sha256(value: Any) -> str:
    """Return the canonical UTF-8 JSON SHA256 of ``value``."""

    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


__all__ = ["canonical_json_dumps", "stable_sha256"]
