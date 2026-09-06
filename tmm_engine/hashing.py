"""Canonical UTF-8 JSON digest helpers: the only hash entry points in the package.

Every deterministic content identity in this package uses one convention:
``ensure_ascii=False`` with sorted keys, compact separators, ``allow_nan=False``
and ``default=str``.  ASCII-escaped serialization would silently diverge for
tasks containing characters such as micrometre symbols, Greek letters, or
Chinese text, so it is deliberately never used for content digests.  Non-finite
floats are rejected instead of serialized as non-standard ``NaN``/``Infinity``
tokens.

Business modules must not implement their own stable hashes; they call the
helpers here.  ``tests/test_hashing_architecture.py`` enforces this boundary.

Every artifact set that contains canonical-JSON-derived identities records
``identity_scheme`` so a future canonicalization change can be explained
machine-readably instead of being inferred from a package version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

IDENTITY_SCHEME = "veritmm-canonical-json-v1"
"""Canonicalization convention for every canonical-JSON-derived identity."""


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON content with the stable UTF-8 canonical convention."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 encoding used for every content identity."""

    return canonical_json_dumps(value).encode("utf-8")


def stable_sha256(value: Any) -> str:
    """Return the canonical UTF-8 JSON SHA256 of ``value``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 of a file's raw bytes, streamed in 1 MiB blocks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "IDENTITY_SCHEME",
    "canonical_json_bytes",
    "canonical_json_dumps",
    "file_sha256",
    "stable_sha256",
]
