"""Bounded material lookup and optical-constant sampling.

The registry deliberately keeps data-source policy separate from interpolation:
local CSV files are the default source, while the bundled refractiveindex.info
SQLite cache is available through an explicit provider or dataset selection.
Wavelengths in this module are expressed in micrometres (um), matching both
the local CSV files and the RII cache.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MATERIALS_DIR = PACKAGE_ROOT / "materials"
DEFAULT_RII_DB_PATH = PACKAGE_ROOT / "rii_cache.db"


_MATERIAL_ALIASES = {
    "silica": "sio2",
    "fusedsilica": "sio2",
    "sio2": "sio2",
    "titania": "tio2",
    "tio2": "tio2",
    "silver": "ag",
    "ag": "ag",
    "alumina": "al2o3",
    "al2o3": "al2o3",
}

# Natural-language names are useful for the *new candidate feed*, but are not
# added to the historical ``normalize_material_name`` alias table: the outer
# catalogue deliberately rejects prose in ``verify`` and existing callers rely
# on that contract.  The ranker can still use these aliases to find evidence and
# then returns a canonical dataset name for the execution path.
_SEMANTIC_MATERIAL_ALIASES = {
    "fusedquartz": "sio2",
    "silicondioxide": "sio2",
    "siliconmonoxide": "sio",
    "siliconnitride": "si3n4",
    "titaniumdioxide": "tio2",
    "zincselenide": "znse",
    "zincsulfide": "zns",
    "zincsulphide": "zns",
}

_UNICODE_SUBSCRIPT_TRANSLATION = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉₊₋₍₎",
    "0123456789+-()",
)

MATERIAL_CANDIDATE_RANKING_SCHEMA_VERSION = "material-candidate-ranking.v1"
MAX_MATERIAL_CANDIDATES = 10

# A route may name a material together with a page/state/source qualifier, for
# example ``ZnSe (Querry)``.  The qualifier is deliberately kept out of the
# normalised chemical key: it is a selector for a real dataset, not a new
# chemical formula.  Keeping the raw spelling in the route lets the downstream
# registry honour the same page instead of silently falling back to another
# measurement.
_MATERIAL_SELECTOR_RE = re.compile(
    r"(?:\(([^()]*)\)|\[([^\[\]]*)\]|\{([^{}]*)\})"
)


def normalize_material_name(name: str) -> str:
    """Return a stable, case-insensitive material key.

    Separators are ignored so that ``fused silica``, ``fused_silica`` and
    ``fused-silica`` resolve through the same alias table.
    """

    if name is None:
        return ""
    translated = str(name).translate(_UNICODE_SUBSCRIPT_TRANSLATION)
    token = re.sub(r"[^a-z0-9]+", "", translated.strip().casefold())
    return _MATERIAL_ALIASES.get(token, token)


def _material_selector_parts(value: Any) -> tuple[str, tuple[str, ...]]:
    """Return the chemical/base key and any parenthesised selectors.

    Parenthetical annotations are common in refractiveindex.info page names
    and in literature (state, preparation, temperature, or source).  They are
    not discarded for callers: the second return value is used to constrain
    RII page matching.  Bare names retain the old normalisation exactly.
    """

    raw = str(value or "").translate(_UNICODE_SUBSCRIPT_TRANSLATION).strip()
    if not raw:
        return "", ()
    selectors: list[str] = []
    for match in _MATERIAL_SELECTOR_RE.finditer(raw):
        for group in match.groups():
            text = " ".join(str(group or "").split()).strip()
            if text:
                selectors.append(text)
    base = _MATERIAL_SELECTOR_RE.sub(" ", raw)
    base = " ".join(base.split()).strip()
    return normalize_material_name(base), tuple(dict.fromkeys(selectors))


def _semantic_material_parts(value: Any) -> tuple[str, tuple[str, ...]]:
    base, selectors = _material_selector_parts(value)
    return _SEMANTIC_MATERIAL_ALIASES.get(base, base), selectors


def _selector_matches_page(page: Mapping[str, Any], selectors: Sequence[str]) -> bool:
    """Whether a page carries every requested state/source qualifier."""

    if not selectors:
        return True
    page_tokens = [
        str(page.get("shelf") or ""),
        str(page.get("book") or ""),
        str(page.get("page") or ""),
        str(page.get("filepath") or ""),
    ]
    folded = [normalize_material_name(token) for token in page_tokens if token]
    raw_folded = [token.casefold() for token in page_tokens if token]
    for selector in selectors:
        selector_key = normalize_material_name(selector)
        selector_raw = str(selector).casefold()
        if not selector_key and not selector_raw:
            continue
        if not any(
            selector_key == token
            or selector_key in token
            or selector_raw in token
            for token in (*folded, *raw_folded)
        ):
            return False
    return True


# The registry cannot import the harness' formula parser: the engine is also a
# standalone package.  This deliberately small chemistry vocabulary is used
# only for semantic candidate matching, never to manufacture optical data.
_ELEMENT_SYMBOLS: frozenset[str] = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I
    Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt
    Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U
    """.split()
)
_ELEMENTS_BY_FOLD: dict[str, str] = {
    symbol.casefold(): symbol for symbol in _ELEMENT_SYMBOLS
}


def _formula_signature(value: Any) -> tuple[tuple[str, int], ...]:
    """Return a stoichiometric formula signature, or ``()`` when not one.

    Formula counts are retained, so the semantic gate can distinguish SiO from
    SiO2 even when both have the same element set.  Names such as ``silica``
    still work through the alias table before parsing.
    """

    raw = str(value or "").translate(_UNICODE_SUBSCRIPT_TRANSLATION).strip()
    raw = _MATERIAL_SELECTOR_RE.sub(" ", raw)
    raw_compact = re.sub(r"[^a-zA-Z0-9]", "", raw)
    raw_key = raw_compact.casefold()
    # Common material words are allowed to enter the formula comparison only
    # through an explicit alias.  This prevents a page title such as
    # ``HIKARI-multipurpose`` from being decomposed into fictitious elements.
    alias = _MATERIAL_ALIASES.get(raw_key) or _SEMANTIC_MATERIAL_ALIASES.get(raw_key)
    if alias:
        compact = alias
        strict_case = False
    else:
        compact = raw_compact
        strict_case = True
        if not compact:
            return ()
        # Lower-case canonical spellings (for example ``sio2`` in a local
        # reference) are accepted when they are short or carry stoichiometric
        # digits.  Long prose-like labels are not formula candidates.
        if compact == compact.casefold():
            if not any(char.isdigit() for char in compact) and len(compact) > 5:
                return ()
            strict_case = False
    if not compact:
        return ()
    groups: dict[str, int] = {}
    index = 0
    while index < len(compact):
        pair = compact[index : index + 2]
        pair_is_valid = (
            len(pair) == 2
            and pair.casefold() in _ELEMENTS_BY_FOLD
            and (not strict_case or (pair[0].isupper() and pair[1].islower()))
        )
        if pair_is_valid:
            symbol = _ELEMENTS_BY_FOLD[pair.casefold()]
            index += 2
        elif (
            compact[index].casefold() in _ELEMENTS_BY_FOLD
            and (not strict_case or compact[index].isupper())
        ):
            symbol = _ELEMENTS_BY_FOLD[compact[index].casefold()]
            index += 1
        else:
            return ()
        start = index
        while index < len(compact) and compact[index].isdigit():
            index += 1
        count = int(compact[start:index] or "1")
        if count <= 0:
            return ()
        groups[symbol] = groups.get(symbol, 0) + count
    return tuple(sorted(groups.items()))


def _formula_elements(signature: Sequence[tuple[str, int]]) -> frozenset[str]:
    return frozenset(symbol for symbol, _ in signature)


def _candidate_formula_signature(candidate: Any) -> tuple[tuple[str, int], ...]:
    reference = getattr(candidate, "ref", None)
    for value in (
        getattr(candidate, "book", None),
        getattr(reference, "name", None),
        getattr(candidate, "material", None),
        getattr(candidate, "normalized_name", None),
    ):
        signature = _formula_signature(value)
        if signature:
            return signature
    filepath = str(getattr(candidate, "filepath", None) or "")
    for token in re.split(r"[\\/._-]+", filepath):
        signature = _formula_signature(token)
        if signature:
            return signature
    return ()


def _candidate_text(candidate: Any) -> str:
    reference = getattr(candidate, "ref", None)
    values = (
        getattr(candidate, "material", None),
        getattr(candidate, "book", None),
        getattr(candidate, "page", None),
        getattr(candidate, "shelf", None),
        getattr(candidate, "filepath", None),
        getattr(reference, "name", None),
    )
    return " ".join(str(value) for value in values if value).casefold()


def _material_match_kind(material: Any, candidate: Any) -> str | None:
    """Classify a candidate without treating a nearby formula as equivalent."""

    query_key, _ = _semantic_material_parts(material)
    query_signature = _formula_signature(material)
    candidate_signature = _candidate_formula_signature(candidate)
    candidate_keys = {
        _semantic_material_parts(value)[0]
        for value in (
            getattr(candidate, "material", None),
            getattr(candidate, "book", None),
            getattr(getattr(candidate, "ref", None), "name", None),
        )
        if value
    }
    if query_signature and candidate_signature:
        if query_signature == candidate_signature:
            return "exact_formula"
        if _formula_elements(query_signature) == _formula_elements(candidate_signature):
            return "same_element_family"
        return None
    if query_key and query_key in candidate_keys:
        return "exact_name"
    if query_signature:
        # A formula request must never be rescued by an unparsed text label.
        return None
    query_text = str(material or "").casefold().strip()
    candidate_text = _candidate_text(candidate)
    if query_text and (
        query_text in candidate_text or candidate_text in query_text
    ):
        return "text_related"
    return None


def _normalise_target_ranges(
    wavelength_range: Sequence[float] | np.ndarray | None = None,
    wavelength_ranges: Sequence[Sequence[float]] | None = None,
) -> tuple[tuple[float, float], ...]:
    if wavelength_range is not None and wavelength_ranges is not None:
        raise ValueError("use wavelength_range or wavelength_ranges, not both")
    if wavelength_range is not None:
        one = _coerce_wavelength_range(wavelength_range)
        return (one,) if one is not None else ()
    if wavelength_ranges is None:
        return ()
    values = list(wavelength_ranges)
    # Be forgiving at this public boundary: callers sometimes pass one pair
    # under the plural name.
    if len(values) == 2 and all(np.isscalar(item) for item in values):
        one = _coerce_wavelength_range(values)
        return (one,) if one is not None else ()
    result: list[tuple[float, float]] = []
    for item in values:
        one = _coerce_wavelength_range(item)
        if one is not None:
            result.append(one)
    return tuple(result)


def _merge_intervals(
    intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    if not intervals:
        return ()
    ordered = sorted((float(low), float(high)) for low, high in intervals)
    merged: list[list[float]] = []
    for low, high in ordered:
        if not merged or low > merged[-1][1]:
            merged.append([low, high])
        else:
            merged[-1][1] = max(merged[-1][1], high)
    return tuple((item[0], item[1]) for item in merged)


def _interval_coverage(
    source_range: tuple[float, float] | None,
    target_ranges: Sequence[tuple[float, float]],
) -> float:
    if source_range is None:
        return 0.0
    low, high = source_range
    if high < low:
        low, high = high, low
    bands = _merge_intervals(target_ranges)
    total = sum(max(0.0, band_high - band_low) for band_low, band_high in bands)
    if total <= 0.0:
        return 0.0
    covered = sum(
        max(0.0, min(high, band_high) - max(low, band_low))
        for band_low, band_high in bands
    )
    return float(max(0.0, min(1.0, covered / total)))


def _points_in_intervals(
    wavelengths: np.ndarray,
    intervals: Sequence[tuple[float, float]],
) -> int:
    if wavelengths.size == 0 or not intervals:
        return 0
    mask = np.zeros(wavelengths.size, dtype=bool)
    for low, high in intervals:
        mask |= (wavelengths >= low) & (wavelengths <= high)
    return int(np.count_nonzero(mask))


def _median_step_in_intervals(
    wavelengths: np.ndarray,
    intervals: Sequence[tuple[float, float]],
) -> float | None:
    if wavelengths.size < 2:
        return None
    pieces: list[np.ndarray] = []
    for low, high in intervals:
        overlap = wavelengths[(wavelengths >= low) & (wavelengths <= high)]
        if overlap.size >= 2:
            pieces.append(np.diff(overlap))
    if not pieces:
        return None
    steps = np.concatenate(pieces)
    steps = steps[np.isfinite(steps) & (steps > 0.0)]
    return float(np.median(steps)) if steps.size else None


def _signature_as_json(signature: Sequence[tuple[str, int]]) -> list[list[Any]]:
    return [[str(symbol), int(count)] for symbol, count in signature]


def _coerce_wavelengths(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim != 1:
        raise ValueError("wavelengths_um must be a one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("wavelengths_um must contain only finite values")
    return array.copy()


def _coerce_wavelength_range(
    wavelength_range: Sequence[float] | np.ndarray | None = None,
    *,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
) -> tuple[float, float] | None:
    if wavelength_range is None and wavelength_min is None and wavelength_max is None:
        return None
    if wavelength_range is not None:
        values = np.asarray(wavelength_range, dtype=np.float64).reshape(-1)
        if values.size != 2:
            raise ValueError("wavelength_range must contain exactly (min_um, max_um)")
        low, high = float(values[0]), float(values[1])
        if wavelength_min is not None or wavelength_max is not None:
            raise ValueError("use wavelength_range or wavelength_min/max, not both")
    else:
        if wavelength_min is None or wavelength_max is None:
            raise ValueError("wavelength_min and wavelength_max must be supplied together")
        low, high = float(wavelength_min), float(wavelength_max)
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError("wavelength_range must be finite and ordered")
    return low, high


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _first_error_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:300]


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    return str(value)


def _dataset_id_as_int(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if token.startswith("+"):
            token = token[1:]
        if token.isdigit():
            return int(token)
    return None


def _provider_key(value: Any) -> str:
    token = str(value).strip().casefold().replace("-", "_")
    aliases = {
        "local": "local_csv",
        "csv": "local_csv",
        "localcsv": "local_csv",
        "local_csv": "local_csv",
        "rii": "rii_sqlite",
        "sqlite": "rii_sqlite",
        "rii_sqlite": "rii_sqlite",
    }
    return aliases.get(token, token)


def _sort_series(
    wavelengths: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(wavelengths, kind="mergesort")
    wavelengths = np.asarray(wavelengths, dtype=np.float64)[order]
    values = np.asarray(values, dtype=np.float64)[order]
    unique, indices = np.unique(wavelengths, return_index=True)
    if unique.size != wavelengths.size:
        # The bundled sources are unique.  Keeping the first value makes a
        # malformed external cache deterministic without silently averaging it.
        wavelengths = unique
        values = values[indices]
    return wavelengths, values


def _interpolate_with_endpoint_audit(
    query: np.ndarray,
    source_wavelengths: np.ndarray,
    source_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    extrapolated = (query < source_wavelengths[0]) | (query > source_wavelengths[-1])
    values = np.interp(query, source_wavelengths, source_values)
    return values.astype(np.float64), extrapolated.astype(bool)


def _range_provenance(
    provenance: dict[str, Any],
    low: float,
    high: float,
    *,
    data_low: float | None = None,
    data_high: float | None = None,
) -> dict[str, Any]:
    result = dict(provenance)
    result["range"] = (float(low), float(high))
    result["range_um"] = (float(low), float(high))
    result["range_min"] = float(low)
    result["range_max"] = float(high)
    result["range_min_um"] = float(low)
    result["range_max_um"] = float(high)
    if data_low is not None and data_high is not None:
        result["data_range_um"] = (float(data_low), float(data_high))
    return result


class MaterialRegistryError(Exception):
    """Base class for material registry failures."""


class MaterialNotFoundError(MaterialRegistryError):
    """No material or dataset matched the requested reference."""

    def __init__(
        self,
        material: str | None = None,
        *,
        provider: str | None = None,
        dataset_id: Any = None,
        message: str | None = None,
    ) -> None:
        self.material = material
        self.provider = provider
        self.dataset_id = dataset_id
        if message is None:
            bits = []
            if material:
                bits.append(f"material={material!r}")
            if provider:
                bits.append(f"provider={provider!r}")
            if dataset_id is not None:
                bits.append(f"dataset_id={dataset_id!r}")
            message = "No optical-constant dataset found" + (f" ({', '.join(bits)})" if bits else "")
        super().__init__(message)


class MaterialRangeError(MaterialRegistryError):
    """The requested wavelengths exceed the selected dataset range."""

    def __init__(
        self,
        material: str | None = None,
        requested_range: tuple[float, float] | None = None,
        available_range: tuple[float, float] | None = None,
        *,
        message: str | None = None,
    ) -> None:
        self.material = material
        self.requested_range = requested_range
        self.available_range = available_range
        if message is None:
            message = (
                f"Requested wavelength range {requested_range!r} for {material!r} "
                f"is outside available range {available_range!r}; "
                "pass allow_extrapolation=True to use endpoint extension"
            )
        super().__init__(message)


class MaterialAmbiguityError(MaterialRegistryError):
    """Several top-ranked datasets are equally suitable."""

    def __init__(
        self,
        material: str | None,
        candidates: Iterable["MaterialCandidate"],
        *,
        message: str | None = None,
    ) -> None:
        self.material = material
        self.candidates = tuple(candidates)
        self.matches = self.candidates
        if message is None:
            labels = ", ".join(
                f"{candidate.provider}:{candidate.dataset_id}"
                for candidate in self.candidates
            )
            message = f"Material reference {material!r} is ambiguous among: {labels}"
        super().__init__(message)


@dataclass(frozen=True)
class MaterialRef:
    """A resolved, source-specific material reference."""

    name: str
    provider: str = "local_csv"
    dataset_id: str | int | None = None
    shelf: str | None = None
    book: str | None = None
    page: str | None = None
    filepath: str | None = None
    normalized_name: str = ""
    range_min: float | None = None
    range_max: float | None = None

    def __post_init__(self) -> None:
        if not self.normalized_name:
            object.__setattr__(
                self,
                "normalized_name",
                _material_selector_parts(self.name)[0]
                or normalize_material_name(self.name),
            )
        object.__setattr__(self, "provider", _provider_key(self.provider or "local_csv"))

    @property
    def canonical_name(self) -> str:
        return self.normalized_name

    @property
    def material(self) -> str:
        return self.normalized_name

    @property
    def pageid(self) -> str | int | None:
        return self.dataset_id

    @property
    def range(self) -> tuple[float, float] | None:
        if self.range_min is None or self.range_max is None:
            return None
        return float(self.range_min), float(self.range_max)

    @property
    def range_um(self) -> tuple[float, float] | None:
        return self.range


@dataclass(frozen=True)
class MaterialCandidate:
    """A searchable dataset plus the ranking evidence used to select it."""

    ref: MaterialRef
    score: float = 0.0
    exact_book: bool = False
    full_coverage: bool = False
    has_n: bool = False
    has_k: bool = False
    points: int = 0
    range_min: float | None = None
    range_max: float | None = None
    provider: str | None = None
    dataset_id: str | int | None = None
    shelf: str | None = None
    book: str | None = None
    page: str | None = None
    filepath: str | None = None
    rank_key: tuple[int, int, int, int] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.provider is None:
            object.__setattr__(self, "provider", self.ref.provider)
        if self.dataset_id is None:
            object.__setattr__(self, "dataset_id", self.ref.dataset_id)
        if self.shelf is None:
            object.__setattr__(self, "shelf", self.ref.shelf)
        if self.book is None:
            object.__setattr__(self, "book", self.ref.book)
        if self.page is None:
            object.__setattr__(self, "page", self.ref.page)
        if self.filepath is None:
            object.__setattr__(self, "filepath", self.ref.filepath)
        if self.range_min is None:
            object.__setattr__(self, "range_min", self.ref.range_min)
        if self.range_max is None:
            object.__setattr__(self, "range_max", self.ref.range_max)
        if not self.rank_key:
            object.__setattr__(
                self,
                "rank_key",
                (
                    int(self.exact_book),
                    int(self.full_coverage),
                    int(self.has_n and self.has_k),
                    int(self.points),
                ),
            )

    @property
    def pageid(self) -> str | int | None:
        return self.dataset_id

    @property
    def normalized_name(self) -> str:
        return self.ref.normalized_name

    @property
    def material(self) -> str:
        return self.ref.normalized_name

    @property
    def range(self) -> tuple[float, float] | None:
        if self.range_min is None or self.range_max is None:
            return None
        return float(self.range_min), float(self.range_max)

    @property
    def range_um(self) -> tuple[float, float] | None:
        return self.range


@dataclass
class SampledOpticalConstants:
    """Sampled n/k arrays together with source and extrapolation audit data."""

    wavelengths_um: np.ndarray
    n: np.ndarray
    k: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)
    extrapolated_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    warnings: list[str] = field(default_factory=list)
    ref: MaterialRef | None = None

    def __post_init__(self) -> None:
        self.wavelengths_um = _coerce_wavelengths(self.wavelengths_um)
        self.n = np.asarray(self.n, dtype=np.float64).reshape(-1)
        self.k = np.asarray(self.k, dtype=np.float64).reshape(-1)
        self.extrapolated_mask = np.asarray(self.extrapolated_mask, dtype=bool).reshape(-1)
        size = self.wavelengths_um.size
        if self.n.size != size or self.k.size != size:
            raise ValueError("wavelengths_um, n and k must have equal lengths")
        if self.extrapolated_mask.size == 0 and size:
            self.extrapolated_mask = np.zeros(size, dtype=bool)
        if self.extrapolated_mask.size != size:
            raise ValueError("extrapolated_mask must have the same length as wavelengths_um")
        self.provenance = dict(self.provenance)
        self.warnings = list(self.warnings)

    @property
    def wavelength_um(self) -> np.ndarray:
        return self.wavelengths_um

    @property
    def wavelengths(self) -> np.ndarray:
        return self.wavelengths_um

    @property
    def nk(self) -> np.ndarray:
        return self.n + 1j * self.k

    @property
    def extrapolated(self) -> bool:
        return bool(np.any(self.extrapolated_mask))


class LocalCsvProvider:
    """Read n/k tables from a local ``materials/*.csv`` directory."""

    provider_name = "local_csv"

    def __init__(self, materials_dir: str | Path | None = None) -> None:
        self.materials_dir = Path(materials_dir or DEFAULT_MATERIALS_DIR)

    def _paths(self) -> list[Path]:
        if not self.materials_dir.exists():
            return []
        return sorted(self.materials_dir.glob("*.csv"), key=lambda path: path.name.casefold())

    @staticmethod
    def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unable to read optical constants CSV {path}: {exc}") from exc
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Optical constants CSV must contain wavelength and n columns: {path}")
        if not np.all(np.isfinite(data[:, : min(data.shape[1], 3)])):
            raise ValueError(f"Optical constants CSV contains non-finite values: {path}")
        wavelengths, n_values = _sort_series(data[:, 0], data[:, 1])
        has_k = data.shape[1] >= 3
        if has_k:
            _, k_values = _sort_series(data[:, 0], data[:, 2])
        else:
            k_values = np.zeros_like(n_values)
        if wavelengths.size == 0:
            raise ValueError(f"Optical constants CSV is empty: {path}")
        return wavelengths, n_values, k_values, has_k

    def _find_path(self, value: Any, dataset_id: Any = None) -> Path | None:
        ref = value.ref if isinstance(value, MaterialCandidate) else value
        if isinstance(ref, MaterialRef):
            if ref.filepath:
                path = Path(ref.filepath)
                if path.exists():
                    return path
            if dataset_id is None:
                dataset_id = ref.dataset_id
            value = ref.name
        if dataset_id is not None:
            token = str(dataset_id).strip()
            if token.startswith("local_csv:"):
                token = token.split(":", 1)[1]
            direct = Path(token)
            if direct.suffix.casefold() == ".csv" and direct.exists():
                return direct
            for path in self._paths():
                if token.casefold() in {path.stem.casefold(), path.name.casefold(), str(path).casefold()}:
                    return path
        if value is None:
            return None
        canonical = _material_selector_parts(value)[0]
        for path in self._paths():
            if normalize_material_name(path.stem) == canonical:
                return path
        return None

    @staticmethod
    def _material_from_query(material: Any) -> tuple[str | None, MaterialRef | None]:
        if isinstance(material, MaterialCandidate):
            return material.ref.normalized_name, material.ref
        if isinstance(material, MaterialRef):
            return material.normalized_name, material
        if material is None:
            return None, None
        return _material_selector_parts(material)[0], None

    def search(
        self,
        material: str | MaterialRef | MaterialCandidate | None = None,
        wavelength_range: Sequence[float] | np.ndarray | None = None,
        *,
        book: str | None = None,
        page: str | None = None,
        dataset_id: str | int | None = None,
        pageid: str | int | None = None,
        wavelength_min: float | None = None,
        wavelength_max: float | None = None,
    ) -> list[MaterialCandidate]:
        requested = _coerce_wavelength_range(
            wavelength_range,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
        canonical, ref = self._material_from_query(material)
        if ref is not None:
            if book is None:
                book = ref.book
            if page is None:
                page = ref.page
            if dataset_id is None:
                dataset_id = ref.dataset_id
        if dataset_id is None:
            dataset_id = pageid

        paths = self._paths()
        if canonical is not None:
            paths = [path for path in paths if normalize_material_name(path.stem) == canonical]
        if dataset_id is not None:
            paths = [
                path
                for path in paths
                if str(dataset_id).casefold()
                in {path.stem.casefold(), path.name.casefold(), str(path).casefold()}
            ]
        if book is not None:
            book_key = normalize_material_name(book)
            paths = [
                path
                for path in paths
                if book_key in {normalize_material_name(path.stem), "local", "csv"}
            ]
        if page is not None:
            page_key = str(page).casefold()
            paths = [
                path
                for path in paths
                if page_key in {path.name.casefold(), path.stem.casefold(), str(path).casefold()}
            ]

        candidates: list[MaterialCandidate] = []
        for path in paths:
            wavelengths, n_values, _, has_k = self._read_csv(path)
            low, high = float(wavelengths[0]), float(wavelengths[-1])
            full = requested is not None and low <= requested[0] and high >= requested[1]
            exact = canonical is not None and normalize_material_name(path.stem) == canonical
            rank_key = (int(exact), int(full), int(has_k), int(wavelengths.size))
            ref_obj = MaterialRef(
                name=normalize_material_name(path.stem),
                provider=self.provider_name,
                dataset_id=path.stem,
                page=path.name,
                filepath=str(path),
                range_min=low,
                range_max=high,
            )
            candidates.append(
                MaterialCandidate(
                    ref=ref_obj,
                    score=float(
                        int(exact) * 1_000_000_000_000
                        + int(full) * 1_000_000_000
                        + int(has_k) * 1_000_000
                        + int(wavelengths.size)
                    ),
                    exact_book=exact,
                    full_coverage=full,
                    has_n=True,
                    has_k=has_k,
                    points=int(wavelengths.size),
                    range_min=low,
                    range_max=high,
                    rank_key=rank_key,
                )
            )
        candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
        return candidates

    def resolve(
        self,
        material: str | MaterialRef | MaterialCandidate | None = None,
        *,
        dataset_id: str | int | None = None,
        wavelength_range: Sequence[float] | np.ndarray | None = None,
    ) -> MaterialRef:
        if isinstance(material, MaterialCandidate):
            return material.ref
        if isinstance(material, MaterialRef):
            return material
        candidates = self.search(material, wavelength_range, dataset_id=dataset_id)
        if not candidates:
            raise MaterialNotFoundError(str(material), provider=self.provider_name, dataset_id=dataset_id)
        return candidates[0].ref

    def sample(
        self,
        material: str | MaterialRef | MaterialCandidate,
        wavelengths_um: Sequence[float] | np.ndarray | None = None,
        *,
        allow_extrapolation: bool = False,
        wavelengths: Sequence[float] | np.ndarray | None = None,
        wavelength_um: Sequence[float] | np.ndarray | None = None,
        dataset_id: str | int | None = None,
        pageid: str | int | None = None,
    ) -> SampledOpticalConstants:
        if wavelengths_um is None:
            wavelengths_um = wavelengths if wavelengths is not None else wavelength_um
        elif wavelengths is not None or wavelength_um is not None:
            raise ValueError("use wavelengths_um, wavelengths or wavelength_um, not more than one")
        if wavelengths_um is None:
            raise ValueError("wavelengths_um is required")
        query = _coerce_wavelengths(wavelengths_um)
        if dataset_id is None:
            dataset_id = pageid

        if isinstance(material, MaterialCandidate):
            ref = material.ref
        elif isinstance(material, MaterialRef):
            ref = material
        else:
            ref = self.resolve(material, dataset_id=dataset_id)
        if ref.provider != self.provider_name:
            raise MaterialNotFoundError(ref.name, provider=self.provider_name, dataset_id=ref.dataset_id)
        path = self._find_path(ref, dataset_id=dataset_id)
        if path is None:
            raise MaterialNotFoundError(ref.name, provider=self.provider_name, dataset_id=dataset_id or ref.dataset_id)

        source_wavelengths, n_source, k_source, has_k = self._read_csv(path)
        available = (float(source_wavelengths[0]), float(source_wavelengths[-1]))
        if query.size:
            requested = (float(np.min(query)), float(np.max(query)))
        else:
            requested = available
        extrapolated = (query < available[0]) | (query > available[1])
        if np.any(extrapolated) and not allow_extrapolation:
            raise MaterialRangeError(ref.name, requested, available)

        n_values, n_mask = _interpolate_with_endpoint_audit(query, source_wavelengths, n_source)
        k_values, k_mask = _interpolate_with_endpoint_audit(query, source_wavelengths, k_source)
        extrapolated = n_mask | k_mask
        warnings: list[str] = []
        if np.any(extrapolated) and allow_extrapolation:
            warnings.append(
                f"Local CSV {path.name}: explicit extrapolation via endpoint extension used outside "
                f"{available[0]:g}-{available[1]:g} um"
            )
        provenance = _range_provenance(
            {
                "provider": self.provider_name,
                "dataset_id": path.stem,
                "pageid": None,
                "shelf": None,
                "book": None,
                "page": path.name,
                "filepath": str(path),
                "wavelength_unit": "um",
                "has_n": True,
                "has_k": bool(has_k),
                "points": int(source_wavelengths.size),
            },
            available[0],
            available[1],
        )
        return SampledOpticalConstants(
            wavelengths_um=query,
            n=n_values,
            k=k_values,
            provenance=provenance,
            extrapolated_mask=extrapolated,
            warnings=warnings,
            ref=ref,
        )


class RiiSqliteProvider:
    """Read the bundled RII cache using only Python's ``sqlite3`` module."""

    provider_name = "rii_sqlite"

    def catalog_status(self) -> dict[str, Any]:
        """Return reproducibility metadata for the bundled RII mirror."""

        if not self.db_path.exists():
            return {
                "provider": self.provider_name,
                "database_path": str(self.db_path),
                "available": False,
                "page_count": 0,
            }
        connection = sqlite3.connect(str(self.db_path))
        try:
            page_count = int(connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0])
            nk_points = int(
                connection.execute("SELECT COUNT(*) FROM refractiveindex").fetchone()[0]
            )
            k_points = int(connection.execute("SELECT COUNT(*) FROM extcoeff").fetchone()[0])
        finally:
            connection.close()
        stat = self.db_path.stat()
        digest = hashlib.sha256()
        with self.db_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "provider": self.provider_name,
            "database_path": str(self.db_path.resolve()),
            "available": True,
            "page_count": page_count,
            "refractive_index_point_count": nk_points,
            "extinction_coefficient_point_count": k_points,
            "database_size_bytes": int(stat.st_size),
            "database_mtime_epoch": float(stat.st_mtime),
            "database_sha256": digest.hexdigest(),
            "upstream": "https://refractiveindex.info and https://github.com/polyanskiy/refractiveindex.info-database",
            "license": "CC0-1.0",
        }

    _PAGE_COLUMNS = (
        "pageid",
        "shelf",
        "book",
        "page",
        "filepath",
        "hasrefractive",
        "hasextinction",
        "rangeMin",
        "rangeMax",
        "points",
    )

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_RII_DB_PATH)

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"RII SQLite cache does not exist: {self.db_path}")
        return sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)

    @staticmethod
    def _row_to_page(row: Sequence[Any]) -> dict[str, Any]:
        return dict(zip(RiiSqliteProvider._PAGE_COLUMNS, row))

    @staticmethod
    def _matches_material(page: Mapping[str, Any], material_key: str | None) -> bool:
        if material_key is None:
            return True
        tokens = [page.get("book"), page.get("page")]
        filepath = str(page.get("filepath") or "")
        tokens.extend(re.split(r"[\\/]", filepath))
        return any(normalize_material_name(str(token)) == material_key for token in tokens if token)

    @staticmethod
    def _matches_text(value: Any, query: str | None) -> bool:
        if query is None:
            return True
        raw = str(value or "")
        return raw.casefold() == str(query).casefold() or normalize_material_name(raw) == normalize_material_name(query)

    def search(
        self,
        material: str | MaterialRef | MaterialCandidate | None = None,
        wavelength_range: Sequence[float] | np.ndarray | None = None,
        *,
        book: str | None = None,
        page: str | int | None = None,
        dataset_id: str | int | None = None,
        pageid: str | int | None = None,
        wavelength_min: float | None = None,
        wavelength_max: float | None = None,
    ) -> list[MaterialCandidate]:
        requested = _coerce_wavelength_range(
            wavelength_range,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
        ref: MaterialRef | None = None
        selector_parts: tuple[str, ...] = ()
        if isinstance(material, MaterialCandidate):
            ref = material.ref
        elif isinstance(material, MaterialRef):
            ref = material
        if ref is not None:
            material_key = ref.normalized_name
            if book is None:
                book = ref.book
            if page is None:
                page = ref.page
            if dataset_id is None:
                dataset_id = ref.dataset_id
        elif material is None:
            material_key = None
        else:
            material_key, selector_parts = _material_selector_parts(str(material))
        if dataset_id is None:
            dataset_id = pageid
        if dataset_id is None and isinstance(page, (int, np.integer)):
            dataset_id = int(page)
            page = None
        requested_pageid = _dataset_id_as_int(dataset_id)

        if not self.db_path.exists():
            return []
        candidates: list[MaterialCandidate] = []
        with self._connect() as connection:
            sql = (
                "SELECT pageid,shelf,book,page,filepath,hasrefractive,hasextinction,"
                "rangeMin,rangeMax,points FROM pages"
            )
            parameters: list[Any] = []
            if requested_pageid is not None:
                sql += " WHERE pageid = ?"
                parameters.append(requested_pageid)
            rows = connection.execute(sql, parameters).fetchall()

        for raw_row in rows:
            page_record = self._row_to_page(raw_row)
            if requested_pageid is None and dataset_id is not None:
                if str(page_record["pageid"]).casefold() != str(dataset_id).casefold():
                    continue
            if not self._matches_material(page_record, material_key):
                continue
            if selector_parts and not _selector_matches_page(page_record, selector_parts):
                continue
            if book is not None and not self._matches_text(page_record["book"], book):
                continue
            if page is not None and not (
                self._matches_text(page_record["page"], page)
                or self._matches_text(page_record["filepath"], page)
            ):
                continue
            low = _maybe_float(page_record["rangeMin"])
            high = _maybe_float(page_record["rangeMax"])
            full = (
                requested is not None
                and low is not None
                and high is not None
                and low <= requested[0]
                and high >= requested[1]
            )
            has_n = bool(page_record["hasrefractive"])
            has_k = bool(page_record["hasextinction"])
            points = int(page_record["points"] or 0)
            exact_book = material_key is not None and normalize_material_name(str(page_record["book"])) == material_key
            rank_key = (int(exact_book), int(full), int(has_n and has_k), points)
            name = material_key or normalize_material_name(str(page_record["book"]))
            ref_obj = MaterialRef(
                name=name,
                provider=self.provider_name,
                dataset_id=int(page_record["pageid"]),
                shelf=str(page_record["shelf"]) if page_record["shelf"] is not None else None,
                book=str(page_record["book"]) if page_record["book"] is not None else None,
                page=str(page_record["page"]) if page_record["page"] is not None else None,
                filepath=str(page_record["filepath"]) if page_record["filepath"] is not None else None,
                range_min=low,
                range_max=high,
            )
            candidates.append(
                MaterialCandidate(
                    ref=ref_obj,
                    score=float(
                        int(exact_book) * 1_000_000_000_000
                        + int(full) * 1_000_000_000
                        + int(has_n and has_k) * 1_000_000
                        + points
                    ),
                    exact_book=exact_book,
                    full_coverage=full,
                    has_n=has_n,
                    has_k=has_k,
                    points=points,
                    range_min=low,
                    range_max=high,
                    rank_key=rank_key,
                )
            )
        candidates.sort(key=lambda candidate: str(candidate.dataset_id))
        candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
        return candidates

    def resolve(
        self,
        material: str | MaterialRef | MaterialCandidate,
        *,
        dataset_id: str | int | None = None,
        pageid: str | int | None = None,
        book: str | None = None,
        page: str | None = None,
        wavelength_range: Sequence[float] | np.ndarray | None = None,
    ) -> MaterialRef:
        if isinstance(material, MaterialCandidate):
            return material.ref
        if isinstance(material, MaterialRef):
            return material
        candidates = self.search(
            material,
            wavelength_range,
            book=book,
            page=page,
            dataset_id=dataset_id,
            pageid=pageid,
        )
        return _choose_best(candidates, str(material), provider=self.provider_name).ref

    def _load_page_and_series(
        self,
        pageid: int,
    ) -> tuple[dict[str, Any], list[tuple[float, float]], list[tuple[float, float]]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT pageid,shelf,book,page,filepath,hasrefractive,hasextinction,"
                "rangeMin,rangeMax,points FROM pages WHERE pageid = ?",
                (pageid,),
            ).fetchone()
            if row is None:
                raise MaterialNotFoundError(
                    provider=self.provider_name,
                    dataset_id=pageid,
                    message=f"RII dataset pageid={pageid!r} was not found",
                )
            page_record = self._row_to_page(row)
            n_rows = connection.execute(
                "SELECT wave,refindex FROM refractiveindex WHERE pageid = ? ORDER BY wave",
                (pageid,),
            ).fetchall()
            k_rows = connection.execute(
                "SELECT wave,coeff FROM extcoeff WHERE pageid = ? ORDER BY wave",
                (pageid,),
            ).fetchall()
        return page_record, n_rows, k_rows

    @staticmethod
    def _clean_rows(rows: Iterable[Sequence[Any]]) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(list(rows), dtype=np.float64)
        if values.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        values = values.reshape(-1, 2)
        finite = np.isfinite(values).all(axis=1)
        values = values[finite]
        if values.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        return _sort_series(values[:, 0], values[:, 1])

    def sample(
        self,
        material: str | int | MaterialRef | MaterialCandidate | None = None,
        wavelengths_um: Sequence[float] | np.ndarray | None = None,
        *,
        allow_extrapolation: bool = False,
        wavelengths: Sequence[float] | np.ndarray | None = None,
        wavelength_um: Sequence[float] | np.ndarray | None = None,
        dataset_id: str | int | None = None,
        pageid: str | int | None = None,
    ) -> SampledOpticalConstants:
        if wavelengths_um is None:
            wavelengths_um = wavelengths if wavelengths is not None else wavelength_um
        elif wavelengths is not None or wavelength_um is not None:
            raise ValueError("use wavelengths_um, wavelengths or wavelength_um, not more than one")
        if wavelengths_um is None:
            raise ValueError("wavelengths_um is required")
        query = _coerce_wavelengths(wavelengths_um)

        ref: MaterialRef
        if isinstance(material, MaterialCandidate):
            ref = material.ref
        elif isinstance(material, MaterialRef):
            ref = material
        else:
            selected_id = dataset_id if dataset_id is not None else pageid
            if selected_id is None and isinstance(material, (int, np.integer)):
                selected_id = int(material)
            if selected_id is not None:
                candidates = self.search(material if not isinstance(material, (int, np.integer)) else None, dataset_id=selected_id)
                ref = _choose_best(candidates, str(material), provider=self.provider_name).ref
            elif material is not None:
                ref = self.resolve(material)
            else:
                raise MaterialNotFoundError(provider=self.provider_name)
        if ref.provider != self.provider_name:
            raise MaterialNotFoundError(ref.name, provider=self.provider_name, dataset_id=ref.dataset_id)
        selected_pageid = dataset_id if dataset_id is not None else pageid
        if selected_pageid is None:
            selected_pageid = ref.dataset_id
        pageid_int = _dataset_id_as_int(selected_pageid)
        if pageid_int is None:
            raise MaterialNotFoundError(ref.name, provider=self.provider_name, dataset_id=selected_pageid)

        page_record, n_rows, k_rows = self._load_page_and_series(pageid_int)
        n_wavelengths, n_source = self._clean_rows(n_rows)
        k_wavelengths, k_source = self._clean_rows(k_rows)
        if n_wavelengths.size == 0:
            raise MaterialNotFoundError(
                ref.name,
                provider=self.provider_name,
                dataset_id=pageid_int,
                message=f"RII dataset pageid={pageid_int!r} has no refractive-index data",
            )
        has_k = k_wavelengths.size > 0
        if has_k:
            data_low = max(float(n_wavelengths[0]), float(k_wavelengths[0]))
            data_high = min(float(n_wavelengths[-1]), float(k_wavelengths[-1]))
            if data_low > data_high:
                raise MaterialRangeError(
                    ref.name,
                    (data_low, data_high),
                    (float(n_wavelengths[0]), float(n_wavelengths[-1])),
                    message=f"RII dataset pageid={pageid_int!r} has no common n/k wavelength range",
                )
        else:
            data_low, data_high = float(n_wavelengths[0]), float(n_wavelengths[-1])
        available_low = _maybe_float(page_record["rangeMin"])
        available_high = _maybe_float(page_record["rangeMax"])
        if available_low is None or available_high is None:
            available_low, available_high = data_low, data_high
        # Sampling uses the actual common n/k range.  The page metadata is
        # retained separately in provenance for auditability.
        available = (data_low, data_high)
        requested = (float(np.min(query)), float(np.max(query))) if query.size else available
        extrapolated = (query < available[0]) | (query > available[1])
        if np.any(extrapolated) and not allow_extrapolation:
            raise MaterialRangeError(ref.name, requested, available)

        n_values, n_mask = _interpolate_with_endpoint_audit(query, n_wavelengths, n_source)
        if has_k:
            k_values, k_mask = _interpolate_with_endpoint_audit(query, k_wavelengths, k_source)
        else:
            k_values = np.zeros_like(query)
            k_mask = np.zeros_like(query, dtype=bool)
        extrapolated = n_mask | k_mask
        warnings: list[str] = []
        if np.any(extrapolated) and allow_extrapolation:
            warnings.append(
                f"RII dataset pageid={pageid_int}: explicit extrapolation via endpoint extension used outside "
                f"{available[0]:g}-{available[1]:g} um"
            )
        provenance = _range_provenance(
            {
                "provider": self.provider_name,
                "dataset_id": pageid_int,
                "pageid": pageid_int,
                "shelf": page_record["shelf"],
                "book": page_record["book"],
                "page": page_record["page"],
                "filepath": page_record["filepath"],
                "wavelength_unit": "um",
                "has_n": True,
                "has_k": bool(has_k),
                "points": int(page_record["points"] or n_wavelengths.size),
                "metadata_range_um": (float(available_low), float(available_high)),
            },
            float(available_low),
            float(available_high),
            data_low=data_low,
            data_high=data_high,
        )
        final_ref = MaterialRef(
            name=ref.name or normalize_material_name(str(page_record["book"])),
            provider=self.provider_name,
            dataset_id=pageid_int,
            shelf=str(page_record["shelf"]) if page_record["shelf"] is not None else None,
            book=str(page_record["book"]) if page_record["book"] is not None else None,
            page=str(page_record["page"]) if page_record["page"] is not None else None,
            filepath=str(page_record["filepath"]) if page_record["filepath"] is not None else None,
            range_min=float(available_low),
            range_max=float(available_high),
        )
        return SampledOpticalConstants(
            wavelengths_um=query,
            n=n_values,
            k=k_values,
            provenance=provenance,
            extrapolated_mask=extrapolated,
            warnings=warnings,
            ref=final_ref,
        )


def _choose_best(
    candidates: Sequence[MaterialCandidate],
    material: str | None,
    *,
    provider: str | None = None,
) -> MaterialCandidate:
    if not candidates:
        raise MaterialNotFoundError(material, provider=provider)
    top_key = candidates[0].rank_key
    tied = [candidate for candidate in candidates if candidate.rank_key == top_key]
    if len(tied) > 1:
        raise MaterialAmbiguityError(material, tied)
    return candidates[0]


class MaterialRegistry:
    """Unified search/resolve/sample facade with local-first policy."""

    def __init__(
        self,
        materials_dir: str | Path | None = None,
        rii_db_path: str | Path | None = None,
        *,
        local_provider: LocalCsvProvider | None = None,
        rii_provider: RiiSqliteProvider | None = None,
    ) -> None:
        self.local_provider = local_provider or LocalCsvProvider(materials_dir)
        self.rii_provider = rii_provider or RiiSqliteProvider(rii_db_path)
        self.providers = {
            "local_csv": self.local_provider,
            "rii_sqlite": self.rii_provider,
        }

    def _provider(self, provider: Any) -> Any:
        if provider is None:
            return None
        if hasattr(provider, "search") and hasattr(provider, "sample"):
            return provider
        key = _provider_key(provider)
        try:
            return self.providers[key]
        except KeyError as exc:
            raise ValueError(f"Unknown material provider: {provider!r}") from exc

    def catalog_status(self) -> dict[str, Any]:
        return {
            "local_csv": {
                "provider": "local_csv",
                "materials_dir": str(self.local_provider.materials_dir.resolve()),
                "dataset_count": len(self.local_provider._paths()),
            },
            "rii_sqlite": self.rii_provider.catalog_status(),
        }

    @staticmethod
    def _ref_provider(ref: MaterialRef) -> str:
        return _provider_key(ref.provider)

    def search(
        self,
        material: str | MaterialRef | MaterialCandidate | None = None,
        wavelength_range: Sequence[float] | np.ndarray | None = None,
        *,
        provider: Any = None,
        book: str | None = None,
        page: str | int | None = None,
        dataset_id: str | int | None = None,
        pageid: str | int | None = None,
        wavelength_min: float | None = None,
        wavelength_max: float | None = None,
    ) -> list[MaterialCandidate]:
        if isinstance(material, MaterialCandidate):
            material_ref = material.ref
        elif isinstance(material, MaterialRef):
            material_ref = material
        else:
            material_ref = None
        if provider is None and material_ref is not None:
            provider = material_ref.provider
        selected = self._provider(provider)
        if selected is not None:
            return selected.search(
                material,
                wavelength_range,
                book=book,
                page=page,
                dataset_id=dataset_id,
                pageid=pageid,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
            )

        # Parenthesised state/source/page qualifiers belong to the RII
        # selector namespace.  Do not let the local generic CSV shadow a
        # request such as ``SiO2 (Gao)``.
        name_selectors = (
            _material_selector_parts(material)[1]
            if isinstance(material, str)
            else ()
        )
        if name_selectors:
            return self.rii_provider.search(
                material,
                wavelength_range,
                book=book,
                page=page,
                dataset_id=dataset_id,
                pageid=pageid,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
            )

        # Explicit RII selectors are intentionally not applied to local CSVs.
        if dataset_id is not None or pageid is not None or page is not None or book is not None:
            return self.rii_provider.search(
                material,
                wavelength_range,
                book=book,
                page=page,
                dataset_id=dataset_id,
                pageid=pageid,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
            )

        local_candidates = self.local_provider.search(
            material,
            wavelength_range,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
        rii_candidates = self.rii_provider.search(
            material,
            wavelength_range,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
        return local_candidates + rii_candidates

    @staticmethod
    def _candidate_identity(candidate: MaterialCandidate) -> tuple[str, str]:
        provider = _provider_key(getattr(candidate, "provider", None) or "")
        dataset_id = getattr(candidate, "dataset_id", None)
        if dataset_id is None:
            dataset_id = getattr(candidate, "pageid", None)
        if dataset_id is None:
            dataset_id = getattr(candidate, "filepath", None) or getattr(
                candidate, "page", None
            )
        return provider, str(dataset_id)

    def _candidate_grid(
        self,
        candidate: MaterialCandidate,
    ) -> tuple[np.ndarray, np.ndarray, str, str | None]:
        """Load the actual n and k wavelength grids for a search hit.

        Ranking must inspect the data that will later be sampled rather than
        trusting only the page metadata.  A metadata-only fallback is kept for
        a damaged cache or a small injected provider, but it is surfaced in
        the returned audit row and never pretends to be a measured grid.
        """

        provider = _provider_key(getattr(candidate, "provider", None) or "")
        try:
            if provider == "local_csv":
                path = self.local_provider._find_path(
                    candidate, dataset_id=getattr(candidate, "dataset_id", None)
                )
                if path is None:
                    raise MaterialNotFoundError(
                        candidate.material,
                        provider=provider,
                        dataset_id=candidate.dataset_id,
                    )
                wavelengths, _, _, has_k = self.local_provider._read_csv(path)
                return (
                    np.asarray(wavelengths, dtype=np.float64),
                    np.asarray(wavelengths if has_k else (), dtype=np.float64),
                    "source_grid",
                    None,
                )
            if provider == "rii_sqlite":
                pageid = _dataset_id_as_int(getattr(candidate, "dataset_id", None))
                if pageid is None:
                    raise MaterialNotFoundError(
                        candidate.material,
                        provider=provider,
                        dataset_id=getattr(candidate, "dataset_id", None),
                    )
                _, n_rows, k_rows = self.rii_provider._load_page_and_series(pageid)
                n_wavelengths, _ = self.rii_provider._clean_rows(n_rows)
                k_wavelengths, _ = self.rii_provider._clean_rows(k_rows)
                return (
                    n_wavelengths,
                    k_wavelengths,
                    "source_grid",
                    None,
                )
            raise MaterialNotFoundError(
                candidate.material,
                provider=provider or None,
                dataset_id=getattr(candidate, "dataset_id", None),
            )
        except Exception as exc:
            low = _maybe_float(getattr(candidate, "range_min", None))
            high = _maybe_float(getattr(candidate, "range_max", None))
            points = max(1, int(getattr(candidate, "points", 0) or 0))
            if low is not None and high is not None and low <= high:
                fallback = np.linspace(low, high, points, dtype=np.float64)
                return (
                    fallback if bool(getattr(candidate, "has_n", False)) else np.zeros(0),
                    fallback if bool(getattr(candidate, "has_k", False)) else np.zeros(0),
                    "metadata_fallback",
                    _first_error_line(exc),
                )
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), "unavailable", _first_error_line(exc)

    def rank_candidates(
        self,
        material: str,
        wavelength_range: Sequence[float] | np.ndarray | None = None,
        *,
        wavelength_ranges: Sequence[Sequence[float]] | None = None,
        limit: int = MAX_MATERIAL_CANDIDATES,
    ) -> list[dict[str, Any]]:
        """Return up to ten physically ranked local dataset candidates.

        The method is intentionally separate from :meth:`search` and
        :meth:`resolve`: those APIs retain their historical source-priority
        and equal-rank ambiguity semantics, while this interface is a
        documented candidate feed for the route-material selector.  The score
        has exactly three equal-weight components -- band coverage, n/k
        completeness, and sampling density -- and uses no material name as a
        quality signal.
        """

        if material is None or not str(material).strip():
            return []
        target_ranges = _normalise_target_ranges(wavelength_range, wavelength_ranges)
        limit_value = min(MAX_MATERIAL_CANDIDATES, max(1, int(limit)))
        _, selector_parts = _semantic_material_parts(str(material))

        candidates: list[MaterialCandidate] = []
        seen: set[tuple[str, str]] = set()

        def add(candidate: MaterialCandidate) -> None:
            identity = self._candidate_identity(candidate)
            if identity in seen:
                return
            kind = _material_match_kind(material, candidate)
            if kind is None:
                return
            seen.add(identity)
            candidates.append(candidate)

        # The exact/base query is both faster and more faithful to a named
        # material.  It also preserves a page qualifier such as ``(Querry)``.
        for candidate in self.search(material):
            add(candidate)

        # If the exact query did not fill the bounded feed, inspect the full
        # local registry for spelling/alias/formula-family relatives.  A
        # qualifier is a hard constraint and therefore never falls through to
        # a different page or state.
        has_exact_identity = any(
            _material_match_kind(material, candidate) in {"exact_formula", "exact_name"}
            for candidate in candidates
        )
        if not selector_parts and not has_exact_identity:
            for candidate in self.search(None):
                add(candidate)
                if len(candidates) >= limit_value:
                    break
        if not candidates:
            return []

        quality_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            n_grid, k_grid, grid_source, grid_error = self._candidate_grid(candidate)
            n_grid = np.asarray(n_grid, dtype=np.float64).reshape(-1)
            k_grid = np.asarray(k_grid, dtype=np.float64).reshape(-1)
            n_grid = n_grid[np.isfinite(n_grid)]
            k_grid = k_grid[np.isfinite(k_grid)]
            n_range = (
                (float(np.min(n_grid)), float(np.max(n_grid)))
                if n_grid.size
                else None
            )
            k_range = (
                (float(np.min(k_grid)), float(np.max(k_grid)))
                if k_grid.size
                else None
            )
            metadata_range = candidate.range
            if metadata_range is None:
                metadata_range = n_range or k_range
            grid_ranges = [item for item in (n_range, k_range) if item is not None]
            grid_range = (
                (min(item[0] for item in grid_ranges), max(item[1] for item in grid_ranges))
                if grid_ranges
                else None
            )
            ranking_ranges = target_ranges or (
                (metadata_range,) if metadata_range is not None else ()
            )
            coverage_score = _interval_coverage(metadata_range, ranking_ranges)
            n_coverage = _interval_coverage(n_range, ranking_ranges)
            k_coverage = _interval_coverage(k_range, ranking_ranges)
            n_points = _points_in_intervals(n_grid, ranking_ranges)
            k_points = _points_in_intervals(k_grid, ranking_ranges)
            steps = [
                step
                for step in (
                    _median_step_in_intervals(n_grid, ranking_ranges),
                    _median_step_in_intervals(k_grid, ranking_ranges),
                )
                if step is not None and step > 0.0
            ]
            effective_step = max(steps) if steps else None
            if effective_step is not None:
                density = 1.0 / effective_step
            else:
                target_length = sum(
                    max(0.0, high - low) for low, high in _merge_intervals(ranking_ranges)
                )
                in_band_points = max(n_points, k_points)
                density = (
                    float(in_band_points) / target_length
                    if target_length > 0.0
                    else 0.0
                )
            match_kind = _material_match_kind(material, candidate) or "unknown"
            query_signature = _formula_signature(material)
            candidate_signature = _candidate_formula_signature(candidate)
            formula_exact = bool(
                query_signature
                and candidate_signature
                and query_signature == candidate_signature
            )
            if not query_signature:
                formula_exact = match_kind in {"exact_name", "exact_formula"}
            quality_rows.append(
                {
                    "_candidate": candidate,
                    "_density": float(max(0.0, density)),
                    "_coverage_score": float(coverage_score),
                    "_n_coverage_fraction": float(n_coverage),
                    "_k_coverage_fraction": float(k_coverage),
                    "_n_points_in_band": int(n_points),
                    "_k_points_in_band": int(k_points),
                    "_effective_step_um": (
                        None if effective_step is None else float(effective_step)
                    ),
                    "_match_kind": match_kind,
                    "_formula_exact": formula_exact,
                    "_query_signature": query_signature,
                    "_candidate_signature": candidate_signature,
                    "_n_range": n_range,
                    "_k_range": k_range,
                    "_grid_range": grid_range,
                    "_grid_source": grid_source,
                    "_grid_error": grid_error,
                }
            )

        # A complete n/k dataset is a physical eligibility gate for this
        # route-binding interface.  Partial datasets remain available when no
        # candidate can cover the requested bands, but a very fine partial
        # table must not outrank a complete table merely through the sampling
        # component and then cause a later fail-closed execution.
        native_rows = [
            row
            for row in quality_rows
            if row["_grid_source"] == "source_grid"
            and row["_coverage_score"] >= 1.0 - 1e-12
            and row["_n_coverage_fraction"] >= 1.0 - 1e-12
            and row["_k_coverage_fraction"] >= 1.0 - 1e-12
        ]
        native_pool_used = bool(native_rows)
        ranking_rows = native_rows or quality_rows
        max_density = max((row["_density"] for row in ranking_rows), default=0.0)
        for row in ranking_rows:
            sampling_score = (
                min(1.0, row["_density"] / max_density)
                if max_density > 0.0
                else 0.0
            )
            nk_completeness = (
                row["_n_coverage_fraction"] + row["_k_coverage_fraction"]
            ) / 2.0
            physical_score = (
                row["_coverage_score"] + nk_completeness + sampling_score
            ) / 3.0
            row["_sampling_score"] = float(sampling_score)
            row["_n_k_completeness_score"] = float(nk_completeness)
            row["_physical_score"] = float(physical_score)

        # Only the three physical components participate before the final
        # provider/id tie break.  In particular, exact_book and material name
        # never rescue a physically inferior dataset.
        ranking_rows.sort(
            key=lambda row: (
                -row["_physical_score"],
                -row["_coverage_score"],
                -row["_n_k_completeness_score"],
                -row["_sampling_score"],
                _provider_key(getattr(row["_candidate"], "provider", None) or ""),
                str(getattr(row["_candidate"], "dataset_id", "")),
            )
        )

        output: list[dict[str, Any]] = []
        for rank, row in enumerate(ranking_rows[:limit_value], 1):
            candidate = row["_candidate"]
            provider = _provider_key(getattr(candidate, "provider", None) or "")
            dataset_id = getattr(candidate, "dataset_id", None)
            book = getattr(candidate, "book", None)
            page = getattr(candidate, "page", None)
            material_display = str(book or getattr(candidate, "material", "") or material)
            metadata_range = candidate.range
            if metadata_range is None:
                metadata_range = row["_n_range"] or row["_k_range"]
            full_coverage = bool(row["_coverage_score"] >= 1.0 - 1e-12)
            nk_full = bool(
                row["_n_coverage_fraction"] >= 1.0 - 1e-12
                and row["_k_coverage_fraction"] >= 1.0 - 1e-12
            )
            interpolation_available = bool(
                row["_grid_source"] == "source_grid"
                and row["_n_range"] is not None
                and row["_k_range"] is not None
            )
            output.append(
                {
                    "ranking_schema_version": MATERIAL_CANDIDATE_RANKING_SCHEMA_VERSION,
                    "rank": int(rank),
                    "material": material_display,
                    "resolved_material": str(getattr(candidate, "material", "") or material),
                    "semantic_match": row["_match_kind"],
                    "formula_exact": bool(row["_formula_exact"]),
                    "formula_signature": _signature_as_json(row["_candidate_signature"]),
                    "requested_formula_signature": _signature_as_json(row["_query_signature"]),
                    "provider": provider,
                    "dataset_id": _json_scalar(dataset_id),
                    "shelf": _json_scalar(getattr(candidate, "shelf", None)),
                    "book": _json_scalar(book),
                    "page": _json_scalar(page),
                    "filepath": _json_scalar(getattr(candidate, "filepath", None)),
                    "range_um": (
                        [float(metadata_range[0]), float(metadata_range[1])]
                        if metadata_range is not None
                        else None
                    ),
                    "grid_range_um": (
                        [float(row["_grid_range"][0]), float(row["_grid_range"][1])]
                        if row["_grid_range"] is not None
                        else None
                    ),
                    "n_range_um": (
                        [float(row["_n_range"][0]), float(row["_n_range"][1])]
                        if row["_n_range"] is not None
                        else None
                    ),
                    "k_range_um": (
                        [float(row["_k_range"][0]), float(row["_k_range"][1])]
                        if row["_k_range"] is not None
                        else None
                    ),
                    "full_coverage": full_coverage,
                    "native_nk_full_coverage": nk_full,
                    "has_n": bool(row["_n_range"] is not None),
                    "has_k": bool(row["_k_range"] is not None),
                    "points": int(max(row["_n_points_in_band"], row["_k_points_in_band"], getattr(candidate, "points", 0) or 0)),
                    "n_points_in_band": int(row["_n_points_in_band"]),
                    "k_points_in_band": int(row["_k_points_in_band"]),
                    "band_coverage_fraction": float(row["_coverage_score"]),
                    "target_band_coverage_score": float(row["_coverage_score"]),
                    "n_coverage_fraction": float(row["_n_coverage_fraction"]),
                    "k_coverage_fraction": float(row["_k_coverage_fraction"]),
                    "n_k_completeness_score": float(row["_n_k_completeness_score"]),
                    "effective_step_um": row["_effective_step_um"],
                    "sampling_density": float(row["_density"]),
                    "sampling_score": float(row["_sampling_score"]),
                    "score": float(row["_physical_score"]),
                    "physical_score": float(row["_physical_score"]),
                    "grid_source": row["_grid_source"],
                    "grid_load_error": row["_grid_error"],
                    "interpolation_available": interpolation_available,
                    "interpolation_mode": (
                        "native"
                        if nk_full
                        else "endpoint_extension_available"
                        if interpolation_available
                        else "unavailable"
                    ),
                    "native_candidate_pool_used": native_pool_used,
                    "partial_candidate_fallback": not native_pool_used,
                    "rank_key": [int(value) for value in getattr(candidate, "rank_key", ())],
                }
            )
        return output

    def interpolate_nk(
        self,
        material: str | MaterialRef | MaterialCandidate,
        wavelengths_um: Sequence[float] | np.ndarray,
        *,
        allow_extrapolation: bool = False,
        provider: Any = None,
        dataset_id: str | int | None = None,
        pageid: str | int | None = None,
        book: str | None = None,
        page: str | int | None = None,
    ) -> SampledOpticalConstants:
        """Named n/k interpolation entry point with explicit extrapolation.

        ``sample`` already performs the deterministic interpolation used by
        the engine.  This name makes the candidate resolver's interpolation
        capability discoverable without changing the existing execution
        contract: extending a partial dataset is still opt-in and remains
        visible in ``extrapolated_mask`` and ``provenance``.
        """

        return self.sample(
            material,
            wavelengths_um,
            allow_extrapolation=allow_extrapolation,
            provider=provider,
            dataset_id=dataset_id,
            pageid=pageid,
            book=book,
            page=page,
        )

    def resolve(
        self,
        material: str | MaterialRef | MaterialCandidate,
        *,
        provider: Any = None,
        dataset_id: str | int | None = None,
        pageid: str | int | None = None,
        book: str | None = None,
        page: str | int | None = None,
        wavelength_range: Sequence[float] | np.ndarray | None = None,
        wavelength_min: float | None = None,
        wavelength_max: float | None = None,
    ) -> MaterialRef:
        if isinstance(material, MaterialCandidate):
            return material.ref
        if isinstance(material, MaterialRef):
            return material
        if material is None and not any(value is not None for value in (dataset_id, pageid, book, page, provider)):
            raise MaterialNotFoundError(str(material))
        if material is not None and not str(material).strip():
            raise MaterialNotFoundError(str(material))
        # A page/state qualifier in the material spelling is an explicit RII
        # selector too.  Without this flag, ``SiO2 (Gao)`` could be shadowed by
        # the local generic SiO2 CSV before the requested page is considered.
        name_selectors = (
            _material_selector_parts(material)[1]
            if isinstance(material, str)
            else ()
        )
        explicit_selector = any(value is not None for value in (provider, dataset_id, pageid, book, page)) or bool(name_selectors)
        candidates = self.search(
            material,
            wavelength_range,
            provider=provider,
            book=book,
            page=page,
            dataset_id=dataset_id,
            pageid=pageid,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
        if not candidates:
            provider_name = _provider_key(provider) if provider is not None and not hasattr(provider, "search") else None
            raise MaterialNotFoundError(str(material), provider=provider_name, dataset_id=dataset_id or pageid)

        # The default registry policy is source-level priority: a local file
        # wins over RII, even when RII has alternative pages.  Selectors make
        # that policy explicit when the caller needs a specific RII dataset.
        if provider is None and not explicit_selector:
            local_candidates = [c for c in candidates if c.provider == "local_csv"]
            if local_candidates:
                candidates = local_candidates
        return _choose_best(candidates, str(material), provider=_provider_key(provider) if provider else None).ref

    def sample(
        self,
        material: str | MaterialRef | MaterialCandidate | None,
        wavelengths_um: Sequence[float] | np.ndarray | None = None,
        *,
        allow_extrapolation: bool = False,
        provider: Any = None,
        dataset_id: str | int | None = None,
        pageid: str | int | None = None,
        book: str | None = None,
        page: str | int | None = None,
        wavelength_range: Sequence[float] | np.ndarray | None = None,
        wavelengths: Sequence[float] | np.ndarray | None = None,
        wavelength_um: Sequence[float] | np.ndarray | None = None,
    ) -> SampledOpticalConstants:
        if wavelengths_um is None:
            wavelengths_um = wavelengths if wavelengths is not None else wavelength_um
        elif wavelengths is not None or wavelength_um is not None:
            raise ValueError("use wavelengths_um, wavelengths or wavelength_um, not more than one")
        if wavelengths_um is None:
            raise ValueError("wavelengths_um is required")
        query = _coerce_wavelengths(wavelengths_um)
        if wavelength_range is None and query.size:
            resolve_range: tuple[float, float] | None = (float(np.min(query)), float(np.max(query)))
        else:
            resolve_range = wavelength_range

        if isinstance(material, MaterialCandidate):
            ref = material.ref
        elif isinstance(material, MaterialRef):
            ref = material
        else:
            ref = self.resolve(
                material,
                provider=provider,
                dataset_id=dataset_id,
                pageid=pageid,
                book=book,
                page=page,
                wavelength_range=resolve_range,
            )
        selected_provider = self._provider(provider) if provider is not None else self.providers[self._ref_provider(ref)]
        return selected_provider.sample(
            ref,
            query,
            allow_extrapolation=allow_extrapolation,
            dataset_id=dataset_id,
            pageid=pageid,
        )


__all__ = [
    "LocalCsvProvider",
    "MATERIAL_CANDIDATE_RANKING_SCHEMA_VERSION",
    "MAX_MATERIAL_CANDIDATES",
    "MaterialAmbiguityError",
    "MaterialCandidate",
    "MaterialNotFoundError",
    "MaterialRangeError",
    "MaterialRef",
    "MaterialRegistry",
    "RiiSqliteProvider",
    "SampledOpticalConstants",
    "normalize_material_name",
]
