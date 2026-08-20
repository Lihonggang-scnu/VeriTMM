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


def normalize_material_name(name: str) -> str:
    """Return a stable, case-insensitive material key.

    Separators are ignored so that ``fused silica``, ``fused_silica`` and
    ``fused-silica`` resolve through the same alias table.
    """

    if name is None:
        return ""
    token = re.sub(r"[^a-z0-9]+", "", str(name).strip().casefold())
    return _MATERIAL_ALIASES.get(token, token)


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
            object.__setattr__(self, "normalized_name", normalize_material_name(self.name))
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
        canonical = normalize_material_name(str(value))
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
        return normalize_material_name(str(material)), None

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
            material_key = normalize_material_name(str(material))
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
        explicit_selector = any(value is not None for value in (provider, dataset_id, pageid, book, page))
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
