from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmm_engine.material_registry import (
    LocalCsvProvider,
    MaterialAmbiguityError,
    MaterialNotFoundError,
    MaterialRangeError,
    MaterialRegistry,
    RiiSqliteProvider,
    normalize_material_name,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("silica", "sio2"), ("fused silica", "sio2"), ("titania", "tio2"), ("silver", "ag"), ("alumina", "al2o3")],
)
def test_common_material_aliases_are_normalized(alias, canonical):
    assert normalize_material_name(alias) == canonical


def test_local_sampling_and_aliases_include_source_provenance():
    registry = MaterialRegistry()
    result = registry.sample("fused silica", [0.40, 0.55, 0.70])

    assert result.n.shape == (3,)
    assert result.k.shape == (3,)
    assert result.provenance["provider"] == "local_csv"
    assert result.provenance["filepath"].endswith("sio2.csv")
    assert result.provenance["range_min"] == pytest.approx(0.2)
    assert result.ref is not None
    assert result.ref.normalized_name == "sio2"


def test_local_sampling_rejects_out_of_range_by_default():
    provider = LocalCsvProvider(ROOT / "tmm_engine" / "materials")

    with pytest.raises(MaterialRangeError) as exc_info:
        provider.sample("sio2", [0.19, 0.40])

    assert exc_info.value.available_range == pytest.approx((0.2, 25.0))
    assert "allow_extrapolation=True" in str(exc_info.value)


def test_local_extrapolation_is_explicit_and_audited():
    provider = LocalCsvProvider(ROOT / "tmm_engine" / "materials")
    result = provider.sample("sio2", [0.10, 0.20, 25.0, 25.10], allow_extrapolation=True)

    np.testing.assert_array_equal(result.extrapolated_mask, [True, False, False, True])
    assert result.warnings
    assert result.extrapolated
    assert result.n[0] == pytest.approx(result.n[1])
    assert result.n[-1] == pytest.approx(result.n[-2])


def test_rii_search_ranks_exact_book_full_coverage_nk_and_points():
    provider = RiiSqliteProvider(ROOT / "tmm_engine" / "rii_cache.db")
    candidates = provider.search("silica", wavelength_range=(0.40, 0.70))

    assert candidates
    top = candidates[0]
    assert top.book == "SiO2"
    assert top.exact_book
    assert top.full_coverage
    assert top.has_n and top.has_k
    assert all(candidates[i].rank_key >= candidates[i + 1].rank_key for i in range(len(candidates) - 1))


def test_rii_pageid_sampling_has_complete_provenance():
    provider = RiiSqliteProvider(ROOT / "tmm_engine" / "rii_cache.db")
    page = next(c for c in provider.search(book="SiO2", page="Gao") if c.page == "Gao")

    result = provider.sample(None, [0.40, 0.70], dataset_id=page.dataset_id)

    assert result.provenance["provider"] == "rii_sqlite"
    assert result.provenance["pageid"] == page.dataset_id
    assert result.provenance["shelf"] == "main"
    assert result.provenance["book"] == "SiO2"
    assert result.provenance["page"] == "Gao"
    assert result.provenance["filepath"].endswith("main\\SiO2\\Gao.yml")
    assert result.provenance["range"] == pytest.approx((0.252, 1.25))


def test_registry_can_explicitly_select_rii_dataset_and_keeps_local_first():
    registry = MaterialRegistry()
    local = registry.resolve("silver")
    assert local.provider == "local_csv"
    assert local.normalized_name == "ag"

    result = registry.sample(
        "sio2",
        [0.40, 0.70],
        provider="rii",
        dataset_id=410,
    )
    assert result.provenance["provider"] == "rii_sqlite"
    assert result.provenance["pageid"] == 410


def test_registry_does_not_silently_choose_an_equal_rii_match():
    registry = MaterialRegistry()

    with pytest.raises(MaterialAmbiguityError):
        registry.resolve("alumina", provider="rii", wavelength_range=(0.40, 0.70))


def test_unknown_material_is_explicitly_reported():
    registry = MaterialRegistry()

    with pytest.raises(MaterialNotFoundError):
        registry.resolve("unobtanium")


def test_rii_catalog_status_is_auditable():
    status = MaterialRegistry().catalog_status()["rii_sqlite"]
    assert status["available"]
    assert status["page_count"] > 2_000
    assert status["refractive_index_point_count"] > 200_000
    assert status["license"] == "CC0-1.0"
