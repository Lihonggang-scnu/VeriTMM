from __future__ import annotations

from pathlib import Path

from tmm_engine import MaterialRegistry, TMMWorkbench
from tmm_engine.validation_cases import validate_pmc9147317


def test_published_dbr_case_reproduces_stopband_cavity_dip_and_field(tmp_path: Path) -> None:
    report = validate_pmc9147317(TMMWorkbench(MaterialRegistry()), tmp_path)
    assert report["status"] == "passed"
    assert report["stopband_interval_iou"] >= 0.75
    assert report["cavity_dip_error_nm"] <= 10.0
    assert report["maximum_normalized_field_intensity"] >= 50.0
    assert (tmp_path / "VALIDATION_REPORT.json").exists()
