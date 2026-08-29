from __future__ import annotations

from pathlib import Path

from tmm_engine.physics_runtime import physics_python_candidates, runtime_has_torch


def test_physics_runtime_candidates_put_explicit_path_first(tmp_path: Path) -> None:
    explicit = tmp_path / "python.exe"
    candidates = list(physics_python_candidates(str(explicit)))
    assert candidates[0] == explicit.resolve()


def test_missing_runtime_does_not_claim_torch(tmp_path: Path) -> None:
    assert not runtime_has_torch(tmp_path / "missing-python.exe")
