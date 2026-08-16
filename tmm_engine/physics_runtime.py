"""Discovery of an isolated PyTorch runtime for differentiable TMM tasks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def runtime_has_torch(executable: str | Path, timeout_seconds: float = 20.0) -> bool:
    path = Path(executable)
    if not path.exists():
        return False
    try:
        result = subprocess.run(
            [str(path), "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def physics_python_candidates(explicit: Optional[str] = None) -> Iterable[Path]:
    seen = set()
    values = [
        explicit,
        os.environ.get("VERITMM_PHYSICS_PYTHON"),
        sys.executable,
        str(Path.cwd() / ".venv-physics" / "Scripts" / "python.exe"),
        str(Path.cwd() / ".venv-physics" / "bin" / "python"),
    ]
    for value in values:
        if not value:
            continue
        path = Path(value).resolve()
        key = str(path).casefold()
        if key not in seen:
            seen.add(key)
            yield path


def discover_physics_python(explicit: Optional[str] = None) -> Path:
    attempted = []
    for candidate in physics_python_candidates(explicit):
        attempted.append(str(candidate))
        if runtime_has_torch(candidate):
            return candidate
    raise RuntimeError(
        "No PyTorch physics runtime is available. Set VERITMM_PHYSICS_PYTHON "
        "or install the optimization optional dependency. Attempted: %s"
        % attempted
    )


__all__ = ["discover_physics_python", "physics_python_candidates", "runtime_has_torch"]
