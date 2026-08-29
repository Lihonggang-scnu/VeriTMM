"""pytest configuration shared across the VeriTMM test suite.

Marker auto-skip
----------------
Tests decorated with ``@pytest.mark.requires_torch`` are automatically
skipped in environments where PyTorch is not installed.  This keeps the
core CI job clean without requiring each test module to guard torch imports
individually.

In the dedicated ``torch`` CI job, ``pip install -e ".[test,optimize]"``
installs torch first, so these marks never skip there.
"""

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip requires_torch tests when torch is not importable."""
    if item.get_closest_marker("requires_torch") is not None:
        pytest.importorskip("torch")
