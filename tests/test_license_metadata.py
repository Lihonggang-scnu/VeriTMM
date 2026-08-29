"""Regression tests for declared distribution license metadata.

A previous change replaced the SPDX ``license`` expression in
``pyproject.toml`` with a ``{file = "LICENSE"}`` table.  That form is accepted
by the build backend, but it drops ``License-Expression`` from the built wheel
metadata and instead emits the first line of the license text as the ``License``
field.  The distribution then no longer machine-declares Apache-2.0, and no
existing test caught it because the physics suite is unaffected.

These tests pin the declared license so the same silent regression cannot
recur.  They read ``pyproject.toml`` directly rather than building a wheel, so
they stay fast and do not require network access or a build backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10 has no stdlib tomllib; tomli is its upstream implementation.
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
LICENSE_PATH = REPO_ROOT / "LICENSE"

EXPECTED_LICENSE_EXPRESSION = "Apache-2.0"


def _project_table() -> dict:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_license_is_declared_as_an_spdx_expression() -> None:
    """``license`` must be the SPDX string form, not a legacy file table.

    PEP 639 expects a machine-readable SPDX expression.  The ``{file = ...}``
    table form is what caused ``License-Expression`` to disappear from wheel
    metadata.
    """
    license_field = _project_table()["license"]
    assert isinstance(license_field, str), (
        "project.license must be an SPDX expression string; a table form such as "
        '{file = "LICENSE"} drops License-Expression from wheel metadata'
    )
    assert license_field == EXPECTED_LICENSE_EXPRESSION


def test_license_files_reference_the_repository_license() -> None:
    """The license text must be shipped alongside the expression."""
    project = _project_table()
    assert project["license-files"] == ["LICENSE"]
    assert LICENSE_PATH.is_file()


def test_license_text_is_apache_2_0() -> None:
    """Guard against the expression and the shipped text disagreeing."""
    text = LICENSE_PATH.read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text


def test_no_license_classifier_conflicts_with_the_expression() -> None:
    """PEP 639 forbids pairing a license classifier with an SPDX expression.

    setuptools>=77 raises on this combination, so a stray ``License ::``
    classifier would break the build rather than merely degrade metadata.
    """
    classifiers = _project_table().get("classifiers", [])
    license_classifiers = [item for item in classifiers if item.startswith("License ::")]
    assert license_classifiers == [], (
        "License classifiers must not be combined with an SPDX license expression: "
        f"{license_classifiers}"
    )


def test_build_backend_supports_pep_639_expressions() -> None:
    """``License-Expression`` requires a setuptools new enough to emit it."""
    with PYPROJECT_PATH.open("rb") as handle:
        requires = tomllib.load(handle)["build-system"]["requires"]

    setuptools_pins = [item for item in requires if item.startswith("setuptools")]
    assert setuptools_pins, "build-system.requires must pin setuptools"

    pin = setuptools_pins[0]
    assert ">=" in pin, f"setuptools pin must declare a minimum version: {pin}"
    minimum = int(pin.split(">=", 1)[1].split(".", 1)[0])
    assert minimum >= 77, (
        "setuptools>=77 is required for PEP 639 License-Expression metadata; "
        f"found {pin}"
    )
