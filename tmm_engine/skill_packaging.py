"""Packaged Agent Skill distribution (``veritmm skill-path`` / ``install-skill``).

The skill source of truth is ``tmm_engine/skills/veritmm-tmm`` — it ships
inside the wheel, so ``skill-path`` always points at the installed copy.
``install_skill`` copies that fixed source into a host skill directory (for
example ``~/.claude/skills``); the source is not a parameter, the destination
must not be or traverse a symlink, and overwriting an existing installation
requires the explicit ``force`` confirmation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
SKILL_DIR_NAME = "veritmm-tmm"


class SkillInstallError(ValueError):
    """Raised when a skill installation cannot proceed safely."""


def skill_source_dir() -> Path:
    """Return the packaged skill directory (the single distribution source)."""

    return PACKAGE_ROOT / "skills" / SKILL_DIR_NAME


def skill_manifest() -> dict[str, Any]:
    source = skill_source_dir()
    files = sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    )
    return {
        "ok": True,
        "skill_name": SKILL_DIR_NAME,
        "skill_path": str(source),
        "files": files,
    }


def install_skill(target: str | Path, *, force: bool = False) -> dict[str, Any]:
    """Install the packaged skill into a host skill directory.

    ``target`` is the host's skills *parent* directory (for example
    ``~/.claude/skills``); the skill is installed as
    ``<target>/veritmm-tmm``.  The source is fixed by construction, symlinked
    targets are refused, and an existing installation is only replaced after
    an explicit ``force=True``.
    """

    source = skill_source_dir()
    if not source.is_dir():
        raise SkillInstallError(
            "the packaged skill directory is missing from this installation"
        )
    target_path = Path(target).expanduser()
    if not target_path.is_absolute():
        target_path = Path.cwd() / target_path
    if target_path.is_symlink():
        raise SkillInstallError(
            f"install target must not be a symbolic link: {target_path}"
        )
    resolved_target = target_path.resolve()
    destination = resolved_target / SKILL_DIR_NAME
    overwritten = destination.exists()
    if overwritten:
        if not force:
            raise SkillInstallError(
                f"destination already exists: {destination}; pass --force to "
                "overwrite it explicitly"
            )
        if destination.is_symlink():
            raise SkillInstallError(
                "existing destination is a symbolic link; refusing to replace it"
            )
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    if (destination / "SKILL.md").read_bytes() != (source / "SKILL.md").read_bytes():
        shutil.rmtree(destination, ignore_errors=True)
        raise SkillInstallError(
            "the copied skill does not match the packaged skill; installation "
            "was rolled back"
        )
    files = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    )
    return {
        "ok": True,
        "skill_name": SKILL_DIR_NAME,
        "destination": str(destination),
        "files": files,
        "overwritten": overwritten,
    }


__all__ = [
    "SKILL_DIR_NAME",
    "SkillInstallError",
    "install_skill",
    "skill_manifest",
    "skill_source_dir",
]
