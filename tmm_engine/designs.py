"""Reusable stack constructors for common thin-film and 1D-PhC experiments."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Tuple

from .schemas import LayerSpec, MediumSpec, StackSpec


def periodic_stack(
    high_material: str,
    high_thickness_nm: float,
    low_material: str,
    low_thickness_nm: float,
    periods: int,
    *,
    incident: Optional[MediumSpec] = None,
    exit: Optional[MediumSpec] = None,
    start_with: str = "high",
    name: str = "periodic_1d_photonic_crystal",
) -> StackSpec:
    if int(periods) < 1:
        raise ValueError("periods must be positive")
    pair = (
        (LayerSpec(high_material, high_thickness_nm), LayerSpec(low_material, low_thickness_nm))
        if start_with == "high"
        else (LayerSpec(low_material, low_thickness_nm), LayerSpec(high_material, high_thickness_nm))
    )
    if start_with not in ("high", "low"):
        raise ValueError("start_with must be high or low")
    return StackSpec(
        layers=tuple(layer for _ in range(int(periods)) for layer in pair),
        incident=incident or MediumSpec.air(),
        exit=exit or MediumSpec(material="sio2"),
        name=name,
    )


def defect_cavity_stack(
    high_material: str,
    high_thickness_nm: float,
    low_material: str,
    low_thickness_nm: float,
    periods_each_side: int,
    defect_material: str,
    defect_thickness_nm: float,
    *,
    incident: Optional[MediumSpec] = None,
    exit: Optional[MediumSpec] = None,
    mirror_symmetric: bool = True,
    name: str = "defect_cavity_1d_photonic_crystal",
) -> StackSpec:
    left = periodic_stack(
        high_material,
        high_thickness_nm,
        low_material,
        low_thickness_nm,
        periods_each_side,
        incident=incident or MediumSpec.air(),
        exit=exit or MediumSpec(material="sio2"),
    ).layers
    right = tuple(reversed(left)) if mirror_symmetric else left
    return StackSpec(
        layers=left + (LayerSpec(defect_material, defect_thickness_nm),) + right,
        incident=incident or MediumSpec.air(),
        exit=exit or MediumSpec(material="sio2"),
        name=name,
    )


def chirped_stack(
    material_pairs: Sequence[Tuple[str, str]],
    thickness_pairs_nm: Sequence[Tuple[float, float]],
    *,
    incident: Optional[MediumSpec] = None,
    exit: Optional[MediumSpec] = None,
    name: str = "chirped_multilayer",
) -> StackSpec:
    if len(material_pairs) != len(thickness_pairs_nm) or not material_pairs:
        raise ValueError("material_pairs and thickness_pairs_nm must have equal non-zero length")
    layers = []
    for (mat_a, mat_b), (d_a, d_b) in zip(material_pairs, thickness_pairs_nm):
        layers.extend((LayerSpec(mat_a, d_a), LayerSpec(mat_b, d_b)))
    return StackSpec(
        layers=tuple(layers),
        incident=incident or MediumSpec.air(),
        exit=exit or MediumSpec(material="sio2"),
        name=name,
    )


def with_finite_substrate(
    stack: StackSpec,
    substrate_material: str,
    substrate_thickness_nm: float,
    *,
    backing_medium: Optional[MediumSpec] = None,
    coherent: bool = False,
) -> StackSpec:
    """Represent a finite substrate explicitly as the final finite layer."""

    substrate = LayerSpec(
        substrate_material,
        substrate_thickness_nm,
        coherence="coherent" if coherent else "incoherent",
        optimizable=False,
        label="finite_substrate",
    )
    return replace(
        stack,
        layers=stack.layers + (substrate,),
        exit=backing_medium or MediumSpec.air(),
        name=stack.name + "_finite_substrate",
    )


__all__ = ["chirped_stack", "defect_cavity_stack", "periodic_stack", "with_finite_substrate"]
