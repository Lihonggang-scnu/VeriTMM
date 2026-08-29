from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from tmm_engine.research import (
    ConstraintSpec,
    ConstraintStatus,
    ContinuousThicknessVariable,
    DesignSpace,
    DesignSpaceContract,
    DiscreteThicknessVariable,
    MaterialChoiceVariable,
    MaterialOption,
    ObjectiveScore,
    ObjectiveSet,
    ObjectiveSetResult,
    ObjectiveSpec,
    ObjectiveValue,
)
from tmm_engine.schemas import (
    IlluminationSpec,
    LayerSpec,
    MediumSpec,
    SimulationTask,
    SpectralGrid,
    StackSpec,
)


def _base_task() -> SimulationTask:
    return SimulationTask(
        stack=StackSpec(
            layers=(
                LayerSpec(
                    None,
                    100.0,
                    constant_n=1.45,
                    min_thickness_nm=50.0,
                    max_thickness_nm=200.0,
                    label="continuous",
                ),
                LayerSpec(
                    "sio2",
                    250.0,
                    provider="builtin",
                    optimizable=False,
                    label="fixed",
                ),
                LayerSpec(
                    None,
                    80.0,
                    constant_n=2.0,
                    min_thickness_nm=60.0,
                    max_thickness_nm=120.0,
                    label="discrete",
                ),
                LayerSpec("tio2", 90.0, provider="builtin", label="material"),
            ),
            incident=MediumSpec(constant_n=1.0),
            exit=MediumSpec(material="sio2", provider="builtin"),
            name="research-stack",
        ),
        spectrum=SpectralGrid(start_nm=400.0, stop_nm=800.0, points=41),
        illumination=IlluminationSpec(
            angles_deg=(0.0, 30.0), polarizations=("s", "p")
        ),
        requested_outputs=("R", "T", "A"),
    )


def _variables() -> tuple[
    ContinuousThicknessVariable,
    DiscreteThicknessVariable,
    MaterialChoiceVariable,
]:
    return (
        ContinuousThicknessVariable(
            name="front_nm", layer_index=0, lower_nm=60.0, upper_nm=180.0
        ),
        DiscreteThicknessVariable(
            name="rear_nm", layer_index=2, values_nm=(60.0, 80.0, 120.0)
        ),
        MaterialChoiceVariable(
            name="material",
            layer_index=3,
            options=(
                MaterialOption(
                    name="catalog-tio2",
                    material="tio2",
                    provider="rii",
                    dataset_id="main/Devore-o",
                ),
                MaterialOption(name="constant-index", constant_n=1.8, constant_k=0.02),
            ),
        ),
    )


def _space() -> DesignSpace:
    return DesignSpace(
        DesignSpaceContract(
            base_task=_base_task(),
            variables=_variables(),
            metadata={"campaign": "deterministic", "revision": 1},
        )
    )


def _values() -> dict[str, float | str]:
    return {
        "front_nm": 120.0,
        "rear_nm": 80.0,
        "material": "catalog-tio2",
    }


def test_contract_json_round_trip_and_stable_design_space_identity() -> None:
    first = _space().contract
    encoded = first.canonical_json()
    restored = DesignSpaceContract.model_validate_json(encoded)
    rebuilt = DesignSpaceContract(
        base_task=_base_task(),
        variables=_variables(),
        metadata={"revision": 1, "campaign": "deterministic"},
    )

    assert json.loads(encoded)["schema_version"] == "veritmm-design-space-v1"
    assert restored == first
    assert restored.canonical_json() == encoded
    assert rebuilt.design_space_id == first.design_space_id


def test_candidate_conversion_preserves_fixed_layer_and_material_metadata() -> None:
    space = _space()
    candidate = space.candidate(_values())
    task = space.to_simulation_task(candidate)
    base = _base_task()

    assert isinstance(task, SimulationTask)
    assert task.stack.layers[0].thickness_nm == 120.0
    assert task.stack.layers[2].thickness_nm == 80.0
    assert task.stack.layers[3].material == "tio2"
    assert task.stack.layers[3].provider == "rii"
    assert task.stack.layers[3].dataset_id == "main/Devore-o"
    assert task.stack.layers[3].constant_n is None
    assert task.stack.layers[1] == base.stack.layers[1]
    assert task.stack.incident == base.stack.incident
    assert task.stack.exit == base.stack.exit


def test_constant_material_choice_preserves_n_and_k() -> None:
    space = _space()
    values = {**_values(), "material": "constant-index", "rear_nm": 120.0}
    task = space.to_simulation_task(values)

    assert task.stack.layers[3].material is None
    assert task.stack.layers[3].provider is None
    assert task.stack.layers[3].dataset_id is None
    assert task.stack.layers[3].constant_n == 1.8
    assert task.stack.layers[3].constant_k == 0.02


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"front_nm": 120.0, "rear_nm": 80.0}, "missing"),
        ({**_values(), "unknown": 1.0}, "extra"),
        ({**_values(), "front_nm": "120"}, "numeric thickness"),
        ({**_values(), "front_nm": True}, "numeric thickness"),
        ({**_values(), "front_nm": 200.0}, "inclusive bounds"),
        ({**_values(), "front_nm": float("nan")}, "finite"),
        ({**_values(), "rear_nm": 81.0}, "allowed discrete"),
        ({**_values(), "material": "unknown"}, "material choice"),
    ],
)
def test_candidate_assignment_rejection(
    values: dict[str, object], error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _space().candidate(values)


def test_duplicate_names_and_conflicting_layer_properties_are_rejected() -> None:
    duplicate_name = (
        ContinuousThicknessVariable(
            name="same", layer_index=0, lower_nm=60.0, upper_nm=150.0
        ),
        MaterialChoiceVariable(
            name="same",
            layer_index=3,
            options=(MaterialOption(name="x", constant_n=1.5),),
        ),
    )
    conflicting_target = (
        ContinuousThicknessVariable(
            name="a", layer_index=0, lower_nm=60.0, upper_nm=150.0
        ),
        DiscreteThicknessVariable(
            name="b", layer_index=0, values_nm=(70.0, 100.0)
        ),
    )

    with pytest.raises(ValidationError, match="names must be unique"):
        DesignSpaceContract(base_task=_base_task(), variables=duplicate_name)
    with pytest.raises(ValidationError, match="same layer property"):
        DesignSpaceContract(base_task=_base_task(), variables=conflicting_target)


def test_invalid_variable_declarations_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ContinuousThicknessVariable(
            name="bad", layer_index=0, lower_nm=100.0, upper_nm=100.0
        )
    with pytest.raises(ValidationError):
        DiscreteThicknessVariable(
            name="bad", layer_index=0, values_nm=(80.0, float("inf"))
        )
    with pytest.raises(ValidationError):
        MaterialOption(name="bad", material="sio2", constant_n=1.5)
    with pytest.raises(ValidationError):
        DesignSpaceContract(
            base_task=_base_task(),
            variables=(
                ContinuousThicknessVariable(
                    name="missing-layer",
                    layer_index=99,
                    lower_nm=1.0,
                    upper_nm=2.0,
                ),
            ),
        )


def test_seeded_indexed_sampling_is_reproducible_and_call_order_independent() -> None:
    space = _space()
    first = space.sample_indices((7, 2, 11), seed=9182)
    unrelated = space.sample(5, seed=3)
    second = space.sample_indices((7, 2, 11), seed=9182)
    reordered = space.sample_indices((11, 7), seed=9182)

    assert unrelated
    assert first == second
    assert [item.sample_index for item in first] == [7, 2, 11]
    assert reordered[0] == first[2]
    assert reordered[1] == first[0]
    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]


def test_candidate_identity_ignores_provenance_and_is_json_round_trippable() -> None:
    space = _space()
    direct = space.candidate(_values())
    indexed = space.candidate(
        _values(), sample_index=123, sampler="external", seed=99
    )
    restored = type(direct).model_validate_json(direct.canonical_json())

    assert direct.candidate_id == indexed.candidate_id
    assert restored == direct
    assert space.validate_candidate(restored) == direct


def test_normalized_design_is_ordered_finite_bounded_and_decodable() -> None:
    space = _space()
    candidate = space.candidate(_values())
    decoded = space.candidate_from_normalized(candidate.normalized_design)

    assert candidate.normalized_design == (0.5, 0.5, 0.0)
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in candidate.normalized_design)
    assert decoded.values == candidate.values
    assert decoded.candidate_id == candidate.candidate_id


def test_future_variable_layer_count_is_not_silently_accepted() -> None:
    payload = json.loads(_space().contract.canonical_json())
    payload["capabilities"]["variable_layer_count"] = True

    with pytest.raises(ValidationError, match="False"):
        DesignSpaceContract.model_validate(payload)

    payload = json.loads(_space().contract.canonical_json())
    payload["variable_layer_count"] = {"minimum": 1, "maximum": 10}
    with pytest.raises(ValidationError, match="Extra inputs"):
        DesignSpaceContract.model_validate(payload)


def test_existing_simulation_task_validate_runs_after_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    space = _space()
    original = SimulationTask.validate
    calls = 0

    def counting_validate(self: SimulationTask) -> None:
        nonlocal calls
        calls += 1
        original(self)

    monkeypatch.setattr(SimulationTask, "validate", counting_validate)
    converted = space.to_simulation_task(_values())

    assert calls >= 1
    assert converted.stack.layers[0].thickness_nm == 120.0


def _objective_set() -> ObjectiveSet:
    return ObjectiveSet(
        objectives=(
            ObjectiveSpec(
                name="high-reflection",
                direction="maximize",
                observable="R",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
                angle_deg=0.0,
                polarization="s",
                aggregation="mean",
                weight=2.0,
            ),
            ObjectiveSpec(
                name="low-absorption",
                direction="minimize",
                observable="A",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
                aggregation="max",
                weight=0.5,
            ),
            ObjectiveSpec(
                name="target-transmission",
                direction="target",
                observable="T",
                wavelength_min_nm=650.0,
                wavelength_max_nm=700.0,
                aggregation="min",
                target=0.8,
                weight=1.25,
            ),
        ),
        constraints=(
            ConstraintSpec(
                name="minimum-transmission",
                relation="at_least",
                observable="T",
                wavelength_min_nm=650.0,
                wavelength_max_nm=700.0,
                aggregation="min",
                threshold=0.7,
                tolerance=0.01,
            ),
            ConstraintSpec(
                name="maximum-absorption",
                relation="at_most",
                observable="A",
                wavelength_min_nm=500.0,
                wavelength_max_nm=600.0,
                aggregation="max",
                threshold=0.1,
                tolerance=0.005,
            ),
        ),
        metadata={"study": "weighted"},
    )


def test_objective_set_supports_weighted_directions_constraints_and_round_trip() -> None:
    objective_set = _objective_set()
    restored = ObjectiveSet.model_validate_json(objective_set.canonical_json())

    assert [item.direction for item in objective_set.objectives] == [
        "maximize",
        "minimize",
        "target",
    ]
    assert [item.weight for item in objective_set.objectives] == [2.0, 0.5, 1.25]
    assert [item.relation for item in objective_set.constraints] == [
        "at_least",
        "at_most",
    ]
    assert restored == objective_set
    assert restored.canonical_json() == objective_set.canonical_json()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ObjectiveSpec(
            name="band",
            direction="maximize",
            observable="R",
            wavelength_min_nm=600.0,
            wavelength_max_nm=500.0,
        ),
        lambda: ObjectiveSpec(
            name="missing-target",
            direction="target",
            observable="R",
            wavelength_min_nm=500.0,
            wavelength_max_nm=600.0,
        ),
        lambda: ObjectiveSpec(
            name="unexpected-target",
            direction="maximize",
            observable="R",
            wavelength_min_nm=500.0,
            wavelength_max_nm=600.0,
            target=0.5,
        ),
        lambda: ObjectiveSpec(
            name="weight",
            direction="minimize",
            observable="A",
            wavelength_min_nm=500.0,
            wavelength_max_nm=600.0,
            weight=0.0,
        ),
        lambda: ConstraintSpec(
            name="relation",
            relation="equal",  # type: ignore[arg-type]
            observable="T",
            wavelength_min_nm=500.0,
            wavelength_max_nm=600.0,
            threshold=0.5,
        ),
        lambda: ConstraintSpec(
            name="tolerance",
            relation="at_least",
            observable="T",
            wavelength_min_nm=500.0,
            wavelength_max_nm=600.0,
            threshold=0.5,
            tolerance=-0.1,
        ),
    ],
)
def test_invalid_objective_and_constraint_contracts_are_rejected(factory: object) -> None:
    with pytest.raises(ValidationError):
        factory()  # type: ignore[operator]


def test_result_contracts_are_deterministic_and_do_not_imply_physics_validity() -> None:
    objective_set = _objective_set()
    result = ObjectiveSetResult(
        objective_set_id=objective_set.objective_set_id,
        candidate_id=_space().candidate(_values()).candidate_id,
        values=(ObjectiveValue(objective_name="high-reflection", value=0.9),),
        scores=(
            ObjectiveScore(
                objective_name="high-reflection",
                value=0.9,
                score=0.9,
                weighted_score=1.8,
            ),
        ),
        total_score=1.8,
        constraints=(
            ConstraintStatus(
                constraint_name="minimum-transmission",
                relation="at_least",
                value=0.8,
                threshold=0.7,
                tolerance=0.01,
                satisfied=True,
            ),
        ),
        feasible=True,
    )
    restored = ObjectiveSetResult.model_validate_json(result.canonical_json())

    assert restored == result
    assert restored.physics_validity == "not_assessed"
    with pytest.raises(ValidationError):
        ObjectiveSetResult.model_validate(
            {**result.model_dump(mode="python"), "physics_validity": "valid"}
        )


def test_research_layer_does_not_execute_numerics_or_create_certificates() -> None:
    task = _space().to_simulation_task(_values())

    assert isinstance(task, SimulationTask)
    assert not hasattr(task, "result")
    assert not hasattr(task, "certificate")
