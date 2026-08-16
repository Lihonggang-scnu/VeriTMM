from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tmm_engine import (
    IlluminationSpec,
    LayerSpec,
    MaterialRegistry,
    MediumSpec,
    SimulationConfig,
    SimulationTask,
    SpectralGrid,
    StackSpec,
    TMMEngine,
    TMMWorkbench,
    bloch_trace_bilayer_normal_incidence,
    find_threshold_bands,
    periodic_stack,
    phase_dispersion_from_amplitude,
    thickness_tolerance_monte_carlo,
)


def _workbench() -> TMMWorkbench:
    return TMMWorkbench(MaterialRegistry())


def _constant_task(*, solver: str = "smatrix", angle: float = 0.0) -> SimulationTask:
    return SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 100.0, constant_n=2.1),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=450.0, stop_nm=750.0, points=61),
        illumination=IlluminationSpec(angles_deg=(angle,), polarizations=("s", "p", "unpolarized")),
        solver=solver,
    )


def test_constant_index_layer_and_fresnel_limit() -> None:
    task = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 20.0, constant_n=1.0),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=600.0, points=3),
    )
    result = _workbench().simulate(task)
    np.testing.assert_allclose(result.channel()["R"], 0.04, atol=1e-12)
    np.testing.assert_allclose(result.channel()["T"], 0.96, atol=1e-12)
    assert result.audit["passivity_check_passed"]


@pytest.mark.parametrize("angle", [0.0, 37.0, 67.0])
def test_internal_smatrix_matches_independent_byrnes(angle: float) -> None:
    workbench = _workbench()
    internal = workbench.simulate(_constant_task(solver="smatrix", angle=angle))
    reference = workbench.simulate(_constant_task(solver="byrnes", angle=angle))
    for pol in ("s", "p", "unpolarized"):
        actual = internal.channel(angle, pol)
        expected = reference.channel(angle, pol)
        np.testing.assert_allclose(actual["R"], expected["R"], rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(actual["T"], expected["T"], rtol=2e-10, atol=2e-11)


def test_characteristic_and_smatrix_agree_for_benign_stack() -> None:
    workbench = _workbench()
    smatrix = workbench.simulate(_constant_task(solver="smatrix"))
    characteristic = workbench.simulate(_constant_task(solver="characteristic"))
    np.testing.assert_allclose(
        smatrix.channel()["R"], characteristic.channel()["R"], rtol=2e-10, atol=2e-11
    )


def test_finite_substrate_is_an_explicit_layer_and_changes_spectrum() -> None:
    base = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 90.0, constant_n=2.0),),
            incident=MediumSpec.air(),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=450.0, stop_nm=750.0, points=101),
    )
    finite = replace(
        base,
        stack=StackSpec(
            layers=base.stack.layers + (LayerSpec(None, 1250.0, constant_n=1.5, optimizable=False),),
            incident=MediumSpec.air(),
            exit=MediumSpec.air(),
        ),
    )
    semi_r = _workbench().simulate(base).channel()["R"]
    finite_r = _workbench().simulate(finite).channel()["R"]
    assert float(np.max(np.abs(semi_r - finite_r))) > 0.02


def test_mixed_coherence_runs_through_reference_backend() -> None:
    task = SimulationTask(
        stack=StackSpec(
            layers=(
                LayerSpec(None, 100.0, constant_n=2.0),
                LayerSpec("sio2", 1_000_000.0, coherence="incoherent", optimizable=False),
            ),
            exit=MediumSpec.air(),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=700.0, points=11),
    )
    result = _workbench().simulate(task)
    assert result.solver == "byrnes"
    assert result.audit["mixed_coherence"] is True
    assert result.audit["passivity_check_passed"]


def test_layer_absorption_and_ellipsometry_have_explicit_shapes() -> None:
    task = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 120.0, constant_n=2.0, constant_k=0.08),),
            exit=MediumSpec(constant_n=1.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=700.0, points=7),
        illumination=IlluminationSpec(angles_deg=(60.0,), polarizations=("s",)),
        solver="byrnes",
        requested_outputs=("R", "T", "A", "layer_absorption", "ellipsometry"),
    )
    result = _workbench().simulate(task)
    absorption = result.extras["layer_absorption|angle=60|pol=s"]
    assert absorption.shape == (1, 7)
    np.testing.assert_allclose(absorption[0], result.channel(60.0, "s")["A"], atol=2e-12)
    ellipsometry = result.extras["ellipsometry|angle=60"]
    assert len(ellipsometry["psi_rad"]) == 7
    assert np.all(np.isfinite(ellipsometry["delta_rad"]))


def test_tolerance_monte_carlo_is_reproducible() -> None:
    task = _constant_task()
    first = thickness_tolerance_monte_carlo(
        _workbench(), task, sigma_nm=2.0, samples=5, seed=13
    )
    second = thickness_tolerance_monte_carlo(
        _workbench(), task, sigma_nm=2.0, samples=5, seed=13
    )
    np.testing.assert_allclose(first["R_mean"], second["R_mean"])
    np.testing.assert_allclose(first["thickness_draws_nm"], second["thickness_draws_nm"])


def test_system_emissivity_is_distinct_from_finite_layer_absorptance() -> None:
    task = SimulationTask(
        stack=StackSpec(
            layers=(LayerSpec(None, 80.0, constant_n=1.8),),
            exit=MediumSpec(constant_n=2.0, constant_k=0.5),
        ),
        spectrum=SpectralGrid(start_nm=500.0, stop_nm=700.0, points=5),
        requested_outputs=("R", "T", "A", "system_emissivity"),
    )
    channel = _workbench().simulate(task).channel()
    np.testing.assert_allclose(channel["E_system"], 1.0 - channel["R"])
    assert float(np.max(np.abs(channel["E_system"] - channel["A"]))) > 0.1


def test_material_unit_conversion_is_auditable() -> None:
    task = SimulationTask(
        stack=StackSpec((LayerSpec("tio2", 100.0),), exit=MediumSpec(material="sio2")),
        spectrum=SpectralGrid(start_nm=400.0, stop_nm=700.0, points=5),
    )
    result = _workbench().simulate(task)
    film = result.material_provenance[1]
    assert film["wavelength_unit"] == "um"
    assert film["range_min_um"] <= 0.4
    assert film["range_max_um"] >= 0.7


def test_quarter_wave_periodic_stack_has_expected_stop_band() -> None:
    center_nm = 600.0
    stack = periodic_stack(
        "unused_high",
        center_nm / (4 * 2.2),
        "unused_low",
        center_nm / (4 * 1.45),
        periods=6,
        exit=MediumSpec(constant_n=1.45),
    )
    layers = tuple(
        replace(layer, material=None, constant_n=2.2 if index % 2 == 0 else 1.45)
        for index, layer in enumerate(stack.layers)
    )
    task = SimulationTask(
        stack=replace(stack, layers=layers),
        spectrum=SpectralGrid(start_nm=350.0, stop_nm=900.0, points=551),
    )
    result = _workbench().simulate(task)
    R = result.channel()["R"]
    peak_nm = result.wavelengths_nm[int(np.argmax(R))]
    assert 540.0 <= peak_nm <= 660.0
    assert float(np.max(R)) > 0.98
    bands = find_threshold_bands(result.wavelengths_nm, R, threshold=0.9, min_width_nm=20.0)
    assert any(b.start_nm < center_nm < b.stop_nm for b in bands)


def test_bloch_trace_marks_quarter_wave_stop_band() -> None:
    wavelengths = np.linspace(400.0, 800.0, 401)
    n_high = np.full_like(wavelengths, 2.2, dtype=np.complex128)
    n_low = np.full_like(wavelengths, 1.45, dtype=np.complex128)
    data = bloch_trace_bilayer_normal_incidence(
        n_high, 600.0 / (4 * 2.2), n_low, 600.0 / (4 * 1.45), wavelengths
    )
    center_index = int(np.argmin(np.abs(wavelengths - 600.0)))
    assert bool(data["forbidden_band_mask"][center_index])


def test_legacy_engine_does_not_mutate_config_and_rejects_fake_finite_substrate() -> None:
    engine = TMMEngine()
    config = SimulationConfig(wl_points=5, solver="smatrix")
    engine.simulate(["tio2"], [100.0], config=config, solver="cm")
    assert config.solver == "smatrix"
    with pytest.raises(ValueError, match="finite substrate"):
        engine.simulate(["tio2"], [100.0], substrate_mode="finite")


def test_phase_group_delay_and_gdd_follow_declared_convention() -> None:
    wavelengths = np.linspace(500.0, 700.0, 501)
    omega = 2.0 * np.pi * 299_792_458.0 / (wavelengths * 1e-9)
    delay_s = 2.5e-15
    amplitude = np.exp(1j * omega * delay_s)
    result = phase_dispersion_from_amplitude(wavelengths, amplitude)
    np.testing.assert_allclose(result["group_delay_fs"][5:-5], 2.5, atol=2e-8)
    assert float(np.max(np.abs(result["gdd_fs2"][5:-5]))) < 1e-5


def test_workbench_emits_phase_dispersion_for_each_coherent_channel() -> None:
    task = replace(
        _constant_task(),
        requested_outputs=("R", "T", "A", "phase_dispersion"),
    )
    result = _workbench().simulate(task)
    item = result.extras["phase_dispersion|angle=0|pol=s"]
    assert item["reflection"]["phase_rad"].shape == (61,)
    assert np.all(np.isfinite(item["transmission"]["group_delay_fs"]))


def test_legacy_cache_key_distinguishes_nonuniform_grids() -> None:
    engine = TMMEngine()
    first = np.array([0.4, 0.5, 0.7])
    second = np.array([0.4, 0.6, 0.7])
    assert engine.nk_db._cache_key(first) != engine.nk_db._cache_key(second)


def test_forward_result_serializes_nested_complex_arrays_losslessly() -> None:
    import json

    from tmm_engine.workbench import ForwardSimulationResult

    result = ForwardSimulationResult(
        wavelengths_nm=np.asarray([500.0, 600.0]),
        channels={"angle=0|pol=s": {"R": np.asarray([0.1, 0.2])}},
        material_provenance=[],
        solver="smatrix",
        extras={
            "amplitudes|angle=0|pol=s": {
                "r": np.asarray([1 + 2j, 3 - 4j]),
            }
        },
    )
    payload = result.to_dict()
    encoded = payload["extras"]["amplitudes|angle=0|pol=s"]["r"]
    assert encoded == {
        "encoding": "complex_array_cartesian",
        "shape": [2],
        "real": [1.0, 3.0],
        "imag": [2.0, -4.0],
    }
    json.dumps(payload, allow_nan=False)
