"""General-purpose forward simulation workbench for multilayer optics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .analysis import phase_dispersion_from_amplitude
from .capabilities import PhysicsEngineError, assess_tmm_capability
from .schemas import MediumSpec, SimulationTask
from .tmm_solver import TMM, TMMConfig


def _channel_key(angle_deg: float, polarization: str) -> str:
    return "angle=%g|pol=%s" % (float(angle_deg), polarization)


def _json_safe_result_value(value: Any) -> Any:
    """Recursively encode numerical output, preserving complex values."""

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if np.iscomplexobj(array):
            return {
                "encoding": "complex_array_cartesian",
                "shape": list(array.shape),
                "real": np.real(array).tolist(),
                "imag": np.imag(array).tolist(),
            }
        return array.tolist()
    if isinstance(value, np.generic):
        return _json_safe_result_value(value.item())
    if isinstance(value, complex):
        return {
            "encoding": "complex_scalar_cartesian",
            "real": float(value.real),
            "imag": float(value.imag),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe_result_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_result_value(item) for item in value]
    return value


@dataclass
class ForwardSimulationResult:
    wavelengths_nm: np.ndarray
    channels: Dict[str, Dict[str, np.ndarray]]
    material_provenance: List[Dict[str, Any]]
    solver: str
    audit: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def channel(self, angle_deg: float = 0.0, polarization: str = "unpolarized") -> Dict[str, np.ndarray]:
        return self.channels[_channel_key(angle_deg, polarization)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wavelengths_nm": self.wavelengths_nm.tolist(),
            "channels": _json_safe_result_value(self.channels),
            "material_provenance": _json_safe_result_value(self.material_provenance),
            "solver": self.solver,
            "audit": _json_safe_result_value(self.audit),
            "extras": _json_safe_result_value(self.extras),
        }


class TMMWorkbench:
    """Run validated coherent or mixed-coherence multilayer simulations."""

    def __init__(self, material_registry: Any) -> None:
        self.registry = material_registry

    def _sample_material(
        self,
        name: str,
        wavelengths_nm: np.ndarray,
        provider: Optional[str],
        dataset_id: Optional[str],
        allow_extrapolation: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # MaterialRegistry's public unit is micrometres.  The workbench
        # contract is nanometres, so conversion happens exactly once here.
        sampled = self.registry.sample(
            name,
            wavelengths_nm * 1e-3,
            provider=provider,
            dataset_id=dataset_id,
            allow_extrapolation=allow_extrapolation,
        )
        if hasattr(sampled, "nk"):
            nk = np.asarray(sampled.nk, dtype=np.complex128)
        else:
            nk = np.asarray(sampled.n, dtype=np.float64) + 1j * np.asarray(sampled.k, dtype=np.float64)
        provenance = dict(getattr(sampled, "provenance", {}) or {})
        warnings = list(getattr(sampled, "warnings", []) or [])
        extrapolated_mask = np.asarray(
            getattr(sampled, "extrapolated_mask", np.zeros(wavelengths_nm.size, dtype=bool)),
            dtype=bool,
        )
        provenance["extrapolated"] = bool(np.any(extrapolated_mask))
        provenance["extrapolated_point_count"] = int(np.count_nonzero(extrapolated_mask))
        provenance["requested_range_um"] = [
            float(np.min(wavelengths_nm) * 1e-3),
            float(np.max(wavelengths_nm) * 1e-3),
        ]
        if warnings:
            provenance["warnings"] = warnings
        return nk, provenance

    def _sample_layer(
        self,
        layer: Any,
        wavelengths_nm: np.ndarray,
        allow_extrapolation: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if layer.constant_n is not None:
            value = complex(float(layer.constant_n), float(layer.constant_k))
            return np.full(wavelengths_nm.shape, value, dtype=np.complex128), {
                "provider": "constant",
                "n": float(layer.constant_n),
                "k": float(layer.constant_k),
                "label": layer.label,
            }
        return self._sample_material(
            str(layer.material),
            wavelengths_nm,
            layer.provider,
            layer.dataset_id,
            allow_extrapolation,
        )

    def _sample_medium(
        self,
        medium: MediumSpec,
        wavelengths_nm: np.ndarray,
        allow_extrapolation: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if medium.constant_n is not None:
            value = complex(float(medium.constant_n), float(medium.constant_k))
            return np.full(wavelengths_nm.shape, value, dtype=np.complex128), {
                "provider": "constant",
                "n": float(medium.constant_n),
                "k": float(medium.constant_k),
            }
        return self._sample_material(
            str(medium.material),
            wavelengths_nm,
            medium.provider,
            medium.dataset_id,
            allow_extrapolation,
        )

    def _resolve_stack(self, task: SimulationTask) -> Tuple[List[np.ndarray], np.ndarray, List[Dict[str, Any]]]:
        wavelengths_nm = task.spectrum.wavelengths_nm()
        provenance: List[Dict[str, Any]] = []
        incident, meta = self._sample_medium(
            task.stack.incident, wavelengths_nm, task.allow_material_extrapolation
        )
        provenance.append(dict(meta, stack_position="incident"))
        films: List[np.ndarray] = []
        for index, layer in enumerate(task.stack.layers):
            nk, meta = self._sample_layer(
                layer, wavelengths_nm, task.allow_material_extrapolation
            )
            films.append(nk)
            provenance.append(dict(meta, stack_position="layer_%d" % index, thickness_nm=layer.thickness_nm))
        exit_nk, meta = self._sample_medium(
            task.stack.exit, wavelengths_nm, task.allow_material_extrapolation
        )
        provenance.append(dict(meta, stack_position="exit"))
        return [incident] + films + [exit_nk], wavelengths_nm, provenance

    @staticmethod
    def _audit_channels(
        channels: Mapping[str, Mapping[str, np.ndarray]],
        wavelengths_nm: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        max_energy_error = 0.0
        energy_worst_channel: Optional[str] = None
        energy_worst_lam_idx: int = 0
        min_value = float("inf")
        max_value = float("-inf")
        nonfinite = 0
        for ch_key, values in channels.items():
            R = np.asarray(values["R"], dtype=np.float64)
            T = np.asarray(values["T"], dtype=np.float64)
            A = np.asarray(values["A"], dtype=np.float64)
            err = np.abs(R + T + A - 1.0)
            ch_max_err = float(np.max(err))
            if ch_max_err > max_energy_error:
                max_energy_error = ch_max_err
                energy_worst_channel = ch_key
                energy_worst_lam_idx = int(np.argmax(err))
            merged = np.concatenate([R, T, A])
            nonfinite += int(np.sum(~np.isfinite(merged)))
            if merged.size:
                min_value = min(min_value, float(np.nanmin(merged)))
                max_value = max(max_value, float(np.nanmax(merged)))
        passive = bool(nonfinite == 0 and min_value >= -1e-7 and max_value <= 1.0 + 1e-7)
        # Resolve worst-case wavelength in nm when the wavelength array is available
        energy_worst_lam_nm: Optional[float] = None
        if energy_worst_channel is not None and wavelengths_nm is not None:
            if 0 <= energy_worst_lam_idx < len(wavelengths_nm):
                energy_worst_lam_nm = float(wavelengths_nm[energy_worst_lam_idx])
        return {
            "energy_conservation_max_abs_error": max_energy_error,
            "energy_worst_case_channel": energy_worst_channel,
            "energy_worst_case_wavelength_idx": energy_worst_lam_idx if energy_worst_channel is not None else None,
            "energy_worst_case_wavelength_nm": energy_worst_lam_nm,
            "minimum_observable": min_value,
            "maximum_observable": max_value,
            "nonfinite_value_count": nonfinite,
            "passivity_check_passed": passive,
        }

    def simulate(self, task: SimulationTask) -> ForwardSimulationResult:
        assessment = assess_tmm_capability(task)
        if not assessment.supported:
            raise PhysicsEngineError(assessment.failures[0])
        media, wavelengths_nm, provenance = self._resolve_stack(task)
        if assessment.resolved_solver == "byrnes":
            result = self._simulate_byrnes(task, media, wavelengths_nm, provenance)
        else:
            result = self._simulate_internal(task, media, wavelengths_nm, provenance)
        result.audit.update(self._audit_channels(result.channels, result.wavelengths_nm))
        result.audit["material_extrapolation_allowed"] = bool(task.allow_material_extrapolation)
        result.audit["incoherent_layers_present"] = bool(task.stack.has_incoherent_layers)
        result.audit["absorptance_definition"] = "A_finite_layers=1-R-T_into_exit_medium"
        result.audit["capability_assessment"] = assessment.to_dict()
        if "system_emissivity" in task.requested_outputs:
            result.audit["system_emissivity_definition"] = (
                "E_system=1-R; physically appropriate when the semi-infinite exit medium is part "
                "of the emitting system and absorbs transmitted power"
            )
        return result

    def field_profile(
        self,
        task: SimulationTask,
        wavelength_nm: float,
        *,
        angle_deg: float = 0.0,
        polarization: str = "s",
        points_per_layer: int = 80,
    ) -> Dict[str, np.ndarray]:
        """Return a position-resolved coherent field and absorption profile.

        The profile uses the independently maintained Byrnes implementation.  It
        is useful for cavity modes, absorption localization, and 1D-PhC defect
        states, and also serves as a second implementation for validation.
        """

        if task.stack.has_incoherent_layers:
            raise ValueError("position-resolved fields require a fully coherent stack")
        if polarization not in ("s", "p"):
            raise ValueError("field_profile polarization must be s or p")
        if int(points_per_layer) < 2:
            raise ValueError("points_per_layer must be >= 2")
        try:
            from tmm import coh_tmm, position_resolved
        except ImportError as exc:  # pragma: no cover
            raise ImportError("The 'tmm' package is required for field profiles") from exc

        # Resolve at one wavelength through the same material contract used by
        # the spectrum solver, preventing material-source drift.
        single_task = SimulationTask(
            stack=task.stack,
            spectrum=type(task.spectrum)(values_nm=(float(wavelength_nm),)),
            illumination=task.illumination,
            solver="byrnes",
            allow_material_extrapolation=task.allow_material_extrapolation,
            requested_outputs=task.requested_outputs,
        )
        media, _, _ = self._resolve_stack(single_task)
        n_list = [complex(values[0]) for values in media]
        d_list = [np.inf] + [float(layer.thickness_nm) for layer in task.stack.layers] + [np.inf]
        data = coh_tmm(
            polarization,
            n_list,
            d_list,
            math.radians(float(angle_deg)),
            float(wavelength_nm),
        )
        z_values: List[float] = []
        layer_values: List[int] = []
        fields: Dict[str, List[complex]] = {"Ex": [], "Ey": [], "Ez": []}
        poyn: List[float] = []
        absor: List[float] = []
        offset = 0.0
        for layer_index, layer in enumerate(task.stack.layers, start=1):
            distances = np.linspace(0.0, float(layer.thickness_nm), int(points_per_layer), endpoint=False)
            for distance in distances:
                point = position_resolved(layer_index, float(distance), data)
                z_values.append(offset + float(distance))
                layer_values.append(layer_index - 1)
                for name in fields:
                    fields[name].append(complex(point.get(name, 0.0)))
                poyn.append(float(np.real(point.get("poyn", 0.0))))
                absor.append(float(np.real(point.get("absor", 0.0))))
            offset += float(layer.thickness_nm)
        return {
            "z_nm": np.asarray(z_values, dtype=np.float64),
            "layer_index": np.asarray(layer_values, dtype=np.int64),
            "Ex": np.asarray(fields["Ex"], dtype=np.complex128),
            "Ey": np.asarray(fields["Ey"], dtype=np.complex128),
            "Ez": np.asarray(fields["Ez"], dtype=np.complex128),
            "poynting_normalized": np.asarray(poyn, dtype=np.float64),
            "absorption_density_per_nm": np.asarray(absor, dtype=np.float64),
        }

    def _simulate_internal(
        self,
        task: SimulationTask,
        media: List[np.ndarray],
        wavelengths_nm: np.ndarray,
        provenance: List[Dict[str, Any]],
    ) -> ForwardSimulationResult:
        if task.stack.has_incoherent_layers:
            raise ValueError("internal coherent solvers cannot process incoherent layers")
        wavelengths_um = wavelengths_nm * 1e-3
        incident = media[0]
        n_list = media[1:]
        d_list = [float(layer.thickness_nm) * 1e-9 for layer in task.stack.layers] + [0.0]
        solver = TMM(
            TMMConfig(
                wavelength_unit="um",
                n_incident=incident,
                treat_last_layer_as_substrate=True,
                clip_R=False,
                clip_T=False,
            )
        )
        channels: Dict[str, Dict[str, np.ndarray]] = {}
        extras: Dict[str, Any] = {}

        def one(angle: float, pol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            if task.solver == "characteristic":
                return solver.rt_spectrum_cm(
                    n_list, d_list, wavelengths_um, angle, pol, "deg", return_amplitudes=True
                )
            return solver.rt_spectrum(
                n_list, d_list, wavelengths_um, angle, pol, "deg", return_amplitudes=True
            )

        for angle in task.illumination.angles_deg:
            for requested_pol in task.illumination.polarizations:
                if requested_pol == "unpolarized":
                    Rs, Ts, rs, ts = one(float(angle), "s")
                    Rp, Tp, rp, tp = one(float(angle), "p")
                    R = 0.5 * (Rs + Rp)
                    T = 0.5 * (Ts + Tp)
                    r = np.stack([rs, rp], axis=0)
                    t = np.stack([ts, tp], axis=0)
                else:
                    R, T, r0, t0 = one(float(angle), requested_pol)
                    r = np.asarray(r0)
                    t = np.asarray(t0)
                values = {"R": np.asarray(R), "T": np.asarray(T), "A": 1.0 - R - T}
                if "system_emissivity" in task.requested_outputs:
                    values["E_system"] = 1.0 - R
                if "amplitudes" in task.requested_outputs:
                    values["r"] = r
                    values["t"] = t
                channel_key = _channel_key(float(angle), requested_pol)
                channels[channel_key] = values
                if "phase_dispersion" in task.requested_outputs:
                    extras["phase_dispersion|%s" % channel_key] = {
                        "reflection": phase_dispersion_from_amplitude(wavelengths_nm, r),
                        "transmission": phase_dispersion_from_amplitude(wavelengths_nm, t),
                        "convention": "fields exp(i*k*z-i*omega*t); group_delay=d(phase)/d(omega)",
                    }
        return ForwardSimulationResult(
            wavelengths_nm=wavelengths_nm,
            channels=channels,
            material_provenance=provenance,
            solver=task.solver,
            audit={"backend": "internal_numpy"},
            extras=extras,
        )

    def _simulate_byrnes(
        self,
        task: SimulationTask,
        media: List[np.ndarray],
        wavelengths_nm: np.ndarray,
        provenance: List[Dict[str, Any]],
    ) -> ForwardSimulationResult:
        try:
            from tmm import (
                absorp_in_each_layer,
                coh_tmm,
                ellips,
                inc_absorp_in_each_layer,
                inc_tmm,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("The 'tmm' package is required for the Byrnes backend") from exc

        d_list = [np.inf] + [float(layer.thickness_nm) for layer in task.stack.layers] + [np.inf]
        c_list = ["i"] + ["c" if layer.coherence == "coherent" else "i" for layer in task.stack.layers] + ["i"]
        mixed = task.stack.has_incoherent_layers
        channels: Dict[str, Dict[str, np.ndarray]] = {}
        extras: Dict[str, Any] = {}

        def one(angle_deg: float, pol: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
            R_values: List[float] = []
            T_values: List[float] = []
            absorption_values: List[np.ndarray] = []
            for wi, wavelength_nm in enumerate(wavelengths_nm):
                n_at_wavelength = [complex(values[wi]) for values in media]
                theta0 = math.radians(float(angle_deg))
                if mixed:
                    data = inc_tmm(pol, n_at_wavelength, d_list, c_list, theta0, float(wavelength_nm))
                    R_values.append(float(data["R"]))
                    T_values.append(float(data["T"]))
                    if "layer_absorption" in task.requested_outputs:
                        full = np.asarray(inc_absorp_in_each_layer(data), dtype=np.float64)
                        absorption_values.append(full[1:-1])
                else:
                    data = coh_tmm(pol, n_at_wavelength, d_list, theta0, float(wavelength_nm))
                    R_values.append(float(data["R"]))
                    T_values.append(float(data["T"]))
                    if "layer_absorption" in task.requested_outputs:
                        full = np.asarray(absorp_in_each_layer(data), dtype=np.float64)
                        absorption_values.append(full[1:-1])
            absorption = None
            if absorption_values:
                absorption = np.stack(absorption_values, axis=1)
            return np.asarray(R_values), np.asarray(T_values), absorption

        for angle in task.illumination.angles_deg:
            for requested_pol in task.illumination.polarizations:
                if requested_pol == "unpolarized":
                    Rs, Ts, As_layers = one(float(angle), "s")
                    Rp, Tp, Ap_layers = one(float(angle), "p")
                    R, T = 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)
                    if As_layers is not None and Ap_layers is not None:
                        extras["layer_absorption|%s" % _channel_key(float(angle), requested_pol)] = 0.5 * (As_layers + Ap_layers)
                else:
                    R, T, layer_abs = one(float(angle), requested_pol)
                    if layer_abs is not None:
                        extras["layer_absorption|%s" % _channel_key(float(angle), requested_pol)] = layer_abs
                channels[_channel_key(float(angle), requested_pol)] = {
                    "R": R,
                    "T": T,
                    "A": 1.0 - R - T,
                }
                if "system_emissivity" in task.requested_outputs:
                    channels[_channel_key(float(angle), requested_pol)]["E_system"] = 1.0 - R

        if "ellipsometry" in task.requested_outputs and not mixed:
            for angle in task.illumination.angles_deg:
                psi, delta = [], []
                for wi, wavelength_nm in enumerate(wavelengths_nm):
                    n_at_wavelength = [complex(values[wi]) for values in media]
                    data = ellips(n_at_wavelength, d_list, math.radians(float(angle)), float(wavelength_nm))
                    psi.append(float(data["psi"]))
                    delta.append(float(data["Delta"]))
                extras["ellipsometry|angle=%g" % float(angle)] = {
                    "psi_rad": psi,
                    "delta_rad": delta,
                }
        if "layer_absorption" in task.requested_outputs:
            extras["layer_absorption_definition"] = (
                "fraction of incident power absorbed in each finite layer; incident and exit media excluded"
            )
        return ForwardSimulationResult(
            wavelengths_nm=wavelengths_nm,
            channels=channels,
            material_provenance=provenance,
            solver="byrnes",
            audit={"backend": "byrnes_tmm", "mixed_coherence": mixed},
            extras=extras,
        )


__all__ = ["ForwardSimulationResult", "TMMWorkbench"]
