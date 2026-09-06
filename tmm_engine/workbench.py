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


INDEPENDENCE_AUDIT_FIELDS = (
    "energy_independent_max_abs_error",
    "energy_independent_worst_channel",
    "energy_independent_worst_wavelength_nm",
    "absorption_independence_class",
)
"""Audit fields derived from the independent absorption accounting.

They are computed by ``_audit_channels`` but routed into
``ForwardSimulationResult.independence_audit`` instead of ``result.audit``:
the evidence/certificate path serializes ``result.audit`` verbatim, so any
key added there would change every ``certificate_id`` and break the golden
verification-equivalence fixtures without a declared behaviour change.
"""


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
    independence_audit: Dict[str, Any] = field(default_factory=dict)
    independent_absorption: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    """Per-channel A_independent arrays; in-memory verifier input only (never
    serialized — declared channels and task hashes must stay byte-identical)."""

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
        independent_absorption: Optional[Mapping[str, np.ndarray]] = None,
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
        audit = {
            "energy_conservation_max_abs_error": max_energy_error,
            "energy_worst_case_channel": energy_worst_channel,
            "energy_worst_case_wavelength_idx": energy_worst_lam_idx if energy_worst_channel is not None else None,
            "energy_worst_case_wavelength_nm": energy_worst_lam_nm,
            "minimum_observable": min_value,
            "maximum_observable": max_value,
            "nonfinite_value_count": nonfinite,
            "passivity_check_passed": passive,
        }
        independence: Dict[str, Any] = {
            "energy_independent_max_abs_error": None,
            "energy_independent_worst_channel": None,
            "energy_independent_worst_wavelength_nm": None,
            "absorption_independence_class": "unavailable",
        }
        if independent_absorption:
            worst_error = 0.0
            independent_worst_channel: Optional[str] = None
            independent_worst_idx = 0
            complete = bool(channels) and set(independent_absorption.keys()) == set(channels.keys())
            for ch_key, values in channels.items():
                a_independent = independent_absorption.get(ch_key)
                if a_independent is None:
                    complete = False
                    break
                R = np.asarray(values["R"], dtype=np.float64)
                T = np.asarray(values["T"], dtype=np.float64)
                error = np.abs(1.0 - R - T - np.asarray(a_independent, dtype=np.float64))
                ch_max_error = float(np.max(error))
                if independent_worst_channel is None or ch_max_error > worst_error:
                    worst_error = ch_max_error
                    independent_worst_channel = ch_key
                    independent_worst_idx = int(np.argmax(error))
            if complete and independent_worst_channel is not None:
                independence = {
                    "energy_independent_max_abs_error": worst_error,
                    "energy_independent_worst_channel": independent_worst_channel,
                    "energy_independent_worst_wavelength_nm": (
                        float(wavelengths_nm[independent_worst_idx])
                        if wavelengths_nm is not None
                        else None
                    ),
                    "absorption_independence_class": "layer_absorption_integral",
                }
        audit.update(independence)
        return audit

    @staticmethod
    def _layer_absorption_integrals(
        media: List[np.ndarray],
        wavelengths_nm: np.ndarray,
        d_nm: List[float],
        angle_deg: float,
        polarization: str,
    ) -> np.ndarray:
        """Per-film absorbed power fractions from tangential-field flux integration.

        Independent absorption accounting for the internal solvers: the layer
        absorption is the decrease of the net complex Poynting flux across each
        finite film in the Abeles admittance framework (s: ``Y = n cos(theta)``,
        p: ``Y = n / cos(theta)``).  It never consumes the solver's R/T outputs
        and never uses the ``1 - R - T`` closure; the reflection amplitude only
        fixes the field scale through the routine's own transfer-matrix solve.
        Returns shape ``(films, wavelengths)``; raises ``ValueError`` on
        degenerate input so callers can degrade honestly instead of guessing.
        """

        if polarization not in ("s", "p"):
            raise ValueError("polarization must be s or p")
        wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
        if wavelengths.size == 0 or np.any(wavelengths <= 0.0):
            raise ValueError("wavelengths must be positive and non-empty")
        n = np.asarray([np.asarray(row, dtype=np.complex128) for row in media])
        if n.ndim != 2 or n.shape[0] < 2:
            raise ValueError("media must include incident and exit media")
        films = n.shape[0] - 2
        if len(d_nm) != films:
            raise ValueError("one thickness per finite film is required")
        if films == 0:
            return np.zeros((0, wavelengths.size), dtype=np.float64)
        sin0 = np.sin(math.radians(float(angle_deg)))
        kx = n[0] * sin0
        cos_t = np.sqrt(1.0 - (kx[None, :] / n) ** 2 + 0j)
        if polarization == "s":
            y = n * cos_t
        else:
            y = n / cos_t
        k0 = 2.0 * np.pi / wavelengths
        delta = k0[None, :] * np.asarray(d_nm, dtype=np.float64)[:, None] * n[1:-1, :] * cos_t[1:-1, :]
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        y_film = y[1:-1, :]
        y0 = y[0]
        y_exit = y[-1]
        # Total transfer with V_exit = T @ V_0 and V = [E; H] tangential, where
        # the film matrix is [[cos d, i sin d / Y], [i Y sin d, cos d]].
        one = np.ones_like(cos_d[0])
        zero = np.zeros_like(cos_d[0])
        t11, t12, t21, t22 = one, zero, zero, one
        for j in range(films):
            a, b, yj = cos_d[j], sin_d[j], y_film[j]
            l12 = 1j * b / yj
            l21 = 1j * yj * b
            t11, t12, t21, t22 = (
                a * t11 + l12 * t21,
                a * t12 + l12 * t22,
                l21 * t11 + a * t21,
                l21 * t12 + a * t22,
            )
        # Solve [t; Y_exit t] = T [1+r; Y0 (1-r)] with u = 1+r, v = 1-r, u+v = 2.
        p = y_exit * t11 - t21
        q = y0 * (t22 - y_exit * t12)
        denom = p + q
        if not np.all(np.isfinite(denom)) or np.any(denom == 0):
            raise ValueError("degenerate stack transfer; field scale is undetermined")
        v = 2.0 * p / denom
        u = 2.0 - v
        v0 = np.stack([u, y0 * v])
        flux_norm = np.real(y0)
        if not np.all(np.isfinite(flux_norm)) or np.any(flux_norm <= 0.0):
            raise ValueError("incident medium must carry a positive real flux")
        absorbed = np.empty((films, wavelengths.size), dtype=np.float64)
        c11, c12, c21, c22 = one, zero, zero, one
        for j in range(films):
            # C = L_{j-1} ... L_1 maps the incident-side fields to the top of
            # film j (tangential fields are continuous across each interface).
            e_top = c11 * v0[0] + c12 * v0[1]
            h_top = c21 * v0[0] + c22 * v0[1]
            a, b, yj = cos_d[j], sin_d[j], y_film[j]
            e_bot = a * e_top + (1j * b / yj) * h_top
            h_bot = 1j * yj * b * e_top + a * h_top
            absorbed[j] = np.real(e_top * np.conj(h_top)) - np.real(e_bot * np.conj(h_bot))
            l12 = 1j * b / yj
            l21 = 1j * yj * b
            c11, c12, c21, c22 = (
                a * c11 + l12 * c21,
                a * c12 + l12 * c22,
                l21 * c11 + a * c21,
                l21 * c12 + a * c22,
            )
        result = absorbed / flux_norm[None, :]
        if not np.all(np.isfinite(result)):
            raise ValueError("flux integration produced non-finite values")
        return result

    def _independent_channel_absorption(
        self,
        media: List[np.ndarray],
        wavelengths_nm: np.ndarray,
        d_nm: List[float],
        angle_deg: float,
        polarization: str,
    ) -> Optional[np.ndarray]:
        """Total finite-film absorption for one channel, or ``None`` if unavailable."""

        try:
            integrals = self._layer_absorption_integrals(
                media, wavelengths_nm, d_nm, angle_deg, polarization
            )
        except Exception:
            return None
        if integrals.shape[0] == 0:
            return np.zeros(wavelengths_nm.shape, dtype=np.float64)
        total = integrals.sum(axis=0)
        if not np.all(np.isfinite(total)):
            return None
        return total

    def simulate(self, task: SimulationTask) -> ForwardSimulationResult:
        assessment = assess_tmm_capability(task)
        if not assessment.supported:
            raise PhysicsEngineError(assessment.failures[0])
        media, wavelengths_nm, provenance = self._resolve_stack(task)
        if assessment.resolved_solver == "byrnes":
            result, independent_absorption = self._simulate_byrnes(
                task, media, wavelengths_nm, provenance
            )
        else:
            result, independent_absorption = self._simulate_internal(
                task, media, wavelengths_nm, provenance
            )
        audit = self._audit_channels(
            result.channels, result.wavelengths_nm, independent_absorption=independent_absorption
        )
        result.independence_audit = {
            key: audit.pop(key) for key in sorted(INDEPENDENCE_AUDIT_FIELDS) if key in audit
        }
        result.independent_absorption = dict(independent_absorption)
        result.audit.update(audit)
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
    ) -> Tuple[ForwardSimulationResult, Dict[str, Optional[np.ndarray]]]:
        if task.stack.has_incoherent_layers:
            raise ValueError("internal coherent solvers cannot process incoherent layers")
        wavelengths_um = wavelengths_nm * 1e-3
        incident = media[0]
        n_list = media[1:]
        d_list = [float(layer.thickness_nm) * 1e-9 for layer in task.stack.layers] + [0.0]
        d_nm = [float(layer.thickness_nm) for layer in task.stack.layers]
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
        independent_absorption: Dict[str, Optional[np.ndarray]] = {}

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
                    a_ind_s = self._independent_channel_absorption(
                        media, wavelengths_nm, d_nm, float(angle), "s"
                    )
                    a_ind_p = self._independent_channel_absorption(
                        media, wavelengths_nm, d_nm, float(angle), "p"
                    )
                    a_independent = (
                        None if a_ind_s is None or a_ind_p is None else 0.5 * (a_ind_s + a_ind_p)
                    )
                else:
                    R, T, r0, t0 = one(float(angle), requested_pol)
                    r = np.asarray(r0)
                    t = np.asarray(t0)
                    a_independent = self._independent_channel_absorption(
                        media, wavelengths_nm, d_nm, float(angle), requested_pol
                    )
                values = {"R": np.asarray(R), "T": np.asarray(T), "A": 1.0 - R - T}
                if "system_emissivity" in task.requested_outputs:
                    values["E_system"] = 1.0 - R
                if "amplitudes" in task.requested_outputs:
                    values["r"] = r
                    values["t"] = t
                channel_key = _channel_key(float(angle), requested_pol)
                channels[channel_key] = values
                independent_absorption[channel_key] = a_independent
                if "phase_dispersion" in task.requested_outputs:
                    extras["phase_dispersion|%s" % channel_key] = {
                        "reflection": phase_dispersion_from_amplitude(wavelengths_nm, r),
                        "transmission": phase_dispersion_from_amplitude(wavelengths_nm, t),
                        "convention": "fields exp(i*k*z-i*omega*t); group_delay=d(phase)/d(omega)",
                    }
        return (
            ForwardSimulationResult(
                wavelengths_nm=wavelengths_nm,
                channels=channels,
                material_provenance=provenance,
                solver=task.solver,
                audit={"backend": "internal_numpy"},
                extras=extras,
            ),
            independent_absorption,
        )

    def _simulate_byrnes(
        self,
        task: SimulationTask,
        media: List[np.ndarray],
        wavelengths_nm: np.ndarray,
        provenance: List[Dict[str, Any]],
    ) -> Tuple[ForwardSimulationResult, Dict[str, Optional[np.ndarray]]]:
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
        independent_absorption: Dict[str, Optional[np.ndarray]] = {}

        def one(
            angle_deg: float, pol: str
        ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
            R_values: List[float] = []
            T_values: List[float] = []
            absorption_values: List[np.ndarray] = []
            independent_values: List[float] = []
            for wi, wavelength_nm in enumerate(wavelengths_nm):
                n_at_wavelength = [complex(values[wi]) for values in media]
                theta0 = math.radians(float(angle_deg))
                if mixed:
                    data = inc_tmm(pol, n_at_wavelength, d_list, c_list, theta0, float(wavelength_nm))
                    full = np.asarray(inc_absorp_in_each_layer(data), dtype=np.float64)
                else:
                    data = coh_tmm(pol, n_at_wavelength, d_list, theta0, float(wavelength_nm))
                    full = np.asarray(absorp_in_each_layer(data), dtype=np.float64)
                R_values.append(float(data["R"]))
                T_values.append(float(data["T"]))
                independent_values.append(float(np.sum(full[1:-1])))
                if "layer_absorption" in task.requested_outputs:
                    absorption_values.append(full[1:-1])
            absorption = None
            if absorption_values:
                absorption = np.stack(absorption_values, axis=1)
            return (
                np.asarray(R_values),
                np.asarray(T_values),
                absorption,
                np.asarray(independent_values, dtype=np.float64),
            )

        for angle in task.illumination.angles_deg:
            for requested_pol in task.illumination.polarizations:
                if requested_pol == "unpolarized":
                    Rs, Ts, As_layers, A_ind_s = one(float(angle), "s")
                    Rp, Tp, Ap_layers, A_ind_p = one(float(angle), "p")
                    R, T = 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)
                    if As_layers is not None and Ap_layers is not None:
                        extras["layer_absorption|%s" % _channel_key(float(angle), requested_pol)] = 0.5 * (As_layers + Ap_layers)
                    independent_absorption[_channel_key(float(angle), requested_pol)] = (
                        0.5 * (A_ind_s + A_ind_p)
                    )
                else:
                    R, T, layer_abs, a_independent = one(float(angle), requested_pol)
                    if layer_abs is not None:
                        extras["layer_absorption|%s" % _channel_key(float(angle), requested_pol)] = layer_abs
                    independent_absorption[_channel_key(float(angle), requested_pol)] = a_independent
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
        return (
            ForwardSimulationResult(
                wavelengths_nm=wavelengths_nm,
                channels=channels,
                material_provenance=provenance,
                solver="byrnes",
                audit={"backend": "byrnes_tmm", "mixed_coherence": mixed},
                extras=extras,
            ),
            independent_absorption,
        )


__all__ = ["ForwardSimulationResult", "TMMWorkbench"]
