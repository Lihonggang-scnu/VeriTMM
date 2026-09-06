"""First-class gradient / sensitivity / batch-proposal API.

The proposal side of "simulate → gradient → decide": for one validated
simulation task, return per-layer thickness gradients of a band objective,
normalized sensitivities (objective change per characteristic thickness
scale), and batched proposal forwards with per-candidate independent
verification.

Boundary (deliberate): gradients and sensitivities are *proposal evidence* —
the machine-readable basis for "which layer is worth changing next".  They
never constitute, upgrade, or replace a physics acceptance certificate;
verification claims live only in the certificate and its evidence.  The torch
backend is required for gradients; an absent torch is reported as an explicit
``backend_unavailable`` typed failure, never silently downgraded to a
finite-difference numpy path.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .schemas import SimulationTask

GRADIENT_API_SCHEMA_VERSION = "veritmm-gradient-result-v1"
SENSITIVITY_API_SCHEMA_VERSION = "veritmm-sensitivity-result-v1"
BATCH_SIMULATE_SCHEMA_VERSION = "veritmm-batch-simulate-result-v1"

DEFAULT_CHARACTERISTIC_SCALE_NM = 10.0
_OBJECTIVE_PATTERN = re.compile(
    r"^mean_(?P<observable>R|T|A)(:(?P<min>[0-9.]+)\.\.(?P<max>[0-9.]+))?$"
)


class GradientRequestError(ValueError):
    """Typed failure of a gradient/sensitivity/batch request.

    ``code`` is machine-readable (for example ``no_optimizable_layers`` or
    ``backend_unavailable``) so the CLI can surface the same typed-failure
    contract as the rest of the protocol.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ObjectiveSpec:
    """Parsed band objective: mean of one observable over an optional band."""

    observable: str
    band_nm: tuple | None
    name: str
    custom: bool = False


def parse_objective(text: str) -> ObjectiveSpec:
    match = _OBJECTIVE_PATTERN.match(text.strip())
    if not match:
        raise GradientRequestError(
            "objective_unsupported",
            "objective must be mean_R, mean_T, or mean_A with an optional "
            ":min..max band (for example mean_R:500..700)",
        )
    band = None
    if match.group("min") is not None:
        band = (float(match.group("min")), float(match.group("max")))
        if not band[1] > band[0]:
            raise GradientRequestError(
                "objective_unsupported", "objective band max must exceed min"
            )
    name = text.strip()
    return ObjectiveSpec(
        observable=match.group("observable"), band_nm=band, name=name
    )


def _objective_notes(objective: ObjectiveSpec) -> list:
    notes = [
        "gradients are proposal evidence; physics validity is decided only by "
        "the acceptance certificate and its verification evidence"
    ]
    if objective.observable == "A":
        notes.append(
            "mean_A uses the closure convention A = 1 - R - T (acceptable for "
            "gradient purposes); the independent absorption ledger is not "
            "differentiable in this release"
        )
    return notes


def _require_torch_backend():
    from .backends import BackendNotRegisteredError, get_backend

    try:
        return get_backend("torch")
    except BackendNotRegisteredError:
        raise GradientRequestError(
            "backend_unavailable",
            "the torch differentiable backend is required for gradients; "
            "install the 'veritmm[optimize]' extra. Gradients are not "
            "silently downgraded to a finite-difference numpy path.",
        ) from None


def _validate_task(task: SimulationTask, variables) -> list:
    if task.stack.has_incoherent_layers:
        raise GradientRequestError(
            "incoherent_stack_unsupported",
            "gradients require a fully coherent stack",
        )
    optimizable = [
        index
        for index, layer in enumerate(task.stack.layers)
        if layer.optimizable
    ]
    if not optimizable:
        raise GradientRequestError(
            "no_optimizable_layers",
            "the task has no optimizable layers; mark layers with "
            "optimizable=true to differentiate thicknesses",
        )
    if variables is None:
        return list(optimizable)
    if isinstance(variables, int) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in variables
    ):
        raise GradientRequestError(
            "unknown_variable",
            "variables must be a list of integer layer indices",
        )
    for index in variables:
        if not 0 <= index < len(task.stack.layers):
            raise GradientRequestError(
                "unknown_variable",
                f"layer index {index} is outside the stack",
            )
        if index not in optimizable:
            raise GradientRequestError(
                "variable_not_optimizable",
                f"layer {index} is not marked optimizable",
            )
    return list(variables)


def _resolve_illumination(task: SimulationTask) -> tuple:
    return (
        float(task.illumination.angles_deg[0]),
        str(task.illumination.polarizations[0]),
    )


def _make_objective_value(
    optimizer: Any,
    task: SimulationTask,
    nk_stack: Any,
    objective: ObjectiveSpec,
    angle_deg: float,
    polarization: str,
    custom_objective: Callable | None,
) -> Callable:
    """Return f(thicknesses_row) -> scalar tensor over all layer thicknesses."""

    torch = optimizer.torch
    wavelengths_nm = task.spectrum.wavelengths_nm()
    wavelength_t = torch.tensor(
        wavelengths_nm * 1e-3, dtype=optimizer.real_dtype, device=optimizer.device
    )
    solver = optimizer._solver_class(
        polarization=polarization,
        dtype_real=optimizer.real_dtype,
        dtype_complex=optimizer.complex_dtype,
    ).to(optimizer.device)
    theta_rad = float(angle_deg) * math.pi / 180.0
    mask_np = np.ones(wavelengths_nm.shape, dtype=bool)
    if objective.band_nm is not None:
        mask_np &= wavelengths_nm >= float(objective.band_nm[0])
        mask_np &= wavelengths_nm <= float(objective.band_nm[1])
    if not bool(np.any(mask_np)):
        raise GradientRequestError(
            "band_outside_grid", "objective band does not overlap the grid"
        )
    mask = torch.tensor(mask_np, dtype=torch.bool, device=optimizer.device)

    def value(thicknesses_row: Any) -> Any:
        result = solver(
            thicknesses_row.reshape(1, -1) * 1e-3,
            nk_stack.unsqueeze(0),
            wavelength_t,
            theta_rad=theta_rad,
        )
        observables = {
            "R": result.R[0],
            "T": result.T[0],
            "A": result.A[0],
        }
        if custom_objective is not None:
            return custom_objective(
                dict(observables, wavelengths_nm=wavelengths_nm.copy())
            )
        return torch.mean(observables[objective.observable][mask])

    return value


def _finite_difference_cross_check(
    value: Callable,
    thicknesses_row: Any,
    indices: list,
    torch: Any,
) -> tuple:
    """Central differences through the same backend; returns per-layer pairs."""

    rows = []
    max_relative = None
    with torch.no_grad():
        base = [float(v) for v in thicknesses_row.detach().reshape(-1)]
        autograd = [float(v) for v in thicknesses_row.grad.reshape(-1)]
        for index in indices:
            h = max(1e-3, abs(base[index]) * 1e-4)
            h = min(h, max(1e-6, abs(base[index]) * 0.49))
            plus = list(base)
            minus = list(base)
            plus[index] = base[index] + h
            minus[index] = base[index] - h
            plus_value = value(torch.tensor([plus], dtype=thicknesses_row.dtype))
            minus_value = value(torch.tensor([minus], dtype=thicknesses_row.dtype))
            fd = (float(plus_value) - float(minus_value)) / (2.0 * h)
            ad = autograd[index]
            absolute = abs(ad - fd)
            near_zero = max(abs(ad), abs(fd)) < 1e-8
            relative = (
                None
                if near_zero
                else absolute / max(abs(ad), abs(fd), 1e-15)
            )
            if relative is not None:
                max_relative = (
                    relative if max_relative is None else max(max_relative, relative)
                )
            rows.append(
                {
                    "layer_index": index,
                    "finite_difference_per_nm": float(fd),
                    "absolute_deviation": float(absolute),
                    "relative_deviation": (
                        None if relative is None else float(relative)
                    ),
                    "near_zero_gradient": bool(near_zero),
                }
            )
    return rows, max_relative


def compute_gradient(
    task: SimulationTask,
    objective: str = "mean_R",
    variables: Sequence | None = None,
    *,
    registry: Any = None,
    device: str = "cpu",
    custom_objective: Callable | None = None,
) -> "GradientResult":
    """Compute d(objective)/d(thickness) for the task's optimizable layers."""

    if objective == "custom" or custom_objective is not None:
        if custom_objective is None or objective != "custom":
            raise GradientRequestError(
                "objective_unsupported",
                "custom objectives require objective='custom' and a "
                "custom_objective callable; custom callables are Python-API "
                "only and are not exposed through JSON/CLI",
            )
        spec = ObjectiveSpec(
            observable="custom", band_nm=None, name="custom", custom=True
        )
    else:
        spec = parse_objective(objective)
    indices = _validate_task(task, variables)
    _require_torch_backend()

    from .optimization import DifferentiableThicknessOptimizer

    optimizer = DifferentiableThicknessOptimizer(registry or _default_registry(), device=device)
    nk_stack, _ = optimizer._build_nk_stack(task)
    angle_deg, polarization = _resolve_illumination(task)
    value = _make_objective_value(
        optimizer, task, nk_stack, spec, angle_deg, polarization, custom_objective
    )
    torch = optimizer.torch
    thicknesses_row = torch.tensor(
        [[float(layer.thickness_nm) for layer in task.stack.layers]],
        dtype=optimizer.real_dtype,
        device=optimizer.device,
        requires_grad=True,
    )
    scalar = value(thicknesses_row)
    scalar.backward()
    metric_value = float(scalar.detach().item())

    fd_rows, max_relative = _finite_difference_cross_check(
        value, thicknesses_row, indices, torch
    )
    fd_by_index = {row["layer_index"]: row for row in fd_rows}

    layers = task.stack.layers
    layer_gradients = []
    for index in indices:
        layer = layers[index]
        gradient = float(thicknesses_row.grad.reshape(-1)[index])
        row = fd_by_index.get(index, {})
        layer_gradients.append(
            {
                "layer_index": index,
                "label": layer.label,
                "thickness_nm": float(layer.thickness_nm),
                "gradient_per_nm": gradient,
                "finite_difference_per_nm": row.get("finite_difference_per_nm"),
                "absolute_deviation": row.get("absolute_deviation"),
                "relative_deviation": row.get("relative_deviation"),
            }
        )
    ranking = sorted(
        layer_gradients,
        key=lambda item: (-abs(float(item["gradient_per_nm"])), item["layer_index"]),
    )
    return GradientResult(
        objective=spec.name,
        observable=spec.observable,
        band_nm=list(spec.band_nm) if spec.band_nm else None,
        angle_deg=angle_deg,
        polarization=polarization,
        method="autodiff",
        backend="torch",
        metric_value=metric_value,
        variables=list(indices),
        layer_gradients=layer_gradients,
        top_influential_layers=[
            {"layer_index": item["layer_index"], "gradient_per_nm": item["gradient_per_nm"]}
            for item in ranking
        ],
        finite_difference_max_relative_deviation=max_relative,
        custom_objective_python_only=spec.custom,
        notes=_objective_notes(spec),
    )


def compute_sensitivity(
    task: SimulationTask,
    variables: Sequence | None = None,
    *,
    objective: str = "mean_R",
    characteristic_scale_nm: float = DEFAULT_CHARACTERISTIC_SCALE_NM,
    registry: Any = None,
    device: str = "cpu",
) -> "SensitivityResult":
    """Normalized per-layer sensitivities of a band objective.

    Sensitivity is the autodiff gradient scaled by ``characteristic_scale_nm``
    — the objective change for a +10 nm thickness move by default.  The
    per-nm gradients and the built-in finite-difference cross-check align
    with the thickness-sensitivity audit in ``scientific_analysis`` (which
    additionally gates each derivative against an independent NumPy central
    difference under tolerance decisions on a certified run); this API is the
    lightweight proposal-stage view of the same quantities.
    """

    gradient = compute_gradient(
        task,
        objective,
        variables,
        registry=registry,
        device=device,
    )
    scale = float(characteristic_scale_nm)
    if not scale > 0.0:
        raise GradientRequestError(
            "objective_unsupported", "characteristic_scale_nm must be positive"
        )
    layer_sensitivities = []
    for item in gradient.layer_gradients:
        sensitivity = float(item["gradient_per_nm"]) * scale
        layer_sensitivities.append(
            {
                "layer_index": item["layer_index"],
                "label": item["label"],
                "thickness_nm": item["thickness_nm"],
                "gradient_per_nm": item["gradient_per_nm"],
                "sensitivity_per_scale_nm": sensitivity,
                "absolute_importance": abs(sensitivity),
                "finite_difference_per_nm": item["finite_difference_per_nm"],
            }
        )
    ranking = sorted(
        layer_sensitivities,
        key=lambda item: (-float(item["absolute_importance"]), item["layer_index"]),
    )
    return SensitivityResult(
        objective=gradient.objective,
        observable=gradient.observable,
        band_nm=gradient.band_nm,
        angle_deg=gradient.angle_deg,
        polarization=gradient.polarization,
        method="autodiff",
        backend="torch",
        metric_value=gradient.metric_value,
        characteristic_scale_nm=scale,
        variables=list(gradient.variables),
        layer_sensitivities=layer_sensitivities,
        top_influential_layers=[
            {"layer_index": item["layer_index"], "sensitivity_per_scale_nm": item["sensitivity_per_scale_nm"]}
            for item in ranking
        ],
        finite_difference_max_relative_deviation=(
            gradient.finite_difference_max_relative_deviation
        ),
        notes=list(gradient.notes)
        + [
            "aligns with the scientific_analysis thickness-sensitivity audit "
            "(autodiff derivative per nm cross-checked against an independent "
            "numpy central difference); that audit runs on a certified run "
            "with tolerance gates, this API is the proposal-stage view"
        ],
    )


def batch_simulate(
    tasks: Sequence[SimulationTask],
    *,
    registry: Any = None,
    settings: Any = None,
) -> "BatchSimulateResult":
    """Batched proposal forwards plus per-candidate independent verification.

    Phase one runs one batched torch forward per physics group (tasks sharing
    media, grid, and illumination differ only in thickness).  Phase two
    reuses the existing per-candidate certification pattern
    (``certify_simulation`` under the default acceptance settings) — batch
    execution never changes a verification decision.  Proposal values are
    proposal evidence; only the certificate speaks about validity.
    """

    if not tasks:
        raise GradientRequestError("empty_batch", "batch_simulate requires tasks")
    _require_torch_backend()
    from .acceptance import AcceptanceSettings, certify_simulation
    from .optimization import DifferentiableThicknessOptimizer
    from .run_artifacts import stable_payload_sha256

    workbench_registry = registry or _default_registry()
    workbench = _workbench(workbench_registry)
    optimizer = DifferentiableThicknessOptimizer(workbench_registry, device="cpu")

    groups: dict = {}
    for index, task in enumerate(tasks):
        if task.stack.has_incoherent_layers:
            raise GradientRequestError(
                "incoherent_stack_unsupported",
                f"task {index} is not fully coherent",
            )
        angle_deg, polarization = _resolve_illumination(task)
        wavelengths_nm = task.spectrum.wavelengths_nm()
        nk_stack, _ = optimizer._build_nk_stack(task)
        key = stable_payload_sha256(
            {
                "wavelengths_nm": [float(w) for w in wavelengths_nm],
                "angle_deg": angle_deg,
                "polarization": polarization,
                "nk": np.asarray(nk_stack.detach().numpy()).tolist(),
            }
        )
        groups.setdefault(key, {"tasks": [], "nk_stack": nk_stack, "angle_deg": angle_deg, "polarization": polarization, "wavelengths_nm": wavelengths_nm})
        groups[key]["tasks"].append((index, task))

    torch = optimizer.torch
    entries: list = []
    for group in groups.values():
        stacked_tasks = group["tasks"]
        wavelength_t = torch.tensor(
            group["wavelengths_nm"] * 1e-3,
            dtype=optimizer.real_dtype,
            device=optimizer.device,
        )
        nk_batched = group["nk_stack"].unsqueeze(0).expand(len(stacked_tasks), -1, -1)
        thicknesses = torch.tensor(
            [[float(layer.thickness_nm) for layer in task.stack.layers] for _, task in stacked_tasks],
            dtype=optimizer.real_dtype,
            device=optimizer.device,
        )
        solver = optimizer._solver_class(
            polarization=group["polarization"],
            dtype_real=optimizer.real_dtype,
            dtype_complex=optimizer.complex_dtype,
        ).to(optimizer.device)
        proposal = solver(
            thicknesses * 1e-3,
            nk_batched,
            wavelength_t,
            theta_rad=float(group["angle_deg"]) * math.pi / 180.0,
        )
        proposal_means = {
            name: proposal_value.detach().numpy()
            for name, proposal_value in (
                ("R", proposal.R),
                ("T", proposal.T),
                ("A", proposal.A),
            )
        }
        for position, (index, task) in enumerate(stacked_tasks):
            certified = certify_simulation(
                workbench, task, settings or AcceptanceSettings()
            )
            certified_means = {}
            if certified.result is not None:
                channel = certified.result.channel(
                    group["angle_deg"], group["polarization"]
                )
                certified_means = {
                    name: float(np.mean(np.asarray(channel[name], dtype=np.float64)))
                    for name in ("R", "T", "A")
                }
            proposal_row = {
                name: float(np.mean(values[position])) for name, values in proposal_means.items()
            }
            matches = bool(certified_means) and all(
                abs(proposal_row[name] - certified_means[name]) <= 1e-6
                for name in ("R", "T", "A")
            )
            certificate = certified.certificate
            entries.append(
                {
                    "index": index,
                    "status": certificate.get("status"),
                    "physics_accepted": bool(certificate.get("accepted")),
                    "certificate_id": certificate.get("certificate_id"),
                    "proposal_means": proposal_row,
                    "certified_means": certified_means,
                    "proposal_matches_certified": matches,
                    "failure_codes": [
                        item.get("code") for item in (certificate.get("failures") or [])
                    ],
                }
            )
    entries.sort(key=lambda item: item["index"])
    return BatchSimulateResult(
        count=len(entries),
        backend="torch",
        verification="per-candidate certify_simulation under the default acceptance settings",
        entries=entries,
    )


def _default_registry():
    from .material_registry import MaterialRegistry

    return MaterialRegistry()


def _workbench(registry):
    from .workbench import TMMWorkbench

    return TMMWorkbench(registry)


@dataclass
class GradientResult:
    objective: str
    observable: str
    band_nm: list | None
    angle_deg: float
    polarization: str
    method: str
    backend: str
    metric_value: float
    variables: list
    layer_gradients: list
    top_influential_layers: list
    finite_difference_max_relative_deviation: float | None
    custom_objective_python_only: bool
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": GRADIENT_API_SCHEMA_VERSION,
            "objective": self.objective,
            "observable": self.observable,
            "band_nm": self.band_nm,
            "angle_deg": self.angle_deg,
            "polarization": self.polarization,
            "method": self.method,
            "backend": self.backend,
            "metric_value": self.metric_value,
            "variables": self.variables,
            "layer_gradients": self.layer_gradients,
            "top_influential_layers": self.top_influential_layers,
            "finite_difference_max_relative_deviation": (
                self.finite_difference_max_relative_deviation
            ),
            "custom_objective_python_only": self.custom_objective_python_only,
            "notes": self.notes,
        }


@dataclass
class SensitivityResult:
    objective: str
    observable: str
    band_nm: list | None
    angle_deg: float
    polarization: str
    method: str
    backend: str
    metric_value: float
    characteristic_scale_nm: float
    variables: list
    layer_sensitivities: list
    top_influential_layers: list
    finite_difference_max_relative_deviation: float | None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": SENSITIVITY_API_SCHEMA_VERSION,
            "objective": self.objective,
            "observable": self.observable,
            "band_nm": self.band_nm,
            "angle_deg": self.angle_deg,
            "polarization": self.polarization,
            "method": self.method,
            "backend": self.backend,
            "metric_value": self.metric_value,
            "characteristic_scale_nm": self.characteristic_scale_nm,
            "variables": self.variables,
            "layer_sensitivities": self.layer_sensitivities,
            "top_influential_layers": self.top_influential_layers,
            "finite_difference_max_relative_deviation": (
                self.finite_difference_max_relative_deviation
            ),
            "notes": self.notes,
        }


@dataclass
class BatchSimulateResult:
    count: int
    backend: str
    verification: str
    entries: list

    def to_dict(self) -> dict:
        return {
            "schema_version": BATCH_SIMULATE_SCHEMA_VERSION,
            "count": self.count,
            "backend": self.backend,
            "verification": self.verification,
            "entries": self.entries,
        }


__all__ = [
    "BATCH_SIMULATE_SCHEMA_VERSION",
    "GRADIENT_API_SCHEMA_VERSION",
    "GradientRequestError",
    "GradientResult",
    "ObjectiveSpec",
    "SensitivityResult",
    "batch_simulate",
    "compute_gradient",
    "compute_sensitivity",
    "parse_objective",
]
