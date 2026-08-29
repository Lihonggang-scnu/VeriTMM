"""Thickness inverse design built on the SpecFormer-derived PyTorch backend."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schemas import LayerSpec, MediumSpec, OptimizationTask, SimulationTask
from .uncertainty import (
    apply_thickness_boundary_policy_torch,
    final_robustness_seed,
    sample_normal_offsets,
)


@dataclass
class OptimizationResult:
    status: str
    initial_thicknesses_nm: List[float]
    optimized_thicknesses_nm: List[float]
    quantized_thicknesses_nm: Optional[List[float]]
    initial_loss: float
    optimized_loss: float
    quantized_loss: Optional[float]
    steps_executed: int
    best_start_index: int
    stop_reason: str
    wall_seconds: float
    loss_history: List[float] = field(default_factory=list)
    target_metrics: Dict[str, float] = field(default_factory=dict)
    candidate_designs: List[Dict[str, Any]] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)
    evaluation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "initial_thicknesses_nm": self.initial_thicknesses_nm,
            "optimized_thicknesses_nm": self.optimized_thicknesses_nm,
            "quantized_thicknesses_nm": self.quantized_thicknesses_nm,
            "initial_loss": self.initial_loss,
            "optimized_loss": self.optimized_loss,
            "quantized_loss": self.quantized_loss,
            "steps_executed": self.steps_executed,
            "best_start_index": self.best_start_index,
            "stop_reason": self.stop_reason,
            "wall_seconds": self.wall_seconds,
            "loss_history": self.loss_history,
            "target_metrics": self.target_metrics,
            "candidate_designs": self.candidate_designs,
            "audit": self.audit,
            "evaluation_count": self.evaluation_count,
        }


class DifferentiableThicknessOptimizer:
    """Optimize continuous layer thicknesses for arbitrary spectral bands.

    Material choice remains discrete and fixed during this stage.  A future outer
    search may propose material sequences and call this optimizer for each sequence.
    """

    def __init__(self, material_registry: Any, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise ImportError("PyTorch is required for differentiable optimization") from exc
        from .differentiable import DifferentiableTMM

        self.torch = torch
        self.device = torch.device(device)
        self.registry = material_registry
        self.real_dtype = torch.float64
        self.complex_dtype = torch.complex128
        self._solver_class = DifferentiableTMM
        self._evaluation_count = 0

    @property
    def evaluation_count(self) -> int:
        """Number of design rows evaluated by this optimizer instance."""

        return int(self._evaluation_count)

    def _sample_material(
        self,
        name: str,
        wavelengths_nm: np.ndarray,
        provider: Optional[str],
        dataset_id: Optional[str],
        allow_extrapolation: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        return nk, provenance

    def _sample_layer(
        self,
        layer: LayerSpec,
        wavelengths_nm: np.ndarray,
        allow_extrapolation: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if layer.constant_n is not None:
            nk = np.full(
                wavelengths_nm.shape,
                complex(float(layer.constant_n), float(layer.constant_k)),
                dtype=np.complex128,
            )
            return nk, {
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
            nk = np.full(
                wavelengths_nm.shape,
                complex(float(medium.constant_n), float(medium.constant_k)),
                dtype=np.complex128,
            )
            return nk, {"provider": "constant", "n": medium.constant_n, "k": medium.constant_k}
        return self._sample_material(
            str(medium.material), wavelengths_nm, medium.provider, medium.dataset_id, allow_extrapolation
        )

    def _build_nk_stack(self, task: SimulationTask) -> Tuple[Any, List[Dict[str, Any]]]:
        wavelengths_nm = task.spectrum.wavelengths_nm()
        samples: List[np.ndarray] = []
        provenance: List[Dict[str, Any]] = []
        nk, meta = self._sample_medium(task.stack.incident, wavelengths_nm, task.allow_material_extrapolation)
        samples.append(nk)
        provenance.append(meta)
        for layer in task.stack.layers:
            nk, meta = self._sample_layer(
                layer, wavelengths_nm, task.allow_material_extrapolation
            )
            samples.append(nk)
            provenance.append(meta)
        nk, meta = self._sample_medium(task.stack.exit, wavelengths_nm, task.allow_material_extrapolation)
        samples.append(nk)
        provenance.append(meta)
        stacked = np.stack(samples, axis=0)
        tensor = self.torch.tensor(stacked, dtype=self.complex_dtype, device=self.device)
        return tensor, provenance

    @staticmethod
    def _range_to_raw(value: Any, lo: Any, hi: Any) -> Any:
        span = (hi - lo).clamp_min(1e-9)
        ratio = ((value - lo) / span).clamp(1e-7, 1.0 - 1e-7)
        return (ratio / (1.0 - ratio)).log()

    @staticmethod
    def _raw_to_range(raw: Any, lo: Any, hi: Any) -> Any:
        return lo + (hi - lo) * raw.sigmoid()

    def _loss_for_thicknesses(
        self,
        task: OptimizationTask,
        thicknesses_nm: Any,
        nk_stack: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        wavelengths_nm = task.simulation.spectrum.wavelengths_nm()
        wavelength_t = self.torch.tensor(
            wavelengths_nm * 1e-3, dtype=self.real_dtype, device=self.device
        )
        batch_size = int(thicknesses_nm.shape[0])
        self._evaluation_count += batch_size
        nk_batched = nk_stack.unsqueeze(0).expand(batch_size, -1, -1)
        details: Dict[str, Any] = {}
        total = self.torch.zeros(batch_size, dtype=self.real_dtype, device=self.device)
        total_weight = 0.0
        grouped: Dict[Tuple[float, str], List[Any]] = {}
        for target in task.targets:
            grouped.setdefault((float(target.angle_deg), str(target.polarization)), []).append(target)

        for (angle_deg, polarization), targets in grouped.items():
            solver = self._solver_class(
                polarization=polarization,
                dtype_real=self.real_dtype,
                dtype_complex=self.complex_dtype,
            ).to(self.device)
            out = solver(
                thicknesses_nm * 1e-3,
                nk_batched,
                wavelength_t,
                theta_rad=float(angle_deg) * math.pi / 180.0,
            )
            observables = {"R": out.R, "T": out.T, "A": out.A}
            for index, target in enumerate(targets):
                mask_np = (wavelengths_nm >= float(target.wavelength_min_nm)) & (
                    wavelengths_nm <= float(target.wavelength_max_nm)
                )
                if not bool(np.any(mask_np)):
                    raise ValueError("target band does not overlap the simulation grid")
                mask = self.torch.tensor(mask_np, dtype=self.torch.bool, device=self.device)
                values = observables[target.observable][:, mask]
                if target.constraint == "match":
                    errors = (values - float(target.target)) ** 2
                elif target.constraint == "at_least":
                    # Directional preferences remain continuous beyond the stated
                    # reference target; the target is not a hard acceptance plateau.
                    errors = (1.0 - values) ** 2
                else:
                    errors = values**2
                term = (
                    self.torch.max(errors, dim=1).values
                    if target.aggregation == "worst_case"
                    else self.torch.mean(errors, dim=1)
                )
                total = total + float(target.weight) * term
                total_weight += float(target.weight)
                key = target.name or "%s_%g_%g_%s_%s" % (
                    target.observable,
                    target.wavelength_min_nm,
                    target.wavelength_max_nm,
                    angle_deg,
                    polarization,
                )
                details[key] = self.torch.mean(values, dim=1)
        return total / max(total_weight, 1e-12), details

    def _robust_loss_for_thicknesses(
        self,
        task: OptimizationTask,
        thicknesses_nm: Any,
        nk_stack: Any,
        perturbations_nm: Any,
    ) -> Any:
        """Aggregate a fixed differentiable perturbation ensemble per design."""

        if task.robustness is None or not task.robustness.enabled:
            return self._loss_for_thicknesses(task, thicknesses_nm, nk_stack)[0]
        batch, layers = thicknesses_nm.shape
        samples = int(perturbations_nm.shape[0])
        expanded = apply_thickness_boundary_policy_torch(
            thicknesses_nm.reshape(batch, 1, layers)
            + perturbations_nm.reshape(1, samples, layers),
            boundary_policy=task.robustness.boundary_policy,
            min_thickness_physical_nm=float(
                task.robustness.min_thickness_physical_nm
            ),
        )
        losses, _ = self._loss_for_thicknesses(
            task,
            expanded.reshape(batch * samples, layers),
            nk_stack,
        )
        losses = losses.reshape(batch, samples)
        objective = task.robustness.objective
        if objective == "expected_loss":
            return self.torch.mean(losses, dim=1)
        if objective == "worst_case_loss":
            return self.torch.max(losses, dim=1).values
        if objective == "cvar":
            if task.robustness.cvar_alpha is None:
                raise ValueError("cvar objective requires cvar_alpha")
            tail_count = max(
                1,
                int(np.ceil(float(task.robustness.cvar_alpha) * samples)),
            )
            sorted_losses = self.torch.sort(losses, dim=1).values
            return self.torch.mean(sorted_losses[:, -tail_count:], dim=1)
        return self.torch.mean(losses, dim=1) + float(task.robustness.k_sigma) * self.torch.std(
            losses, dim=1, unbiased=False
        )

    def evaluate(self, task: OptimizationTask, thicknesses_nm: Sequence[float]) -> Tuple[float, Dict[str, float]]:
        task.validate()
        nk_stack, _ = self._build_nk_stack(task.simulation)
        value = self.torch.tensor(
            np.asarray(thicknesses_nm, dtype=np.float64).reshape(1, -1),
            dtype=self.real_dtype,
            device=self.device,
        )
        with self.torch.no_grad():
            loss, details = self._loss_for_thicknesses(task, value, nk_stack)
        return float(loss[0].item()), {key: float(val[0].item()) for key, val in details.items()}

    def optimize(self, task: OptimizationTask) -> OptimizationResult:
        self._evaluation_count = 0
        task.validate()
        if task.simulation.stack.has_incoherent_layers:
            raise ValueError("differentiable optimizer currently supports coherent stacks only")
        torch = self.torch
        cfg = task.optimizer
        torch.manual_seed(int(cfg.seed))
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))

        initial = np.asarray([layer.thickness_nm for layer in task.simulation.stack.layers], dtype=np.float64)
        bounds = np.asarray(
            [layer.bounds_nm(cfg.thickness_window_nm) for layer in task.simulation.stack.layers],
            dtype=np.float64,
        )
        fixed = np.asarray([not layer.optimizable for layer in task.simulation.stack.layers], dtype=bool)
        bounds[fixed, 0] = initial[fixed]
        bounds[fixed, 1] = initial[fixed]
        # A tiny span avoids singular inverse-sigmoid values while fixed layers are
        # explicitly overwritten after the transform.
        safe_hi = np.maximum(bounds[:, 1], bounds[:, 0] + 1e-6)
        starts = int(cfg.starts)
        lo = torch.tensor(bounds[:, 0], dtype=self.real_dtype, device=self.device).reshape(1, -1)
        hi = torch.tensor(safe_hi, dtype=self.real_dtype, device=self.device).reshape(1, -1)
        init = torch.tensor(initial, dtype=self.real_dtype, device=self.device).reshape(1, -1)
        raw_init = self._range_to_raw(init, lo, hi)
        raw_data = raw_init.repeat(starts, 1)
        if starts > 1:
            raw_data[1:] = raw_data[1:] + 0.75 * torch.randn_like(raw_data[1:])
        raw = torch.nn.Parameter(raw_data)
        fixed_mask = torch.tensor(fixed, dtype=torch.bool, device=self.device).reshape(1, -1)
        nk_stack, provenance = self._build_nk_stack(task.simulation)
        robustness_offsets = None
        if task.robustness is not None and task.robustness.enabled:
            offsets = sample_normal_offsets(
                seed=int(task.robustness.seed),
                sample_count=int(task.robustness.samples_per_step),
                layer_count=initial.size,
                sigma_nm=float(task.robustness.thickness_sigma_nm),
            )
            robustness_offsets = torch.tensor(
                offsets,
                dtype=self.real_dtype,
                device=self.device,
            )

        optimizer = torch.optim.Adam([raw], lr=float(cfg.learning_rate))
        best_loss = float("inf")
        best_thickness = init[0].detach().clone()
        best_start = 0
        best_step = 0
        best_per_start_loss = torch.full(
            (starts,), float("inf"), dtype=self.real_dtype, device=self.device
        )
        best_per_start_thickness = init.repeat(starts, 1).detach().clone()
        no_improve = 0
        history: List[float] = []
        stop_reason = "max_steps"
        started = time.perf_counter()

        for step in range(1, int(cfg.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            thicknesses = self._raw_to_range(raw, lo.expand_as(raw), hi.expand_as(raw))
            thicknesses = torch.where(fixed_mask.expand_as(thicknesses), init.expand_as(thicknesses), thicknesses)
            per_start = (
                self._robust_loss_for_thicknesses(
                    task, thicknesses, nk_stack, robustness_offsets
                )
                if robustness_offsets is not None
                else self._loss_for_thicknesses(task, thicknesses, nk_stack)[0]
            )
            loss = torch.mean(per_start)
            if not torch.isfinite(loss):
                stop_reason = "non_finite_loss"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], float(cfg.gradient_clip_norm))
            optimizer.step()
            current, current_index = torch.min(per_start.detach(), dim=0)
            improved_per_start = per_start.detach() < best_per_start_loss
            if bool(torch.any(improved_per_start)):
                best_per_start_loss = torch.where(
                    improved_per_start, per_start.detach(), best_per_start_loss
                )
                best_per_start_thickness[improved_per_start] = thicknesses.detach()[
                    improved_per_start
                ]
            current_value = float(current.item())
            history.append(current_value)
            if current_value < best_loss - float(cfg.improvement_tolerance):
                best_loss = current_value
                best_start = int(current_index.item())
                best_step = step
                best_thickness = thicknesses.detach()[best_start].clone()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= int(cfg.early_stop_patience):
                stop_reason = "no_improvement"
                break

        # Optional deterministic local finish.  It runs only from the best Adam
        # solution and therefore does not inflate all multi-start trajectories.
        if cfg.method == "adam_lbfgs" and stop_reason != "non_finite_loss":
            raw_best = torch.nn.Parameter(
                self._range_to_raw(best_thickness.reshape(1, -1), lo, hi).detach().clone()
            )
            lbfgs = torch.optim.LBFGS(
                [raw_best], lr=0.5, max_iter=20, tolerance_grad=1e-9, tolerance_change=1e-10
            )

            def closure() -> Any:
                lbfgs.zero_grad(set_to_none=True)
                candidate = self._raw_to_range(raw_best, lo, hi)
                candidate = torch.where(fixed_mask, init, candidate)
                value = (
                    self._robust_loss_for_thicknesses(
                        task, candidate, nk_stack, robustness_offsets
                    )
                    if robustness_offsets is not None
                    else self._loss_for_thicknesses(task, candidate, nk_stack)[0]
                )
                value[0].backward()
                return value[0]

            try:
                lbfgs.step(closure)
                candidate = self._raw_to_range(raw_best, lo, hi)
                candidate = torch.where(fixed_mask, init, candidate)
                with torch.no_grad():
                    candidate_loss = (
                        self._robust_loss_for_thicknesses(
                            task, candidate, nk_stack, robustness_offsets
                        )
                        if robustness_offsets is not None
                        else self._loss_for_thicknesses(task, candidate, nk_stack)[0]
                    )
                if float(candidate_loss[0].item()) < best_loss:
                    best_loss = float(candidate_loss[0].item())
                    best_thickness = candidate.detach()[0].clone()
                    stop_reason = "lbfgs_finish"
            except RuntimeError:
                # Adam result remains valid and auditable if LBFGS encounters an
                # unsupported complex-gradient edge case.
                stop_reason = "lbfgs_failed_adam_kept"

        optimized = best_thickness.detach().cpu().numpy().astype(np.float64)
        initial_loss, _ = self.evaluate(task, initial)
        optimized_loss, metrics = self.evaluate(task, optimized)
        quantized_values: Optional[np.ndarray] = None
        quantized_loss: Optional[float] = None
        if cfg.quantization_nm is not None:
            q = float(cfg.quantization_nm)
            quantized_values = np.round(optimized / q) * q
            quantized_values = np.clip(quantized_values, bounds[:, 0], bounds[:, 1])
            quantized_values[fixed] = initial[fixed]
            quantized_loss, _ = self.evaluate(task, quantized_values)

        raw_candidates: List[Tuple[str, np.ndarray, float]] = [
            (
                "initial",
                initial.copy(),
                float(
                    initial_loss
                    if robustness_offsets is None
                    else self._robust_loss_for_thicknesses(
                        task,
                        torch.tensor(
                            initial.reshape(1, -1),
                            dtype=self.real_dtype,
                            device=self.device,
                        ),
                        nk_stack,
                        robustness_offsets,
                    )[0].detach().item()
                ),
            ),
            ("optimized_best", optimized.copy(), float(best_loss)),
        ]
        for index in range(starts):
            values = best_per_start_thickness[index].detach().cpu().numpy().astype(np.float64)
            raw_candidates.append(
                (f"multistart_{index}", values, float(best_per_start_loss[index].item()))
            )
        if quantized_values is not None and quantized_loss is not None:
            quantized_selection_loss = float(quantized_loss)
            if robustness_offsets is not None:
                quantized_selection_loss = float(
                    self._robust_loss_for_thicknesses(
                        task,
                        torch.tensor(
                            quantized_values.reshape(1, -1),
                            dtype=self.real_dtype,
                            device=self.device,
                        ),
                        nk_stack,
                        robustness_offsets,
                    )[0]
                    .detach()
                    .item()
                )
            raw_candidates.append(
                ("quantized_best", quantized_values.copy(), quantized_selection_loss)
            )

        # Keep distinct trajectories; repeated starts often converge to the same basin.
        candidate_designs: List[Dict[str, Any]] = []
        seen: set[Tuple[float, ...]] = set()
        for source, values, loss in sorted(raw_candidates, key=lambda item: item[2]):
            signature = tuple(float(x) for x in np.round(values, decimals=8))
            if signature in seen or not np.isfinite(loss):
                continue
            seen.add(signature)
            candidate_nominal_loss, candidate_metrics = self.evaluate(task, values)
            candidate_designs.append(
                {
                    "candidate_id": f"candidate_{len(candidate_designs) + 1:02d}",
                    "source": source,
                    "thicknesses_nm": values.tolist(),
                    "objective_loss": float(candidate_nominal_loss),
                    "training_selection_loss": float(loss),
                    "training_objective": (
                        "nominal_loss"
                        if task.robustness is None or not task.robustness.enabled
                        else task.robustness.objective
                    ),
                    "target_metrics": candidate_metrics,
                }
            )

        return OptimizationResult(
            status="completed" if np.isfinite(optimized_loss) else "failed",
            initial_thicknesses_nm=initial.tolist(),
            optimized_thicknesses_nm=optimized.tolist(),
            quantized_thicknesses_nm=None if quantized_values is None else quantized_values.tolist(),
            initial_loss=float(initial_loss),
            optimized_loss=float(optimized_loss),
            quantized_loss=None if quantized_loss is None else float(quantized_loss),
            steps_executed=len(history),
            best_start_index=best_start,
            stop_reason=stop_reason,
            wall_seconds=float(time.perf_counter() - started),
            loss_history=history,
            target_metrics=metrics,
            candidate_designs=candidate_designs,
            audit={
                "best_step": best_step,
                "device": str(self.device),
                "dtype": str(self.real_dtype),
                "material_provenance": provenance,
                "coherent_only": True,
                "specformer_derived_backend": True,
                "evaluation_count": self.evaluation_count,
                "robust_training": (
                    None
                    if task.robustness is None
                    else {
                        **task.robustness.__dict__,
                        "distribution": task.robustness.distribution,
                        "training_seed": int(task.robustness.seed),
                        "final_seed": final_robustness_seed(
                            int(task.robustness.seed)
                        ),
                        "final_validation_is_independent": True,
                    }
                ),
            },
            evaluation_count=self.evaluation_count,
        )

    def validate_result(
        self,
        task: OptimizationTask,
        result: OptimizationResult,
        workbench: Any,
        *,
        loss_tolerance: float = 1e-7,
    ) -> Tuple[SimulationTask, Any, Dict[str, Any]]:
        """Re-simulate an optimized design with the independent NumPy backend.

        The differentiable solver is allowed to propose a design but is never
        allowed to certify itself.  This closes the most important inverse-design
        loophole: exploiting a gradient/backend bug that looks good only to the
        optimizer.
        """

        chosen = (
            result.quantized_thicknesses_nm
            if result.quantized_thicknesses_nm is not None
            else result.optimized_thicknesses_nm
        )
        layers = tuple(
            replace(layer, thickness_nm=float(value))
            for layer, value in zip(task.simulation.stack.layers, chosen)
        )
        simulation = replace(
            task.simulation,
            stack=replace(task.simulation.stack, layers=layers),
            solver="smatrix",
        )
        forward = workbench.simulate(simulation)
        weighted_loss = 0.0
        total_weight = 0.0
        target_metrics: Dict[str, float] = {}
        target_attainment: Dict[str, Dict[str, Any]] = {}
        all_reportable_targets_met = True
        wavelengths = simulation.spectrum.wavelengths_nm()
        for target in task.targets:
            channel = forward.channel(float(target.angle_deg), str(target.polarization))
            mask = (wavelengths >= float(target.wavelength_min_nm)) & (
                wavelengths <= float(target.wavelength_max_nm)
            )
            values = np.asarray(channel[target.observable], dtype=np.float64)[mask]
            if target.constraint == "match":
                errors = (values - float(target.target)) ** 2
            elif target.constraint == "at_least":
                errors = (1.0 - values) ** 2
            else:
                errors = values**2
            objective_loss = float(
                np.max(errors) if target.aggregation == "worst_case" else np.mean(errors)
            )
            weighted_loss += float(target.weight) * objective_loss
            total_weight += float(target.weight)
            key = target.name or "%s_%g_%g_%g_%s" % (
                target.observable,
                target.wavelength_min_nm,
                target.wavelength_max_nm,
                target.angle_deg,
                target.polarization,
            )
            target_metrics[key + "_mean"] = float(np.mean(values))
            target_metrics[key + "_objective_loss"] = objective_loss
            tolerance = float(target.tolerance or 0.0)
            if target.constraint == "at_most":
                observed = float(np.max(values) if target.aggregation == "worst_case" else np.mean(values))
                satisfied = observed <= float(target.target) + tolerance
                certifiable = True
                shortfall = max(0.0, observed - (float(target.target) + tolerance))
            elif target.constraint == "at_least":
                observed = float(np.min(values) if target.aggregation == "worst_case" else np.mean(values))
                satisfied = observed >= float(target.target) - tolerance
                certifiable = True
                shortfall = max(0.0, (float(target.target) - tolerance) - observed)
            elif target.tolerance is not None:
                deviations = np.abs(values - float(target.target))
                observed = float(np.max(deviations) if target.aggregation == "worst_case" else np.mean(deviations))
                satisfied = observed <= tolerance
                certifiable = True
                shortfall = max(0.0, observed - tolerance)
            else:
                observed = float(np.mean(values))
                satisfied = None
                certifiable = False
                shortfall = None
            if certifiable:
                all_reportable_targets_met = all_reportable_targets_met and bool(satisfied)
            target_attainment[key] = {
                "constraint": target.constraint,
                "aggregation": target.aggregation,
                "target": float(target.target),
                "tolerance": tolerance if certifiable else None,
                "observed": observed,
                "certifiable": certifiable,
                "satisfied": satisfied,
                "shortfall": shortfall,
                "weight": float(target.weight),
                "role": "soft_scoring_objective",
            }
        numpy_loss = weighted_loss / max(total_weight, 1e-12)
        torch_loss, _ = self.evaluate(task, chosen)
        difference = abs(float(torch_loss) - float(numpy_loss))
        audit = {
            "status": "passed"
            if difference <= float(loss_tolerance)
            and forward.audit.get("passivity_check_passed")
            else "failed",
            "differentiable_loss": float(torch_loss),
            "independent_numpy_loss": float(numpy_loss),
            "absolute_loss_difference": float(difference),
            "loss_tolerance": float(loss_tolerance),
            "passivity_check_passed": bool(forward.audit.get("passivity_check_passed")),
            "energy_conservation_max_abs_error": forward.audit.get(
                "energy_conservation_max_abs_error"
            ),
            "target_metrics": target_metrics,
            "target_acceptance": target_attainment,
            "target_attainment": target_attainment,
            "all_reportable_targets_met": all_reportable_targets_met,
            "design_outcome_status": (
                "target_met" if all_reportable_targets_met else "physically_valid_best_effort"
            ),
        }
        return simulation, forward, audit


__all__ = ["DifferentiableThicknessOptimizer", "OptimizationResult"]
