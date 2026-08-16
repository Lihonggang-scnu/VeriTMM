# Multilayer Optics Workbench Skill

Use this skill only after a scientific request has been converted into a
standard `SimulationTask` or `OptimizationTask`. Do not infer missing units.

## Non-negotiable rules

1. Wavelength and thickness fields in the public JSON contract are nanometres.
2. A finite substrate is a finite `LayerSpec` followed by an exit medium. Never
   set a flag and silently treat it as semi-infinite.
3. Select material datasets by provider and `dataset_id` when reproducibility
   matters. Never silently extrapolate optical constants.
4. `A = 1 - R - T` means absorption in finite layers, where `T` is power entering
   the semi-infinite exit medium. `E_system = 1 - R` is a different quantity and
   is valid only when the absorbing exit medium belongs to the emitting system.
5. Use the stable S-matrix backend by default. Use the characteristic matrix only
   for diagnosis and the Byrnes backend for mixed coherent/incoherent stacks,
   field profiles, layer absorption, or ellipsometry.
6. A differentiable optimizer may propose a design but may not certify itself.
   Accept optimization only after `INDEPENDENT_VALIDATION.json` says `passed`.
7. Never clip raw spectra to hide negative absorption or values outside physical
   bounds. Treat them as diagnostics.

## Commands

Search the bundled refractiveindex.info mirror and export an exact dataset:

```powershell
python scripts/search_optical_material.py TiO2 --provider rii `
  --start-nm 500 --stop-nm 800 --dataset-id 418 `
  --export-csv outputs/tio2.csv
```

Run a forward task:

```powershell
python scripts/run_tmm_task.py `
  --input examples/tmm_tasks/periodic_dbr_simulation.json `
  --output-dir outputs/tmm_run
```

Run differentiable thickness optimization. The launcher automatically discovers
the runtime named by `VERITMM_PHYSICS_PYTHON` or `.venv-physics`:

```powershell
python scripts/run_tmm_task.py `
  --input examples/tmm_tasks/antireflection_optimization.json `
  --output-dir outputs/tmm_optimization
```

Reproduce the published DBR/cavity trend case:

```powershell
python scripts/validate_tmm_against_pmc9147317.py
```

## Supported experiment families

- coherent thin-film stacks;
- thick/incoherent substrates and mixed-coherence stacks;
- periodic 1D photonic crystals, defect cavities, and chirped stacks;
- wavelength-, angle-, and polarization-resolved R/T/A;
- system emissivity with explicit semantics;
- position-resolved fields and absorption;
- finite-layer absorption and ellipsometry;
- Bloch forbidden-band analysis;
- thickness-tolerance Monte Carlo analysis;
- multi-band, multi-angle differentiable thickness optimization with equality,
  lower-bound, or upper-bound targets, mean or worst-case aggregation, fixed
  layers, fabrication bounds, multistart, and thickness quantization.

## Out of scope

This is an isotropic planar 1D solver. It does not claim to model lateral
patterning, diffraction gratings, metasurface unit cells, diffuse scattering,
anisotropic tensor layers, nonlinear optics, thermal transport, or fabrication
chemistry. Route those tasks to a suitable RCWA/FDTD/FEM/thermal tool.
