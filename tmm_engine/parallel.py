"""Deterministic CPU parallel execution helpers (Windows spawn safe).

Scope: embarrassingly parallel study loops whose units are independent
managed/child runs — sweep children, verified batches of candidates.  The
serial implementations are untouched; this module is a wrapper that only
activates when the caller passes ``workers > 1``.

Determinism contract:
- child work units are pure functions of their payload (task JSON + output
  path + logical index + seed); nothing depends on completion order;
- child RNG seeds (where a study uses them) derive as
  ``derive_child_seed(parent_seed, child_index) = int(sha256(canonical
  {parent_seed, child_index, domain})[:16], 16)`` — stable across processes,
  platforms, and run order;
- scientific fingerprints of results must be identical between workers=1 and
  workers>1 (``tests/test_parallel_equivalence.py``); only ``run_id``,
  timestamps, output paths, and completion order may differ, and they are
  explicitly excluded from the comparison.

Environment: workers force single-threaded BLAS inside child processes
(``OMP_NUM_THREADS=1``, ``MKL_NUM_THREADS=1``) so nested parallelism cannot
oversubscribe the machine; the variables are exported by the parent before
the pool spawns and re-asserted in the worker initializer.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Sequence

from .hashing import stable_sha256

_THREADS_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def derive_child_seed(parent_seed: int, child_index: int, domain: str = "sweep") -> int:
    """Deterministic child seed: first 64 bits of the canonical hash of
    ``(parent_seed, child_index, domain)``.  Documented derivation used by
    every parallel study loop."""

    digest = stable_sha256(
        {
            "parent_seed": int(parent_seed),
            "child_index": int(child_index),
            "domain": domain,
        }
    )
    return int(digest[:16], 16)


def force_single_thread_blas() -> None:
    """Pin BLAS pools to one thread (call in the parent before spawning and
    re-assert inside each worker)."""

    for name in _THREADS_ENV:
        os.environ[name] = "1"


def _worker_init() -> None:
    force_single_thread_blas()


def map_payloads(
    payloads: Sequence[Dict[str, Any]],
    worker: Callable[[Dict[str, Any]], Any],
    *,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    """Map ``worker`` over ``payloads`` and return one result record per
    payload, in input order.

    ``workers == 1`` runs everything in-process (the serial reference path).
    ``workers > 1`` dispatches to a spawn-based process pool; exceptions are
    captured per payload as ``{"ok": False, "error": ...}`` records so one bad
    unit can never fail the batch.  Payloads and results must be picklable
    (task JSON, paths, indices — never engine objects).
    """

    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    payloads = list(payloads)
    if workers == 1 or len(payloads) <= 1:
        results = []
        for payload in payloads:
            try:
                results.append({"ok": True, "result": worker(payload)})
            except Exception as exc:  # noqa: BLE001 - failure isolation
                results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return results

    force_single_thread_blas()
    with ProcessPoolExecutor(
        max_workers=min(workers, len(payloads)), mp_context=_spawn_context()
    ) as pool:
        futures = [pool.submit(worker, payload) for payload in payloads]
        results = []
        for future in futures:
            try:
                results.append({"ok": True, "result": future.result()})
            except Exception as exc:  # noqa: BLE001 - failure isolation
                results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return results


def _spawn_context():
    import multiprocessing

    try:
        return multiprocessing.get_context("spawn")
    except ValueError:  # pragma: no cover - platform without spawn
        return multiprocessing.get_context()


__all__ = [
    "derive_child_seed",
    "force_single_thread_blas",
    "map_payloads",
]
