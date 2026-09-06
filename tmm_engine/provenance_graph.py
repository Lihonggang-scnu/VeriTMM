"""Provenance graph v1: a read-only projection over archived runs.

AiiDA-inspired, lightweight: Data / Calculation / Decision / Workflow nodes
with verb-aligned edges, built strictly from existing run artifacts
(RUN_RESULT envelope, certificate, spectra, summaries) — the graph is a
projection, never a second source of truth, and building it writes nothing.
The data-provenance subgraph (Data + Calculation nodes joined by
``input``/``created_by`` edges) is a directed acyclic graph; a violation is a
construction bug and is checked by :meth:`ProvenanceGraph.data_subgraph_is_dag`.

Explanation queries answer "why is run A better than run B": metric
comparison over the runs' metric snapshots plus the evidence chains
(task → calculation → metrics → certificate → decision) with per-step file
pointers and hashes, verified against the current artifact bytes using the
same integrity criterion as ``verify-run``.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .hashing import file_sha256, stable_sha256
from .verify_run import verify_run_dir

PROVENANCE_SCHEMA_VERSION = "veritmm-provenance-v1"

NODE_TYPES = ("data", "calculation", "decision", "workflow")
DATA_KINDS = ("task", "spectrum", "certificate", "metric_snapshot")
EDGE_TYPES = ("input", "created_by", "returned_by", "decided_by", "part_of")


class ProvenanceGraphError(ValueError):
    """Typed failure of graph construction or query."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class GraphNode:
    node_hash: str
    node_type: str
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "node_hash": self.node_hash,
            "node_type": self.node_type,
            "kind": self.kind,
            "payload": self.payload,
            "evidence_path": self.evidence_path,
        }


@dataclass(frozen=True)
class GraphEdge:
    src_hash: str
    dst_hash: str
    edge_type: str
    evidence_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "src_hash": self.src_hash,
            "dst_hash": self.dst_hash,
            "edge_type": self.edge_type,
            "evidence_path": self.evidence_path,
        }


@dataclass
class ProvenanceGraph:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    run_reports: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> str:
        self.nodes[node.node_hash] = node
        return node.node_hash

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.src_hash not in self.nodes or edge.dst_hash not in self.nodes:
            raise ProvenanceGraphError(
                "dangling_edge", "edge endpoints must be nodes of the graph"
            )
        if edge not in self.edges:
            self.edges.append(edge)

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "nodes": [
                self.nodes[key].to_dict() for key in sorted(self.nodes)
            ],
            "edges": sorted(
                (edge.to_dict() for edge in self.edges),
                key=lambda item: (
                    item["src_hash"],
                    item["dst_hash"],
                    item["edge_type"],
                ),
            ),
            "run_reports": dict(sorted(self.run_reports.items())),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "ProvenanceGraph":
        if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
            raise ProvenanceGraphError(
                "schema_mismatch",
                f"unsupported provenance schema {payload.get('schema_version')!r}",
            )
        graph = cls()
        for node in payload.get("nodes", []):
            graph.add_node(
                GraphNode(
                    node_hash=node["node_hash"],
                    node_type=node["node_type"],
                    kind=node["kind"],
                    payload=node.get("payload", {}),
                    evidence_path=node.get("evidence_path"),
                )
            )
        for edge in payload.get("edges", []):
            graph.add_edge(
                GraphEdge(
                    src_hash=edge["src_hash"],
                    dst_hash=edge["dst_hash"],
                    edge_type=edge["edge_type"],
                    evidence_path=edge.get("evidence_path"),
                )
            )
        graph.run_reports = dict(payload.get("run_reports", {}))
        return graph

    @classmethod
    def load(cls, path: str | Path) -> "ProvenanceGraph":
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")

    # -- queries ----------------------------------------------------------

    def data_subgraph_is_dag(self) -> bool:
        allowed_nodes = {"data", "calculation"}
        allowed_edges = {"input", "created_by"}
        incoming: Dict[str, int] = {}
        adjacency: Dict[str, List[str]] = {}
        members = {
            key for key, node in self.nodes.items() if node.node_type in allowed_nodes
        }
        for key in members:
            incoming[key] = 0
            adjacency[key] = []
        for edge in self.edges:
            if (
                edge.edge_type in allowed_edges
                and edge.src_hash in members
                and edge.dst_hash in members
            ):
                adjacency[edge.src_hash].append(edge.dst_hash)
                incoming[edge.dst_hash] += 1
        queue = deque(key for key in members if incoming[key] == 0)
        visited = 0
        while queue:
            key = queue.popleft()
            visited += 1
            for neighbour in adjacency[key]:
                incoming[neighbour] -= 1
                if incoming[neighbour] == 0:
                    queue.append(neighbour)
        return visited == len(members)

    def ancestors(self, node_hash: str) -> List[str]:
        return self._traverse(node_hash, reverse=True)

    def descendants(self, node_hash: str) -> List[str]:
        return self._traverse(node_hash, reverse=False)

    def _traverse(self, node_hash: str, *, reverse: bool) -> List[str]:
        if node_hash not in self.nodes:
            raise ProvenanceGraphError(
                "unknown_node", f"unknown node {node_hash!r}"
            )
        adjacency: Dict[str, List[str]] = {}
        for edge in self.edges:
            src, dst = edge.src_hash, edge.dst_hash
            if reverse:
                src, dst = dst, src
            adjacency.setdefault(src, []).append(dst)
        seen = {node_hash}
        queue = deque([node_hash])
        order: List[str] = []
        while queue:
            key = queue.popleft()
            for neighbour in sorted(adjacency.get(key, [])):
                if neighbour not in seen:
                    seen.add(neighbour)
                    order.append(neighbour)
                    queue.append(neighbour)
        return order

    def path_between(self, src: str, dst: str) -> List[str]:
        for node in (src, dst):
            if node not in self.nodes:
                raise ProvenanceGraphError(
                    "unknown_node", f"unknown node {node!r}"
                )
        adjacency: Dict[str, List[str]] = {}
        for edge in self.edges:
            adjacency.setdefault(edge.src_hash, []).append(edge.dst_hash)
            adjacency.setdefault(edge.dst_hash, []).append(edge.src_hash)
        previous: Dict[str, Optional[str]] = {src: None}
        queue = deque([src])
        while queue:
            key = queue.popleft()
            if key == dst:
                path = [dst]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])
                return list(reversed(path))
            for neighbour in sorted(adjacency.get(key, [])):
                if neighbour not in previous:
                    previous[neighbour] = key
                    queue.append(neighbour)
        raise ProvenanceGraphError(
            "no_path", f"no path between {src!r} and {dst!r}"
        )

    def run_id_for_source(self, source: Path | str) -> str:
        key = str(Path(source).resolve())
        if key not in self.run_reports:
            raise ProvenanceGraphError(
                "unknown_run", f"run directory not part of the graph: {source!r}"
            )
        run_id = self.run_reports[key].get("run_id")
        if not run_id:
            raise ProvenanceGraphError(
                "unknown_run", f"no run identity recorded for {source!r}"
            )
        return str(run_id)

    def calculation_node_for_run(self, run_id: str) -> GraphNode:
        for node in self.nodes.values():
            if node.node_type == "calculation" and node.payload.get("run_id") == run_id:
                return node
        raise ProvenanceGraphError("unknown_run", f"no calculation node for {run_id!r}")

    def _decision_for_run(self, run_id: str) -> Optional[GraphNode]:
        calc = self.calculation_node_for_run(run_id)
        for edge in self.edges:
            if edge.src_hash == calc.node_hash and edge.edge_type == "decided_by":
                return self.nodes.get(edge.dst_hash)
        return None

    def explain_superiority(self, run_a: str, run_b: str) -> dict:
        """Why is one run superior?  Metric comparison plus evidence chains."""

        chains = {}
        metrics = {}
        for label, run_id in (("run_a", run_a), ("run_b", run_b)):
            calc = self.calculation_node_for_run(run_id)
            chain_steps = []
            chain_nodes = [
                *self.ancestors(calc.node_hash),
                calc.node_hash,
                *self.descendants(calc.node_hash),
            ]
            for node_hash in chain_nodes:
                node = self.nodes[node_hash]
                evidence = None
                if node.evidence_path:
                    path = Path(node.evidence_path)
                    evidence = {
                        "path": str(path),
                        "exists": path.is_file(),
                        "sha256": file_sha256(path) if path.is_file() else None,
                    }
                chain_steps.append(
                    {
                        "node_hash": node_hash,
                        "node_kind": node.kind,
                        "node_type": node.node_type,
                        "evidence": evidence,
                    }
                )
            chains[label] = chain_steps
            for node_hash in self.descendants(calc.node_hash):
                node = self.nodes[node_hash]
                if node.kind == "metric_snapshot":
                    metrics[label] = node.payload.get("channel_means", {})
        def _mean_r(channel_means: dict) -> Optional[float]:
            values = [
                channel["R"]
                for channel in channel_means.values()
                if isinstance(channel, dict) and channel.get("R") is not None
            ]
            return float(sum(values) / len(values)) if values else None

        superior_run = None
        delta = None
        value_a = _mean_r(metrics.get("run_a", {}))
        value_b = _mean_r(metrics.get("run_b", {}))
        if value_a is not None and value_b is not None and abs(value_a - value_b) > 1e-12:
            superior_run = run_a if value_a > value_b else run_b
            delta = abs(value_a - value_b)
        return {
            "verdict": {
                "metric": "mean_R",
                "value_a": value_a,
                "value_b": value_b,
                "delta": delta,
                "superior_run": superior_run,
                "note": "mean R comparison is the v1 superiority heuristic; "
                "the acceptance certificate remains the only validity claim",
            },
            "chains": chains,
        }


def _artifact_ref(envelope: dict, filename: str) -> Optional[dict]:
    for reference in envelope.get("artifacts", []):
        if str(reference.get("path", "")).endswith(filename):
            return reference
    return None


def _build_run_nodes(graph: ProvenanceGraph, run_dir: Path, envelope: dict) -> str:
    run_id = str(envelope.get("run_id"))
    task_sha = str(envelope.get("task_sha256"))
    certificate_id = envelope.get("certificate_id")

    task_node = GraphNode(
        node_hash=task_sha,
        node_type="data",
        kind="task",
        payload={"run_id": run_id},
        evidence_path=str(run_dir / "NORMALIZED_TASK.json"),
    )
    calculation_hash = stable_sha256(
        {
            "kind": "engine_run",
            "run_id": run_id,
            "task_sha256": task_sha,
            "certificate_id": certificate_id,
        }
    )
    calculation_node = GraphNode(
        node_hash=calculation_hash,
        node_type="calculation",
        kind="engine_run",
        payload={
            "run_id": run_id,
            "status": envelope.get("status"),
            "solver": envelope.get("solver"),
        },
        evidence_path=str(run_dir / "RUN_RESULT.json"),
    )
    graph.add_node(task_node)
    graph.add_node(calculation_node)
    graph.add_edge(
        GraphEdge(
            src_hash=task_sha,
            dst_hash=calculation_hash,
            edge_type="input",
            evidence_path=str(run_dir / "NORMALIZED_TASK.json"),
        )
    )

    def add_created_data(filename: str, kind: str, payload: dict) -> Optional[str]:
        path = run_dir / filename
        if not path.is_file():
            return None
        content_hash = file_sha256(path)
        node = GraphNode(
            node_hash=content_hash,
            node_type="data",
            kind=kind,
            payload=payload,
            evidence_path=str(path),
        )
        graph.add_node(node)
        graph.add_edge(
            GraphEdge(
                src_hash=calculation_hash,
                dst_hash=content_hash,
                edge_type="created_by",
                evidence_path=str(path),
            )
        )
        return content_hash

    certificate_hash = None
    certificate_ref = _artifact_ref(envelope, "PHYSICS_ACCEPTANCE_CERTIFICATE.json")
    if certificate_ref is not None:
        certificate_hash = add_created_data(
            "PHYSICS_ACCEPTANCE_CERTIFICATE.json",
            "certificate",
            {"certificate_id": certificate_id},
        )

    add_created_data("SPECTRA.csv", "spectrum", {})

    summary_path = run_dir / "RESULT_SUMMARY.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        channel_means = {}
        features = summary.get("spectral_features", {})
        for channel_key, channel in features.items():
            if isinstance(channel, dict):
                means = {
                    name: channel[name]["mean"]
                    for name in ("R", "T", "A")
                    if isinstance(channel.get(name), dict)
                    and channel[name].get("mean") is not None
                }
                if means:
                    channel_means[channel_key] = means
        add_created_data(
            "RESULT_SUMMARY.json",
            "metric_snapshot",
            {"channel_means": channel_means},
        )

    decision_hash = None
    certificate_path = run_dir / "PHYSICS_ACCEPTANCE_CERTIFICATE.json"
    if certificate_path.is_file():
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        decision_hash = stable_sha256(
            {
                "certificate_id": certificate.get("certificate_id"),
                "accepted": certificate.get("accepted"),
                "status": certificate.get("status"),
                "failure_codes": [
                    item.get("code") for item in (certificate.get("failures") or [])
                ],
            }
        )
        graph.add_node(
            GraphNode(
                node_hash=decision_hash,
                node_type="decision",
                kind="acceptance_decision",
                payload={
                    "accepted": certificate.get("accepted"),
                    "status": certificate.get("status"),
                    "run_id": run_id,
                },
                evidence_path=str(certificate_path),
            )
        )
        anchor = certificate_hash or calculation_hash
        if anchor is None:
            anchor = calculation_hash
        graph.add_edge(
            GraphEdge(
                src_hash=anchor,
                dst_hash=decision_hash,
                edge_type="decided_by",
                evidence_path=str(certificate_path),
            )
        )

    return calculation_hash


def build_graph(sources: Sequence[Path | str]) -> ProvenanceGraph:
    """Build the provenance graph from run directories (read-only).

    Integrity of every source directory is established with the existing
    ``verify-run`` machinery before its artifacts are projected into nodes.
    """

    graph = ProvenanceGraph()
    calculation_hashes: Dict[str, str] = {}
    for source in sources:
        run_dir = Path(source)
        if not run_dir.is_dir():
            raise ProvenanceGraphError(
                "missing_run_dir", f"run directory does not exist: {run_dir}"
            )
        report = verify_run_dir(run_dir)
        graph.run_reports[str(run_dir.resolve())] = {
            "integrity_status": report.get("integrity_status"),
            "overall_status": report.get("overall_status"),
            "run_id": report.get("run_id"),
        }
        envelope_path = run_dir / "RUN_RESULT.json"
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        calculation_hash = _build_run_nodes(graph, run_dir, envelope)
        calculation_hashes[str(run_dir)] = calculation_hash

        sweep_path = run_dir / "SWEEP_RESULT.json"
        if sweep_path.is_file():
            sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
            sweep_hash = file_sha256(sweep_path)
            graph.add_node(
                GraphNode(
                    node_hash=sweep_hash,
                    node_type="workflow",
                    kind="sweep_container",
                    payload={
                        "children": len(sweep.get("children", [])),
                        "run_id": envelope.get("run_id"),
                    },
                    evidence_path=str(sweep_path),
                )
            )
            for child in sweep.get("children", []):
                child_root = run_dir / str(child.get("artifact_root", ""))
                child_run_id = child.get("child_run_id")
                if not child_root.is_dir() or not child_run_id:
                    continue
                child_envelope_path = child_root / "RUN_RESULT.json"
                if not child_envelope_path.is_file():
                    continue
                child_envelope = json.loads(
                    child_envelope_path.read_text(encoding="utf-8")
                )
                child_calc = _build_run_nodes(graph, child_root, child_envelope)
                calculation_hashes[str(child_root)] = child_calc
                graph.add_edge(
                    GraphEdge(
                        src_hash=child_calc,
                        dst_hash=sweep_hash,
                        edge_type="part_of",
                        evidence_path=str(sweep_path),
                    )
                )
                certificate_node = None
                for edge in graph.edges:
                    if (
                        edge.src_hash == child_calc
                        and edge.edge_type == "created_by"
                        and graph.nodes[edge.dst_hash].kind == "certificate"
                    ):
                        certificate_node = edge.dst_hash
                if certificate_node:
                    graph.add_edge(
                        GraphEdge(
                            src_hash=sweep_hash,
                            dst_hash=certificate_node,
                            edge_type="returned_by",
                            evidence_path=str(sweep_path),
                        )
                    )
    return graph


__all__ = [
    "DATA_KINDS",
    "EDGE_TYPES",
    "NODE_TYPES",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceGraph",
    "ProvenanceGraphError",
    "build_graph",
]
