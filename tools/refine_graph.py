"""Refine Graphify output into a quieter, evidence-preserving project graph.

Run this after Graphify has produced ``.graphify_extract.json`` and
``.graphify_detect.json`` but before its normal temporary-file cleanup::

    $(cat graphify-out/.graphify_python) tools/refine_graph.py

For later code-only refreshes, reuse the saved semantic evidence::

    tools/update_graph.sh

The default ``graph.json`` remains an undirected traversal graph so Graphify
queries can walk both callers and callees, but it becomes a MultiGraph. Edge
endpoint order and ``_src``/``_tgt`` retain direction, while parallel relation
types remain separate. A fuller audit copy is written to ``graph.full.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from graphify.analyze import god_nodes, suggest_questions, surprising_connections
from graphify.build import _normalize_id, build_from_json
from graphify.cli import _stamped_manifest_files
from graphify.cluster import (
    cluster,
    label_communities_by_hub,
    remap_communities_to_previous,
    score_all,
)
from graphify.detect import detect, save_manifest
from graphify.export import to_json
from graphify.extract import collect_files, extract
from graphify.report import generate


GENERIC_RELATIONS = {"conceptually_related_to", "references", "uses"}
NAVIGATION_INFERRED_USES_MIN_CONFIDENCE = 0.75
DOC_SUFFIXES = {".md", ".mdx", ".markdown", ".txt"}
WORD_RE = re.compile(r"[a-z0-9]+")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*$")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def _normalized_source_file(value: object, root: Path) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(root)
        except (OSError, ValueError):
            return path.as_posix()
    return path.as_posix()


def _unique_index(rows: list[tuple[str, str]]) -> dict[str, str]:
    candidates: dict[str, set[str]] = defaultdict(set)
    for key, node_id in rows:
        if key:
            candidates[key].add(node_id)
    return {key: next(iter(values)) for key, values in candidates.items() if len(values) == 1}


def _endpoint_mapper(
    extraction: dict[str, Any],
    canonical: nx.DiGraph,
    root: Path,
) -> Callable[[object], str | None]:
    canonical_ids = {str(node_id) for node_id in canonical.nodes}
    normalized = _unique_index(
        [(_normalize_id(node_id), node_id) for node_id in canonical_ids]
    )
    signatures = _unique_index(
        [
            (
                f"{_normalized_source_file(data.get('source_file'), root)}\0"
                f"{str(data.get('label') or '').strip()}",
                str(node_id),
            )
            for node_id, data in canonical.nodes(data=True)
        ]
    )
    extracted_nodes = {
        str(node.get("id")): node
        for node in extraction.get("nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
    }

    def resolve(raw_id: object) -> str | None:
        if not isinstance(raw_id, (str, int, float)):
            return None
        raw = str(raw_id)
        if raw in canonical_ids:
            return raw
        normalized_match = normalized.get(_normalize_id(raw))
        if normalized_match:
            return normalized_match
        node = extracted_nodes.get(raw)
        if not node:
            return None
        signature = (
            f"{_normalized_source_file(node.get('source_file'), root)}\0"
            f"{str(node.get('label') or '').strip()}"
        )
        return signatures.get(signature)

    return resolve


def _safe_number(value: object, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number >= 0 else default


def refresh_code_extraction(
    root: Path,
    extraction_path: Path,
    detection_path: Path,
    output_dir: Path,
) -> None:
    """Refresh deterministic code evidence while reusing semantic document evidence."""
    semantic_path = output_dir / ".graphify_semantic.json"
    saved_semantic_path = output_dir / "semantic.evidence.json"
    if semantic_path.exists():
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    elif saved_semantic_path.exists():
        semantic = json.loads(saved_semantic_path.read_text(encoding="utf-8"))
    else:
        raise SystemExit("No saved semantic evidence; run /graphify once before refreshing.")

    detection = detect(root)
    detection_path.write_text(
        json.dumps(detection, ensure_ascii=False), encoding="utf-8"
    )
    code_files: list[Path] = []
    for file_name in detection.get("files", {}).get("code", []):
        path = Path(file_name)
        code_files.extend(collect_files(path) if path.is_dir() else [path])
    ast = (
        extract(code_files, cache_root=root)
        if code_files
        else {"nodes": [], "edges": []}
    )
    (output_dir / ".graphify_ast.json").write_text(
        json.dumps(ast, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    seen = {node["id"] for node in ast.get("nodes", [])}
    nodes = list(ast.get("nodes", []))
    for node in semantic.get("nodes", []):
        if node["id"] not in seen:
            nodes.append(node)
            seen.add(node["id"])
    merged = {
        "nodes": nodes,
        "edges": ast.get("edges", []) + semantic.get("edges", []),
        "hyperedges": semantic.get("hyperedges", []),
        "input_tokens": semantic.get("input_tokens", 0),
        "output_tokens": semantic.get("output_tokens", 0),
    }
    extraction_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    saved_semantic_path.write_text(
        json.dumps(semantic, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    corpus = detection.get("all_files") or detection["files"]
    manifest_files = _stamped_manifest_files(corpus, merged, root)
    scan_corpus = {file_name for files in corpus.values() for file_name in files}
    save_manifest(manifest_files, root=root, scan_corpus=scan_corpus)
    print(
        f"Code extraction refreshed: {len(ast.get('nodes', []))} nodes · "
        f"{len(ast.get('edges', []))} edges; reused "
        f"{len(semantic.get('nodes', []))} semantic nodes."
    )


def build_evidence_multigraph(
    raw_extraction: dict[str, Any], root: Path
) -> tuple[nx.MultiGraph, dict[str, int]]:
    """Canonicalize with Graphify, then restore distinct accepted relationships."""
    extraction = copy.deepcopy(raw_extraction)
    canonical = build_from_json(extraction, directed=True, root=root)
    resolve = _endpoint_mapper(extraction, canonical, root)

    graph = nx.MultiGraph()
    graph.add_nodes_from((node_id, dict(data)) for node_id, data in canonical.nodes(data=True))
    graph.graph.update(canonical.graph)

    accepted_pairs = set(canonical.edges())
    seen: set[str] = set()
    stats = {
        "raw_edges": len(extraction.get("edges", [])),
        "kept_edges": 0,
        "exact_duplicates_removed": 0,
        "unresolved_or_rejected_edges": 0,
    }

    for edge in extraction.get("edges", []):
        if not isinstance(edge, dict):
            stats["unresolved_or_rejected_edges"] += 1
            continue
        source = resolve(edge.get("source", edge.get("from")))
        target = resolve(edge.get("target", edge.get("to")))
        if not source or not target or (source, target) not in accepted_pairs:
            stats["unresolved_or_rejected_edges"] += 1
            continue

        attrs = {
            key: value
            for key, value in edge.items()
            if key
            not in {
                "source",
                "target",
                "from",
                "to",
                "target_file",
                "local_alias",
            }
        }
        attrs["weight"] = _safe_number(attrs.get("weight"), 1.0)
        attrs["confidence_score"] = _safe_number(
            attrs.get("confidence_score"),
            1.0 if attrs.get("confidence") == "EXTRACTED" else 0.5,
        )
        attrs["source_file"] = _normalized_source_file(attrs.get("source_file"), root)
        if not attrs["source_file"]:
            attrs["source_file"] = (
                canonical.nodes[source].get("source_file")
                or canonical.nodes[target].get("source_file")
                or ""
            )
        attrs["_src"] = source
        attrs["_tgt"] = target

        dedupe_payload = {
            "source": source,
            "target": target,
            **{key: value for key, value in attrs.items() if key not in {"_src", "_tgt"}},
        }
        dedupe_key = json.dumps(
            dedupe_payload, ensure_ascii=False, sort_keys=True, default=str
        )
        if dedupe_key in seen:
            stats["exact_duplicates_removed"] += 1
            continue
        seen.add(dedupe_key)
        graph.add_edge(source, target, **attrs)

    stats["kept_edges"] = graph.number_of_edges()
    return graph, stats


def _words(value: str) -> set[str]:
    return {word for word in WORD_RE.findall(value.lower()) if word not in STOP_WORDS}


def _find_document_anchor(label: str, file_type: str, lines: list[str]) -> tuple[str, str] | None:
    label_words = _words(label)
    normalized_label = " ".join(WORD_RE.findall(label.lower()))
    if normalized_label:
        for index, line in enumerate(lines, start=1):
            normalized_line = " ".join(WORD_RE.findall(line.lower()))
            if normalized_label in normalized_line:
                return f"L{index}", "exact_text"

    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            headings.append((index, match.group(1)))
    if file_type == "document" and headings:
        return f"L{headings[0][0]}", "document_heading"

    if label_words and headings:
        scored = []
        for index, heading in headings:
            heading_words = _words(heading)
            overlap = len(label_words & heading_words)
            coverage = overlap / len(label_words)
            scored.append((coverage, overlap, -index, index))
        coverage, overlap, _neg_index, index = max(scored)
        if overlap >= 2 or coverage >= 0.6:
            return f"L{index}", "heading_match"

    if file_type == "document" and lines:
        return "L1", "document_start"
    return None


def add_document_source_anchors(graph: nx.MultiGraph, root: Path) -> dict[str, int]:
    cache: dict[str, list[str]] = {}
    node_anchors = 0
    edge_anchors = 0

    for _node_id, data in graph.nodes(data=True):
        source_file = str(data.get("source_file") or "")
        if data.get("source_location") or Path(source_file).suffix.lower() not in DOC_SUFFIXES:
            continue
        if source_file not in cache:
            path = root / source_file
            try:
                cache[source_file] = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                cache[source_file] = []
        anchor = _find_document_anchor(
            str(data.get("label") or ""),
            str(data.get("file_type") or ""),
            cache[source_file],
        )
        if anchor:
            data["source_location"], data["source_location_method"] = anchor
            node_anchors += 1

    for source, target, _key, data in graph.edges(keys=True, data=True):
        if data.get("source_location"):
            continue
        source_file = str(data.get("source_file") or "")
        for endpoint in (source, target):
            endpoint_data = graph.nodes[endpoint]
            if (
                endpoint_data.get("source_file") == source_file
                and endpoint_data.get("source_location")
            ):
                data["source_location"] = endpoint_data["source_location"]
                data["source_location_method"] = "endpoint_anchor"
                edge_anchors += 1
                break

    return {"node_source_anchors_added": node_anchors, "edge_source_anchors_added": edge_anchors}


def build_navigation_graph(full_graph: nx.MultiGraph) -> tuple[nx.MultiGraph, dict[str, int]]:
    semantic_bridge_nodes: set[str] = set()
    for source, target, data in full_graph.edges(data=True):
        if (
            data.get("relation") == "semantically_similar_to"
            and float(data.get("confidence_score") or 0) >= 0.75
        ):
            semantic_bridge_nodes.update((source, target))

    removed_rationale = {
        node_id
        for node_id, data in full_graph.nodes(data=True)
        if data.get("file_type") == "rationale" and node_id not in semantic_bridge_nodes
    }
    removed_unresolved = {
        node_id
        for node_id, data in full_graph.nodes(data=True)
        if not data.get("source_file")
    }
    removed = removed_rationale | removed_unresolved
    keep = [node_id for node_id in full_graph if node_id not in removed]
    navigation = full_graph.subgraph(keep).copy()

    weak_inferred_uses: list[tuple[str, str, int]] = []
    redundant_weak_inferred_uses = 0
    for source, target, key, data in navigation.edges(keys=True, data=True):
        if not (
            data.get("confidence") == "INFERRED"
            and data.get("relation") == "uses"
            and float(data.get("confidence_score") or 0)
            < NAVIGATION_INFERRED_USES_MIN_CONFIDENCE
        ):
            continue
        peers = navigation.get_edge_data(source, target, default={})
        if any(
            peer_key != key and peer.get("confidence") == "EXTRACTED"
            for peer_key, peer in peers.items()
        ):
            redundant_weak_inferred_uses += 1
        weak_inferred_uses.append((source, target, key))
    navigation.remove_edges_from(weak_inferred_uses)

    hyperedges = []
    for hyperedge in full_graph.graph.get("hyperedges", []):
        members = [node_id for node_id in hyperedge.get("nodes", []) if node_id in navigation]
        if members:
            hyperedges.append({**hyperedge, "nodes": members})
    navigation.graph["hyperedges"] = hyperedges
    navigation.graph["refinement"] = {
        "purpose": "source-backed navigation graph",
        "parallel_relations_preserved": True,
        "edge_direction_stored_in_endpoints": True,
        "filtered_node_types": ["rationale", "unresolved_without_source_file"],
        "filtered_edge_types": ["weak_inferred_uses"],
        "semantic_bridge_rationale_retained": True,
    }
    return navigation, {
        "rationale_nodes_filtered": len(removed_rationale),
        "semantic_bridge_rationale_nodes_kept": sum(
            full_graph.nodes[node_id].get("file_type") == "rationale"
            for node_id in semantic_bridge_nodes
            if node_id in full_graph
        ),
        "unresolved_nodes_filtered": len(removed_unresolved - removed_rationale),
        "weak_inferred_uses_filtered": len(weak_inferred_uses),
        "redundant_weak_inferred_uses_filtered": redundant_weak_inferred_uses,
    }


def _analysis_graph(multigraph: nx.MultiGraph) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from((node_id, dict(data)) for node_id, data in multigraph.nodes(data=True))
    graph.graph.update(multigraph.graph)

    def rank(data: dict[str, Any]) -> tuple[int, float, int]:
        confidence = {"EXTRACTED": 2, "INFERRED": 1, "AMBIGUOUS": 0}.get(
            str(data.get("confidence")), 0
        )
        specific = int(str(data.get("relation")) not in GENERIC_RELATIONS)
        return specific, float(data.get("confidence_score") or 0), confidence

    for source, target, data in multigraph.edges(data=True):
        if not graph.has_edge(source, target) or rank(data) > rank(graph[source][target]):
            graph.add_edge(source, target, **dict(data))
    return graph


def _surprise_graph(analysis: nx.Graph, min_inferred_confidence: float) -> nx.Graph:
    graph = analysis.copy()
    to_remove = []
    for source, target, data in graph.edges(data=True):
        source_files = {
            str(graph.nodes[source].get("source_file") or ""),
            str(graph.nodes[target].get("source_file") or ""),
        }
        is_test_reference = any(path.startswith("tests/") for path in source_files)
        is_weak_inference = (
            data.get("confidence") == "INFERRED"
            and float(data.get("confidence_score") or 0) < min_inferred_confidence
        )
        if is_test_reference or is_weak_inference:
            to_remove.append((source, target))
    graph.remove_edges_from(to_remove)
    return graph


def _load_old_state(output_dir: Path) -> tuple[dict[str, int], str | None]:
    old_communities: dict[str, int] = {}
    built_at_commit: str | None = None
    graph_path = output_dir / "graph.json"
    if graph_path.exists():
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        built_at_commit = data.get("built_at_commit")
        for node in data.get("nodes", []):
            if node.get("community") is not None:
                old_communities[str(node["id"])] = int(node["community"])
    return old_communities, built_at_commit


def _plain_label(value: str) -> str:
    value = re.sub(r"\.(?:csv|json|md|py|toml|txt)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\(.*?\)", "", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    words = [word for word in re.split(r"[^A-Za-z0-9]+", value) if word]
    acronyms = {"api", "csv", "id", "json", "mc", "pomdp", "rl", "ui", "vae"}
    label = " ".join(
        word.upper() if word.lower() in acronyms else word.title() for word in words[:5]
    )
    return label.replace("2 D", "2D")


def _source_label(source_file: str) -> str:
    stem = Path(source_file).stem
    if stem == "refine_graph":
        return "Graph Refinement"
    is_test = stem.startswith("test_")
    if is_test:
        stem = stem.removeprefix("test_")
    label = _plain_label(stem)
    return f"{label} Tests" if is_test else label


def _community_labels(
    graph: nx.Graph,
    communities: dict[int, list[str]],
) -> dict[int, str]:
    hub_labels = label_communities_by_hub(graph, communities)
    labels: dict[int, str] = {}
    for community_id, members in communities.items():
        source_counts = Counter(
            str(graph.nodes[node_id].get("source_file") or "")
            for node_id in members
            if graph.nodes[node_id].get("source_file")
        )
        source_file, count = source_counts.most_common(1)[0] if source_counts else ("", 0)
        if source_file and count >= 3 and count / max(len(members), 1) >= 0.2:
            labels[community_id] = _source_label(source_file)
        else:
            labels[community_id] = _plain_label(hub_labels[community_id])

    duplicates = Counter(labels.values())
    for community_id, label in list(labels.items()):
        if duplicates[label] > 1:
            labels[community_id] = _plain_label(hub_labels[community_id])
    return labels


def _mark_unknown_cost(output_dir: Path) -> None:
    cost_path = output_dir / "cost.json"
    if not cost_path.exists():
        return
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    for run in cost.get("runs", []):
        if run.get("input_tokens") == 0 and run.get("output_tokens") == 0:
            run["usage_available"] = False
            run["usage_note"] = "Agent token usage metadata was unavailable; zero is not a measured cost."
    cost["measurement_status"] = "unknown"
    cost["measurement_note"] = "Token usage was not exposed by the semantic extraction agent."
    cost_path.write_text(json.dumps(cost, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _refinement_section(stats: dict[str, Any], min_confidence: float) -> str:
    return "\n".join(
        [
            "## Graph Refinement",
            f"- Default navigation graph: {stats['navigation_nodes']} source-backed nodes · "
            f"{stats['navigation_edges']} distinct relationships · "
            f"{stats['navigation_pairs']} connected node pairs.",
            f"- Full evidence graph: {stats['full_nodes']} nodes · {stats['full_edges']} "
            "distinct relationships (`graph.full.json`).",
            f"- Filtered from the default view: {stats['rationale_nodes_filtered']} rationale nodes · "
            f"{stats['unresolved_nodes_filtered']} unresolved/type-placeholder nodes.",
            f"- Retained {stats['semantic_bridge_rationale_nodes_kept']} rationale nodes that support "
            "high-confidence semantic bridges.",
            f"- Moved {stats['weak_inferred_uses_filtered']} low-confidence inferred `uses` edges "
            "out of navigation and into the full evidence graph only; "
            f"{stats['redundant_weak_inferred_uses_filtered']} duplicated extracted endpoint pairs.",
            f"- Preserved {stats['parallel_relationships']} parallel relationships beyond one-edge-per-pair "
            "storage; endpoint order retains source-to-target direction.",
            f"- Added {stats['node_source_anchors_added']} document-node and "
            f"{stats['edge_source_anchors_added']} document-edge source anchors.",
            f"- Surprising inferred connections require confidence >= {min_confidence:.2f}.",
            "- Routine test references are excluded from surprising-connection ranking.",
            "- Token cost: unknown because semantic-agent usage metadata was unavailable.",
        ]
    )


def refine(
    root: Path,
    extraction_path: Path,
    detection_path: Path,
    output_dir: Path,
    hub_percentile: float,
    surprise_min_confidence: float,
) -> dict[str, Any]:
    raw_extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    detection = json.loads(detection_path.read_text(encoding="utf-8"))
    semantic_path = output_dir / ".graphify_semantic.json"
    if semantic_path.exists():
        shutil.copy2(semantic_path, output_dir / "semantic.evidence.json")
    old_communities, built_at_commit = _load_old_state(output_dir)

    graph_path = output_dir / "graph.json"
    report_path = output_dir / "GRAPH_REPORT.md"
    if graph_path.exists() and not (output_dir / "graph.pre-refinement.json").exists():
        shutil.copy2(graph_path, output_dir / "graph.pre-refinement.json")
    if report_path.exists() and not (output_dir / "GRAPH_REPORT.pre-refinement.md").exists():
        shutil.copy2(report_path, output_dir / "GRAPH_REPORT.pre-refinement.md")

    full_graph, edge_stats = build_evidence_multigraph(raw_extraction, root)
    anchor_stats = add_document_source_anchors(full_graph, root)
    navigation, filter_stats = build_navigation_graph(full_graph)
    analysis = _analysis_graph(navigation)

    communities = cluster(analysis, exclude_hubs_percentile=hub_percentile)
    if old_communities:
        communities = remap_communities_to_previous(communities, old_communities)
    cohesion = score_all(analysis, communities)
    labels = _community_labels(analysis, communities)

    surprises = surprising_connections(
        _surprise_graph(analysis, surprise_min_confidence), communities, top_n=5
    )
    gods = god_nodes(analysis)
    questions = suggest_questions(analysis, communities, labels)

    removed_nodes = [node_id for node_id in full_graph if node_id not in navigation]
    evidence_community = max(communities, default=-1) + 1
    full_communities = {**communities, evidence_community: removed_nodes}
    full_labels = {**labels, evidence_community: "Evidence Detail"}

    stats: dict[str, Any] = {
        **edge_stats,
        **anchor_stats,
        **filter_stats,
        "full_nodes": full_graph.number_of_nodes(),
        "full_edges": full_graph.number_of_edges(),
        "navigation_nodes": navigation.number_of_nodes(),
        "navigation_edges": navigation.number_of_edges(),
        "navigation_pairs": analysis.number_of_edges(),
        "parallel_relationships": navigation.number_of_edges() - analysis.number_of_edges(),
        "communities": len(communities),
        "cohesion_mean": round(sum(cohesion.values()) / len(cohesion), 3) if cohesion else 0.0,
        "cohesion_below_0_2": sum(value < 0.2 for value in cohesion.values()),
        "surprise_min_confidence": surprise_min_confidence,
        "hub_exclusion_percentile": hub_percentile,
    }
    navigation.graph["refinement"].update(stats)
    full_graph.graph["refinement"] = {**stats, "purpose": "full evidence graph"}

    to_json(
        full_graph,
        full_communities,
        str(output_dir / "graph.full.json"),
        force=True,
        built_at_commit=built_at_commit,
        community_labels=full_labels,
    )
    to_json(
        navigation,
        communities,
        str(graph_path),
        force=True,
        built_at_commit=built_at_commit,
        community_labels=labels,
    )

    report = generate(
        navigation,
        communities,
        cohesion,
        labels,
        gods,
        surprises,
        detection,
        {"input": 0, "output": 0},
        str(root),
        suggested_questions=questions,
        built_at_commit=built_at_commit,
    )
    report = report.replace(
        "- Token cost: 0 input · 0 output",
        "- Token cost: unknown (semantic-agent usage metadata unavailable)",
    )
    report = report.replace(
        "- Run `graphify update .` after code changes (no API cost).",
        "- Run `tools/update_graph.sh` after code changes; run `/graphify --update` "
        "after document changes.",
    )
    marker = "\n## Community Hubs (Navigation)"
    report = report.replace(marker, "\n" + _refinement_section(stats, surprise_min_confidence) + marker)
    report_path.write_text(report, encoding="utf-8")

    (output_dir / ".graphify_labels.json").write_text(
        json.dumps({str(key): value for key, value in labels.items()}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / ".graphify_analysis.json").write_text(
        json.dumps(
            {
                "communities": {str(key): value for key, value in communities.items()},
                "cohesion": {str(key): value for key, value in cohesion.items()},
                "gods": gods,
                "surprises": surprises,
                "questions": questions,
                "refinement": stats,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "GRAPH_REFINEMENT.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _mark_unknown_cost(output_dir)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--extraction", type=Path, default=Path("graphify-out/.graphify_extract.json")
    )
    parser.add_argument(
        "--detection", type=Path, default=Path("graphify-out/.graphify_detect.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("graphify-out"))
    parser.add_argument("--hub-percentile", type=float, default=99.0)
    parser.add_argument("--surprise-min-confidence", type=float, default=0.75)
    parser.add_argument(
        "--refresh-code",
        action="store_true",
        help="refresh deterministic code extraction and reuse saved semantic evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.refresh_code:
        refresh_code_extraction(
            root=root,
            extraction_path=args.extraction,
            detection_path=args.detection,
            output_dir=args.output_dir,
        )
    stats = refine(
        root=root,
        extraction_path=args.extraction,
        detection_path=args.detection,
        output_dir=args.output_dir,
        hub_percentile=args.hub_percentile,
        surprise_min_confidence=args.surprise_min_confidence,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
