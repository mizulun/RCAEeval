"""Diagnostic payload and additive timing schema for MERIT."""

import sys


CANONICAL_STAGE_NAMES = (
    "trace",
    "preprocess",
    "metric_evidence",
    "candidates",
    "banzhaf",
    "ranking",
)


def canonical_stage_timing(core_timing, trace_timing):
    """Return non-overlapping top-level stages; details stay in trace timing."""
    merged = {}
    if trace_timing:
        merged["trace"] = float(sum(trace_timing.values()))
    for name in CANONICAL_STAGE_NAMES[1:]:
        if name in (core_timing or {}):
            merged[name] = float(core_timing[name])
    return merged


def build_result(ranks, cols, config, kwargs, state, stage_timing):
    result = {"ranks": ranks, "node_names": cols}
    if kwargs.get("return_diagnostics", False):
        game_candidates = state["game_candidates"]
        base_game_candidates = state["base_game_candidates"]
        game_services = state["game_services"]
        local_contributions = state["local_contributions"]
        edge_removal = state["edge_removal_utility_drop"]
        rank_mode = config.rank_mode
        diagnostics = {
            "rank_mode": rank_mode,
            "configuration": {
                "banzhaf_mode": config.banzhaf_mode,
                "use_early_full_fusion": config.use_early_full_fusion,
                "use_metric_type_dominance": config.use_metric_type_dominance,
                "use_metrics": config.use_metrics,
                "use_trace": config.use_trace,
                "use_trace_topology": config.use_trace_topology,
                "trace_signal": config.trace_signal,
                "trace_control": config.trace_control,
                "trace_topology_max_lag": config.trace_topology_max_lag,
                "trace_dropout_fraction": config.trace_dropout_fraction,
                "trace_noise_fraction": config.trace_noise_fraction,
                "trace_edge_noise_fraction": config.trace_edge_noise_fraction,
                "use_victim_demotion": config.use_victim_demotion,
                "protect_coverage_parents": config.protect_coverage_parents,
                "use_coverage": bool(kwargs.get("use_coverage", True)),
            },
            "candidate_pool": list(state["cand_list"]),
            "game_candidates": list(game_candidates),
            "game_services": list(game_services),
            "candidate_pool_k": len(state["cand_list"]),
            "base_game_candidates": list(base_game_candidates),
            "base_candidate_k": len(base_game_candidates),
            "coverage_rescued_candidates": sorted(
                set(game_candidates) - set(base_game_candidates)
            ),
            "coverage_rescued_k": len(
                set(game_candidates) - set(base_game_candidates)
            ),
            "final_candidate_k": len(game_candidates),
            "coverage_strength_matrix": state["prob_matrix"].tolist(),
            "causal_probability_matrix": state["prob_matrix"].tolist(),
            "metric_evidence": dict(state["metric_ev"]),
            "metric_service_evidence": dict(state["svc_anom_dict"]),
            "direct_metric_service_evidence": dict(state["direct_metric_evidence"]),
            "direct_trace_service_evidence": dict(state["direct_trace_evidence"]),
            "direct_service_scores": dict(state["direct_service_scores"]),
            "service_ranks": list(state["service_ranks"]),
            "service_banzhaf": dict(state["svc_banzhaf"]),
            "player_target_contributions": ({
                candidate: {
                    target: float(local_contributions[i, j])
                    for j, target in enumerate(game_services)
                }
                for i, candidate in enumerate(game_candidates)
            } if rank_mode == "banzhaf" else {}),
            "edge_removal_utility_drop": ({
                candidate: {
                    target: float(edge_removal[i, j])
                    for j, target in enumerate(game_services)
                }
                for i, candidate in enumerate(game_candidates)
            } if rank_mode == "banzhaf" else {}),
            "trace_topology_edges": list(state["trace_topology_edges"]),
            "trace_edge_count": len(state["trace_topology_edges"]),
            "trace_candidate_added_count": state["trace_added_candidates"],
            "trace_service_evidence": dict(state["trace_service_evidence"]),
            "trace_latency_evidence": dict(state["trace_latency_evidence"]),
            "trace_error_evidence": dict(state["trace_error_evidence"]),
            "trace_stage_timing": dict(state["trace_stage_timing"]),
            "causal_edges": list(state["causal_edges"]),
            "causal_edge_count": len(state["causal_edges"]),
            "graph_provenance": dict(kwargs.get("graph_provenance", {}) or {}),
            "call_graph_edges": sorted([
                [caller, callee]
                for caller, callees in (kwargs.get("call_graph", {}) or {}).items()
                for callee in callees
            ]),
            "input_boundary": dict(kwargs.get("input_boundary", {}) or {}),
        }
        result["diagnostics"] = diagnostics
        result.update(diagnostics)
    if kwargs.get("return_stage_timing", False):
        result["stage_timing"] = canonical_stage_timing(
            stage_timing, state["trace_stage_timing"]
        )
    return result


def finish_result(result, dataset, profile):
    """Attach one additive timing view and optional legacy stderr profiling."""
    diagnostics = result.get("diagnostics")
    trace_timing = diagnostics.get("trace_stage_timing", {}) if diagnostics else {}
    timing = canonical_stage_timing(result.get("stage_timing", {}), trace_timing)
    if diagnostics is not None:
        diagnostics["stage_timing"] = dict(timing)
    if "stage_timing" in result:
        result["stage_timing"] = dict(timing)
    if profile and timing:
        total = sum(timing.values())
        label = str(dataset) or "?"
        parts = [f"{key}={value:.3f}s" for key, value in timing.items()]
        parts.append(f"total={total:.3f}s")
        print(f"[PROFILE] {label}  " + "  ".join(parts), file=sys.stderr)
    return result
