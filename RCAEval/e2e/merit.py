"""MERIT orchestration and compatibility entry points.

Metric evidence, trace ingestion, coverage-game math, diagnostics, experiment
switches, and defaults live in dedicated sibling modules.  Private aliases are
retained because frozen research scripts monkeypatch these names at runtime.
"""

from time import perf_counter
import warnings

import numpy as np

from RCAEval.e2e.merit_config import _GabanzhafConfig
from RCAEval.e2e import merit_ablation as _ablation
from RCAEval.e2e import merit_diagnostics as _diagnostics
from RCAEval.e2e import merit_evidence as _evidence
from RCAEval.e2e import merit_game as _game
from RCAEval.e2e import merit_trace as _trace

warnings.filterwarnings("ignore")


# Compatibility aliases used by tests and frozen research interventions.
_parse_svc = _evidence._parse_svc
_norm01_arr = _evidence._norm01_arr
_norm_dict = _evidence._norm_dict
_norm_weights = _evidence._norm_weights
_metric_evidence = _evidence._metric_evidence
_service_evidence = _evidence._service_evidence
_service_metric_family_scores = _evidence._service_metric_family_scores
_service_latency_scores = _evidence._service_latency_scores
_service_latency_family_scores = _evidence._service_latency_family_scores
_split_raw_windows = _evidence._split_raw_windows
_preprocess_windows = _evidence._preprocess_windows
_finite_nonnegative = _evidence._finite_nonnegative
_rank_metrics = _evidence._rank_metrics
_service_ranks_from_metric_ranks = _evidence._service_ranks_from_metric_ranks

_noisy_or_coverage = _game._noisy_or_coverage
_probabilistic_banzhaf = _game._probabilistic_banzhaf
_probabilistic_banzhaf_contributions = _game._probabilistic_banzhaf_contributions
_edge_removal_faithfulness = _game._edge_removal_faithfulness
_coverage_parents = _game._coverage_parents
_apply_coverage_edges = _game._apply_coverage_edges
_apply_trace_topology_edges = _game._apply_trace_topology_edges
_causal_edges = _game._causal_edges
_victim_factors = _game._victim_factors
_select_exact_candidates = _game._select_exact_candidates
_causal_explanation_matrix = _game._causal_explanation_matrix
_filter_trace_topology_edges = _game._filter_trace_topology_edges
_augment_coverage_candidates = _game._augment_coverage_candidates

_TraceTopologyFeatures = _trace._TraceTopologyFeatures
_CALL_GRAPH_CACHE = _trace._CALL_GRAPH_CACHE
_match_service = _trace._match_service
_build_call_graph_from_traces = _trace._build_call_graph_from_traces
_service_trace_series_from_frame = _trace._service_trace_series_from_frame
_norm_trace_arr = _trace._norm_trace_arr
_lagged_cosine_score = _trace._lagged_cosine_score
_trace_multilag_edges = _trace._trace_multilag_edges
_merge_trace_topology_edges = _trace._merge_trace_topology_edges
_trace_service_evidence = _trace._trace_service_evidence
_trace_topology_from_case_dir = _trace._trace_topology_from_case_dir


def _cusum_metric_evidence(normal_mat, anomal_mat, cols, config):
    return _ablation._cusum_metric_evidence(normal_mat, anomal_mat, cols, config)


def _bruteforce_probabilistic_banzhaf(prob_matrix, anomaly_weights):
    return _ablation._bruteforce_probabilistic_banzhaf(
        prob_matrix, anomaly_weights, _noisy_or_coverage=_noisy_or_coverage
    )


def _banzhaf_power(prob_matrix, anomaly_weights, config):
    return _ablation._banzhaf_power(
        prob_matrix,
        anomaly_weights,
        config,
        _probabilistic_banzhaf=_probabilistic_banzhaf,
        _bruteforce_probabilistic_banzhaf=_bruteforce_probabilistic_banzhaf,
    )


def _build_union_call_graph(
    dataset_root, exposure_manifest, service_names, boundary_version,
    dropout_fraction=0.0,
):
    return _ablation._build_union_call_graph(
        dataset_root,
        exposure_manifest,
        service_names,
        boundary_version,
        dropout_fraction,
        _CALL_GRAPH_CACHE=_CALL_GRAPH_CACHE,
        _build_call_graph_from_traces=_build_call_graph_from_traces,
    )


def _time_shuffle_trace_series(trace_series, salt):
    return _ablation._time_shuffle_trace_series(trace_series, salt)


def _deterministic_row_dropout(frame, fraction, salt):
    return _ablation._deterministic_row_dropout(frame, fraction, salt)


def _add_trace_series_noise(trace_series, fraction, salt):
    return _ablation._add_trace_series_noise(
        trace_series, fraction, salt, _finite_nonnegative=_finite_nonnegative
    )


def _degree_matched_random_edges(edges):
    return _ablation._degree_matched_random_edges(edges)


def _inject_random_trace_edges(edges, services, fraction):
    return _ablation._inject_random_trace_edges(
        edges, services, fraction, _finite_nonnegative=_finite_nonnegative
    )


def _direct_service_score_sum(metric_service_evidence, trace_service_evidence):
    return _ablation._direct_service_score_sum(
        metric_service_evidence,
        trace_service_evidence,
        _finite_nonnegative=_finite_nonnegative,
        _norm_dict=_norm_dict,
    )


def _rank_evidence_metrics(
    all_cols, metric_ev, svc_anom_dict, rank_mode, trace_service_evidence=None,
):
    return _ablation._rank_evidence_metrics(
        all_cols,
        metric_ev,
        svc_anom_dict,
        rank_mode,
        trace_service_evidence,
        _norm_dict=_norm_dict,
        _finite_nonnegative=_finite_nonnegative,
        _parse_svc=_parse_svc,
        _direct_service_score_sum=_direct_service_score_sum,
    )


def _merit_core(data, inject_time, dataset, anomalies, kwargs):
    """Run the shared evidence -> coverage game -> attribution pipeline."""
    config = _GabanzhafConfig.from_kwargs(kwargs)
    if not config.use_trace:
        # Keep direct private-core research calls consistent with the public
        # input gate: trace data must not leak into diagnostics or ranking.
        kwargs = _ablation.disable_trace_inputs(kwargs)
    trace_topology_edges = (
        list(kwargs.get("trace_topology_edges", []))
        if config.use_trace and config.use_trace_topology else []
    )
    trace_service_evidence = (
        kwargs.get("trace_service_evidence", {}) or {} if config.use_trace else {}
    )
    trace_latency_evidence = (
        kwargs.get("trace_latency_evidence", {}) or {} if config.use_trace else {}
    )
    trace_error_evidence = (
        kwargs.get("trace_error_evidence", {}) or {} if config.use_trace else {}
    )
    trace_stage_timing = (
        kwargs.get("trace_stage_timing", {}) or {} if config.use_trace else {}
    )
    return_stage_timing = bool(kwargs.get("return_stage_timing", False))
    stage_timing = {}
    last_tick = perf_counter()

    def mark_stage(name):
        nonlocal last_tick
        if return_stage_timing:
            now = perf_counter()
            stage_timing[name] = float(now - last_tick)
            last_tick = now

    normal_raw, anomal_raw = _split_raw_windows(data, inject_time, anomalies)
    if config.use_metrics:
        normal_df, anomal_df, no_time = _preprocess_windows(
            normal_raw, anomal_raw, dataset, config
        )
        cols = list(no_time.columns)
        normal_mat = normal_df.drop(columns=["time"], errors="ignore").values.astype(float)
        anomal_mat = anomal_df.drop(columns=["time"], errors="ignore").values.astype(float)
    else:
        cols, normal_mat, anomal_mat = _ablation.metric_free_windows(
            data, normal_raw, anomal_raw
        )
    mark_stage("preprocess")

    if config.use_metrics:
        metric_ev = _metric_evidence(normal_mat, anomal_mat, cols, config=config)
        svc_anom_dict = _service_evidence(
            cols,
            metric_ev,
            use_metric_type_dominance=config.use_metric_type_dominance,
            config=config,
        )
    else:
        metric_ev, svc_anom_dict = _ablation.metric_free_evidence(
            cols,
            trace_service_evidence,
            parse_service=_parse_svc,
            nonnegative=_finite_nonnegative,
        )
    svc_names = sorted(svc_anom_dict)
    mark_stage("metric_evidence")
    _ablation.validate_rank_mode(config.rank_mode)

    if config.rank_mode == "banzhaf":
        svc_ranked = sorted(
            svc_names, key=lambda svc: svc_anom_dict.get(svc, 0.0), reverse=True
        )
        cand_list = sorted(set(svc_ranked[: config.top_k_svcs]))
        mark_stage("candidates")
        svc_latency = _service_latency_scores(cols, metric_ev)
        latency_family = _service_latency_family_scores(cols, metric_ev)
        if config.use_metrics:
            trace_topology_edges = _filter_trace_topology_edges(
                trace_topology_edges, latency_family, config.coverage_lat_floor
            )
        if config.use_metrics and bool(kwargs.get("use_coverage", True)):
            call_graph = kwargs.get("call_graph", {}) or {} if config.use_trace else {}
            coverage_parents = _coverage_parents(
                call_graph, svc_anom_dict, svc_latency, config
            )
        else:
            coverage_parents = {}
        causal_edges = _causal_edges(
            coverage_parents, trace_topology_edges, config.coverage_conf
        )
        base_game_candidates = _select_exact_candidates(
            cand_list, svc_anom_dict, config
        )
        game_candidates = _augment_coverage_candidates(
            base_game_candidates, coverage_parents, svc_anom_dict
        )
        trace_added_candidates = 0
        game_services = sorted(svc_anom_dict)
        prob_matrix = _causal_explanation_matrix(game_candidates, game_services)
        prob_matrix = _apply_coverage_edges(
            prob_matrix,
            game_candidates,
            game_services,
            coverage_parents,
            config.coverage_conf,
        )
        prob_matrix = _apply_trace_topology_edges(
            prob_matrix, game_candidates, game_services, trace_topology_edges
        )
        if config.use_victim_demotion:
            victim_factor = _victim_factors(
                causal_edges,
                svc_anom_dict,
                config,
                protected=(
                    set(coverage_parents) if config.protect_coverage_parents else set()
                ),
            )
            service_columns = {svc: index for index, svc in enumerate(game_services)}
            for index, candidate in enumerate(game_candidates):
                factor = victim_factor.get(candidate)
                if factor is not None and candidate in service_columns:
                    prob_matrix[index, service_columns[candidate]] *= factor
        anomaly_weights = np.asarray(
            [svc_anom_dict.get(svc, 0.0) for svc in game_services], dtype=float
        )
    else:
        cand_list, coverage_parents, causal_edges = [], {}, []
        game_candidates, base_game_candidates, trace_added_candidates = [], [], 0
        game_services = []
        prob_matrix = np.zeros((0, 0), dtype=float)
        anomaly_weights = np.zeros(0, dtype=float)

    svc_banzhaf = {}
    local_contributions = np.zeros_like(prob_matrix)
    edge_removal = np.zeros_like(prob_matrix)
    if config.rank_mode == "banzhaf":
        values = _banzhaf_power(prob_matrix, anomaly_weights, config)
        svc_banzhaf = {
            game_candidates[index]: float(values[index])
            for index in range(len(game_candidates))
        }
        if kwargs.get("return_diagnostics", False):
            local_contributions = _ablation.attribution_contributions(
                prob_matrix,
                anomaly_weights,
                config.banzhaf_mode,
                closed_form=_probabilistic_banzhaf_contributions,
            )
            edge_removal = _edge_removal_faithfulness(
                prob_matrix, anomaly_weights
            )
        mark_stage("banzhaf")

    if config.rank_mode == "banzhaf":
        ranks = _rank_metrics(
            cols, svc_banzhaf, metric_ev if config.use_metrics else None
        )
    else:
        ranks = _rank_evidence_metrics(
            cols,
            metric_ev,
            svc_anom_dict,
            config.rank_mode,
            trace_service_evidence=trace_service_evidence,
        )
    mark_stage("ranking")

    direct_metric, direct_trace, direct_scores = {}, {}, {}
    if config.rank_mode == "service_evidence_trace":
        direct_metric, direct_trace, direct_scores = _direct_service_score_sum(
            svc_anom_dict, trace_service_evidence
        )
    state = {
        "cand_list": cand_list,
        "game_candidates": game_candidates,
        "base_game_candidates": base_game_candidates,
        "game_services": game_services,
        "prob_matrix": prob_matrix,
        "metric_ev": metric_ev,
        "svc_anom_dict": svc_anom_dict,
        "svc_banzhaf": svc_banzhaf,
        "service_ranks": _service_ranks_from_metric_ranks(ranks),
        "local_contributions": local_contributions,
        "edge_removal_utility_drop": edge_removal,
        "trace_topology_edges": trace_topology_edges,
        "trace_added_candidates": trace_added_candidates,
        "trace_service_evidence": trace_service_evidence,
        "trace_latency_evidence": trace_latency_evidence,
        "trace_error_evidence": trace_error_evidence,
        "trace_stage_timing": trace_stage_timing,
        "causal_edges": causal_edges,
        "direct_metric_evidence": direct_metric,
        "direct_trace_evidence": direct_trace,
        "direct_service_scores": direct_scores,
    }
    return _diagnostics.build_result(
        ranks, cols, config, kwargs, state, stage_timing
    )


def _prepare_case_inputs(data, inject_time, kwargs, runtime_kwargs, disable_trace):
    """Resolve the modality gate before any graph or trace file access."""
    args_ns = kwargs.get("args")
    config = _GabanzhafConfig.from_kwargs(runtime_kwargs)
    if disable_trace or not config.use_trace:
        input_boundary = (
            getattr(args_ns, "input_boundary", None) if args_ns else None
        )
        runtime_kwargs = _ablation.disable_trace_inputs(runtime_kwargs)
        runtime_kwargs["input_boundary"] = input_boundary or {}
        return runtime_kwargs
    if not args_ns or not hasattr(args_ns, "data_path"):
        return runtime_kwargs
    graph_source = _ablation.resolve_graph_source(args_ns)
    return _trace.prepare_case_inputs(
        data,
        inject_time,
        args_ns,
        runtime_kwargs,
        config,
        graph_source,
        build_graph=_build_call_graph_from_traces,
        build_trace_features=_trace_topology_from_case_dir,
    )


def merit(data, inject_time=None, dataset=None, num_loop=None,
          sli=None, anomalies=None, **kwargs):
    """Rank root-cause metrics with MERIT's directed coverage game."""
    runtime_kwargs, profile, disable_trace = _ablation.resolve_runtime_kwargs(kwargs)
    runtime_kwargs = _prepare_case_inputs(
        data, inject_time, kwargs, runtime_kwargs, disable_trace
    )
    if profile:
        runtime_kwargs["return_stage_timing"] = True
    result = _merit_core(data, inject_time, dataset, anomalies, runtime_kwargs)
    return _diagnostics.finish_result(result, dataset, profile)
