"""Tests for the pure-Banzhaf gabanzhaf pipeline.

After ablation showed the old metric-derived trace lag to be mostly redundant,
the pipeline now keeps metric evidence as anomaly weight and adds trace/topology
signals as causal edges plus a conservative abnormal-frontier bonus. These tests
cover the math core, configuration, diagnostics, and entry point.
"""
import importlib.util
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

# Load the module directly from its file so the tests do not drag in the
# torch-dependent RCAEval.e2e package __init__ (gabanzhaf itself needs none of it).
_spec = importlib.util.spec_from_file_location(
    "gabanzhaf_under_test",
    pathlib.Path(__file__).resolve().parent.parent / "RCAEval" / "e2e" / "gabanzhaf.py",
)
gabanzhaf_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gabanzhaf_mod)

_GabanzhafConfig = gabanzhaf_mod._GabanzhafConfig
_gabanzhaf_core = gabanzhaf_mod._gabanzhaf_core
_metric_evidence = gabanzhaf_mod._metric_evidence
_banzhaf_power = gabanzhaf_mod._banzhaf_power
_noisy_or_coverage = gabanzhaf_mod._noisy_or_coverage
_probabilistic_banzhaf = gabanzhaf_mod._probabilistic_banzhaf
_select_exact_candidates = gabanzhaf_mod._select_exact_candidates
_causal_explanation_matrix = gabanzhaf_mod._causal_explanation_matrix
_rank_metrics = gabanzhaf_mod._rank_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metric evidence: Winsorization and early-window sensitivity
# (_metric_evidence returns {col: score}; score blends 0.6*early + 0.4*full)
# ─────────────────────────────────────────────────────────────────────────────

def test_metric_evidence_winsorizes_near_zero_baseline():
    """Near-zero baseline metric must not produce unbounded rz scores."""
    rng = np.random.default_rng(0)
    n_rows, n_cols = 100, 3

    normal = np.zeros((n_rows, n_cols))
    normal[:, 0] = rng.uniform(0, 0.001, n_rows)   # near-zero baseline

    anomal = np.zeros((n_rows, n_cols))
    anomal[:, 0] = rng.uniform(100, 200, n_rows)    # large anomaly

    cols = ["svc_latency", "svc_cpu", "svc_mem"]
    ev = _metric_evidence(normal, anomal, cols)

    scores = [ev[c] for c in cols]
    assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of [0,1]: {scores}"
    assert ev["svc_latency"] == max(scores), "Anomalous column should win"


def test_metric_evidence_selective_winsorization():
    """A near-constant-baseline column (capped at rz=10) must not dominate a
    column whose real variation gives a higher uncapped rz."""
    n_rows = 100

    normal_0 = np.zeros(n_rows)        # exactly constant → std=0, IQR=0
    anomal_0 = np.full(n_rows, 50.0)   # rz would be 50 without cap

    rng = np.random.default_rng(0)
    normal_1 = rng.normal(100.0, 5.0, n_rows)  # std~5, IQR~6.7
    anomal_1 = np.full(n_rows, 200.0)           # |200-100|/6.7 ≈ 14.9

    normal = np.column_stack([normal_0, normal_1])
    anomal = np.column_stack([anomal_0, anomal_1])
    cols = ["near_zero_svc_lat", "normal_svc_lat"]

    ev = _metric_evidence(normal, anomal, cols)

    assert ev["near_zero_svc_lat"] <= ev["normal_svc_lat"] + 0.01


def test_metric_evidence_rewards_early_onset():
    """Score weights the early window 0.6, so an early-onset signal outscores a
    late-onset victim of equal magnitude."""
    normal = np.column_stack([np.zeros(40), np.zeros(40)])
    early_root = np.r_[np.full(8, 8.0), np.zeros(32)]
    slow_victim = np.r_[np.zeros(12), np.full(28, 4.0)]
    anomal = np.column_stack([early_root, slow_victim])

    ev = _metric_evidence(normal, anomal, ["root_latency", "victim_latency"])

    assert ev["root_latency"] > ev["victim_latency"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Banzhaf power index
# ─────────────────────────────────────────────────────────────────────────────

def test_gabanzhaf_config_centralizes_runtime_parameters():
    cfg = _GabanzhafConfig.from_kwargs({
        "top_k_svcs": "9",
        "dk_select_useful": True,
        "rank_mode": "service_evidence",
    })
    assert cfg.top_k_svcs == 9
    assert cfg.dk_select_useful is True
    assert cfg.rank_mode == "service_evidence"


def test_gabanzhaf_config_defaults_to_closed_form_banzhaf():
    assert _GabanzhafConfig.from_kwargs({}).banzhaf_mode == "closed_form"


def test_exact_candidate_k_invalid_values_fall_back_to_default():
    for value in ["nan", np.nan, np.inf, "-2", 0, "0.5", 0.5]:
        cfg = _GabanzhafConfig.from_kwargs({"exact_candidate_k": value})
        assert cfg.exact_candidate_k == _GabanzhafConfig.exact_candidate_k


def test_trace_topology_max_lag_allows_zero():
    cfg = _GabanzhafConfig.from_kwargs({"trace_topology_max_lag": 0})

    assert cfg.trace_topology_max_lag == 0


def test_banzhaf_power_uses_probabilistic_game_single_flow():

    prob = np.array([[1.0, 0.7], [0.0, 1.0]], dtype=float)
    weights = np.array([0.9, 0.5], dtype=float)

    values = _banzhaf_power(prob, weights, _GabanzhafConfig.from_kwargs({}))

    assert values.shape == (2,)
    assert values[0] > values[1]
    assert np.all(values >= 0.0)


def test_banzhaf_power_bruteforce_matches_closed_form():

    prob = np.array(
        [[1.0, 0.4, 0.2], [0.0, 0.7, 0.5], [0.1, 0.2, 0.9]],
        dtype=float,
    )
    weights = np.array([0.9, 0.6, 0.4], dtype=float)

    closed = _banzhaf_power(prob, weights, _GabanzhafConfig.from_kwargs({"banzhaf_mode": "closed_form"}))
    brute = _banzhaf_power(prob, weights, _GabanzhafConfig.from_kwargs({"banzhaf_mode": "bruteforce"}))

    assert np.allclose(brute, closed, atol=1e-10)
    assert np.all(brute >= 0.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_small_noisy_or_games_match_bruteforce(seed):
    rng = np.random.default_rng(seed)
    prob = rng.uniform(0.0, 1.0, size=(4, 5))
    weights = rng.uniform(0.0, 1.0, size=5)

    closed = _banzhaf_power(
        prob,
        weights,
        _GabanzhafConfig.from_kwargs({"banzhaf_mode": "closed_form"}),
    )
    brute = _banzhaf_power(
        prob,
        weights,
        _GabanzhafConfig.from_kwargs({"banzhaf_mode": "bruteforce"}),
    )

    assert np.allclose(closed, brute, atol=1e-10)


def test_banzhaf_power_rejects_unknown_mode():

    with pytest.raises(ValueError, match="Unsupported banzhaf_mode"):
        _banzhaf_power(
            np.array([[1.0]], dtype=float),
            np.array([0.5], dtype=float),
            _GabanzhafConfig.from_kwargs({"banzhaf_mode": "invalid"}),
        )


def test_noisy_or_coverage_is_monotone_and_bounded():

    prob = np.array([[1.0, 0.6, 0.0], [0.0, 0.5, 0.8]], dtype=float)

    empty = _noisy_or_coverage(prob, [])
    one = _noisy_or_coverage(prob, [0])
    both = _noisy_or_coverage(prob, [0, 1])
    expected_both = 1.0 - (1.0 - prob[0]) * (1.0 - prob[1])

    assert np.all(empty == 0.0)
    assert np.all(one >= empty)
    assert np.all(both >= one)
    assert np.all(both <= 1.0)
    assert np.allclose(both, expected_both)


def test_noisy_or_coverage_deduplicates_coalition_players():

    prob = np.array([[0.2, 0.5], [0.7, 0.4]], dtype=float)
    deduped = _noisy_or_coverage(prob, [0, 1])
    duplicated = _noisy_or_coverage(prob, [0, 1, 0])
    assert np.allclose(duplicated, deduped)


def test_noisy_or_coverage_rejects_invalid_coalition_indices():

    prob = np.array([[0.2, 0.5], [0.7, 0.4]], dtype=float)
    for bad in ([-1], [0, 1.5], [2], [[0]]):
        with pytest.raises(ValueError, match="coalition"):
            _noisy_or_coverage(prob, bad)


def test_noisy_or_payoff_has_diminishing_marginal_returns():

    prob = np.array([[1.0, 0.8], [0.0, 0.7], [0.0, 0.6]], dtype=float)
    weights = np.array([1.0, 1.0], dtype=float)

    def payoff(coalition):
        return float(np.dot(weights, _noisy_or_coverage(prob, coalition)))

    assert (payoff([0]) - payoff([])) > (payoff([0, 1, 2]) - payoff([1, 2]))


def test_probabilistic_banzhaf_matches_bruteforce_noisy_or_game():
    from itertools import combinations

    prob = np.array(
        [[1.0, 0.4, 0.2], [0.0, 0.7, 0.5], [0.1, 0.2, 0.9]],
        dtype=float,
    )
    weights = np.array([0.9, 0.6, 0.4], dtype=float)

    def payoff(coalition):
        return float(np.dot(weights, _noisy_or_coverage(prob, coalition)))

    brute = np.zeros(prob.shape[0], dtype=float)
    players = list(range(prob.shape[0]))
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        count = 0
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                total += payoff(list(subset) + [i]) - payoff(list(subset))
                count += 1
        brute[i] = total / count

    closed = _probabilistic_banzhaf(prob, weights)
    assert np.allclose(closed, brute, atol=1e-10)
    assert np.all(closed >= 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Candidate selection (anomaly-ranked) and lag-correlation expansion
# ─────────────────────────────────────────────────────────────────────────────

def test_select_exact_candidates_ranks_by_anomaly_score():

    cfg = _GabanzhafConfig.from_kwargs({"exact_candidate_k": 3})
    selected = _select_exact_candidates(
        ["svc-a", "svc-b", "svc-c", "svc-d"],
        svc_anom_dict={"svc-a": 0.9, "svc-b": 0.1, "svc-c": 0.8, "svc-d": 0.7},
        config=cfg,
    )
    assert selected == ["svc-a", "svc-c", "svc-d"]


def test_select_exact_candidates_deduplicates_before_limiting_slots():

    cfg = _GabanzhafConfig.from_kwargs({"exact_candidate_k": 3})
    selected = _select_exact_candidates(
        ["svc-a", "svc-a", "svc-b", "svc-c"],
        svc_anom_dict={"svc-a": 1.0, "svc-b": 0.8, "svc-c": 0.7},
        config=cfg,
    )
    assert selected == ["svc-a", "svc-b", "svc-c"]


def test_service_trace_series_aggregates_operation_columns_by_service():
    df = pd.DataFrame({
        "time": [0, 1, 2, 3],
        "svc-a_GET": [0, 1, 3, 1],
        "svc-a_POST": [0, 0, 2, 0],
        "svc-b_GET": [0, 0, 1, 4],
    })

    series = gabanzhaf_mod._service_trace_series_from_frame(df)

    assert set(series) == {"svc-a", "svc-b"}
    assert np.allclose(series["svc-a"], np.array([0.0, 1.0, 5.0, 1.0]))
    assert np.allclose(series["svc-b"], np.array([0.0, 0.0, 1.0, 4.0]))


def test_service_trace_series_residual_isolates_fault_induced_delta():
    # A service flat across inject (equally active pre and post) must reduce to
    # ~0 residual; a service that only spikes post-inject must keep its bump.
    df = pd.DataFrame({
        "time": [0, 1, 2, 3, 4, 5],
        "flat_GET": [5, 5, 5, 5, 5, 5],
        "spike_GET": [0, 0, 0, 9, 8, 7],
    })

    series = gabanzhaf_mod._service_trace_series_from_frame(
        df, inject_time=3, residual=True
    )

    assert np.allclose(series["flat"], np.zeros(3))
    assert series["spike"].max() > 0.0
    # discrimination property: perturbed service dominates the unperturbed one
    assert series["spike"].max() > series["flat"].max()


def test_service_trace_series_residual_noop_without_inject_time():
    df = pd.DataFrame({"time": [0, 1], "svc_GET": [2, 4]})
    full = gabanzhaf_mod._service_trace_series_from_frame(df)
    no_inject = gabanzhaf_mod._service_trace_series_from_frame(df, residual=True)
    assert np.allclose(full["svc"], no_inject["svc"])  # residual inert w/o inject


def test_trace_min_energy_gate_drops_quiet_residual_service():
    call_graph = {"caller": ["loud", "quiet"]}
    trace_series = {
        "caller": np.array([0.0, 0.0, 4.0, 0.0]),
        "loud": np.array([0.0, 0.0, 5.0, 0.0]),
        "quiet": np.array([0.0, 0.0, 0.2, 0.0]),  # < 10% of loudest peak
    }
    cfg = _GabanzhafConfig.from_kwargs({
        "trace_residual": True,
        "trace_min_energy": 0.1,
        "trace_topology_max_lag": 0,
        "trace_topology_min_conf": 0.1,
    })

    edges = gabanzhaf_mod._trace_multilag_edges(
        call_graph, trace_series, cfg, source_kind="latency"
    )

    sources = {e[0] for e in edges}
    assert "loud" in sources
    assert "quiet" not in sources  # energy gate dropped it


def test_rank_evidence_trace_lets_trace_move_the_order():
    cols = ["svc-a_latency", "svc-b_latency"]
    metric_ev = {"svc-a_latency": 0.5, "svc-b_latency": 0.5}
    svc_anom = {"svc-a": 0.5, "svc-b": 0.5}  # metric tie
    trace_sev = {"svc-b": 0.9, "svc-a": 0.1}  # trace prefers svc-b

    ranks = gabanzhaf_mod._rank_evidence_metrics(
        cols, metric_ev, svc_anom, "service_evidence_trace",
        trace_service_evidence=trace_sev,
    )
    assert ranks[0] == "svc-b_latency"  # trace broke the metric tie


def test_rank_evidence_trace_only_ignores_metric():
    cols = ["svc-a_cpu", "svc-b_cpu"]
    metric_ev = {"svc-a_cpu": 0.9, "svc-b_cpu": 0.1}  # metric prefers svc-a
    svc_anom = {"svc-a": 0.9, "svc-b": 0.1}
    trace_sev = {"svc-a": 0.1, "svc-b": 0.9}  # trace prefers svc-b

    ranks = gabanzhaf_mod._rank_evidence_metrics(
        cols, metric_ev, svc_anom, "trace_evidence", trace_service_evidence=trace_sev,
    )
    assert ranks[0] == "svc-b_cpu"  # metric ignored entirely


def test_rank_evidence_trace_empty_trace_matches_metric_only():
    cols = ["svc-a_cpu", "svc-b_cpu"]
    metric_ev = {"svc-a_cpu": 0.3, "svc-b_cpu": 0.9}
    svc_anom = {"svc-a": 0.3, "svc-b": 0.9}

    combined = gabanzhaf_mod._rank_evidence_metrics(
        cols, metric_ev, svc_anom, "service_evidence_trace", trace_service_evidence={}
    )
    metric_only = gabanzhaf_mod._rank_evidence_metrics(
        cols, metric_ev, svc_anom, "service_evidence"
    )
    assert combined == metric_only  # degrades to metric-only without trace


def test_build_call_graph_from_traces_span_hierarchy(tmp_path):
    pd.DataFrame({
        "spanID": ["root", "a", "b"],
        "parentSpanID": ["", "root", "a"],
        "serviceName": ["frontend", "checkout", "currency"],
    }).to_csv(tmp_path / "traces.csv", index=False)

    g = gabanzhaf_mod._build_call_graph_from_traces(
        str(tmp_path), service_names={"frontend", "checkout", "currency"}
    )
    assert g == {"frontend": ["checkout"], "checkout": ["currency"]}


def test_build_call_graph_missing_traces_returns_empty(tmp_path):
    # No traces.csv (e.g. sock-shop) → empty graph → metric fallback.
    assert gabanzhaf_mod._build_call_graph_from_traces(str(tmp_path)) == {}


def test_match_service_maps_trace_name_to_metric_prefix():
    names = {"frontend", "frontend-external", "checkoutservice"}
    # "frontendservice" → longest metric prefix it starts with = "frontend"
    assert gabanzhaf_mod._match_service("frontendservice", names) == "frontend"
    assert gabanzhaf_mod._match_service("checkoutservice", names) == "checkoutservice"
    assert gabanzhaf_mod._match_service("unknown", names) == "unknown"


def test_trace_multilag_edges_detects_callee_leading_caller():
    call_graph = {"caller": ["callee"]}
    trace_series = {
        "callee": np.array([0.0, 0.0, 5.0, 0.0, 0.0]),
        "caller": np.array([0.0, 0.0, 0.0, 4.0, 0.0]),
    }
    cfg = _GabanzhafConfig.from_kwargs({
        "trace_topology_max_lag": 2,
        "trace_topology_min_conf": 0.1,
        "trace_topology_conf": 0.7,
    })

    edges = gabanzhaf_mod._trace_multilag_edges(
        call_graph,
        trace_series,
        cfg,
        source_kind="latency",
    )

    assert edges == [("callee", "caller", pytest.approx(0.7), "latency", 1)]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Causal explanation matrix (diagonal base; coverage overlaid separately)
# ─────────────────────────────────────────────────────────────────────────────

def test_causal_explanation_matrix_builds_self_probabilities():

    matrix = _causal_explanation_matrix(
        ["svc-a", "svc-b"],
        ["svc-a", "svc-b", "svc-c"],
    )

    assert matrix.shape == (2, 3)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert matrix[0, 1] == 0.0
    assert matrix[0, 2] == 0.0


def test_causal_explanation_matrix_rejects_duplicate_candidates():

    with pytest.raises(ValueError, match="Duplicate candidate"):
        _causal_explanation_matrix(["svc-a", "svc-a"], ["svc-a", "svc-b"])


def test_causal_explanation_matrix_rejects_duplicate_services():

    with pytest.raises(ValueError, match="Duplicate service"):
        _causal_explanation_matrix(["svc-a", "svc-b"], ["svc-a", "svc-b", "svc-b"])


def test_apply_trace_topology_edges_overlays_off_diagonal_probability():
    matrix = gabanzhaf_mod._causal_explanation_matrix(
        ["callee", "other"],
        ["callee", "caller", "other"],
    )
    edges = [("callee", "caller", 0.42, "latency", 1)]

    out = gabanzhaf_mod._apply_trace_topology_edges(
        matrix,
        ["callee", "other"],
        ["callee", "caller", "other"],
        edges,
    )

    assert out[0, 1] == pytest.approx(0.42)
    assert out[0, 0] == 1.0


def test_frontier_bonus_rewards_abnormal_boundary_before_normal_upstream():
    cfg = _GabanzhafConfig.from_kwargs({
        "frontier_abnormal_threshold": 0.45,
        "frontier_normal_threshold": 0.2,
        "frontier_edge_threshold": 0.3,
    })

    bonus, diagnostics = gabanzhaf_mod._frontier_bonus(
        causal_edges=[
            ("svc-b", "svc-c", 0.6, "trace", 1),
            ("svc-a", "svc-b", 0.6, "trace", 1),
        ],
        svc_anom_dict={"svc-a": 0.1, "svc-b": 0.8, "svc-c": 0.9},
        config=cfg,
    )

    assert bonus["svc-b"] > 0.0
    assert bonus.get("svc-c", 0.0) == 0.0
    assert diagnostics["svc-b"]["upstream_state"] == "normal"


def test_frontier_bonus_suppresses_candidate_when_upstream_is_abnormal():
    cfg = _GabanzhafConfig.from_kwargs({
        "frontier_abnormal_threshold": 0.45,
        "frontier_edge_threshold": 0.3,
    })

    bonus, diagnostics = gabanzhaf_mod._frontier_bonus(
        causal_edges=[
            ("svc-b", "svc-c", 0.6, "trace", 1),
            ("svc-a", "svc-b", 0.6, "trace", 1),
        ],
        svc_anom_dict={"svc-a": 0.7, "svc-b": 0.8, "svc-c": 0.9},
        config=cfg,
    )

    assert bonus.get("svc-b", 0.0) == 0.0
    assert diagnostics["svc-b"]["upstream_state"] == "abnormal"


def test_frontier_bonus_ignores_weak_edges():
    cfg = _GabanzhafConfig.from_kwargs({
        "frontier_abnormal_threshold": 0.45,
        "frontier_edge_threshold": 0.3,
    })

    bonus, diagnostics = gabanzhaf_mod._frontier_bonus(
        causal_edges=[("svc-b", "svc-c", 0.2, "trace", 1)],
        svc_anom_dict={"svc-b": 0.8, "svc-c": 0.9},
        config=cfg,
    )

    assert bonus == {}
    assert diagnostics == {}


def test_apply_frontier_bonus_softly_adjusts_banzhaf_scores():
    cfg = _GabanzhafConfig.from_kwargs({"frontier_alpha": 0.5})

    adjusted = gabanzhaf_mod._apply_frontier_bonus(
        svc_banzhaf={"svc-b": 0.4, "svc-c": 0.5},
        frontier_bonus={"svc-b": 1.0},
        config=cfg,
    )

    assert adjusted["svc-b"] > adjusted["svc-c"]
    assert adjusted["svc-c"] == pytest.approx(0.5)


def test_select_frontier_rescue_candidates_requires_high_confidence_boundary():
    cfg = _GabanzhafConfig.from_kwargs({
        "frontier_rescue_min_bonus": 0.25,
        "frontier_rescue_min_child_count": 1,
    })

    rescued = gabanzhaf_mod._select_frontier_rescue_candidates(
        frontier_bonus={"svc-b": 0.36, "svc-d": 0.20},
        frontier_diagnostics={
            "svc-b": {
                "upstream_state": "normal",
                "abnormal_children": [{"service": "svc-c"}],
            },
            "svc-d": {
                "upstream_state": "normal",
                "abnormal_children": [{"service": "svc-e"}],
            },
            "svc-x": {
                "upstream_state": "abnormal",
                "abnormal_children": [{"service": "svc-y"}],
            },
        },
        svc_anom_dict={"svc-b": 0.7, "svc-d": 0.8, "svc-x": 0.9},
        config=cfg,
    )

    assert rescued == ["svc-b"]


def test_gabanzhaf_core_frontier_rescue_adds_high_confidence_candidate():
    data = pd.DataFrame({
        "time": np.arange(10),
        "svc-a_cpu": [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
        "svc-a_latency": [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
        "svc-b_cpu": [1, 1, 1, 1, 1, 6, 6, 6, 6, 6],
        "svc-b_latency": [1, 1, 1, 1, 1, 5, 5, 5, 5, 5],
        "svc-c_cpu": [1, 1, 1, 1, 1, 7, 7, 7, 7, 7],
        "svc-c_latency": [1, 1, 1, 1, 1, 8, 8, 8, 8, 8],
    })

    result = _gabanzhaf_core(
        data,
        inject_time=5,
        dataset="unit",
        anomalies=None,
        kwargs={
            "return_diagnostics": True,
            "exact_candidate_k": 1,
            "trace_topology_edges": [
                ("svc-b", "svc-c", 0.6, "latency", 1),
                ("svc-a", "svc-b", 0.6, "latency", 1),
            ],
            "frontier_alpha": 0.5,
            "frontier_abnormal_threshold": 0.45,
            "frontier_normal_threshold": 0.2,
            "frontier_edge_threshold": 0.3,
            "use_frontier_rescue": True,
            "frontier_rescue_min_bonus": 0.25,
        },
    )

    assert "svc-b" in result["game_candidates"]
    assert result["frontier_candidate_added_count"] == 1
    assert result["frontier_rescue_candidates"] == ["svc-b"]


def test_gabanzhaf_core_does_not_rescue_frontier_candidate_when_disabled():
    data = pd.DataFrame({
        "time": np.arange(10),
        "svc-a_cpu": [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
        "svc-a_latency": [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
        "svc-b_cpu": [1, 1, 1, 1, 1, 6, 6, 6, 6, 6],
        "svc-b_latency": [1, 1, 1, 1, 1, 5, 5, 5, 5, 5],
        "svc-c_cpu": [1, 1, 1, 1, 1, 7, 7, 7, 7, 7],
        "svc-c_latency": [1, 1, 1, 1, 1, 8, 8, 8, 8, 8],
    })

    result = _gabanzhaf_core(
        data,
        inject_time=5,
        dataset="unit",
        anomalies=None,
        kwargs={
            "return_diagnostics": True,
            "exact_candidate_k": 1,
            "trace_topology_edges": [
                ("svc-b", "svc-c", 0.6, "latency", 1),
                ("svc-a", "svc-b", 0.6, "latency", 1),
            ],
            "frontier_alpha": 0.5,
            "frontier_abnormal_threshold": 0.45,
            "frontier_normal_threshold": 0.2,
            "frontier_edge_threshold": 0.3,
            "use_frontier_rescue": False,
            "frontier_rescue_min_bonus": 0.25,
        },
    )

    assert result["game_candidates"] == ["svc-c"]
    assert result["frontier_candidate_added_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Banzhaf-score ranking
# ─────────────────────────────────────────────────────────────────────────────

def test_rank_metrics_orders_by_service_banzhaf_then_column_name():

    ranks = _rank_metrics(
        ["svc-b_mem", "svc-a_z", "svc-a_a"],
        svc_banzhaf={"svc-a": 1.0, "svc-b": 0.2},
    )
    assert ranks == ["svc-a_a", "svc-a_z", "svc-b_mem"]


def test_rank_metrics_keeps_positive_banzhaf_above_non_candidate_service():

    ranks = _rank_metrics(
        ["svc-a_cpu", "svc-c_cpu", "svc-b_cpu"],
        svc_banzhaf={"svc-a": 1.0, "svc-b": 0.2},
    )
    assert ranks.index("svc-b_cpu") < ranks.index("svc-c_cpu")


def test_rank_metrics_single_positive_candidate_does_not_degenerate_to_name_order():
    ranks = _rank_metrics(
        ["aaa-noncandidate_cpu", "root_cpu", "zzz_noncandidate_cpu"],
        svc_banzhaf={"root": 0.7},
    )

    assert ranks[0] == "root_cpu"


def test_service_evidence_ranking_orders_services_then_metrics():
    ranks = gabanzhaf_mod._rank_evidence_metrics(
        ["svc-b_mem", "svc-a_z", "svc-a_a"],
        metric_ev={"svc-b_mem": 1.0, "svc-a_z": 0.1, "svc-a_a": 0.9},
        svc_anom_dict={"svc-a": 1.0, "svc-b": 0.2},
        rank_mode="service_evidence",
    )

    assert ranks == ["svc-a_a", "svc-a_z", "svc-b_mem"]


def test_metric_evidence_ranking_uses_metric_score_and_stable_name_tie_break():
    ranks = gabanzhaf_mod._rank_evidence_metrics(
        ["svc-b_mem", "svc-a_z", "svc-a_a"],
        metric_ev={"svc-b_mem": 1.0, "svc-a_z": 0.5, "svc-a_a": 0.5},
        svc_anom_dict={"svc-a": 1.0, "svc-b": 0.2},
        rank_mode="metric_evidence",
    )

    assert ranks == ["svc-b_mem", "svc-a_a", "svc-a_z"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Core pipeline and entry point
# ─────────────────────────────────────────────────────────────────────────────

def _simple_core_frame():
    return pd.DataFrame({
        "time": np.arange(12),
        "svc-a_cpu": [1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9],
        "svc-b_cpu": [1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6],
    })


def test_gabanzhaf_core_can_return_stage_timing():
    result = _gabanzhaf_core(
        _simple_core_frame(),
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={"return_stage_timing": True},
    )

    assert "stage_timing" in result
    for key in ["preprocess", "metric_evidence", "candidates", "banzhaf", "ranking"]:
        assert key in result["stage_timing"]
        assert isinstance(result["stage_timing"][key], float)
        assert result["stage_timing"][key] >= 0.0


def test_gabanzhaf_core_returns_game_candidates_when_requested():
    data = pd.DataFrame({
        "time": np.arange(12),
        "svc-a_cpu": [1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8],
        "svc-a_latency": [1] * 12,
        "svc-b_cpu": [1] * 12,
        "svc-b_latency": [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
    })

    result = _gabanzhaf_core(
        data,
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={"return_stage_timing": True, "return_diagnostics": True},
    )

    assert "ranks" in result
    assert "stage_timing" in result
    assert "game_candidates" in result
    assert len(result["game_candidates"]) > 0


def test_gabanzhaf_core_uses_identity_only_causal_matrix_without_coverage():
    # "unit" is not a train-ticket dataset, so no call graph loads and the
    # explanation matrix is the pure diagonal.
    result = _gabanzhaf_core(
        _simple_core_frame(),
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={
            "return_diagnostics": True,
        },
    )

    prob = np.asarray(result["causal_probability_matrix"], dtype=float)
    candidates = result["game_candidates"]
    services = result["game_services"]
    expected = np.zeros_like(prob)
    service_idx = {service: i for i, service in enumerate(services)}
    for row, candidate in enumerate(candidates):
        expected[row, service_idx[candidate]] = 1.0
    assert np.array_equal(prob, expected)


def test_gabanzhaf_core_overlays_trace_topology_edges_from_kwargs():
    data = pd.DataFrame({
        "time": np.arange(8),
        "callee_cpu": [1, 1, 1, 1, 5, 5, 5, 5],
        "caller_cpu": [1, 1, 1, 1, 4, 4, 4, 4],
        "caller_latency": [1, 1, 1, 1, 6, 6, 6, 6],
    })

    result = _gabanzhaf_core(
        data,
        inject_time=4,
        dataset="unit",
        anomalies=None,
        kwargs={
            "return_diagnostics": True,
            "trace_topology_conf": 0.7,
            "trace_topology_edges": [("callee", "caller", 0.5, "latency", 1)],
        },
    )

    matrix = np.asarray(result["causal_probability_matrix"], dtype=float)
    candidates = result["game_candidates"]
    services = result["game_services"]
    assert matrix[candidates.index("callee"), services.index("caller")] == pytest.approx(0.5)
    assert result["trace_edge_count"] == 1


def test_gabanzhaf_core_applies_frontier_bonus_to_banzhaf_scores():
    data = pd.DataFrame({
        "time": np.arange(10),
        "svc-a_cpu": [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
        "svc-a_latency": [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
        "svc-b_cpu": [1, 1, 1, 1, 1, 6, 6, 6, 6, 6],
        "svc-b_latency": [1, 1, 1, 1, 1, 5, 5, 5, 5, 5],
        "svc-c_cpu": [1, 1, 1, 1, 1, 7, 7, 7, 7, 7],
        "svc-c_latency": [1, 1, 1, 1, 1, 8, 8, 8, 8, 8],
    })

    result = _gabanzhaf_core(
        data,
        inject_time=5,
        dataset="unit",
        anomalies=None,
        kwargs={
            "return_diagnostics": True,
            "trace_topology_edges": [
                ("svc-b", "svc-c", 0.6, "latency", 1),
                ("svc-a", "svc-b", 0.6, "latency", 1),
            ],
            "frontier_alpha": 0.5,
            "frontier_abnormal_threshold": 0.45,
            "frontier_normal_threshold": 0.2,
            "frontier_edge_threshold": 0.3,
        },
    )

    assert result["frontier_bonus"]["svc-b"] > 0.0
    assert result["frontier_diagnostics"]["svc-b"]["upstream_state"] == "normal"
    assert result["service_banzhaf"]["svc-b"] > result["service_banzhaf_raw"]["svc-b"]


def test_gabanzhaf_entry_loads_trace_multilag_topology_from_case_dir(tmp_path, monkeypatch):
    case_dir = tmp_path / "svc-a_delay" / "1"
    case_dir.mkdir(parents=True)
    metric_path = case_dir / "simple_metrics.csv"
    data = pd.DataFrame({
        "time": np.arange(8),
        "svc-a_cpu": [1, 1, 1, 1, 5, 5, 5, 5],
        "svc-b_cpu": [1, 1, 1, 1, 4, 4, 4, 4],
        "svc-b_latency": [1, 1, 1, 1, 6, 6, 6, 6],
    })
    data.to_csv(metric_path, index=False)
    pd.DataFrame({
        "time": np.arange(8),
        "svc-a_GET": [0, 0, 5, 0, 0, 0, 0, 0],
        "svc-b_GET": [0, 0, 0, 4, 0, 0, 0, 0],
    }).to_csv(case_dir / "tracets_lat.csv", index=False)
    # traces.csv encodes the call graph svc-b -> svc-a (parent span svc-b,
    # child span svc-a); the pipeline builds the graph from it directly.
    pd.DataFrame({
        "spanID": ["p1", "c1"],
        "parentSpanID": ["", "p1"],
        "serviceName": ["svc-b", "svc-a"],
    }).to_csv(case_dir / "traces.csv", index=False)

    args = type("Args", (), {"data_path": str(metric_path)})()
    result = gabanzhaf_mod.gabanzhaf(
        data,
        inject_time=4,
        dataset="unit",
        args=args,
        return_diagnostics=True,
        trace_topology_conf=0.7,
    )

    assert result["diagnostics"]["trace_edge_count"] == 1
    assert result["diagnostics"]["trace_topology_edges"] == [
        ("svc-a", "svc-b", pytest.approx(0.7), "latency", 1)
    ]
    assert "trace" in result["diagnostics"]["stage_timing"]


@pytest.mark.parametrize("rank_mode", ["service_evidence", "metric_evidence"])
def test_gabanzhaf_core_evidence_modes_skip_banzhaf(monkeypatch, rank_mode):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("causal game must not run in evidence-only modes")

    monkeypatch.setattr(gabanzhaf_mod, "_banzhaf_power", fail_if_called)
    monkeypatch.setattr(
        gabanzhaf_mod,
        "_causal_explanation_matrix",
        fail_if_called,
    )
    result = _gabanzhaf_core(
        _simple_core_frame(),
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={
            "rank_mode": rank_mode,
            "return_diagnostics": True,
        },
    )

    assert result["rank_mode"] == rank_mode
    assert result["service_banzhaf"] == {}
    assert result["causal_probability_matrix"] == []


def test_gabanzhaf_entry_reads_banzhaf_mode_env(monkeypatch):
    captured_kwargs = {}

    def fake_core(data, inject_time, dataset, anomalies, kwargs):
        captured_kwargs.update(kwargs)
        return {"ranks": []}

    monkeypatch.setenv("GABANZHAF_BANZHAF_MODE", "bruteforce")
    monkeypatch.setattr(gabanzhaf_mod, "_gabanzhaf_core", fake_core)

    gabanzhaf_mod.gabanzhaf(
        pd.DataFrame({"time": [0, 1], "svc_cpu": [0.0, 1.0]}),
        inject_time=1,
    )

    assert captured_kwargs["banzhaf_mode"] == "bruteforce"


def test_victim_factors_demotes_service_with_abnormal_upstream():
    # Call chain A -> B -> C (cause->effect edges). B and C abnormal, A normal.
    # C has an abnormal predecessor (B) => victim, self-coverage demoted to floor.
    # B's only predecessor (A) is normal => root, keeps 1.0 (absent from dict).
    _victim_factors = gabanzhaf_mod._victim_factors
    edges = [("A", "B", 0.9, "trace", 1), ("B", "C", 0.9, "trace", 1)]
    svc_anom = {"A": 0.05, "B": 0.8, "C": 0.7}
    cfg = _GabanzhafConfig.from_kwargs({"use_victim_demotion": True, "victim_floor": 0.4})

    factors = _victim_factors(edges, svc_anom, cfg)

    assert factors.get("C") == pytest.approx(0.4)   # victim: demoted to floor
    assert "B" not in factors                        # root: undemoted (implicit 1.0)
    assert "A" not in factors


def test_gabanzhaf_contains_no_dataset_or_fault_specific_branches():
    source_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "RCAEval", "e2e", "gabanzhaf.py",
    )
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    forbidden = [
        "ground_truth", "root_cause", "case_id", "if dataset",
        "re2-ob", "re2-ss", "re2-tt", "DELAY", "LOSS",
    ]
    for token in forbidden:
        assert token not in source
