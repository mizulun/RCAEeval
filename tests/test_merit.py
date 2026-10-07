"""Tests for the MERIT (formerly gabanzhaf) pipeline.

After ablation showed the old metric-derived trace lag to be mostly redundant,
the pipeline now keeps metric evidence as anomaly weight and adds trace/topology
signals as causal edges. These tests cover the math core, configuration,
diagnostics, and entry point.
"""
import importlib.util
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

# Load the module directly from its file so the tests do not drag in the
# torch-dependent RCAEval.e2e package __init__ (merit itself needs none of it).
_spec = importlib.util.spec_from_file_location(
    "merit_under_test",
    pathlib.Path(__file__).resolve().parent.parent / "RCAEval" / "e2e" / "merit.py",
)
merit_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merit_mod)

_GabanzhafConfig = merit_mod._GabanzhafConfig
_gabanzhaf_core = merit_mod._merit_core
_metric_evidence = merit_mod._metric_evidence
_banzhaf_power = merit_mod._banzhaf_power
_noisy_or_coverage = merit_mod._noisy_or_coverage
_probabilistic_banzhaf = merit_mod._probabilistic_banzhaf
_select_exact_candidates = merit_mod._select_exact_candidates
_causal_explanation_matrix = merit_mod._causal_explanation_matrix
_rank_metrics = merit_mod._rank_metrics


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


def test_metric_evidence_default_scorer_matches_robust_z_unchanged():
    """Default config.scorer ("robust_z") must produce byte-identical output
    to the pre-CUSUM implementation -- i.e. explicitly requesting robust_z
    must match the config=None / default path."""
    rng = np.random.default_rng(1)
    normal = rng.normal(10.0, 2.0, (50, 4))
    anomal = rng.normal(10.0, 2.0, (30, 4))
    anomal[:, 1] += 20.0
    cols = [f"svc{i}_latency" for i in range(4)]

    ev_default = _metric_evidence(normal, anomal, cols)
    ev_explicit = _metric_evidence(
        normal, anomal, cols, config=_GabanzhafConfig.from_kwargs({"scorer": "robust_z"})
    )
    for c in cols:
        assert ev_default[c] == ev_explicit[c]


def test_cusum_metric_evidence_vectorized_matches_naive_loop():
    """The vectorized CUSUM peak (per column) must equal the naive per-step
    accumulator acc = max(0, acc + dev[t]) tracked to its running maximum."""
    rng = np.random.default_rng(42)
    n_rows, n_cols = 60, 5
    normal = rng.normal(5.0, 1.5, (n_rows, n_cols))
    anomal = rng.normal(5.0, 1.5, (40, n_cols))
    anomal[:, 2] += 3.0  # small persistent shift on one column
    cols = [f"svc{i}_metric" for i in range(n_cols)]
    config = _GabanzhafConfig.from_kwargs({"scorer": "cusum"})

    normal_mat = np.asarray(normal, dtype=float)
    anomal_mat = np.asarray(anomal, dtype=float)

    med = np.nanmedian(normal_mat, axis=0)
    q75, q25 = np.nanpercentile(normal_mat, [75, 25], axis=0)
    sigma = np.where((q75 - q25) > 0, (q75 - q25) / 1.349, 1.0)
    k = config.cusum_slack * sigma
    dev = np.abs(anomal_mat - med) - k
    dev = np.nan_to_num(dev, nan=0.0)

    naive_peak = np.zeros(n_cols)
    for j in range(n_cols):
        acc = 0.0
        peak_j = 0.0
        for t in range(dev.shape[0]):
            acc = max(0.0, acc + dev[t, j])
            peak_j = max(peak_j, acc)
        naive_peak[j] = peak_j / sigma[j]
    naive_log = np.log1p(naive_peak)
    naive_ref = np.percentile(naive_log, 99)
    naive_score = (
        np.clip(naive_log / naive_ref, 0.0, 1.0)
        if naive_ref > 0
        else np.zeros_like(naive_log)
    )

    ev = merit_mod._cusum_metric_evidence(normal_mat, anomal_mat, cols, config)
    vectorized_score = np.array([ev[c] for c in cols])

    assert np.allclose(vectorized_score, naive_score)


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


def test_individual_attribution_keeps_probability_matrix_and_weights():
    prob = np.array([[1.0, 0.4], [0.2, 1.0]], dtype=float)
    weights = np.array([0.7, 0.5], dtype=float)

    values = _banzhaf_power(
        prob,
        weights,
        _GabanzhafConfig.from_kwargs({"banzhaf_mode": "individual"}),
    )

    assert np.allclose(values, prob @ weights)


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

    series = merit_mod._service_trace_series_from_frame(df)

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

    series = merit_mod._service_trace_series_from_frame(
        df, inject_time=3, residual=True
    )

    assert np.allclose(series["flat"], np.zeros(3))
    assert series["spike"].max() > 0.0
    # discrimination property: perturbed service dominates the unperturbed one
    assert series["spike"].max() > series["flat"].max()


def test_service_trace_series_residual_noop_without_inject_time():
    df = pd.DataFrame({"time": [0, 1], "svc_GET": [2, 4]})
    full = merit_mod._service_trace_series_from_frame(df)
    no_inject = merit_mod._service_trace_series_from_frame(df, residual=True)
    assert np.allclose(full["svc"], no_inject["svc"])  # residual inert w/o inject


def test_trace_service_evidence_preserves_cross_service_amplitude():
    evidence = merit_mod._trace_service_evidence({
        "loud": np.array([0.0, 1.0, 9.0, 8.0]),
        "quiet": np.array([0.0, 0.2, 0.5, 0.4]),
    })

    assert evidence["loud"] == pytest.approx(1.0)
    assert evidence["quiet"] == pytest.approx(0.0)


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

    edges = merit_mod._trace_multilag_edges(
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

    ranks = merit_mod._rank_evidence_metrics(
        cols, metric_ev, svc_anom, "service_evidence_trace",
        trace_service_evidence=trace_sev,
    )
    assert ranks[0] == "svc-b_latency"  # trace broke the metric tie


def test_direct_metric_trace_mode_ranks_by_normalized_score_sum():
    cols = ["svc-a_cpu", "svc-b_cpu", "svc-c_cpu"]
    metric_ev = {"svc-a_cpu": 10.0, "svc-b_cpu": 6.0, "svc-c_cpu": 0.0}
    metric_service = {"svc-a": 10.0, "svc-b": 6.0, "svc-c": 0.0}
    trace_service = {"svc-a": 0.0, "svc-b": 5.0, "svc-c": 10.0}

    ranks = merit_mod._rank_evidence_metrics(
        cols,
        metric_ev,
        metric_service,
        "service_evidence_trace",
        trace_service_evidence=trace_service,
    )

    # svc-b wins only after adding normalized metric (0.6) and trace (0.5)
    # scores; the single-source winners are svc-a and svc-c.
    assert ranks[0] == "svc-b_cpu"


def test_rank_evidence_trace_only_ignores_metric():
    cols = ["svc-a_cpu", "svc-b_cpu"]
    metric_ev = {"svc-a_cpu": 0.9, "svc-b_cpu": 0.1}  # metric prefers svc-a
    svc_anom = {"svc-a": 0.9, "svc-b": 0.1}
    trace_sev = {"svc-a": 0.1, "svc-b": 0.9}  # trace prefers svc-b

    ranks = merit_mod._rank_evidence_metrics(
        cols, metric_ev, svc_anom, "trace_evidence", trace_service_evidence=trace_sev,
    )
    assert ranks[0] == "svc-b_cpu"  # metric ignored entirely


def test_rank_evidence_trace_empty_trace_matches_metric_only():
    cols = ["svc-a_cpu", "svc-b_cpu"]
    metric_ev = {"svc-a_cpu": 0.3, "svc-b_cpu": 0.9}
    svc_anom = {"svc-a": 0.3, "svc-b": 0.9}

    combined = merit_mod._rank_evidence_metrics(
        cols, metric_ev, svc_anom, "service_evidence_trace", trace_service_evidence={}
    )
    metric_only = merit_mod._rank_evidence_metrics(
        cols, metric_ev, svc_anom, "service_evidence"
    )
    assert combined == metric_only  # degrades to metric-only without trace


def test_build_call_graph_from_traces_span_hierarchy(tmp_path):
    pd.DataFrame({
        "spanID": ["root", "a", "b"],
        "parentSpanID": ["", "root", "a"],
        "serviceName": ["frontend", "checkout", "currency"],
    }).to_csv(tmp_path / "traces.csv", index=False)

    g = merit_mod._build_call_graph_from_traces(
        str(tmp_path), service_names={"frontend", "checkout", "currency"}
    )
    assert g == {"frontend": ["checkout"], "checkout": ["currency"]}


def test_build_call_graph_missing_traces_returns_empty(tmp_path):
    # No traces.csv (e.g. sock-shop) → empty graph → metric fallback.
    assert merit_mod._build_call_graph_from_traces(str(tmp_path)) == {}


def test_match_service_maps_trace_name_to_metric_prefix():
    names = {"frontend", "frontend-external", "checkoutservice"}
    # "frontendservice" → longest metric prefix it starts with = "frontend"
    assert merit_mod._match_service("frontendservice", names) == "frontend"
    assert merit_mod._match_service("checkoutservice", names) == "checkoutservice"
    assert merit_mod._match_service("unknown", names) == "unknown"


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

    edges = merit_mod._trace_multilag_edges(
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
    matrix = merit_mod._causal_explanation_matrix(
        ["callee", "other"],
        ["callee", "caller", "other"],
    )
    edges = [("callee", "caller", 0.42, "latency", 1)]

    out = merit_mod._apply_trace_topology_edges(
        matrix,
        ["callee", "other"],
        ["callee", "caller", "other"],
        edges,
    )

    assert out[0, 1] == pytest.approx(0.42)
    assert out[0, 0] == 1.0


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
    ranks = merit_mod._rank_evidence_metrics(
        ["svc-b_mem", "svc-a_z", "svc-a_a"],
        metric_ev={"svc-b_mem": 1.0, "svc-a_z": 0.1, "svc-a_a": 0.9},
        svc_anom_dict={"svc-a": 1.0, "svc-b": 0.2},
        rank_mode="service_evidence",
    )

    assert ranks == ["svc-a_a", "svc-a_z", "svc-b_mem"]


def test_metric_evidence_ranking_uses_metric_score_and_stable_name_tie_break():
    ranks = merit_mod._rank_evidence_metrics(
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
    assert "root_service" not in result["diagnostics"]
    assert "root_candidate_recall" not in result["diagnostics"]


def test_no_metric_modality_uses_trace_for_candidates_and_banzhaf():
    trace_evidence = {"svc-a": 0.2, "svc-b": 0.9}
    first = _gabanzhaf_core(
        _simple_core_frame(),
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={
            "use_metrics": False,
            "trace_service_evidence": trace_evidence,
            "return_diagnostics": True,
        },
    )
    metric_values_swapped = _simple_core_frame().copy()
    metric_values_swapped[["svc-a_cpu", "svc-b_cpu"]] = (
        metric_values_swapped[["svc-b_cpu", "svc-a_cpu"]].to_numpy()
    )
    second = _gabanzhaf_core(
        metric_values_swapped,
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={
            "use_metrics": False,
            "trace_service_evidence": trace_evidence,
            "return_diagnostics": True,
        },
    )

    assert first["ranks"] == second["ranks"]
    assert first["ranks"][0].startswith("svc-b_")
    assert first["diagnostics"]["configuration"]["use_metrics"] is False
    assert set(first["diagnostics"]["metric_evidence"].values()) == {0.0}
    assert first["diagnostics"]["service_banzhaf"]


def test_no_metric_modality_keeps_constant_columns_as_schema_only():
    data = _simple_core_frame().assign(svc_c_mem=7.0)

    result = _gabanzhaf_core(
        data,
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={
            "use_metrics": False,
            "trace_service_evidence": {"svc-a": 0.2, "svc-b": 0.9},
            "return_diagnostics": True,
        },
    )

    assert "svc_c_mem" in result["node_names"]
    assert result["diagnostics"]["metric_evidence"]["svc_c_mem"] == 0.0


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
    result = merit_mod.merit(
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


@pytest.mark.parametrize(
    "rank_mode",
    [
        "service_evidence",
        "metric_evidence",
        "trace_evidence",
        "service_evidence_trace",
    ],
)
def test_gabanzhaf_core_evidence_modes_skip_banzhaf(monkeypatch, rank_mode):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("causal game must not run in evidence-only modes")

    for helper in [
        "_service_latency_scores",
        "_service_latency_family_scores",
        "_coverage_parents",
        "_causal_edges",
        "_select_exact_candidates",
        "_causal_explanation_matrix",
        "_banzhaf_power",
    ]:
        monkeypatch.setattr(merit_mod, helper, fail_if_called)
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
    assert result["candidate_pool"] == []
    assert result["causal_edges"] == []


def test_direct_metric_trace_core_reports_service_score_sum_and_ranking():
    result = _gabanzhaf_core(
        _simple_core_frame(),
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={
            "rank_mode": "service_evidence_trace",
            "trace_service_evidence": {"svc-a": 0.0, "svc-b": 1.0},
            "return_diagnostics": True,
            "return_stage_timing": True,
        },
    )

    metric_scores = result["direct_metric_service_evidence"]
    trace_scores = result["direct_trace_service_evidence"]
    combined_scores = result["direct_service_scores"]
    for service in combined_scores:
        assert combined_scores[service] == pytest.approx(
            metric_scores.get(service, 0.0) + trace_scores.get(service, 0.0)
        )

    assert result["service_ranks"] == sorted(
        result["service_ranks"],
        key=lambda service: -combined_scores[service],
    )
    assert set(result["stage_timing"]) == {
        "preprocess",
        "metric_evidence",
        "ranking",
    }


def test_gabanzhaf_entry_reads_banzhaf_mode_env(monkeypatch):
    captured_kwargs = {}

    def fake_core(data, inject_time, dataset, anomalies, kwargs):
        captured_kwargs.update(kwargs)
        return {"ranks": []}

    monkeypatch.setenv("GABANZHAF_BANZHAF_MODE", "bruteforce")
    monkeypatch.setattr(merit_mod, "_merit_core", fake_core)

    merit_mod.merit(
        pd.DataFrame({"time": [0, 1], "svc_cpu": [0.0, 1.0]}),
        inject_time=1,
    )

    assert captured_kwargs["banzhaf_mode"] == "bruteforce"


def test_merit_entry_reads_disable_metrics_env(monkeypatch):
    captured_kwargs = {}

    def fake_core(data, inject_time, dataset, anomalies, kwargs):
        captured_kwargs.update(kwargs)
        return {"ranks": []}

    monkeypatch.setenv("GABANZHAF_DISABLE_METRICS", "1")
    monkeypatch.setattr(merit_mod, "_merit_core", fake_core)

    merit_mod.merit(
        pd.DataFrame({"time": [0, 1], "svc_cpu": [0.0, 1.0]}),
        inject_time=1,
    )

    assert captured_kwargs["use_metrics"] is False


def test_merit_no_trace_skips_all_trace_file_reads(tmp_path, monkeypatch):
    case_dir = tmp_path / "opaque" / "1"
    case_dir.mkdir(parents=True)
    metric_path = case_dir / "simple_metrics.csv"
    data = _simple_core_frame()
    data.to_csv(metric_path, index=False)
    args = type("Args", (), {"data_path": str(metric_path)})()

    def fail_if_trace_read(*_args, **_kwargs):
        raise AssertionError("no-trace variant must not inspect trace files")

    monkeypatch.setattr(merit_mod, "_build_call_graph_from_traces", fail_if_trace_read)
    monkeypatch.setattr(merit_mod, "_trace_topology_from_case_dir", fail_if_trace_read)
    monkeypatch.setenv("GABANZHAF_GRAPH_SOURCE", "must-not-be-parsed")

    result = merit_mod.merit(
        data,
        inject_time=6,
        dataset="unit",
        args=args,
        use_trace=False,
        return_diagnostics=True,
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["configuration"]["use_trace"] is False
    assert diagnostics["trace_topology_edges"] == []
    assert diagnostics["trace_service_evidence"] == {}
    assert diagnostics["call_graph_edges"] == []
    assert diagnostics["graph_provenance"]["source"] == "disabled_no_trace"
    assert diagnostics["game_candidates"]
    assert diagnostics["service_banzhaf"]
    assert result["ranks"]


@pytest.mark.parametrize("explicit_kwarg", [False, True])
def test_merit_no_trace_scrubs_caller_supplied_trace_evidence(
    monkeypatch, explicit_kwarg
):
    if not explicit_kwarg:
        monkeypatch.setenv("GABANZHAF_DISABLE_TRACE", "1")

    result = merit_mod.merit(
        _simple_core_frame(),
        inject_time=6,
        dataset="unit",
        call_graph={"svc-a": ["svc-b"]},
        trace_topology_edges=[("svc-a", "svc-b", 1.0, "latency", 0)],
        trace_service_evidence={"svc-b": 1.0},
        trace_latency_evidence={"svc-b": 1.0},
        trace_error_evidence={"svc-b": 1.0},
        trace_stage_timing={"trace_latency": 1.0},
        **({"use_trace": False} if explicit_kwarg else {}),
        return_diagnostics=True,
        return_stage_timing=True,
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["configuration"]["use_trace"] is False
    assert diagnostics["trace_topology_edges"] == []
    assert diagnostics["trace_service_evidence"] == {}
    assert diagnostics["trace_latency_evidence"] == {}
    assert diagnostics["trace_error_evidence"] == {}
    assert diagnostics["trace_stage_timing"] == {}
    assert diagnostics["call_graph_edges"] == []
    assert diagnostics["graph_provenance"]["source"] == "disabled_no_trace"
    assert diagnostics["causal_edges"] == []
    assert diagnostics["game_candidates"]
    assert diagnostics["coverage_strength_matrix"]
    assert diagnostics["service_banzhaf"]
    assert result["ranks"]
    assert {"candidates", "banzhaf", "ranking"}.issubset(result["stage_timing"])


def test_victim_factors_demotes_service_with_abnormal_upstream():
    # Call chain A -> B -> C (cause->effect edges). B and C abnormal, A normal.
    # C has an abnormal predecessor (B) => victim, self-coverage demoted to floor.
    # B's only predecessor (A) is normal => root, keeps 1.0 (absent from dict).
    _victim_factors = merit_mod._victim_factors
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
        "RCAEval", "e2e", "merit.py",
    )
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    forbidden = [
        "ground_truth", "root_cause", "root_service", "case_id", "if dataset",
        "re2-ob", "re2-ss", "re2-tt", "DELAY", "LOSS",
    ]
    for token in forbidden:
        assert token not in source


def test_pre_fault_graph_excludes_post_fault_span_edges(tmp_path):
    pd.DataFrame(
        {
            "spanID": ["pre-p", "pre-c", "post-p", "post-c"],
            "parentSpanID": ["", "pre-p", "", "post-p"],
            "serviceName": ["caller", "pre-callee", "caller", "post-callee"],
            "startTime": [1_000_000, 2_000_000, 7_000_000, 8_000_000],
            "duration": [0, 0, 0, 0],
        }
    ).to_csv(tmp_path / "traces.csv", index=False)
    boundary = {
        "start": 0,
        "end": 10,
        "inject_time": 5,
        "mode": "full_record_retrospective",
        "version": "paper-full-record-v1",
    }

    graph = merit_mod._build_call_graph_from_traces(
        str(tmp_path),
        {"caller", "pre-callee", "post-callee"},
        cache_key=("per-case", str(tmp_path), "paper-full-record-v1"),
        boundary=boundary,
        pre_fault=True,
    )

    assert graph == {"caller": ["pre-callee"]}


def test_empty_call_graph_is_not_cached(tmp_path):
    cache_key = ("unit-empty", str(tmp_path), "paper-full-record-v1")
    assert merit_mod._build_call_graph_from_traces(
        str(tmp_path), cache_key=cache_key
    ) == {}
    pd.DataFrame(
        {
            "spanID": ["p", "c"],
            "parentSpanID": ["", "p"],
            "serviceName": ["caller", "callee"],
        }
    ).to_csv(tmp_path / "traces.csv", index=False)

    assert merit_mod._build_call_graph_from_traces(
        str(tmp_path), cache_key=cache_key
    ) == {"caller": ["callee"]}


def test_local_contributions_sum_to_each_closed_form_banzhaf_score():
    result = _gabanzhaf_core(
        _simple_core_frame(),
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={"return_diagnostics": True},
    )

    diagnostics = result["diagnostics"]
    for candidate, targets in diagnostics["player_target_contributions"].items():
        assert sum(targets.values()) == pytest.approx(
            diagnostics["service_banzhaf"][candidate]
        )
    assert diagnostics["coverage_strength_matrix"] == diagnostics[
        "causal_probability_matrix"
    ]
    assert any(
        value > 0
        for targets in diagnostics["edge_removal_utility_drop"].values()
        for value in targets.values()
    )


def test_trace_negative_controls_are_deterministic_and_degree_matched():
    series = {
        "a": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "b": np.asarray([3.0, 2.0, 1.0, 0.0]),
    }
    first = merit_mod._time_shuffle_trace_series(series, "unit")
    second = merit_mod._time_shuffle_trace_series(series, "unit")
    assert all(np.array_equal(first[key], second[key]) for key in series)
    assert any(not np.array_equal(first[key], series[key]) for key in series)

    edges = [
        ("a", "c", 0.7, "latency", 1),
        ("a", "d", 0.6, "latency", 2),
        ("b", "d", 0.8, "error", 0),
        ("b", "e", 0.5, "error", 1),
    ]
    randomized = merit_mod._degree_matched_random_edges(edges)

    def degrees(items):
        outgoing = {}
        incoming = {}
        for source, target, *_ in items:
            outgoing[source] = outgoing.get(source, 0) + 1
            incoming[target] = incoming.get(target, 0) + 1
        return outgoing, incoming

    assert degrees(randomized) == degrees(edges)
    assert randomized == merit_mod._degree_matched_random_edges(edges)

    frame = pd.DataFrame({"time": range(20), "value": range(20)})
    dropped = merit_mod._deterministic_row_dropout(frame, 0.5, "unit")
    assert dropped.equals(
        merit_mod._deterministic_row_dropout(frame, 0.5, "unit")
    )
    assert merit_mod._deterministic_row_dropout(
        frame, 1.0, "unit"
    ).empty

    noisy = merit_mod._add_trace_series_noise(series, 0.3, "unit")
    assert all(
        np.array_equal(noisy[key], merit_mod._add_trace_series_noise(
            series, 0.3, "unit"
        )[key])
        for key in series
    )


def test_metric_component_controls_change_only_the_targeted_evidence_stage():
    normal = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    anomalous = np.asarray([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0]])
    cols = ["a_cpu", "b_cpu"]
    fusion = _GabanzhafConfig.from_kwargs(
        {"use_early_full_fusion": True, "early_window_min_len": 1}
    )
    full_only = _GabanzhafConfig.from_kwargs({"use_early_full_fusion": False})

    fused_scores = merit_mod._metric_evidence(
        normal, anomalous, cols, config=fusion
    )
    full_scores = merit_mod._metric_evidence(
        normal, anomalous, cols, config=full_only
    )

    assert fused_scores != full_scores
    dominance_cols = [
        f"{service}_{metric}"
        for service in ("a", "b", "c", "d")
        for metric in ("cpu", "mem", "disk")
    ]
    dominance_scores = {
        "a_cpu": 1.0, "a_mem": 0.0, "a_disk": 0.0,
        "b_cpu": 0.8, "b_mem": 0.8, "b_disk": 0.8,
        "c_cpu": 0.0, "c_mem": 0.9, "c_disk": 0.0,
        "d_cpu": 0.0, "d_mem": 0.0, "d_disk": 0.9,
    }
    with_dominance = merit_mod._service_evidence(
        dominance_cols,
        dominance_scores,
        use_metric_type_dominance=True,
    )
    without_dominance = merit_mod._service_evidence(
        dominance_cols,
        dominance_scores,
        use_metric_type_dominance=False,
    )
    assert with_dominance != without_dominance


def test_blend_weights_default_to_the_previously_hardcoded_values():
    """Parameterising the evidence blends must not change default behaviour.

    Before 2026-08-31 these weights were literals inside _metric_evidence and
    _service_evidence. This pins the default path to the same arithmetic.
    """
    import importlib

    import numpy as np

    merit = importlib.import_module("RCAEval.e2e.merit")
    cfg = merit._GabanzhafConfig.from_kwargs({})

    assert (cfg.ev_w_mag, cfg.ev_w_persistence, cfg.ev_w_peak) == (0.55, 0.30, 0.15)
    assert (cfg.early_w_mag, cfg.early_w_peak) == (0.70, 0.30)
    assert (cfg.fusion_w_early, cfg.fusion_w_full) == (0.60, 0.40)
    assert (cfg.agg_w_p95, cfg.agg_w_top3, cfg.agg_w_max) == (0.50, 0.30, 0.20)
    assert (cfg.svc_w_agg, cfg.svc_w_dominance) == (0.70, 0.30)

    # a weight group already summing to 1 must pass through untouched
    np.testing.assert_allclose(
        merit._norm_weights((0.55, 0.30, 0.15), (1.0, 0.0, 0.0)), [0.55, 0.30, 0.15]
    )
    # an arbitrary group is renormalised, so any non-negative sweep value is legal
    np.testing.assert_allclose(
        merit._norm_weights((0.75, 0.30, 0.15), (1.0, 0.0, 0.0)), [0.625, 0.25, 0.125]
    )
    # degenerate input falls back instead of dividing by zero
    np.testing.assert_allclose(
        merit._norm_weights((0.0, 0.0, 0.0), (0.55, 0.30, 0.15)), [0.55, 0.30, 0.15]
    )
    np.testing.assert_allclose(
        merit._norm_weights((float("nan"), -1.0), (0.7, 0.3)), [0.7, 0.3]
    )


def test_service_evidence_weights_are_configurable():
    """svc_w_dominance actually moves the aggregate/dominance balance."""
    import importlib

    merit = importlib.import_module("RCAEval.e2e.merit")
    # three services: with only two, _norm_dict collapses the result to
    # {top: 1.0, other: 0.0} and any weighting looks identical.
    cols = [
        "svc-a_cpu", "svc-a_mem",
        "svc-b_cpu", "svc-b_mem",
        "svc-c_cpu", "svc-c_mem",
    ]
    metric_ev = {
        "svc-a_cpu": 0.9, "svc-a_mem": 0.1,
        "svc-b_cpu": 0.2, "svc-b_mem": 0.8,
        "svc-c_cpu": 0.5, "svc-c_mem": 0.45,
    }

    base = merit._service_evidence(
        cols, metric_ev, config=merit._GabanzhafConfig.from_kwargs({})
    )
    shifted = merit._service_evidence(
        cols, metric_ev, config=merit._GabanzhafConfig.from_kwargs({"svc_w_dominance": 0.9})
    )
    assert base != shifted, "svc_w_dominance had no effect on service evidence"
