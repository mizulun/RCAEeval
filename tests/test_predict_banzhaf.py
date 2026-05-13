import numpy as np
import pandas as pd

import RCAEval.e2e.predict_banzhaf as predict_banzhaf_module
from RCAEval.e2e.predict_banzhaf import (
    _build_adjacency,
    _build_service_health_scores,
    _candidate_services,
    _forecast_service,
    _granger_edge_weight,
    _mc_banzhaf,
    _norm_dict,
    _parse_service_metric,
    _read_trace_edges,
    _service_groups,
    predict_banzhaf,
)


def test_parse_service_metric_uses_last_underscore():
    assert _parse_service_metric("ts_ui_dashboard_latency") == (
        "ts_ui_dashboard",
        "latency",
    )
    assert _parse_service_metric("frontend") == ("frontend", "unknown")


def test_norm_dict_is_deterministic_for_constant_values():
    assert _norm_dict({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}


def test_service_groups_excludes_time_and_groups_by_service():
    cols = ["time", "svc_a_cpu", "svc_a_mem", "svc_b_latency"]
    assert _service_groups(cols) == {
        "svc_a": ["svc_a_cpu", "svc_a_mem"],
        "svc_b": ["svc_b_latency"],
    }


def test_build_service_health_scores_returns_one_series_per_service():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "svc_a_cpu": np.r_[rng.normal(0, 0.1, 40), rng.normal(3, 0.1, 40)],
            "svc_a_mem": np.r_[rng.normal(0, 0.1, 40), rng.normal(2, 0.1, 40)],
            "svc_b_cpu": rng.normal(0, 0.1, n),
        }
    )
    groups = _service_groups(df.columns)

    scores = _build_service_health_scores(df, groups, split_idx=40)

    assert set(scores) == {"svc_a", "svc_b"}
    assert scores["svc_a"].shape == (n,)
    assert float(np.mean(scores["svc_a"][40:])) > float(np.mean(scores["svc_a"][:40]))


def test_granger_edge_weight_detects_synthetic_direction():
    rng = np.random.default_rng(1)
    n = 120
    a = rng.normal(0, 1, n)
    b = np.zeros(n)
    noise = rng.normal(0, 0.05, n)
    for t in range(2, n):
        b[t] = 0.8 * a[t - 1] + noise[t]

    ab = _granger_edge_weight(a, b, max_lag=2, alpha=0.05)
    ba = _granger_edge_weight(b, a, max_lag=2, alpha=0.05)

    assert ab > 0.5
    assert ab > ba


def test_mc_banzhaf_promotes_root_over_downstream_victim():
    anomaly = np.array([0.8, 1.0], dtype=float)
    adj = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    values = _mc_banzhaf(anomaly, adj, n_samples=2000, random_state=7)

    assert values.shape == (2,)
    assert values[0] > values[1]


def test_forecast_service_ignores_errors_below_normal_median(monkeypatch):
    y = np.zeros(8, dtype=float)
    base = np.array([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0])
    residual = np.zeros_like(y)

    def fake_stl_decompose_health(health, split_idx):
        return base, residual, True

    def fake_residual_prediction(residual, split_idx, max_lag, random_state):
        return np.zeros_like(residual)

    monkeypatch.setattr(
        predict_banzhaf_module,
        "_stl_decompose_health",
        fake_stl_decompose_health,
    )
    monkeypatch.setattr(
        predict_banzhaf_module,
        "_elasticnet_residual_prediction",
        fake_residual_prediction,
    )

    _, _, score = _forecast_service(y, split_idx=4, max_lag=2, random_state=0)

    assert score == 0.0


def test_candidate_services_keeps_low_anomaly_sli_under_small_cap():
    services = ["root", "victim", "noisy"]
    anomaly_scores = {"root": 0.9, "victim": 0.01, "noisy": 0.8}

    candidates = _candidate_services(
        services,
        anomaly_scores,
        sli="victim_latency",
        trace_edges=[],
        top_k=1,
        max_candidate=1,
    )

    assert candidates == ["victim"]


def test_build_adjacency_caps_incoming_edges_after_trace_fusion():
    candidates = ["a", "b", "c", "d"]
    residuals = {svc: np.zeros(20, dtype=float) for svc in candidates}
    trace_edges = [("a", "d", 1.0), ("b", "d", 0.8), ("c", "d", 0.6)]

    adj = _build_adjacency(
        candidates,
        residuals,
        trace_edges,
        max_lag=2,
        alpha=0.05,
        max_in_edges=1,
        trace_prior=1.0,
    )

    assert np.count_nonzero(adj[:, 3]) == 1
    assert adj[0, 3] == 1.0
    assert adj[1, 3] == 0.0
    assert adj[2, 3] == 0.0


def test_read_trace_edges_does_not_cross_join_reused_span_ids(tmp_path):
    traces = pd.DataFrame(
        [
            {
                "traceID": "trace-a",
                "spanID": "1",
                "parentSpanID": "",
                "serviceName": "frontend",
            },
            {
                "traceID": "trace-a",
                "spanID": "2",
                "parentSpanID": "1",
                "serviceName": "checkout",
            },
            {
                "traceID": "trace-b",
                "spanID": "1",
                "parentSpanID": "",
                "serviceName": "payments",
            },
            {
                "traceID": "trace-b",
                "spanID": "2",
                "parentSpanID": "1",
                "serviceName": "shipping",
            },
        ]
    )
    traces.to_csv(tmp_path / "traces.csv", index=False)

    edges = _read_trace_edges(str(tmp_path))

    assert ("checkout", "frontend", 1.0) in edges
    assert ("shipping", "payments", 1.0) in edges
    assert ("checkout", "payments", 1.0) not in edges
    assert ("shipping", "frontend", 1.0) not in edges


def _make_predict_banzhaf_frame(n=100, split=50):
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "time": np.arange(n),
            "root_cpu": np.r_[
                rng.normal(0, 0.1, split), rng.normal(4, 0.2, n - split)
            ],
            "root_mem": np.r_[
                rng.normal(0, 0.1, split), rng.normal(3, 0.2, n - split)
            ],
            "victim_latency": np.r_[
                rng.normal(0, 0.1, split), rng.normal(2, 0.2, n - split)
            ],
            "quiet_cpu": rng.normal(0, 0.1, n),
        }
    )


def test_predict_banzhaf_returns_rcaeval_format_trace_less():
    df = _make_predict_banzhaf_frame()

    result = predict_banzhaf(
        df,
        inject_time=50,
        dataset=None,
        banzhaf_samples=200,
        top_k_svcs=10,
        max_candidate_svcs=10,
        random_state=3,
    )

    assert "ranks" in result
    assert "node_names" in result
    assert "service_ranks" in result
    assert "service_node_names" in result
    assert "adj" in result
    assert set(result["ranks"]) == {
        "root_cpu",
        "root_mem",
        "victim_latency",
        "quiet_cpu",
    }
    assert result["node_names"] == [
        "root_cpu",
        "root_mem",
        "victim_latency",
        "quiet_cpu",
    ]
    assert result["service_node_names"] == ["quiet", "root", "victim"]
    assert result["adj"].shape == (3, 3)
    assert result["ranks"][0].startswith("root_")


def test_predict_banzhaf_is_deterministic_for_fixed_seed():
    df = _make_predict_banzhaf_frame()
    kwargs = {
        "inject_time": 50,
        "dataset": None,
        "banzhaf_samples": 200,
        "random_state": 123,
    }

    first = predict_banzhaf(df, **kwargs)
    second = predict_banzhaf(df, **kwargs)

    assert first["ranks"] == second["ranks"]
    assert first["service_ranks"] == second["service_ranks"]
    assert np.allclose(first["adj"], second["adj"])
