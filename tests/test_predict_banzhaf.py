import numpy as np
import pandas as pd

from RCAEval.e2e.predict_banzhaf import (
    _build_service_health_scores,
    _granger_edge_weight,
    _mc_banzhaf,
    _norm_dict,
    _parse_service_metric,
    _service_groups,
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
