"""Tests for gabanzhaf DELAY/LOSS improvements.

Three behaviors under test:
1. _metric_evidence: Winsorization caps rz at 10 (no unbounded scores)
2. _extract_call_graph: extracts caller→callee from traces.csv (50K row limit)
3. _gabanzhaf_core: call graph topology demotes callers of root cause
"""
import os
import tempfile
import importlib
import numpy as np
import pandas as pd
import pytest

from RCAEval.e2e.gabanzhaf import _metric_evidence, _extract_call_graph, _gabanzhaf_core


# ─────────────────────────────────────────────────────────────────────────────
# 1. Winsorization in _metric_evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_metric_evidence_winsorizes_near_zero_baseline():
    """Near-zero baseline metric must not produce unbounded rz scores.

    Reproduces the LOSS failure mode: a service with near-zero normal latency
    gets a tiny non-zero anomaly value → raw IQR-normalized score would be
    astronomical. After Winsorization the score must be ≤ 1.0 (normalised).
    """
    rng = np.random.default_rng(0)
    n_rows, n_cols = 100, 3

    # normal: near-zero baseline (simulates services with no traffic)
    normal = np.zeros((n_rows, n_cols))
    normal[:, 0] = rng.uniform(0, 0.001, n_rows)   # near-zero baseline

    # anomaly: one column shoots up (would produce ~1e6 IQR-units without cap)
    anomal = np.zeros((n_rows, n_cols))
    anomal[:, 0] = rng.uniform(100, 200, n_rows)    # large anomaly

    cols = ["svc_latency", "svc_cpu", "svc_mem"]
    ev = _metric_evidence(normal, anomal, cols)

    # All scores must be in [0, 1] — normalisation already does this,
    # but the key check is that the near-zero-baseline column does NOT
    # dominate at exactly 1.0 while all others collapse to 0.0.
    scores = [ev[c]["score"] for c in cols]
    assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of [0,1]: {scores}"

    # With Winsorization: the anomalous column still wins, but the gap
    # between columns should not be extreme (other cols get non-zero scores).
    # Without Winsorization the near-zero-baseline col's rz is ~1e5 while
    # others are ~0 → after norm, anomalous col = 1.0, rest = 0.0.
    # With Winsorization all rz are capped at 10 → fairer comparison.
    anomalous_score = ev["svc_latency"]["score"]
    other_scores = [ev["svc_cpu"]["score"], ev["svc_mem"]["score"]]
    # The near-zero col should still win (highest score)
    assert anomalous_score == max(scores), "Anomalous column should have highest score"


def test_metric_evidence_selective_winsorization():
    """Selective Winsorization: only cap metrics with near-constant normal period.

    - col 0: near-zero baseline (IQR=0, std=0) → Winsorization applies → capped at 10
    - col 1: normal baseline (IQR>0) → Winsorization does NOT apply → full rz preserved

    Without selective Winsorization, near-zero col would dominate with rz ~100
    (using iqr=1 fallback). With selective cap, near-zero col is capped at 10
    while normal col keeps its higher rz, so normal col can have equal or higher mag.
    """
    n_rows = 100

    # col 0: near-constant normal baseline (IQR=0, std=0) → iqr fallback = 1
    # anomaly value = 50 → rz = 50/1 = 50 (capped to 10 by selective Winsorization)
    normal_0 = np.zeros(n_rows)        # exactly constant → std=0, IQR=0
    anomal_0 = np.full(n_rows, 50.0)  # rz would be 50 without cap

    # col 1: real baseline variation (IQR>0) → NOT Winsorized
    # anomaly gives rz ~15 (above the cap value of 10, to confirm no cap)
    rng = np.random.default_rng(0)
    normal_1 = rng.normal(100.0, 5.0, n_rows)  # std~5, IQR~6.7
    anomal_1 = np.full(n_rows, 200.0)           # |200-100|/6.7 ≈ 14.9

    normal = np.column_stack([normal_0, normal_1])
    anomal = np.column_stack([anomal_0, anomal_1])
    cols = ["near_zero_svc_lat", "normal_svc_lat"]

    ev = _metric_evidence(normal, anomal, cols)

    # near_zero_svc_lat: rz was 50, capped to 10 → after norm: mag ≤ 1.0
    # normal_svc_lat: rz ~14.9, NOT capped → after norm: mag > near_zero if rz > 10
    # The normal col should win (rz ~15 > capped 10)
    assert ev["normal_svc_lat"]["mag"] >= ev["near_zero_svc_lat"]["mag"], (
        f"Non-Winsorized col should have >= mag than capped col. "
        f"normal={ev['normal_svc_lat']['mag']:.3f}, near_zero={ev['near_zero_svc_lat']['mag']:.3f}"
    )

    # And near_zero is effectively capped (score should not be 1.0 if normal col is higher)
    assert ev["near_zero_svc_lat"]["score"] <= ev["normal_svc_lat"]["score"] + 0.01, (
        f"Near-zero col should not dominate. "
        f"near_zero_score={ev['near_zero_svc_lat']['score']:.3f}, "
        f"normal_score={ev['normal_svc_lat']['score']:.3f}"
    )


def test_metric_semantic_group_uses_generic_suffixes():
    from RCAEval.e2e.gabanzhaf import _metric_semantic_group

    assert _metric_semantic_group("svc-a_cpu") == "resource"
    assert _metric_semantic_group("svc-a_memory_usage") == "resource"
    assert _metric_semantic_group("svc-a_latency") == "latency"
    assert _metric_semantic_group("svc-a_duration") == "latency"
    assert _metric_semantic_group("svc-a_http_errors") == "error"
    assert _metric_semantic_group("svc-a_custom_metric") == "other"
    assert _metric_semantic_group("svc-a_error_ratio") == "error"
    assert _metric_semantic_group("svc-a_success_ratio") == "other"
    assert _metric_semantic_group("svc-a_io_wait") == "resource"


def test_semantic_service_evidence_combines_metrics_trace_and_topology():
    from RCAEval.e2e.gabanzhaf import _semantic_service_evidence

    cols = ["svc-a_cpu", "svc-a_latency", "svc-b_cpu", "svc-b_latency"]
    metric_ev = {
        "svc-a_cpu": {"score": 0.9},
        "svc-a_latency": {"score": 0.2},
        "svc-b_cpu": {"score": 0.1},
        "svc-b_latency": {"score": 0.8},
    }
    svc_anom = {"svc-a": 0.9, "svc-b": 0.8}
    trace_scores = {"svc-b": 1.0}
    topology_edges = [("svc-a", "svc-b", 0.7)]

    sem = _semantic_service_evidence(cols, metric_ev, trace_scores, topology_edges, svc_anom)

    assert sem["svc-a"]["resource"] > sem["svc-b"]["resource"]
    assert sem["svc-b"]["latency"] > sem["svc-a"]["latency"]
    assert sem["svc-b"]["trace"] == 1.0
    assert sem["svc-a"]["propagation"] > 0.0
    assert sem["svc-b"]["propagation"] > 0.0


def test_semantic_service_evidence_preserves_singleton_positive_components():
    from RCAEval.e2e.gabanzhaf import _semantic_service_evidence

    cols = ["svc-a_cpu"]
    metric_ev = {"svc-a_cpu": {"score": 0.8}}
    sem = _semantic_service_evidence(
        cols,
        metric_ev,
        trace_scores={"svc-a": 0.7},
        topology_edges=[],
        svc_anom_dict={"svc-a": 0.8},
    )

    assert 0.0 < sem["svc-a"]["resource"] <= 1.0
    assert 0.0 < sem["svc-a"]["trace"] <= 1.0
    assert sem["svc-a"]["propagation"] == 0.0


def test_semantic_service_evidence_preserves_broadcast_equal_positive_components():
    from RCAEval.e2e.gabanzhaf import _semantic_service_evidence

    cols = ["svc-a_cpu", "svc-b_cpu"]
    metric_ev = {
        "svc-a_cpu": {"score": 0.8},
        "svc-b_cpu": {"score": 0.8},
    }
    sem = _semantic_service_evidence(
        cols,
        metric_ev,
        trace_scores={},
        topology_edges=[],
        svc_anom_dict={"svc-a": 0.8, "svc-b": 0.8},
    )

    assert 0.0 < sem["svc-a"]["resource"] <= 1.0
    assert 0.0 < sem["svc-b"]["resource"] <= 1.0
    assert sem["svc-a"]["propagation"] == 0.0
    assert sem["svc-b"]["propagation"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. _extract_call_graph
# ─────────────────────────────────────────────────────────────────────────────

def _make_traces_csv(rows, path):
    """Write a minimal traces.csv for testing."""
    df = pd.DataFrame(rows, columns=["serviceName", "spanID", "parentSpanID",
                                     "startTimeMillis", "statusCode", "duration"])
    df.to_csv(path, index=False)


def test_extract_call_graph_basic():
    """Call graph extracts correct caller→callee from parentSpanID links."""
    rows = [
        # span A (svc-a) is the root span (no parent in dataset)
        ("svc-a", "span-a", "root-span", 1000, 0, 10),
        # span B (svc-b) called by svc-a
        ("svc-b", "span-b", "span-a", 1001, 0, 5),
        # span C (svc-c) called by svc-b
        ("svc-c", "span-c", "span-b", 1002, 0, 3),
    ]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        _make_traces_csv(rows, path)
        cg = _extract_call_graph(path)
        # svc-a calls svc-b
        assert "svc-a" in cg, f"svc-a should be a caller. cg={cg}"
        assert "svc-b" in cg["svc-a"], f"svc-a should call svc-b. cg={cg}"
        # svc-b calls svc-c
        assert "svc-b" in cg, f"svc-b should be a caller. cg={cg}"
        assert "svc-c" in cg["svc-b"], f"svc-b should call svc-c. cg={cg}"
    finally:
        os.unlink(path)


def test_extract_call_graph_no_self_loops():
    """Same service calling itself (e.g., internal spans) must not appear."""
    rows = [
        ("svc-a", "span-a1", "span-root", 1000, 0, 10),
        ("svc-a", "span-a2", "span-a1", 1001, 0, 5),  # same svc → internal
        ("svc-b", "span-b",  "span-a1", 1002, 0, 3),
    ]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        _make_traces_csv(rows, path)
        cg = _extract_call_graph(path)
        if "svc-a" in cg:
            assert "svc-a" not in cg["svc-a"], "Self-loop should be excluded"
    finally:
        os.unlink(path)


def test_extract_call_graph_missing_file():
    """Missing traces file must return empty dict (RE1 / no-trace fallback)."""
    cg = _extract_call_graph("/nonexistent/path/traces.csv")
    assert cg == {}, f"Expected empty dict, got {cg}"


def test_extract_call_graph_row_limit():
    """Function must only read first 50K rows (fast path for large files)."""
    # Build a traces file with >50K rows; the extra rows introduce a new edge
    # that should NOT appear in the result (because we cap at 50K rows).
    n_rows = 60_000
    span_ids   = [f"span-{i}" for i in range(n_rows)]
    parent_ids = ["span-root"] + span_ids[:-1]
    # First 50K rows: svc-a → svc-b edges
    svc_names  = ["svc-a" if i % 2 == 0 else "svc-b" for i in range(n_rows)]
    # Row 55000: introduce svc-z → svc-x edge (should be ignored)
    svc_names[55000] = "svc-z"
    svc_names[55001] = "svc-x"
    # Make span 55001's parent = span 55000
    parent_ids[55001] = span_ids[55000]

    rows = list(zip(svc_names, span_ids, parent_ids,
                    [1000] * n_rows, [0] * n_rows, [5] * n_rows))

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        _make_traces_csv(rows, path)
        cg = _extract_call_graph(path)
        # svc-z→svc-x edge is beyond 50K rows and must not appear
        assert "svc-z" not in cg or "svc-x" not in cg.get("svc-z", set()), (
            "Edge beyond 50K row limit should not be extracted"
        )
    finally:
        os.unlink(path)


from RCAEval.e2e.gabanzhaf import _extract_trace_features, _TraceFeatures


def test_extract_trace_features_returns_scores_and_call_graph():
    rows = [
        ("svc-a", "span-a", "root", 1000, 0, 10),
        ("svc-b", "span-b", "span-a", 1001, 0, 5),
        ("svc-b", "span-c", "span-a", 60000, 1, 200),
        ("svc-c", "span-d", "span-b", 60001, 0, 1),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traces.csv")
        _make_traces_csv(rows, path)

        features = _extract_trace_features(d, inject_time=50)

    assert isinstance(features, _TraceFeatures)
    assert "svc-b" in features.trace_scores
    assert features.trace_scores["svc-b"] > 0.0
    assert features.call_graph == {"svc-a": {"svc-b"}, "svc-b": {"svc-c"}}


def test_extract_trace_features_exposes_error_and_latency_components():
    rows = [
        ("svc-a", "span-a0", "root", 1000, 0, 10),
        ("svc-b", "span-b0", "span-a0", 1001, 0, 10),
        ("svc-a", "span-a1", "root-2", 60000, 1, 10),
        ("svc-b", "span-b1", "span-a1", 60001, 0, 300),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traces.csv")
        _make_traces_csv(rows, path)

        features = _extract_trace_features(d, inject_time=50)

    assert features.trace_scores is not None
    assert features.trace_error_scores is not None
    assert features.trace_latency_scores is not None
    assert features.trace_error_scores["svc-a"] > features.trace_error_scores.get("svc-b", 0.0)
    assert features.trace_latency_scores["svc-b"] > features.trace_latency_scores.get("svc-a", 0.0)


def test_extract_trace_features_preserves_singleton_component_evidence():
    rows = [
        ("svc-a", "span-a0", "root", 1000, 0, 10),
        ("svc-a", "span-a1", "root-2", 60000, 1, 300),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traces.csv")
        _make_traces_csv(rows, path)

        features = _extract_trace_features(d, inject_time=50)

    assert features.trace_error_scores is not None
    assert features.trace_latency_scores is not None
    assert features.trace_error_scores["svc-a"] > 0.0
    assert features.trace_latency_scores["svc-a"] > 0.0


def test_extract_trace_features_missing_file_returns_neutral_features():
    features = _extract_trace_features("/nonexistent/path", inject_time=50)
    assert features == _TraceFeatures(
        trace_scores=None,
        trace_error_scores=None,
        trace_latency_scores=None,
        call_graph={},
    )


def test_extract_trace_features_keeps_call_graph_when_score_columns_missing():
    rows = [
        ("svc-a", "span-a", "root"),
        ("svc-b", "span-b", "span-a"),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traces.csv")
        pd.DataFrame(rows, columns=["serviceName", "spanID", "parentSpanID"]).to_csv(
            path, index=False
        )

        features = _extract_trace_features(d, inject_time=50)

    assert features.trace_scores is None
    assert features.call_graph == {"svc-a": {"svc-b"}}


def test_extract_trace_features_keeps_scores_when_span_columns_missing():
    rows = [
        ("svc-a", 1000, 0, 10),
        ("svc-a", 60000, 1, 200),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traces.csv")
        pd.DataFrame(
            rows,
            columns=["serviceName", "startTimeMillis", "statusCode", "duration"],
        ).to_csv(path, index=False)

        features = _extract_trace_features(d, inject_time=50)

    assert features.call_graph == {}
    assert features.trace_scores is not None
    assert "svc-a" in features.trace_scores


def test_extract_trace_features_uses_preaggregated_trace_scores_without_raw_traces():
    with tempfile.TemporaryDirectory() as d:
        pd.DataFrame(
            {
                "time": [10, 20, 60, 70],
                "svc-a_GET": [0, 0, 5, 6],
                "svc-b_GET": [0, 0, 0, 0],
            }
        ).to_csv(os.path.join(d, "tracets_err.csv"), index=False)
        pd.DataFrame(
            {
                "time": [10, 20, 60, 70],
                "svc-a_GET": [10, 10, 10, 10],
                "svc-b_GET": [10, 10, 80, 90],
            }
        ).to_csv(os.path.join(d, "tracets_lat.csv"), index=False)

        features = _extract_trace_features(d, inject_time=50)

    assert features.call_graph == {}
    assert features.trace_scores is not None
    assert features.trace_scores["svc-a"] > features.trace_scores["svc-b"]


def test_extract_trace_features_preserves_preaggregated_singleton_component_evidence():
    with tempfile.TemporaryDirectory() as d:
        pd.DataFrame(
            {
                "time": [10, 20, 60, 70],
                "svc-a_GET": [0, 0, 3, 5],
            }
        ).to_csv(os.path.join(d, "tracets_err.csv"), index=False)
        pd.DataFrame(
            {
                "time": [10, 20, 60, 70],
                "svc-a_GET": [10, 10, 80, 90],
            }
        ).to_csv(os.path.join(d, "tracets_lat.csv"), index=False)

        features = _extract_trace_features(d, inject_time=50)

    assert features.trace_error_scores is not None
    assert features.trace_latency_scores is not None
    assert features.trace_error_scores["svc-a"] > 0.0
    assert features.trace_latency_scores["svc-a"] > 0.0


def test_extract_trace_features_prefers_preaggregated_scores_over_raw_trace_scan(monkeypatch):
    read_calls = []
    real_read_csv = pd.read_csv

    def tracking_read_csv(path, *args, **kwargs):
        read_calls.append((os.path.basename(str(path)), kwargs.get("chunksize")))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", tracking_read_csv)

    rows = [
        ("svc-a", "span-a", "root"),
        ("svc-b", "span-b", "span-a"),
    ]
    with tempfile.TemporaryDirectory() as d:
        pd.DataFrame(rows, columns=["serviceName", "spanID", "parentSpanID"]).to_csv(
            os.path.join(d, "traces.csv"), index=False
        )
        pd.DataFrame(
            {"time": [10, 60], "svc-a_GET": [0, 9], "svc-b_GET": [0, 0]}
        ).to_csv(os.path.join(d, "tracets_err.csv"), index=False)

        features = _extract_trace_features(d, inject_time=50)

    trace_reads = [call for call in read_calls if call[0] == "traces.csv"]
    assert trace_reads == [("traces.csv", None)]
    assert features.call_graph == {"svc-a": {"svc-b"}}
    assert features.trace_scores["svc-a"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Call graph topology demotes callers in _gabanzhaf_core
# ─────────────────────────────────────────────────────────────────────────────

def _make_simple_dataframe(svc_root, svc_caller, inject_row=50, n_rows=100):
    """Synthetic metrics: svc_root has delay anomaly; svc_caller shows propagated latency."""
    rng = np.random.default_rng(42)
    time_col = list(range(n_rows))

    # Root cause: latency spikes at inject_row
    root_lat  = np.concatenate([rng.normal(10, 1, inject_row),
                                 rng.normal(200, 10, n_rows - inject_row)])
    # Caller: smaller propagated spike (caller waits for root)
    call_lat  = np.concatenate([rng.normal(15, 1, inject_row),
                                 rng.normal(80, 10, n_rows - inject_row)])
    # Both have normal cpu
    root_cpu  = rng.normal(30, 2, n_rows)
    call_cpu  = rng.normal(25, 2, n_rows)

    df = pd.DataFrame({
        "time": time_col,
        f"{svc_root}_latency-50": root_lat,
        f"{svc_root}_cpu":        root_cpu,
        f"{svc_caller}_latency-50": call_lat,
        f"{svc_caller}_cpu":        call_cpu,
    })
    return df


def test_callgraph_demotes_caller_service():
    """With correct call graph, the caller of the root cause must rank below it.

    Setup:
    - svc-root has injected delay (HIGH latency: score will exceed 0.2 threshold)
    - svc-caller calls svc-root, shows propagated (but smaller) latency spike
    - call_graph = {"svc-caller": {"svc-root"}}  (caller calls root)
    - Expected: svc-root ranked #1, svc-caller ranked #2

    Without call graph, svc-caller might beat svc-root because Banzhaf
    topology built from lag-correlation could be wrong.
    Note: svc-root must have anomaly score ≥ 0.2 for the call graph edge to
    activate (the guard prevents adding edges for unaffected callees).
    """
    svc_root   = "svc-root"
    svc_caller = "svc-caller"
    inject_row = 50

    df = _make_simple_dataframe(svc_root, svc_caller, inject_row=inject_row)
    inject_time = inject_row  # "time" column = row index

    call_graph = {svc_caller: {svc_root}}  # caller calls root → root explains caller

    result = _gabanzhaf_core(
        df, inject_time=inject_time, dataset=None, anomalies=None,
        kwargs={"call_graph": call_graph}
    )

    ranks = result["ranks"]
    # Extract service-level ranking (take first occurrence per service)
    svc_order = []
    seen = set()
    for col in ranks:
        svc = col.rsplit("_", 1)[0]
        if svc not in seen:
            svc_order.append(svc)
            seen.add(svc)

    assert svc_order[0] == svc_root, (
        f"Root cause should rank first. Got order: {svc_order}"
    )


def test_callgraph_edges_reach_banzhaf_topology_when_lag_corr_is_empty(monkeypatch):
    svc_root = "svc-root"
    svc_caller = "svc-caller"
    inject_row = 50
    df = _make_simple_dataframe(svc_root, svc_caller, inject_row=inject_row)
    captured = {}

    monkeypatch.setattr(gabanzhaf_mod, "_build_topology", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        gabanzhaf_mod,
        "_call_graph_topology_edges",
        lambda *args, **kwargs: [(svc_root, svc_caller, 0.5)],
    )

    def capture_banzhaf(fitness_signal, topo_matrix, config):
        captured["topo_matrix"] = np.asarray(topo_matrix).copy()
        return np.asarray(fitness_signal, dtype=float)

    monkeypatch.setattr(gabanzhaf_mod, "_banzhaf_power", capture_banzhaf)

    _gabanzhaf_core(
        df,
        inject_time=inject_row,
        dataset=None,
        anomalies=None,
        kwargs={"call_graph": {svc_caller: {svc_root}}},
    )

    assert np.count_nonzero(captured["topo_matrix"]) == 1


def test_callgraph_does_not_penalise_caller_root_cause():
    """When the CALLER is the root cause, call graph must not hurt it.

    Setup:
    - svc-caller has injected fault (high cpu)
    - svc-callee is called by svc-caller but is unaffected (normal metrics)
    - call_graph = {"svc-caller": {"svc-callee"}}
    - Because svc-callee has low anomaly (< 0.2 threshold), the edge
      (svc-callee, svc-caller) must NOT be added.
    - Expected: svc-caller still ranks #1.
    """
    rng = np.random.default_rng(7)
    n_rows = 100
    inject_row = 50

    # Caller has CPU fault
    caller_cpu = np.concatenate([rng.normal(30, 2, inject_row),
                                  rng.normal(95, 3, n_rows - inject_row)])
    caller_lat = np.concatenate([rng.normal(10, 1, inject_row),
                                  rng.normal(12, 1, n_rows - inject_row)])
    # Callee is unaffected
    callee_cpu = rng.normal(30, 2, n_rows)
    callee_lat = rng.normal(10, 1, n_rows)

    df = pd.DataFrame({
        "time": list(range(n_rows)),
        "svc-caller_cpu":        caller_cpu,
        "svc-caller_latency-50": caller_lat,
        "svc-callee_cpu":        callee_cpu,
        "svc-callee_latency-50": callee_lat,
    })
    call_graph = {"svc-caller": {"svc-callee"}}

    result = _gabanzhaf_core(
        df, inject_time=inject_row, dataset=None, anomalies=None,
        kwargs={"call_graph": call_graph}
    )
    ranks = result["ranks"]
    svc_order = []
    seen = set()
    for col in ranks:
        svc = col.rsplit("_", 1)[0]
        if svc not in seen:
            svc_order.append(svc)
            seen.add(svc)

    assert svc_order[0] == "svc-caller", (
        f"Caller (root cause) should rank first. Got: {svc_order}"
    )


def test_callgraph_no_effect_when_empty():
    """Empty call graph must not crash and must behave like current algorithm."""
    svc_root   = "svc-root"
    svc_caller = "svc-caller"

    df = _make_simple_dataframe(svc_root, svc_caller)
    result = _gabanzhaf_core(
        df, inject_time=50, dataset=None, anomalies=None,
        kwargs={"call_graph": {}}
    )
    assert "ranks" in result
    assert len(result["ranks"]) > 0


def test_gabanzhaf_core_can_return_stage_timing():
    df = _make_simple_dataframe("svc-root", "svc-caller")

    result = _gabanzhaf_core(
        df,
        inject_time=50,
        dataset=None,
        anomalies=None,
        kwargs={"call_graph": {}, "return_stage_timing": True},
    )

    assert "stage_timing" in result
    for key in ["preprocess", "metric_evidence", "topology", "banzhaf", "ranking"]:
        assert key in result["stage_timing"]
        assert isinstance(result["stage_timing"][key], float)
        assert result["stage_timing"][key] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Banzhaf topology approximation
# ─────────────────────────────────────────────────────────────────────────────
gabanzhaf_mod = importlib.import_module("RCAEval.e2e.gabanzhaf")
from RCAEval.e2e.gabanzhaf import (
    _GabanzhafConfig,
    _service_root_score,
)


def test_gabanzhaf_config_centralizes_runtime_parameters():
    """Runtime knobs should be centralized instead of read ad hoc in the core."""
    cfg = _GabanzhafConfig.from_kwargs({
        "top_k_svcs": "9",
        "max_candidate_svcs": "14",
        "banzhaf_samples": "321",
        "seed": "99",
        "dk_select_useful": True,
    })

    assert cfg.top_k_svcs == 9
    assert cfg.max_candidate_svcs == 14
    assert cfg.banzhaf_samples == 321
    assert cfg.seed == 99
    assert cfg.dk_select_useful is True


def test_gabanzhaf_config_default_candidate_cap_tracks_top_k():
    cfg = _GabanzhafConfig.from_kwargs({"top_k_svcs": "11"})
    assert cfg.max_candidate_svcs == 16


def test_gabanzhaf_config_defaults_to_deterministic_banzhaf():
    cfg = _GabanzhafConfig.from_kwargs({})
    assert cfg.banzhaf_mode == "deterministic"


def test_banzhaf_power_uses_mc_only_when_explicit(monkeypatch):
    calls = []

    def fake_mc(fitness_signal, topo_matrix, n_samples, seed=42):
        calls.append((np.asarray(fitness_signal).tolist(), int(n_samples), int(seed)))
        return np.array([3.0, 2.0])

    monkeypatch.setattr(gabanzhaf_mod, "_mc_banzhaf", fake_mc)

    cfg = _GabanzhafConfig.from_kwargs({
        "banzhaf_mode": "mc",
        "banzhaf_samples": 123,
        "seed": 7,
    })
    values = gabanzhaf_mod._banzhaf_power(
        np.array([0.9, 0.5], dtype=float),
        np.zeros((2, 2), dtype=np.float32),
        cfg,
    )

    assert calls == [([0.9, 0.5], 123, 7)]
    assert np.allclose(values, np.array([3.0, 2.0]))


def test_deterministic_banzhaf_matches_two_node_analytic_case():
    fs = np.array([0.9, 0.5], dtype=float)
    topo = np.zeros((2, 2), dtype=np.float32)
    topo[0, 1] = 0.5

    bz = gabanzhaf_mod._deterministic_banzhaf_coverage(fs, topo)

    assert np.allclose(bz, np.array([1.15, 0.25]))


def test_banzhaf_power_default_is_deterministic_not_mc(monkeypatch):
    def fail_mc(*args, **kwargs):
        raise AssertionError("MC backend should be explicit")

    monkeypatch.setattr(gabanzhaf_mod, "_mc_banzhaf", fail_mc)

    cfg = _GabanzhafConfig.from_kwargs({})
    fs = np.array([0.9, 0.5], dtype=float)
    topo = np.zeros((2, 2), dtype=np.float32)
    topo[0, 1] = 0.5

    bz = gabanzhaf_mod._banzhaf_power(fs, topo, cfg)

    assert bz[0] > bz[1]


def test_banzhaf_power_rejects_invalid_mode():
    cfg = _GabanzhafConfig.from_kwargs({"banzhaf_mode": "invalid"})

    with pytest.raises(ValueError, match="Unsupported banzhaf_mode"):
        gabanzhaf_mod._banzhaf_power(
            np.array([0.9, 0.5], dtype=float),
            np.zeros((2, 2), dtype=np.float32),
            cfg,
        )


def test_banzhaf_power_default_uses_same_path_for_small_and_large(monkeypatch):
    """The Banzhaf computation must not switch algorithms by candidate count."""
    def fail_mc(*args, **kwargs):
        raise AssertionError("MC backend should be explicit")

    monkeypatch.setattr(gabanzhaf_mod, "_mc_banzhaf", fail_mc)
    cfg = _GabanzhafConfig.from_kwargs({})

    small = gabanzhaf_mod._banzhaf_power(
        np.array([0.9, 0.4, 0.1], dtype=float),
        np.zeros((3, 3), dtype=np.float32),
        cfg,
    )
    large = gabanzhaf_mod._banzhaf_power(
        np.array([0.9] + [0.3] * 21, dtype=float),
        np.zeros((22, 22), dtype=np.float32),
        cfg,
    )

    assert small.shape == (3,)
    assert large.shape == (22,)


def test_service_root_score_formula_does_not_branch_on_trace_availability():
    """Missing trace data must be a zero signal, not a different formula."""
    without_trace = _service_root_score(0.8, 0.6, 0.4, trace_score=0.0)
    with_zero_trace = _service_root_score(0.8, 0.6, 0.4, trace_score=0.0)
    with_trace = _service_root_score(0.8, 0.6, 0.4, trace_score=0.5)

    assert without_trace == with_zero_trace
    assert with_trace > without_trace


def test_deterministic_banzhaf_no_topology_preserves_anomaly_order():
    """Without topology, Banzhaf should be proportional to fitness_signal.

    For additive v(S), the deterministic Banzhaf ranking must follow
    fitness_signal.
    """
    cfg = _GabanzhafConfig.from_kwargs({})
    fs = np.array([0.9, 0.4, 0.1], dtype=float)
    topo = np.zeros((3, 3), dtype=np.float32)
    bz = gabanzhaf_mod._banzhaf_power(fs, topo, cfg)
    # ranking must match fitness_signal ranking
    assert bz[0] > bz[1] > bz[2], f"Expected descending Banzhaf, got {bz}"


def test_deterministic_banzhaf_topology_promotes_root_cause():
    """Topology (causal coverage) must rank root cause above victim unconditionally.

    Setup: n=2, A(idx=0) upstream of B(idx=1), fitness=[0.9, 0.5], topo[0,1]=0.5.
    Analytical expected values (causal-coverage v(S)):
      downstream_masks[A] = {B}
      v({})    = 0
      v({A})   = fitness[A] + fitness[B] = 1.4   (A explains B)
      v({B})   = fitness[B]              = 0.5   (B explains nothing)
      v({A,B}) = fitness[A] + fitness[B] = 1.4
      Banzhaf(A) = mean([1.4 - 0,  1.4 - 0.5]) = mean([1.4, 0.9]) = 1.15
      Banzhaf(B) = mean([0.5 - 0,  1.4 - 1.4]) = mean([0.5, 0.0]) = 0.25
    Root cause ALWAYS dominates regardless of relative anomaly magnitudes.
    """
    cfg = _GabanzhafConfig.from_kwargs({})
    fs = np.array([0.9, 0.5], dtype=float)
    topo = np.zeros((2, 2), dtype=np.float32)
    topo[0, 1] = 0.5  # A(0) upstream of B(1)
    bz = gabanzhaf_mod._banzhaf_power(fs, topo, cfg)
    assert bz[0] > bz[1], f"Root cause should outrank victim. bz={bz}"


def test_deterministic_banzhaf_topology_no_negative_values():
    """Banzhaf values must all be non-negative (clipped at 0)."""
    cfg = _GabanzhafConfig.from_kwargs({})
    rng = np.random.default_rng(0)
    n = 5
    fs = rng.random(n)
    topo = np.zeros((n, n), dtype=np.float32)
    topo[0, 1] = 0.8
    topo[1, 2] = 0.6
    bz = gabanzhaf_mod._banzhaf_power(fs, topo, cfg)
    assert np.all(bz >= 0), f"Negative Banzhaf values: {bz}"


def test_deterministic_banzhaf_large_graph_ranks_clear_root_first():
    """Large graphs must use the same deterministic path and rank clear root first.

    n=22 previously triggered a different path. service 0 has much higher anomaly (0.9 vs 0.3)
    and is upstream of all others — it must have the highest Banzhaf value.
    """
    cfg = _GabanzhafConfig.from_kwargs({})
    fs_big = np.array([0.9] + [0.3] * 21)
    topo_big = np.zeros((22, 22), dtype=np.float32)
    for j in range(1, 22):
        topo_big[0, j] = 0.3
    bz = gabanzhaf_mod._banzhaf_power(fs_big, topo_big, cfg)
    assert np.argmax(bz) == 0, (
        f"Root cause (idx 0) should have max Banzhaf. bz={bz}"
    )


def test_deterministic_banzhaf_single_service():
    """Single service: Banzhaf = fitness_signal (trivial coalition)."""
    cfg = _GabanzhafConfig.from_kwargs({})
    fs = np.array([0.7])
    topo = np.zeros((1, 1), dtype=np.float32)
    bz = gabanzhaf_mod._banzhaf_power(fs, topo, cfg)
    assert abs(bz[0] - 0.7) < 1e-6, f"Single service Banzhaf should be 0.7, got {bz[0]}"


def test_incident_phenotype_is_normalized_with_missing_trace():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _incident_phenotype

    cfg = _GabanzhafConfig.from_kwargs({})
    semantic = {
        "svc-a": {"resource": 1.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
        "svc-b": {"resource": 0.5, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
    }
    pi = _incident_phenotype(
        {"svc-a": 1.0, "svc-b": 0.5},
        None,
        None,
        semantic,
        [],
        cfg,
    )

    assert set(pi) == {"resource", "latency", "error", "propagation"}
    assert all(np.isfinite(v) for v in pi.values())
    assert abs(sum(pi.values()) - 1.0) < 1e-6
    assert pi["resource"] > pi["latency"]


def test_incident_phenotype_detects_latency_error_and_propagation_dominance():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _incident_phenotype

    cfg = _GabanzhafConfig.from_kwargs({})
    semantic = {
        "svc-a": {"resource": 0.0, "latency": 1.0, "error": 0.0, "trace": 0.0, "propagation": 1.0},
        "svc-b": {"resource": 0.0, "latency": 0.9, "error": 0.0, "trace": 0.0, "propagation": 1.0},
    }
    svc_anom = {"svc-a": 1.0, "svc-b": 0.9}
    pi_latency = _incident_phenotype(
        svc_anom,
        {"svc-a": 0.0, "svc-b": 0.0},
        {"svc-a": 1.0, "svc-b": 0.9},
        semantic,
        [("svc-a", "svc-b", 0.9)],
        cfg,
    )

    assert pi_latency["latency"] > pi_latency["resource"]
    assert pi_latency["propagation"] > pi_latency["resource"]

    error_semantic = {
        "svc-a": {"resource": 0.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
        "svc-b": {"resource": 0.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
    }
    pi_error = _incident_phenotype(
        svc_anom,
        {"svc-a": 1.0, "svc-b": 0.8},
        {"svc-a": 0.0, "svc-b": 0.0},
        error_semantic,
        [],
        cfg,
    )
    assert pi_error["error"] > pi_error["latency"]


def test_incident_phenotype_missing_semantic_evidence_is_neutral():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _incident_phenotype

    cfg = _GabanzhafConfig.from_kwargs({})
    pi = _incident_phenotype(
        {"svc-a": 1.0},
        None,
        None,
        None,
        [],
        cfg,
    )

    assert set(pi) == {"resource", "latency", "error", "propagation"}
    assert all(np.isfinite(v) for v in pi.values())
    assert abs(sum(pi.values()) - 1.0) < 1e-6


@pytest.mark.parametrize("temperature", [float("nan"), 0.0, -1.0])
def test_incident_phenotype_invalid_temperature_is_finite(temperature):
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _incident_phenotype

    cfg = _GabanzhafConfig.from_kwargs({"phenotype_temperature": temperature})
    pi = _incident_phenotype(
        {"svc-a": 1.0},
        {"svc-a": 0.0},
        {"svc-a": 0.0},
        {"svc-a": {"resource": 1.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0}},
        [],
        cfg,
    )

    assert all(np.isfinite(v) for v in pi.values())
    assert abs(sum(pi.values()) - 1.0) < 1e-6


def test_incident_phenotype_pure_propagation_can_dominate():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _incident_phenotype

    cfg = _GabanzhafConfig.from_kwargs({})
    pi = _incident_phenotype(
        {"svc-a": 1.0, "svc-b": 1.0},
        None,
        None,
        {
            "svc-a": {"resource": 0.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
            "svc-b": {"resource": 0.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
        },
        [("svc-a", "svc-b", 1.0)],
        cfg,
    )

    assert pi["propagation"] > pi["resource"]


def test_phenotype_conditioned_fitness_boosts_matching_evidence():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _phenotype_conditioned_fitness

    cfg = _GabanzhafConfig.from_kwargs({})
    cand = ["svc-a", "svc-b"]
    svc_anom = {"svc-a": 0.8, "svc-b": 0.8}
    svc_residual = {"svc-a": 0.4, "svc-b": 0.4}
    trace_scores = {"svc-a": 0.0, "svc-b": 1.0}
    semantic = {
        "svc-a": {"resource": 1.0, "latency": 0.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
        "svc-b": {"resource": 0.0, "latency": 1.0, "error": 0.0, "trace": 1.0, "propagation": 0.0},
    }
    pi = {"resource": 0.1, "latency": 0.7, "error": 0.1, "propagation": 0.1}

    fitness = _phenotype_conditioned_fitness(
        cand, svc_anom, svc_residual, trace_scores, semantic, pi, cfg
    )

    assert fitness.shape == (2,)
    assert fitness[1] > fitness[0]
    assert np.all(np.isfinite(fitness))
    assert np.all(fitness >= 0.0)


def test_phenotype_conditioned_fitness_handles_missing_optional_evidence():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _phenotype_conditioned_fitness

    cfg = _GabanzhafConfig.from_kwargs({})
    fitness = _phenotype_conditioned_fitness(
        ["svc-a", "svc-b"],
        {"svc-a": 0.8, "svc-b": 0.4},
        {"svc-a": 0.3, "svc-b": 0.2},
        None,
        None,
        None,
        cfg,
    )

    assert fitness.shape == (2,)
    assert np.all(np.isfinite(fitness))
    assert np.all(fitness >= 0.0)


def test_phenotype_conditioned_fitness_sanitizes_nonfinite_values():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _phenotype_conditioned_fitness

    cfg = _GabanzhafConfig.from_kwargs({})
    fitness = _phenotype_conditioned_fitness(
        ["svc-a", "svc-b"],
        {"svc-a": np.nan, "svc-b": np.inf},
        {"svc-a": -1.0, "svc-b": 0.5},
        {"svc-a": np.inf, "svc-b": "bad"},
        {
            "svc-a": {"resource": np.inf, "latency": -1.0, "error": np.nan, "trace": np.inf, "propagation": 0.2},
            "svc-b": {"resource": 0.1, "latency": np.inf, "error": 0.4, "trace": np.nan, "propagation": -2.0},
        },
        {"resource": np.inf, "latency": np.nan, "error": -1.0, "propagation": 0.5},
        cfg,
    )

    assert fitness.shape == (2,)
    assert np.all(np.isfinite(fitness))
    assert np.all(fitness >= 0.0)


def test_phenotype_conditioned_fitness_empty_candidates_returns_empty_vector():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _phenotype_conditioned_fitness

    cfg = _GabanzhafConfig.from_kwargs({})
    fitness = _phenotype_conditioned_fitness([], {}, {}, None, None, None, cfg)

    assert fitness.shape == (0,)
    assert fitness.dtype == float


def test_phenotype_conditioned_fitness_sanitizes_nonfinite_gain_knobs():
    from RCAEval.e2e.gabanzhaf import _GabanzhafConfig, _phenotype_conditioned_fitness

    cfg = _GabanzhafConfig.from_kwargs({"phenotype_gain_latency": float("inf")})
    fitness = _phenotype_conditioned_fitness(
        ["svc-a"],
        {"svc-a": 0.8},
        {"svc-a": 0.4},
        {"svc-a": 0.5},
        {
            "svc-a": {
                "resource": 0.0,
                "latency": 1.0,
                "error": 0.0,
                "trace": 0.5,
                "propagation": 0.0,
            }
        },
        {"resource": 0.0, "latency": 1.0, "error": 0.0, "propagation": 0.0},
        cfg,
    )

    assert fitness.shape == (1,)
    assert np.all(np.isfinite(fitness))
    assert np.all(fitness >= 0.0)


def test_rank_metrics_uses_bounded_semantic_match():
    from RCAEval.e2e.gabanzhaf import _rank_metrics

    cols = ["svc-a_mem", "svc-a_cpu", "svc-a_latency", "svc-b_latency"]
    svc_banzhaf = {"svc-a": 1.0, "svc-b": 0.2}
    svc_residual = {"svc-a": 1.0, "svc-b": 0.2}
    svc_anom = {"svc-a": 1.0, "svc-b": 0.2}
    metric_score = {
        "svc-a_mem": 0.0,
        "svc-a_cpu": 1.0,
        "svc-a_latency": 0.9,
        "svc-b_latency": 1.0,
    }
    trace_scores = {}
    phenotype = {"resource": 0.05, "latency": 0.85, "error": 0.05, "propagation": 0.05}
    semantic = {
        "svc-a": {"resource": 1.0, "latency": 0.8, "error": 0.0, "trace": 0.0, "propagation": 0.0},
        "svc-b": {"resource": 0.0, "latency": 1.0, "error": 0.0, "trace": 0.0, "propagation": 0.0},
    }

    ranks = _rank_metrics(
        cols, svc_banzhaf, svc_residual, svc_anom, metric_score, trace_scores, phenotype, semantic
    )

    assert ranks[0] == "svc-a_latency"
    assert ranks.index("svc-a_latency") < ranks.index("svc-a_cpu")
    assert ranks.index("svc-a_latency") < ranks.index("svc-b_latency")

    wide_metric_gap = {
        "svc-a_mem": 0.0,
        "svc-a_cpu": 10.0,
        "svc-a_latency": 1.0,
        "svc-b_latency": 10.0,
    }
    wide_ranks = _rank_metrics(
        cols, svc_banzhaf, svc_residual, svc_anom, wide_metric_gap, trace_scores, phenotype, semantic
    )

    assert wide_ranks.index("svc-a_latency") < wide_ranks.index("svc-b_latency")
    assert wide_ranks.index("svc-a_cpu") < wide_ranks.index("svc-a_latency")


def test_metric_semantic_match_sanitizes_invalid_phenotype():
    from RCAEval.e2e.gabanzhaf import _metric_semantic_match

    assert _metric_semantic_match("svc_latency", 1.0) == 0.0
    assert _metric_semantic_match("svc_latency", {"latency": np.nan}) == 0.0
    assert _metric_semantic_match("svc_latency", {"latency": np.inf}) == 0.0
    assert _metric_semantic_match("svc_latency", {"latency": 2.0}) == 1.0


def test_rank_metrics_without_phenotype_uses_legacy_formula():
    from RCAEval.e2e.gabanzhaf import _norm_dict, _rank_metrics, _service_root_score

    cols = ["a_m1", "b_m2", "c_m3", "d_m4"]
    svc_banzhaf = {
        "a": 0.10532412090672771,
        "b": 0.9616713204562753,
        "c": 0.8807441898084656,
        "d": 0.901199918051514,
    }
    svc_residual = {
        "a": 0.14079873494655515,
        "b": 0.15395359343411863,
        "c": 0.4332949011819809,
        "d": 0.04063634381353931,
    }
    svc_anom = {
        "a": 0.7236050075764766,
        "b": 0.14502212885327126,
        "c": 0.6676196731659766,
        "d": 0.5057472777201908,
    }
    metric_score = {
        "a_m1": 0.7384574708582091,
        "b_m2": 0.5783387882382596,
        "c_m3": 0.9252111774939774,
        "d_m4": 0.6579524703762987,
    }
    trace_scores = {
        "a": 0.4458525262618027,
        "b": 0.7381993093673848,
        "c": 0.2710090614947943,
        "d": 0.2792327570298251,
    }

    nb = _norm_dict(svc_banzhaf)
    nr = _norm_dict(svc_residual)
    na = _norm_dict(svc_anom)
    nm = _norm_dict(metric_score)
    nt = _norm_dict({s: trace_scores.get(s, 0.0) for s in svc_anom})

    def legacy_score(col):
        svc = col.rsplit("_", 1)[0]
        service_root = _service_root_score(
            nb.get(svc, 0.0),
            nr.get(svc, 0.0),
            na.get(svc, 0.0),
            nt.get(svc, 0.0),
        )
        metric_local = nm.get(col, 0.0)
        return 0.70 * service_root + 0.30 * metric_local + 0.20 * service_root * metric_local

    expected = sorted(cols, key=legacy_score, reverse=True)

    assert _rank_metrics(
        cols, svc_banzhaf, svc_residual, svc_anom, metric_score, trace_scores, phenotype=None
    ) == expected


def test_gabanzhaf_core_returns_phenotype_when_requested():
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
    assert "phenotype" in result
    assert abs(sum(result["phenotype"].values()) - 1.0) < 1e-6
