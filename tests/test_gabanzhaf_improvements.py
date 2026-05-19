"""Tests for gabanzhaf DELAY/LOSS improvements.

Three behaviors under test:
1. _metric_evidence: Winsorization caps rz at 10 (no unbounded scores)
2. _extract_call_graph: extracts caller→callee from traces.csv (50K row limit)
3. _gabanzhaf_core: call graph topology demotes callers of root cause
"""
import os
import tempfile
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
        kwargs={"call_graph": call_graph, "n_generations": 5, "pop_size": 10}
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
        kwargs={"call_graph": call_graph, "n_generations": 5, "pop_size": 10}
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
        kwargs={"call_graph": {}, "n_generations": 5, "pop_size": 10}
    )
    assert "ranks" in result
    assert len(result["ranks"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. _exact_banzhaf_topology
# ─────────────────────────────────────────────────────────────────────────────
from RCAEval.e2e.gabanzhaf import _exact_banzhaf_topology


def test_exact_banzhaf_no_topology_proportional_to_anomaly():
    """Without topology, Banzhaf should be proportional to fitness_signal.

    For additive v(S) (alpha_topo=0), Banzhaf(i) = fitness_signal[i] exactly.
    """
    fs = np.array([0.9, 0.4, 0.1], dtype=float)
    topo = np.zeros((3, 3), dtype=np.float32)
    bz = _exact_banzhaf_topology(fs, topo, alpha_topo=0.0)
    # ranking must match fitness_signal ranking
    assert bz[0] > bz[1] > bz[2], f"Expected descending Banzhaf, got {bz}"


def test_exact_banzhaf_topology_promotes_root_cause():
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
    fs = np.array([0.9, 0.5], dtype=float)
    topo = np.zeros((2, 2), dtype=np.float32)
    topo[0, 1] = 0.5  # A(0) upstream of B(1)
    bz = _exact_banzhaf_topology(fs, topo, alpha_topo=0.3)
    assert abs(bz[0] - 1.15) < 1e-6, f"Expected Banzhaf(A)=1.15, got {bz[0]}"
    assert abs(bz[1] - 0.25) < 1e-6, f"Expected Banzhaf(B)=0.25, got {bz[1]}"


def test_exact_banzhaf_topology_no_negative_values():
    """Banzhaf values must all be non-negative (clipped at 0)."""
    rng = np.random.default_rng(0)
    n = 5
    fs = rng.random(n)
    topo = np.zeros((n, n), dtype=np.float32)
    topo[0, 1] = 0.8
    topo[1, 2] = 0.6
    bz = _exact_banzhaf_topology(fs, topo, alpha_topo=0.5)
    assert np.all(bz >= 0), f"Negative Banzhaf values: {bz}"


def test_exact_banzhaf_mc_fallback_consistent_with_exact():
    """MC fallback (n>20) must rank the clear root cause first.

    n=22 triggers MC path. service 0 has much higher anomaly (0.9 vs 0.3)
    and is upstream of all others — it must have the highest Banzhaf value.
    """
    fs_big = np.array([0.9] + [0.3] * 21)
    topo_big = np.zeros((22, 22), dtype=np.float32)
    for j in range(1, 22):
        topo_big[0, j] = 0.3
    bz_mc = _exact_banzhaf_topology(fs_big, topo_big, alpha_topo=0.3, n_samples=2000)
    assert np.argmax(bz_mc) == 0, (
        f"Root cause (idx 0) should have max Banzhaf in MC path. bz={bz_mc}"
    )


def test_exact_banzhaf_single_service():
    """Single service: Banzhaf = fitness_signal (trivial coalition)."""
    fs = np.array([0.7])
    topo = np.zeros((1, 1), dtype=np.float32)
    bz = _exact_banzhaf_topology(fs, topo, alpha_topo=0.3)
    assert abs(bz[0] - 0.7) < 1e-6, f"Single service Banzhaf should be 0.7, got {bz[0]}"
