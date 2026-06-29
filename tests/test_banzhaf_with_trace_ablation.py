import importlib.util
import pathlib

import numpy as np
import pandas as pd


_spec = importlib.util.spec_from_file_location(
    "banzhaf_with_trace_under_test",
    pathlib.Path(__file__).resolve().parent.parent
    / "RCAEval"
    / "e2e"
    / "banzhaf_with_trace.py",
)
trace_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trace_mod)


def _simple_frame():
    return pd.DataFrame({
        "time": np.arange(12),
        "svc-a_cpu": [1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9],
        "svc-b_cpu": [1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6],
    })


def test_trace_ablation_config_controls_sources_independently():
    cfg = trace_mod._GabanzhafConfig.from_kwargs({
        "use_trace_scores": False,
        "use_call_graph": True,
        "use_residual": False,
    })

    assert cfg.use_trace_scores is False
    assert cfg.use_call_graph is True
    assert cfg.use_residual is False


def test_residual_disabled_preserves_service_evidence():
    cfg = trace_mod._GabanzhafConfig.from_kwargs({"use_residual": False})
    scores = trace_mod._residual_scores(
        {"root": 0.8, "victim": 1.0},
        {"root": set(), "victim": {"root"}},
        {("root", "victim"): 0.9},
        cfg,
    )

    assert scores == {"root": 0.8, "victim": 1.0}


def test_trace_scores_disabled_do_not_add_trace_seed():
    result = trace_mod._gabanzhaf_core(
        _simple_frame(),
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={
            "trace_scores": {"svc-b": 1.0},
            "use_trace_scores": False,
            "use_call_graph": False,
            "return_diagnostics": True,
        },
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["trace_seed_services"] == []
    assert diagnostics["trace_added_seed_count"] == 0


def test_call_graph_disabled_has_no_accepted_or_effective_edges():
    result = trace_mod._gabanzhaf_core(
        _simple_frame(),
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={
            "trace_scores": {"svc-a": 1.0},
            "call_graph": {"svc-b": {"svc-a"}},
            "use_call_graph": False,
            "return_diagnostics": True,
        },
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["raw_call_graph_edge_count"] == 1
    assert diagnostics["accepted_call_graph_edge_count"] == 0
    assert diagnostics["effective_call_graph_edge_count"] == 0


def test_trace_diagnostics_report_overlap_and_candidate_recall():
    result = trace_mod._gabanzhaf_core(
        _simple_frame(),
        inject_time=6,
        dataset="unit",
        anomalies=None,
        kwargs={
            "trace_scores": {"svc-a": 0.9, "svc-b": 0.1},
            "root_service": "svc-a",
            "return_diagnostics": True,
        },
    )

    diagnostics = result["diagnostics"]
    assert -1.0 <= diagnostics["trace_metric_spearman"] <= 1.0
    assert 0.0 <= diagnostics["trace_metric_top_k_overlap"] <= 1.0
    assert diagnostics["root_candidate_recall"] is True
    assert diagnostics["game_candidates"] == result["game_candidates"]


def test_no_trace_env_skips_trace_file_extraction(monkeypatch):
    captured = {}

    def fail_extract(*args, **kwargs):
        raise AssertionError("trace extraction must be skipped")

    def fake_core(data, inject_time, dataset, anomalies, kwargs):
        captured.update(kwargs)
        return {"ranks": []}

    monkeypatch.setenv("GABANZHAF_TRACE_DISABLE_TRACE", "1")
    monkeypatch.setattr(trace_mod, "_extract_trace_features", fail_extract)
    monkeypatch.setattr(trace_mod, "_gabanzhaf_core", fake_core)

    trace_mod.gabanzhaf(_simple_frame(), inject_time=6, dataset="unit")

    assert captured["use_trace_scores"] is False
    assert captured["use_call_graph"] is False
    assert captured["trace_scores"] is None
    assert captured["call_graph"] == {}
