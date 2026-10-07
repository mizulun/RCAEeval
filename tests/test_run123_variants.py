import csv
import importlib
import json
import subprocess


run123 = importlib.import_module("run123")


def test_gabanzhaf_pure_banzhaf_variant_runs_base_method_with_env(monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="AC@1-CPU: 0.4\nAvg@5-CPU: 0.8\nAvg speed: 0.12\n",
            stderr="",
        )

    monkeypatch.setattr(run123, "clear_result_dir", lambda method, dataset: None)
    monkeypatch.setattr(run123.subprocess, "run", fake_run)

    record = run123.run_single(
        "gabanzhaf_pure_banzhaf",
        "re2-ob",
        length=20,
        tdelta=0,
        smoke_test=True,
    )

    assert calls["cmd"][calls["cmd"].index("--method") + 1] == "gabanzhaf"
    assert calls["kwargs"]["env"]["GABANZHAF_BANZHAF_MODE"] == "bruteforce"
    assert "output/runs/gabanzhaf_pure_banzhaf/re2-ob/results" in calls["cmd"]
    assert "--test" in calls["cmd"]
    assert record["method"] == "gabanzhaf_pure_banzhaf"
    assert record["base_method"] == "gabanzhaf"
    assert record["banzhaf_mode"] == "bruteforce"
    assert record["AC@1-CPU"] == 0.4
    assert record["Avg@5-CPU"] == 0.8
    assert record["Avg speed"] == 0.12


def test_removed_ablation_variants_are_not_registered():
    removed = [
        "gabanzhaf_approx_banzhaf",
        "merit_approx_banzhaf",
        "gabanzhaf_no_lag_1",
        "merit_no_lag_1",
        "with_trace_full",
        "with_trace_no_scores",
        "with_trace_no_call_graph",
        "with_trace_no_trace",
        "with_trace_no_residual",
    ]

    for method in removed:
        base, env, meta = run123.resolve_method_variant(method)
        assert base == method
        assert env == {}
        assert meta == {}


def test_gabanzhaf_pure_banzhaf_variant_is_available():
    pure_base, pure_env, pure_meta = run123.resolve_method_variant(
        "gabanzhaf_pure_banzhaf"
    )

    assert pure_base == "gabanzhaf"
    assert pure_env["GABANZHAF_BANZHAF_MODE"] == "bruteforce"
    assert pure_meta["banzhaf_mode"] == "bruteforce"
    assert pure_meta["ablation_role"] == "pure_banzhaf"


def test_merit_pure_banzhaf_variant_uses_renamed_base_method():
    pure_base, pure_env, pure_meta = run123.resolve_method_variant(
        "merit_pure_banzhaf"
    )

    assert pure_base == "merit"
    assert pure_env["GABANZHAF_BANZHAF_MODE"] == "bruteforce"
    assert pure_meta["base_method"] == "merit"
    assert pure_meta["banzhaf_mode"] == "bruteforce"
    assert pure_meta["ablation_role"] == "pure_banzhaf"


def test_gabanzhaf_exact_candidate_k_variants_are_available():
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]:
        base, env, meta = run123.resolve_method_variant(f"gabanzhaf_k{k}")

        assert base == "gabanzhaf"
        assert env == {"GABANZHAF_EXACT_CANDIDATE_K": str(k)}
        assert meta["base_method"] == "gabanzhaf"
        assert meta["exact_candidate_k"] == k
        assert meta["ablation_role"] == f"exact_candidate_k={k}"


def test_merit_exact_candidate_k_variants_are_available():
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]:
        base, env, meta = run123.resolve_method_variant(f"merit_k{k}")

        assert base == "merit"
        assert env == {"GABANZHAF_EXACT_CANDIDATE_K": str(k)}
        assert meta["base_method"] == "merit"
        assert meta["exact_candidate_k"] == k
        assert meta["ablation_role"] == f"exact_candidate_k={k}"


def test_gabanzhaf_component_ablation_variants_are_available():
    expected = {
        "gabanzhaf_full": ({}, "trace_frontier_full"),
        "gabanzhaf_no_trace": ({"GABANZHAF_DISABLE_TRACE": "1"}, "no_trace"),
        "gabanzhaf_no_trace_topology": (
            {"GABANZHAF_DISABLE_TRACE_TOPOLOGY": "1"},
            "no_trace_topology",
        ),
        "gabanzhaf_no_coverage": (
            {"GABANZHAF_DISABLE_COVERAGE": "1"},
            "no_coverage",
        ),
        "gabanzhaf_no_victim_demotion": (
            {"GABANZHAF_DISABLE_VICTIM_DEMOTION": "1"},
            "no_victim_demotion",
        ),
        "gabanzhaf_evidence_service": (
            {"GABANZHAF_RANK_MODE": "service_evidence"},
            "evidence_service",
        ),
        "gabanzhaf_evidence_metric": (
            {"GABANZHAF_RANK_MODE": "metric_evidence"},
            "evidence_metric",
        ),
        "gabanzhaf_evidence_trace": (
            {"GABANZHAF_RANK_MODE": "trace_evidence"},
            "evidence_trace",
        ),
        "gabanzhaf_evidence_service_trace": (
            {"GABANZHAF_RANK_MODE": "service_evidence_trace"},
            "evidence_service_trace",
        ),
        "gabanzhaf_residual": (
            {"GABANZHAF_TRACE_RESIDUAL": "1"},
            "trace_residual",
        ),
    }

    for method, (expected_env, expected_role) in expected.items():
        base, env, meta = run123.resolve_method_variant(method)
        assert base == "gabanzhaf"
        assert env == expected_env
        assert meta["base_method"] == "gabanzhaf"
        assert meta["ablation_role"] == expected_role


def test_merit_component_ablation_variants_are_available():
    expected = {
        "merit_full": ({}, "trace_frontier_full"),
        "merit_no_trace": ({"GABANZHAF_DISABLE_TRACE": "1"}, "no_trace"),
        "merit_no_metric": (
            {"GABANZHAF_DISABLE_METRICS": "1"},
            "no_metric",
        ),
        "merit_no_trace_topology": (
            {"GABANZHAF_DISABLE_TRACE_TOPOLOGY": "1"},
            "no_trace_topology",
        ),
        "merit_no_coverage": (
            {"GABANZHAF_DISABLE_COVERAGE": "1"},
            "no_coverage",
        ),
        "merit_no_victim_demotion": (
            {"GABANZHAF_DISABLE_VICTIM_DEMOTION": "1"},
            "no_victim_demotion",
        ),
        "merit_evidence_service": (
            {"GABANZHAF_RANK_MODE": "service_evidence"},
            "evidence_service",
        ),
        "merit_evidence_metric": (
            {"GABANZHAF_RANK_MODE": "metric_evidence"},
            "evidence_metric",
        ),
        "merit_evidence_trace": (
            {"GABANZHAF_RANK_MODE": "trace_evidence"},
            "evidence_trace",
        ),
        "merit_evidence_service_trace": (
            {"GABANZHAF_RANK_MODE": "service_evidence_trace"},
            "evidence_service_trace",
        ),
        "merit_residual": (
            {"GABANZHAF_TRACE_RESIDUAL": "1"},
            "trace_residual",
        ),
    }

    for method, (expected_env, expected_role) in expected.items():
        base, env, meta = run123.resolve_method_variant(method)
        assert base == "merit"
        assert env == expected_env
        assert meta["base_method"] == "merit"
        assert meta["ablation_role"] == expected_role


def test_merit_sensitivity_sweeps_are_available():
    for tag, value in [("07", 0.7), ("085", 0.85), ("095", 0.95)]:
        base, env, meta = run123.resolve_method_variant(f"merit_tconf_{tag}")
        assert base == "merit"
        assert env == {"GABANZHAF_TRACE_TOPOLOGY_CONF": str(value)}
        assert meta["base_method"] == "merit"
        assert meta["trace_topology_conf"] == value
        assert meta["ablation_role"] == f"trace_topology_conf={value}"

    for tag, value in [("03", 0.3), ("05", 0.5), ("07", 0.7)]:
        base, env, meta = run123.resolve_method_variant(f"merit_victim_{tag}")
        assert base == "merit"
        assert env == {
            "GABANZHAF_USE_VICTIM_DEMOTION": "1",
            "GABANZHAF_VICTIM_FLOOR": str(value),
        }
        assert meta["base_method"] == "merit"
        assert meta["victim_floor"] == value
        assert meta["ablation_role"] == f"victim_floor={value}"


def test_save_csv_preserves_variant_metadata(tmp_path):
    output = tmp_path / "ablation.csv"
    records = [
        {
            "method": "merit_tconf_07",
            "base_method": "merit",
            "trace_topology_conf": 0.7,
            "ablation_role": "trace_topology_conf=0.7",
            "dataset": "re2-ob",
            "status": "OK",
            "wall_time_sec": 1.2,
            "AC@1-CPU": 0.4,
            "Avg speed": 0.1,
            "returncode": 0,
        }
    ]

    run123.save_csv(records, str(output))

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["method"] == "merit_tconf_07"
    assert row["base_method"] == "merit"
    assert row["trace_topology_conf"] == "0.7"
    assert row["ablation_role"] == "trace_topology_conf=0.7"


def test_aggregate_repetitions_uses_runtime_median_and_range():
    records = [
        {
            "method": "gabanzhaf",
            "dataset": "re2-ob",
            "status": "OK",
            "wall_time_sec": wall,
            "Avg speed": speed,
            "AC@1-CPU": 0.4,
            "returncode": 0,
        }
        for wall, speed in [(12.0, 0.12), (10.0, 0.10), (20.0, 0.20)]
    ]

    aggregate = run123.aggregate_repetitions(records)

    assert aggregate["wall_time_sec"] == 12.0
    assert aggregate["wall_time_min_sec"] == 10.0
    assert aggregate["wall_time_max_sec"] == 20.0
    assert aggregate["Avg speed"] == 0.12
    assert aggregate["repetitions"] == 3
    assert aggregate["AC@1-CPU"] == 0.4


def test_run_single_enables_diagnostics_sidecars(monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["env"] = kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout="Avg speed: 0.1\n", stderr="")

    monkeypatch.setattr(run123, "clear_result_dir", lambda method, dataset: None)
    monkeypatch.setattr(run123.subprocess, "run", fake_run)

    run123.run_single(
        "gabanzhaf",
        "re2-ob",
        smoke_test=True,
        diagnostics=True,
    )

    assert calls["env"]["GABANZHAF_WRITE_DIAGNOSTICS"] == "1"
