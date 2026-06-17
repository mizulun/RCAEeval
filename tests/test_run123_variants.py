import csv
import importlib
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


def test_gabanzhaf_ablation_aliases_are_available():
    approx_base, approx_env, approx_meta = run123.resolve_method_variant(
        "gabanzhaf_approx_banzhaf"
    )
    pure_base, pure_env, pure_meta = run123.resolve_method_variant(
        "gabanzhaf_pure_banzhaf"
    )

    assert approx_base == "gabanzhaf"
    assert approx_env["GABANZHAF_BANZHAF_MODE"] == "closed_form"
    assert approx_meta["banzhaf_mode"] == "closed_form"
    assert approx_meta["ablation_role"] == "approx_banzhaf"

    assert pure_base == "gabanzhaf"
    assert pure_env["GABANZHAF_BANZHAF_MODE"] == "bruteforce"
    assert pure_meta["banzhaf_mode"] == "bruteforce"
    assert pure_meta["ablation_role"] == "pure_banzhaf"


def test_gabanzhaf_exact_candidate_k_variants_are_available():
    for k in [6, 8, 10, 12, 16]:
        base, env, meta = run123.resolve_method_variant(f"gabanzhaf_k{k}")

        assert base == "gabanzhaf"
        assert env == {"GABANZHAF_EXACT_CANDIDATE_K": str(k)}
        assert meta["base_method"] == "gabanzhaf"
        assert meta["exact_candidate_k"] == k
        assert meta["ablation_role"] == f"exact_candidate_k={k}"


def test_save_csv_preserves_variant_metadata(tmp_path):
    output = tmp_path / "ablation.csv"
    records = [
        {
            "method": "gabanzhaf_approx_banzhaf",
            "base_method": "gabanzhaf",
            "banzhaf_mode": "closed_form",
            "ablation_role": "approx_banzhaf",
            "exact_candidate_k": 12,
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

    assert row["method"] == "gabanzhaf_approx_banzhaf"
    assert row["base_method"] == "gabanzhaf"
    assert row["banzhaf_mode"] == "closed_form"
    assert row["ablation_role"] == "approx_banzhaf"
    assert row["exact_candidate_k"] == "12"
