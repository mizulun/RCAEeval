import json
from pathlib import Path

import pytest

import script.reproduce_paper as reproduce_paper
from script.reproduce_paper import (
    _validate_post_experiment_state,
    build_jobs,
    parse_args,
)


def _protocol():
    path = Path(__file__).resolve().parents[1] / "experiment/protocol.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_smoke_profile_uses_each_baseline_once_and_one_main_job():
    protocol = _protocol()
    jobs = build_jobs(protocol, "smoke")

    assert len(jobs) == 1
    assert jobs[0].name == "main"
    assert jobs[0].methods == ("merit_full", *protocol["baselines"])


def test_confirmatory_profile_expands_seed_repetitions_variants_and_sweeps():
    protocol = _protocol()
    jobs = build_jobs(protocol, "confirmatory")
    main = jobs[0]

    for method in protocol["stochastic_methods"]:
        assert method not in main.methods
        for seed in protocol["seeds"]:
            assert f"{method}_seed{seed}" in main.methods
    assert "merit_full_rep5" in main.methods
    assert "merit_individual_attribution" in main.methods
    assert "merit_victim_01" in main.methods
    assert "merit_tconf_09" in main.methods
    assert "merit_cconf_05" in main.methods
    assert "merit_k15" in main.methods
    assert len(main.methods) == len(set(main.methods))
    assert set(main.methods) == {
        "merit_full",
        *(
            method
            for method in protocol["baselines"]
            if method not in protocol["stochastic_methods"]
        ),
        *(
            f"{method}_seed{seed}"
            for method in protocol["stochastic_methods"]
            for seed in protocol["seeds"]
        ),
        *(
            label
            for label in main.methods
            if label.startswith("merit_") and label != "merit_full"
        ),
    }


def test_confirmatory_runs_expensive_timeout_prone_baselines_last():
    main = build_jobs(_protocol(), "confirmatory")[0]

    assert main.methods.index("merit_no_trace") < main.methods.index(
        "rcd_seed20260713"
    )
    assert main.methods.index("rcd_seed20260713") < main.methods.index(
        "pc_pagerank"
    )
    assert main.methods[-3:] == (
        "causalrca_seed20260713",
        "causalrca_seed20260714",
        "causalrca_seed20260715",
    )


def test_confirmatory_profile_has_independent_delay_and_window_jobs():
    jobs = build_jobs(_protocol(), "confirmatory")
    settings = {(job.name, job.length, job.tdelta) for job in jobs[1:]}

    assert ("delay-60", 20, 60) in settings
    assert ("delay-120", 20, 120) in settings
    assert ("delay-300", 20, 300) in settings
    assert ("window-5", 5, 0) in settings
    assert ("window-10", 10, 0) in settings
    assert ("window-30", 30, 0) in settings


def test_reproduction_cli_exposes_explicit_resume_mode():
    args = parse_args(["--run-id", "frozen", "--resume"])

    assert args.resume is True


def test_post_experiment_stages_revalidate_source_and_environment(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"formal_manifest": True, "protocol_version": "v"}),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        reproduce_paper,
        "_validate_resume_manifest",
        lambda root, saved, protocol: calls.append(
            ("source", root, saved, protocol)
        ),
    )
    monkeypatch.setattr(
        reproduce_paper,
        "_validate_method_environments",
        lambda root, path, protocol: [
            {"ok": True, "configured_environment": "./py310/bin/python"}
        ],
    )

    _validate_post_experiment_state(
        tmp_path,
        tmp_path / "protocol.json",
        {"protocol_version": "v"},
        manifest,
    )

    assert calls[0][0] == "source"


def test_post_experiment_stages_require_main_formal_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="main formal manifest"):
        _validate_post_experiment_state(
            tmp_path,
            tmp_path / "protocol.json",
            {},
            tmp_path / "missing.json",
        )
