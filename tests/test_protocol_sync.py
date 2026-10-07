import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_human_and_machine_protocol_versions_match():
    protocol = json.loads((ROOT / "experiment/protocol.json").read_text(encoding="utf-8"))
    human = (ROOT / "docs/research-protocol.md").read_text(encoding="utf-8")

    assert f"Protocol version: `{protocol['protocol_version']}`" in human


def test_every_frozen_method_uses_shared_python310():
    protocol = json.loads((ROOT / "experiment/protocol.json").read_text(encoding="utf-8"))

    assert set(protocol["method_environments"]) == {"merit", *protocol["baselines"]}
    assert set(protocol["method_environments"].values()) == {"./py310/bin/python"}
    assert set(protocol["stochastic_methods"]) <= set(protocol["baselines"])
    assert protocol["python_environment"] == {
        "version": "3.10.18",
        "lock_file": "requirements_py310.lock",
        "validation": (
            "exact Python patch version and every exact package pin; "
            "extra packages are recorded but allowed"
        ),
    }


def test_banzhaf_and_denominator_red_lines_are_machine_frozen():
    protocol = json.loads((ROOT / "experiment/protocol.json").read_text(encoding="utf-8"))

    assert protocol["datasets"] == ["re2-ob", "re2-ss", "re2-tt"]
    assert protocol["expected_cases_per_dataset"] == 90
    assert protocol["banzhaf_equivalence"]["max_k"] == 15
    assert protocol["banzhaf_equivalence"]["absolute_tolerance"] == 1e-9
    assert protocol["per_case_timeout_scope"] == "whole_case_hard_subprocess_deadline"
    assert protocol["dataset_budget_seconds"] >= (
        protocol["expected_cases_per_dataset"]
        * protocol["per_case_timeout_seconds"]
    )
    assert protocol["paired_inference"]["scopes"] == [
        "per_system",
        "compatible_pooled",
    ]
    assert protocol["paired_inference"]["holm_families"] == [
        "rq1_baselines",
        "rq2_ablations",
        "rq4_robustness",
    ]
    assert protocol["reporting_gates"]["manifest_output_path_required"] is True
    assert protocol["reporting_gates"]["efficiency_cache_conditions"] == [
        "cold_independent_process",
        "warm_after_same_case_call",
    ]
    assert protocol["reporting_gates"]["timeout_observed_duration_required"] is True
