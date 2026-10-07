"""Frozen pre-split behavior and module-boundary regression checks."""

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from RCAEval.e2e import (
    merit_ablation, merit_diagnostics, merit_evidence, merit_game, merit_trace,
)
from RCAEval.e2e.merit_config import _GabanzhafConfig


CORE_CASES = [
    {}, {"use_trace": False}, {"use_metrics": False},
    {"use_metrics": False, "use_trace": False},
    {"use_coverage": False}, {"use_victim_demotion": False},
    {"protect_coverage_parents": False}, {"use_trace_topology": False},
    {"use_early_full_fusion": False}, {"use_metric_type_dominance": False},
    {"banzhaf_mode": "individual"}, {"banzhaf_mode": "bruteforce"},
    {"scorer": "cusum"}, {"exact_candidate_k": 1, "top_k_svcs": 1},
    *[{"rank_mode": mode} for mode in (
        "metric_evidence", "service_evidence", "trace_evidence", "service_evidence_trace"
    )],
]
ENV_CASES = [
    {"GABANZHAF_DISABLE_TRACE": "1"},
    {"GABANZHAF_DISABLE_METRICS": "1"},
    {"GABANZHAF_BANZHAF_MODE": "individual"},
    {"GABANZHAF_BANZHAF_MODE": "bruteforce"},
    {"GABANZHAF_RANK_MODE": "service_evidence_trace"},
    {"GABANZHAF_DISABLE_COVERAGE": "1"},
    {"GABANZHAF_DISABLE_VICTIM_DEMOTION": "1"},
    {"GABANZHAF_USE_VICTIM_DEMOTION": "1", "GABANZHAF_DISABLE_VICTIM_DEMOTION": "1"},
    {"MERIT_SCORER": "cusum", "MERIT_CUSUM_SLACK": "0.3"},
    {"GABANZHAF_WRITE_DIAGNOSTICS": "1"},
    {"GABANZHAF_PROFILE": "1"},
]


def load_direct(path):
    spec = importlib.util.spec_from_file_location("merit_refactor_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    rng = np.random.default_rng(20261005)
    normal = rng.uniform(0, 2, (40, 5))
    abnormal = rng.uniform(0, 2, (30, 5)) + [0, 8, 6, 3, 0]
    data = pd.DataFrame(np.vstack([normal, abnormal]), columns=[
        "root_cpu", "a_latency", "b_latency", "a_cpu", "quiet_cpu"
    ])
    data.insert(0, "time", np.arange(len(data)))
    kwargs = dict(return_diagnostics=True, return_stage_timing=True,
                  call_graph={"root": ["a", "b"]},
                  trace_topology_edges=[("a", "b", .8, "latency", 1)],
                  trace_service_evidence={"root": .8, "a": .4, "b": 1.},
                  trace_latency_evidence={"a": .2}, trace_error_evidence={"b": .6})
    return data, kwargs


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if "timing" not in k}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [stable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, np.integer):
        return int(value)
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(stable(value), sort_keys=True).encode()).hexdigest()


def behavior(module):
    data, kwargs = inputs()
    outputs = {}
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("MERIT_", "GABANZHAF_"))}
    with patch.dict(os.environ, clean_env, clear=True):
        for i, override in enumerate(CORE_CASES):
            result = module.merit(data, inject_time=40, **(kwargs | override))
            assert all(np.isfinite(v) and v >= 0 for v in result["stage_timing"].values())
            stages = {"preprocess", "metric_evidence", "ranking"}
            if override.get("rank_mode", "banzhaf") == "banzhaf":
                stages.update({"candidates", "banzhaf"})
            assert set(result["stage_timing"]) == stages
            outputs[f"core-{i}"] = fingerprint(result)
        for i, env in enumerate(ENV_CASES):
            with patch.dict(os.environ, env):
                outputs[f"env-{i}"] = fingerprint(module.merit(data, inject_time=40, **kwargs))
    series = {"a": np.arange(20.), "b": np.arange(20.)[::-1]}
    edges = [("a", "b", .7, "latency", 1), ("c", "d", .5, "error", 2)]
    outputs["perturbations"] = fingerprint([
        module._time_shuffle_trace_series(series, "frozen"),
        module._add_trace_series_noise(series, .2, "frozen"),
        module._deterministic_row_dropout(data, .3, "frozen").to_dict("list"),
        module._degree_matched_random_edges(edges),
        module._inject_random_trace_edges(edges, list("abcd"), .3),
    ])
    outputs["defaults"] = fingerprint(vars(module._GabanzhafConfig.from_kwargs({})))
    return outputs


def graph_behavior(module, root):
    """Exercise actual span/series loading and cache policies, not mocked dispatch."""
    root = root.resolve()
    data, _ = inputs()
    boundary = dict(start=0, end=70, inject_time=40,
                    mode="full_record_retrospective", version="paper-full-record-v1")
    exposure = {"cases": {}}
    paths = []
    for i, child in enumerate(["a", "b"]):
        case = root / "root_delay" / str(i)
        case.mkdir(parents=True, exist_ok=True)
        paths.append(case)
        data.to_csv(case / "data.csv", index=False)
        pd.DataFrame(dict(spanID=["p", "c", "post-p", "post-c"],
                          parentSpanID=["", "p", "", "post-p"],
                          serviceName=["root", child, "a", "b"],
                          startTime=[1e6, 2e6, 50e6, 51e6], duration=[0] * 4)).to_csv(
                              case / "traces.csv", index=False)
        for filename in ["tracets_lat.csv", "tracets_err.csv"]:
            pd.DataFrame({"time": range(70), "root_GET": [0.] * 40 + [3.] * 30,
                          "a_GET": [0.] * 41 + [5.] * 29,
                          "b_GET": [0.] * 42 + [4.] * 28}).to_csv(case / filename, index=False)
        exposure["cases"][f"root_delay_{i}"] = dict(fault_directory="root_delay",
                                                      case_directory=str(i), metric_file="data.csv",
                                                      boundary=boundary)
    outputs = {}
    clean_env = {k: v for k, v in os.environ.items()
                 if not k.startswith(("MERIT_", "GABANZHAF_"))}
    controls = [
        {}, {"GABANZHAF_DISABLE_TRACE": "1"},
        {"GABANZHAF_TRACE_CONTROL": "time_shuffle"},
        {"GABANZHAF_TRACE_CONTROL": "degree_matched_random_edges"},
        {"GABANZHAF_TRACE_NOISE": "0.2", "GABANZHAF_TRACE_EDGE_NOISE": "0.3"},
    ]
    with patch.dict(os.environ, clean_env, clear=True):
        for source in ["per_case_pre_fault", "per_case_full", "first_case_pre_fault", "union_pre_fault"]:
            for control_id, control in enumerate(controls):
                module._CALL_GRAPH_CACHE.clear()
                with patch.dict(os.environ, {"GABANZHAF_GRAPH_SOURCE": source, **control}):
                    for i, case in enumerate(paths):
                        ns = SimpleNamespace(data_path=str(case / "data.csv"), input_boundary=boundary,
                                             dataset_exposure_manifest=exposure,
                                             graph_reference_case_id="root_delay_0")
                        result = module.merit(data, inject_time=40, args=ns, return_diagnostics=True)
                        # The absolute temporary root is provenance, not an algorithm output.
                        value = json.loads(json.dumps(stable(result)).replace(str(root), "${CASE_ROOT}"))
                        # Compare unaffected computation with the frozen fixture
                        # using its historical provenance schema.
                        if source.startswith("per_case_"):
                            provenance = value["graph_provenance"]
                            if provenance["source"] == "disabled_no_trace":
                                legacy = {"source": "disabled_no_trace", "edge_count": 0,
                                          "empty_graph_cached": False}
                            else:
                                legacy = {
                                    "source": provenance["source"],
                                    "source_scope": provenance["source_scope"],
                                    "source_case": str(i),
                                    "window": provenance["window"],
                                    "boundary_version": provenance["boundary_version"],
                                    "edge_count": provenance["edge_count"],
                                    "empty_graph_cached": False,
                                }
                            value["graph_provenance"] = legacy
                            value["diagnostics"]["graph_provenance"] = legacy
                        outputs[f"{source}-{control_id}-{i}"] = fingerprint(value)
    return outputs


def test_complete_non_timing_behavior_matches_pre_split():
    module = importlib.import_module("RCAEval.e2e.merit")
    expected = json.loads((Path(__file__).parent / "fixtures/merit_refactor.json").read_text())
    actual = behavior(module)
    # Preserve the historical fixture. Explicit Python no-trace cases are the
    # known correction: caller-supplied trace evidence is now scrubbed.
    corrected = {"core-1", "core-3", "env-0"}
    assert {k: v for k, v in actual.items() if k not in corrected} == {
        k: v for k, v in expected["fingerprints"].items() if k not in corrected
    }
    assert all(actual[key] != expected["fingerprints"][key] for key in corrected)


def test_graph_sources_and_trace_controls_match_pre_split(tmp_path):
    module = importlib.import_module("RCAEval.e2e.merit")
    expected = json.loads((Path(__file__).parent / "fixtures/merit_refactor.json").read_text())
    actual = graph_behavior(module, tmp_path)
    # First-case and union provenance/cache behavior are deliberately fixed;
    # per-case computation remains pinned to the pre-split fixture.
    unaffected = {k for k in actual if k.startswith("per_case_")}
    assert {k: actual[k] for k in unaffected} == {
        k: expected["graph_fingerprints"][k] for k in unaffected
    }


def test_direct_file_import_and_config_identity():
    module = load_direct(Path(__file__).resolve().parents[1] / "RCAEval/e2e/merit.py")
    assert module._GabanzhafConfig is _GabanzhafConfig
    assert callable(module.merit)


def test_split_modules_have_no_reverse_core_import_and_core_is_thin():
    import ast
    core = importlib.import_module("RCAEval.e2e.merit")
    assert "os.environ" not in Path(core.__file__).read_text()
    assert len(Path(core.__file__).read_text().splitlines()) <= 450
    for module in [merit_ablation, merit_diagnostics, merit_evidence,
                   merit_game, merit_trace]:
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "RCAEval.e2e.merit"
            elif isinstance(node, ast.Import):
                assert all(alias.name != "RCAEval.e2e.merit" for alias in node.names)


def test_switch_precedence_and_no_mutation():
    kwargs = {"use_metrics": True, "use_trace": True, "call_graph": {"a": ["b"]},
              "return_diagnostics": False}
    actual, _, disabled = merit_ablation.resolve_runtime_kwargs(kwargs, {
        "GABANZHAF_DISABLE_METRICS": "1", "GABANZHAF_DISABLE_TRACE": "1",
        "GABANZHAF_WRITE_DIAGNOSTICS": "1"})
    assert actual["use_metrics"] is True
    assert actual["use_trace"] is False and actual["call_graph"] == {} and disabled
    assert actual["return_diagnostics"] is True
    assert kwargs["use_trace"] is True and kwargs["call_graph"] == {"a": ["b"]}
    assert kwargs["return_diagnostics"] is False


@pytest.mark.parametrize("env", [{"MERIT_CUSUM_SLACK": "bad"}, {"GABANZHAF_TOP_K_SVCS": "bad"}])
def test_invalid_env_keeps_legacy_errors_even_with_explicit_kwarg(env):
    with pytest.raises(ValueError):
        merit_ablation.resolve_runtime_kwargs({"cusum_slack": .5, "top_k_svcs": 15}, env)


def test_parameter_sweep_source_interventions_still_construct():
    from script import merit_param_sweep as sweep
    edges = [("a", "b", .01, "latency", 1), ("b", "c", .4, "error", 0)]
    assert sweep._E5_TRACEFILTER_ONLY_0(edges, {"a": .01}, .2) == edges
    assert sweep._E1_SIGMA_NORMAL is not sweep.m._metric_evidence


def _graph_fixture(root):
    dataset = root / "dataset"
    case0 = dataset / "fault" / "0"
    case1 = dataset / "fault" / "1"
    case0.mkdir(parents=True)
    case1.mkdir(parents=True)
    pd.DataFrame({"time": [0], "only-reference_cpu": [0]}).to_csv(
        case0 / "data.csv", index=False
    )
    pd.DataFrame({"time": [0], "different-current-services_cpu": [0]}).to_csv(
        case1 / "data.csv", index=False
    )
    boundary0 = dict(start=0, end=40, inject_time=20,
                     mode="full_record_retrospective", version="v1")
    boundary1 = dict(start=10, end=70, inject_time=35,
                     mode="full_record_retrospective", version="v1")
    manifest = {"schema_version": "v1", "cases": {
        "fault_0": {"fault_directory": "fault", "case_directory": "0",
                    "metric_file": "data.csv", "boundary": boundary0},
        "fault_1": {"fault_directory": "fault", "case_directory": "1",
                    "metric_file": "data.csv", "boundary": boundary1},
    }}
    return case0, case1, boundary0, boundary1, manifest


def test_first_case_graph_is_fixed_across_boundaries_cache_and_restart(tmp_path):
    case0, case1, boundary0, boundary1, manifest = _graph_fixture(tmp_path)
    calls = []

    def build(case_dir, service_names, *, boundary, pre_fault, dropout_fraction):
        calls.append((Path(case_dir), service_names, dict(boundary)))
        return {Path(case_dir).name: ["downstream"]}

    merit_trace._CALL_GRAPH_CACHE.clear()
    first = merit_trace._resolve_graph_artifact(
        case_dir=str(case0), service_names={"only-reference"},
        input_boundary=boundary0, graph_source="first_case_pre_fault",
        exposure_manifest=manifest, graph_reference_case_id="fault_0",
        dropout_fraction=0.0, build_graph=build,
    )
    second = merit_trace._resolve_graph_artifact(
        case_dir=str(case1), service_names={"different-current-services"},
        input_boundary=boundary1, graph_source="first_case_pre_fault",
        exposure_manifest=manifest, graph_reference_case_id="fault_0",
        dropout_fraction=0.0, build_graph=build,
    )

    assert first.graph == second.graph == {"0": ["downstream"]}
    assert len(calls) == 1
    assert calls[0] == (case0, {"only-reference"}, boundary0)
    assert first.source_case == second.source_case == "fault_0"
    assert first.cache_hit is False and second.cache_hit is True

    merit_trace._CALL_GRAPH_CACHE.clear()  # simulate worker restart
    restarted = merit_trace._resolve_graph_artifact(
        case_dir=str(case1), service_names={"worker-restart-current-case"},
        input_boundary=boundary1, graph_source="first_case_pre_fault",
        exposure_manifest=manifest, graph_reference_case_id="fault_0",
        dropout_fraction=0.0, build_graph=build,
    )
    assert restarted.graph == {"0": ["downstream"]}
    assert calls[-1] == (case0, {"only-reference"}, boundary0)
    assert restarted.source_case == "fault_0"


def test_first_case_requires_reference_and_caches_empty_graph(tmp_path):
    case0, case1, boundary0, boundary1, manifest = _graph_fixture(tmp_path)
    merit_trace._CALL_GRAPH_CACHE.clear()
    with pytest.raises(ValueError, match="requires graph_reference_case_id"):
        merit_trace._resolve_graph_artifact(
            case_dir=str(case1), service_names={"svc"}, input_boundary=boundary1,
            graph_source="first_case_pre_fault", exposure_manifest=manifest,
            graph_reference_case_id=None, dropout_fraction=0.0,
        )

    calls = []
    def empty(*args, **kwargs):
        calls.append((args, kwargs))
        return {}

    first = merit_trace._resolve_graph_artifact(
        case_dir=str(case0), service_names={"svc"}, input_boundary=boundary0,
        graph_source="first_case_pre_fault", exposure_manifest=manifest,
        graph_reference_case_id="fault_0", dropout_fraction=0.0,
        build_graph=empty,
    )
    second = merit_trace._resolve_graph_artifact(
        case_dir=str(case1), service_names={"other"}, input_boundary=boundary1,
        graph_source="first_case_pre_fault", exposure_manifest=manifest,
        graph_reference_case_id="fault_0", dropout_fraction=0.0,
        build_graph=empty,
    )
    assert first.graph == second.graph == {}
    assert first.empty_graph_cached and second.empty_graph_cached
    assert len(calls) == 1 and second.cache_hit


def test_graph_provenance_for_union_and_per_case(tmp_path):
    case0, case1, boundary0, boundary1, manifest = _graph_fixture(tmp_path)

    def build(case_dir, service_names, *, boundary, pre_fault, dropout_fraction):
        return {Path(case_dir).name: ["shared"]}

    merit_trace._CALL_GRAPH_CACHE.clear()
    union = merit_trace._resolve_graph_artifact(
        case_dir=str(case1), service_names={"current"}, input_boundary=boundary1,
        graph_source="union_pre_fault", exposure_manifest=manifest,
        graph_reference_case_id=None, dropout_fraction=0.0, build_graph=build,
    )
    provenance = union.provenance()
    assert provenance["source_case"] is None
    assert provenance["source_cases"] == ["fault_0", "fault_1"]
    assert provenance["source_case_count"] == 2
    assert provenance["manifest_sha256"]

    per_case = merit_trace._resolve_graph_artifact(
        case_dir=str(case1), service_names={"current"}, input_boundary=boundary1,
        graph_source="per_case_pre_fault", exposure_manifest=manifest,
        graph_reference_case_id=None, dropout_fraction=0.0, build_graph=build,
    )
    assert per_case.provenance()["source_case"] == "fault_1"


def test_union_requires_nonempty_exposure_manifest(tmp_path):
    case = tmp_path / "dataset" / "fault" / "0"
    case.mkdir(parents=True)
    boundary = dict(start=0, end=10, inject_time=5,
                    mode="full_record_retrospective", version="v1")
    with pytest.raises(ValueError, match="requires an exposure manifest"):
        merit_trace._resolve_graph_artifact(
            case_dir=str(case), service_names={"svc"}, input_boundary=boundary,
            graph_source="union_pre_fault", exposure_manifest={"cases": {}},
            graph_reference_case_id=None, dropout_fraction=0.0,
        )


if __name__ == "__main__":
    import sys
    import tempfile
    source = Path(sys.argv[1])
    module = load_direct(source)
    with tempfile.TemporaryDirectory() as temporary:
        print(json.dumps({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "fingerprints": behavior(module),
                          "graph_fingerprints": graph_behavior(module, Path(temporary))}, indent=2))
