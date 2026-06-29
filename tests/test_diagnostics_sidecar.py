import json

from RCAEval.benchmark.diagnostics import write_diagnostics_sidecar


def test_diagnostics_sidecar_is_written_below_nested_directory(tmp_path):
    result_file = tmp_path / "service_cpu_1.json"
    result_file.write_text(json.dumps({"0": ["service_cpu"]}), encoding="utf-8")

    sidecar = write_diagnostics_sidecar(
        tmp_path,
        result_file.name,
        {"game_candidates": ["service"]},
    )

    assert sidecar == tmp_path / "diagnostics" / result_file.name
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "game_candidates": ["service"]
    }
    assert list(tmp_path.glob("*.json")) == [result_file]
