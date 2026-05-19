import os

from RCAEval.utility import dump_json, load_json


def test_dump_json_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "results" / "case.json"

    dump_json(str(path), {"ranks": ["svc_cpu"]})

    assert os.path.exists(path)
    assert load_json(str(path)) == {"ranks": ["svc_cpu"]}
