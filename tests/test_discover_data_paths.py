"""Case discovery must feed the curated simple_data.csv, not the raw container-metric data.csv (issue #12)."""
from pathlib import Path

import pytest

from script import run_paper_dataset
from script.run_paper_dataset import discover_data_paths, prefer_simple_data


def _case(root: Path, name: str, *files: str) -> Path:
    case_dir = root / name / "1"
    case_dir.mkdir(parents=True)
    for f in files:
        (case_dir / f).write_text("time\n0\n")
    return case_dir


def test_prefer_simple_data_swaps_only_when_sibling_exists(tmp_path):
    both = _case(tmp_path, "carts_cpu", "data.csv", "simple_data.csv")
    raw_only = _case(tmp_path, "orders_mem", "data.csv")
    paths = [both / "data.csv", raw_only / "data.csv"]
    assert prefer_simple_data(paths) == [both / "simple_data.csv", raw_only / "data.csv"]


def test_prefer_simple_data_leaves_other_inputs_alone(tmp_path):
    re2 = _case(tmp_path, "ts-auth-service_cpu", "simple_metrics.csv")
    assert prefer_simple_data([re2 / "simple_metrics.csv"]) == [re2 / "simple_metrics.csv"]


def test_discover_data_paths_default_keeps_data_csv(tmp_path, monkeypatch):
    dataset_root = Path("data") / "fake-ds"
    monkeypatch.setitem(run_paper_dataset.DATASET_PATHS, "fake", dataset_root)
    root = tmp_path / dataset_root
    _case(root, "carts_cpu", "data.csv", "simple_data.csv")
    _case(root, "orders_mem", "data.csv")
    found = discover_data_paths(tmp_path, "fake")
    assert [p.relative_to(root).as_posix() for p in found] == [
        "carts_cpu/1/data.csv",
        "orders_mem/1/data.csv",
    ]


def test_discover_data_paths_opt_in_prefers_simple_data(tmp_path, monkeypatch):
    dataset_root = Path("data") / "fake-ds"
    monkeypatch.setitem(run_paper_dataset.DATASET_PATHS, "fake", dataset_root)
    root = tmp_path / dataset_root
    _case(root, "carts_cpu", "data.csv", "simple_data.csv")
    _case(root, "orders_mem", "data.csv")
    found = discover_data_paths(tmp_path, "fake", prefer_simple=True)
    assert [p.relative_to(root).as_posix() for p in found] == [
        "carts_cpu/1/simple_data.csv",
        "orders_mem/1/data.csv",
    ]


def test_discover_data_paths_re2_layout_unchanged(tmp_path, monkeypatch):
    dataset_root = Path("data") / "fake-re2"
    monkeypatch.setitem(run_paper_dataset.DATASET_PATHS, "fake-re2", dataset_root)
    root = tmp_path / dataset_root
    _case(root, "ts-auth-service_cpu", "simple_metrics.csv", "traces.csv")
    assert discover_data_paths(tmp_path, "fake-re2") == [root / "ts-auth-service_cpu" / "1" / "simple_metrics.csv"]


@pytest.mark.skipif(not Path("data/train-ticket").is_dir(), reason="RE1 train-ticket dataset not downloaded")
def test_real_re1_tt_resolves_to_simple_data():
    paths = prefer_simple_data(sorted(Path("data/train-ticket").glob("*/*/data.csv")))
    assert len(paths) == 125
    assert all(p.name == "simple_data.csv" for p in paths)
