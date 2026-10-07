import pytest

from script.select_global_sweep import method_label, select_global_value


def test_method_labels_match_runner_sweep_contract():
    assert method_label("victim_floor", 0.1) == "merit_victim_01"
    assert method_label("trace_topology_conf", 0.9) == "merit_tconf_09"
    assert method_label("coverage_conf", 0.5) == "merit_cconf_05"
    assert method_label("exact_candidate_k", 12) == "merit_k12"


def test_joint_flat_region_uses_parameter_specific_tie_break():
    values = [0.1, 0.2, 0.3]
    scores = {
        dataset: {0.1: 0.79, 0.2: 0.8, 0.3: 0.8}
        for dataset in ("re2-ob", "re2-ss", "re2-tt")
    }

    result = select_global_value("victim_floor", values, scores, 1 / 90)

    assert result["strategy"] == "joint_flat_region"
    assert result["selected_value"] == pytest.approx(0.3)


def test_loso_final_value_is_fold_median_not_best_held_out_score():
    values = [0.1, 0.2, 0.3]
    scores = {
        "re2-ob": {0.1: 1.0, 0.2: 0.0, 0.3: 0.0},
        "re2-ss": {0.1: 0.0, 0.2: 1.0, 0.3: 0.0},
        "re2-tt": {0.1: 0.0, 0.2: 0.0, 0.3: 1.0},
    }

    result = select_global_value(
        "trace_topology_conf", values, scores, tolerance=0.0
    )

    assert result["strategy"] == "leave_one_system_out"
    assert [fold["selected_value"] for fold in result["folds"]] == [0.2, 0.1, 0.1]
    assert result["selected_value"] == pytest.approx(0.1)
