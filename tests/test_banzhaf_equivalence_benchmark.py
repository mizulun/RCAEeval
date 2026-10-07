from script.benchmark_banzhaf_equivalence import (
    observed_game_shapes,
    run_benchmark,
    run_observed_shape_benchmark,
)


def test_banzhaf_equivalence_benchmark_preserves_raw_repetitions():
    raw, summaries = run_benchmark(
        k_values=[2, 3],
        target_count=4,
        repetitions=2,
        seed=7,
        absolute_tolerance=1e-9,
    )

    assert len(raw) == 2 * 2 * 2
    assert [item["k"] for item in summaries] == [2, 3]
    assert all(item["equivalent"] for item in summaries)
    assert all(item["max_absolute_error"] <= 1e-9 for item in summaries)
    assert all(
        item["median_speedup_bruteforce_over_closed_form"] > 0
        for item in summaries
    )


def test_observed_shape_benchmark_uses_actual_candidate_target_pairs():
    shapes = observed_game_shapes(
        [
            {
                "method": "merit_full",
                "status": "success",
                "final_candidate_k": 3,
                "target_count": 5,
            },
            {
                "method": "merit_full",
                "status": "success",
                "final_candidate_k": 3,
                "target_count": 5,
            },
            {
                "method": "other",
                "status": "success",
                "final_candidate_k": 4,
                "target_count": 6,
            },
        ]
    )

    assert shapes == [{"k": 3, "target_count": 5, "observed_cases": 2}]
    raw, summaries = run_observed_shape_benchmark(
        shapes,
        repetitions=1,
        seed=7,
        absolute_tolerance=1e-9,
        max_k=15,
    )
    assert {row["target_count"] for row in raw} == {5}
    assert summaries[0]["observed_cases"] == 2
