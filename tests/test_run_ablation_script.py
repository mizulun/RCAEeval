from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_run_ablation_script_supports_main_table_modes():
    script = (ROOT / "run_ablation.sh").read_text(encoding="utf-8")

    expected_cases = {
        "full)": 'echo ""',
        "no-trace)": 'echo "GABANZHAF_DISABLE_TRACE=1"',
        "no-metric)": 'echo "GABANZHAF_DISABLE_METRICS=1"',
        "trace-evidence)": 'echo "GABANZHAF_RANK_MODE=trace_evidence"',
        "metric-evidence)": 'echo "GABANZHAF_RANK_MODE=metric_evidence"',
        "metric-only)": 'echo "GABANZHAF_RANK_MODE=service_evidence"',
        "metric+trace)": 'echo "GABANZHAF_RANK_MODE=service_evidence_trace"',
        "no-coverage)": 'echo "GABANZHAF_DISABLE_COVERAGE=1"',
        "no-victim-demotion)": 'echo "GABANZHAF_DISABLE_VICTIM_DEMOTION=1"',
        "no-trace-topology)": 'echo "GABANZHAF_DISABLE_TRACE_TOPOLOGY=1"',
        "pure-banzhaf)": 'echo "GABANZHAF_BANZHAF_MODE=bruteforce"',
        "residual)": 'echo "GABANZHAF_TRACE_RESIDUAL=1"',
    }

    for case_label, env_line in expected_cases.items():
        assert case_label in script
        assert env_line in script


def test_run_ablation_script_keeps_default_modes_narrow():
    script = (ROOT / "run_ablation.sh").read_text(encoding="utf-8")

    assert 'MODES="${MODES:-full no-trace no-metric}"' in script


def test_run_ablation_script_allows_optional_test_args():
    script = (ROOT / "run_ablation.sh").read_text(encoding="utf-8")

    assert 'TEST_ARGS="${TEST_ARGS:-}"' in script
    assert '--result-dir output/results $TEST_ARGS' in script
