"""metric_zero_iqr_policy: denominators for zero normal-period IQR (Issue #2)."""

import importlib
import inspect
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from RCAEval.e2e import merit_ablation, merit_evidence
from RCAEval.e2e.merit_config import _GabanzhafConfig


POLICIES = ("unit", "std", "exclude")
COLS = ["sparse_mem", "noisy_cpu", "flat_cpu", "flatspike_cpu"]


def clean_env():
    return {k: v for k, v in os.environ.items()
            if not k.startswith(("MERIT_", "GABANZHAF_"))}


def matrices(scale=1.0):
    """Hand-derivable columns; 100 normal rows and 20 constant post rows.

    sparse_mem: 90 zeros + 10 tens -> median/q25/q75 = 0, population std 3.
    noisy_cpu:  0..9 repeated      -> median 4.5, q25 2, q75 7 (IQR 5).
    flat_cpu / flatspike_cpu: all zeros -> IQR 0 and std 0.
    """
    normal = np.column_stack([
        np.r_[np.zeros(90), np.full(10, 10.)],
        np.tile(np.arange(10.), 10),
        np.zeros(100),
        np.zeros(100),
    ])
    post = np.tile([6., 24.5, 4., 50.], (20, 1))
    scaled = np.array([scale, 1., 1., 1.])
    return normal * scaled, post * scaled


def raw_p95_and_score(normal, post, cols, **kwargs):
    """Spy the first _norm01_arr input: the raw per-column p95 of rz."""
    calls = []
    original = merit_evidence._norm01_arr

    def spy(values):
        calls.append(np.asarray(values, dtype=float).copy())
        return original(values)

    config = _GabanzhafConfig.from_kwargs(kwargs)
    with patch.object(merit_evidence, "_norm01_arr", spy):
        score = merit_evidence._metric_evidence(normal, post, cols, config=config)
    return calls[0], score


def test_default_field_is_unit_and_explicit_unit_scores_match():
    assert _GabanzhafConfig().metric_zero_iqr_policy == "unit"
    assert vars(_GabanzhafConfig.from_kwargs({}))["metric_zero_iqr_policy"] == "unit"
    normal, post = matrices()
    default = merit_evidence._metric_evidence(normal, post, COLS)
    explicit = merit_evidence._metric_evidence(
        normal, post, COLS,
        config=_GabanzhafConfig.from_kwargs({"metric_zero_iqr_policy": "unit"}),
    )
    assert default == explicit


@pytest.mark.parametrize("policy", POLICIES)
def test_valid_policies_accepted_directly_and_from_kwargs(policy):
    assert _GabanzhafConfig(metric_zero_iqr_policy=policy).metric_zero_iqr_policy == policy
    config = _GabanzhafConfig.from_kwargs({"metric_zero_iqr_policy": policy})
    assert config.metric_zero_iqr_policy == policy


@pytest.mark.parametrize("bad", ["UNIT", " unit", "", "median", None, 0, ["std"], b"std"])
def test_invalid_policies_raise_instead_of_falling_back(bad):
    with pytest.raises(ValueError, match="metric_zero_iqr_policy"):
        _GabanzhafConfig(metric_zero_iqr_policy=bad)
    with pytest.raises(ValueError, match="metric_zero_iqr_policy"):
        _GabanzhafConfig.from_kwargs({"metric_zero_iqr_policy": bad})


@pytest.mark.parametrize("policy", ["std", "exclude"])
def test_non_unit_policy_rejects_cusum(policy):
    with pytest.raises(ValueError, match="scorer"):
        _GabanzhafConfig(scorer="cusum", metric_zero_iqr_policy=policy)
    with pytest.raises(ValueError, match="scorer"):
        _GabanzhafConfig.from_kwargs({"scorer": "cusum", "metric_zero_iqr_policy": policy})


def test_unit_with_cusum_remains_supported_and_unchanged():
    config = _GabanzhafConfig.from_kwargs({"scorer": "cusum", "metric_zero_iqr_policy": "unit"})
    normal, post = matrices()
    assert merit_evidence._metric_evidence(normal, post, COLS, config=config) == (
        merit_ablation._cusum_metric_evidence(normal, post, COLS, config)
    )


def test_research_source_anchors_remain_unique():
    # script/merit_param_sweep.py and script/merit_design_audit.py replace
    # these literals in inspect.getsource(_metric_evidence); each must occur once.
    merit = importlib.import_module("RCAEval.e2e.merit")
    assert merit._metric_evidence is merit_evidence._metric_evidence
    source = inspect.getsource(merit_evidence._metric_evidence)
    for anchor in (
        "med = np.nanmedian(normal_mat, axis=0)",
        "iqr = np.where((q75 - q25) > 0, q75 - q25, 1.)",
        "rz = np.nan_to_num(rz, nan=0., posinf=0., neginf=0.)",
        "rz = np.where(near_zero_baseline, np.clip(rz, 0., 10.), rz)",
        "np.clip(rz, 0., 10.)",
        "rz > 3.0",
    ):
        assert source.count(anchor) == 1, anchor


def test_hand_computed_raw_deviation_per_policy():
    normal, post = matrices()
    # Inputs behind the hand derivation in matrices().
    sparse = normal[:, 0]
    assert np.percentile(sparse, 25) == np.percentile(sparse, 75) == np.median(sparse) == 0.
    assert np.std(sparse) == pytest.approx(3.)
    assert np.percentile(normal[:, 1], 75) - np.percentile(normal[:, 1], 25) == pytest.approx(5.)

    raw = {policy: raw_p95_and_score(normal, post, COLS, metric_zero_iqr_policy=policy)[0]
           for policy in POLICIES}
    # sparse: unit |6-0|/1; std |6-0|/3 = 2; exclude imputes zero evidence.
    # noisy (positive IQR): |24.5-4.5|/5 = 4 under every policy.
    # flat (IQR=0, std=0): std falls back to 1 like unit; exclude loses the
    #   flat-to-shift signal.
    # flatspike: 50 under the both-zero fallback, then the fixed cap 10.
    np.testing.assert_allclose(raw["unit"], [6., 4., 4., 10.])
    np.testing.assert_allclose(raw["std"], [2., 4., 4., 10.])
    np.testing.assert_allclose(raw["exclude"], [0., 4., 0., 0.])


def test_positive_iqr_raw_is_unchanged_but_final_evidence_can_change():
    normal, post = matrices()
    out = {policy: raw_p95_and_score(normal, post, COLS, metric_zero_iqr_policy=policy)
           for policy in POLICIES}
    assert out["unit"][0][1] == out["std"][0][1] == out["exclude"][0][1]
    # Min-max normalization across columns makes final evidence relative, so a
    # positive-IQR column's final score moves when zero-IQR columns change.
    assert out["std"][1]["noisy_cpu"] > out["unit"][1]["noisy_cpu"]
    # exclude keeps every column in the output with zero evidence.
    assert list(out["exclude"][1]) == COLS
    assert out["exclude"][1]["sparse_mem"] == 0.0
    assert out["exclude"][1]["flat_cpu"] == 0.0


def test_scale_dependence_is_conditional_not_universal():
    # Rescaling a sparse zero-IQR column scales unit-policy raw evidence,
    # while the std denominator rescales with it here.
    for scale in (1., 1000.):
        normal, post = matrices(scale=scale)
        unit = raw_p95_and_score(normal, post, COLS, metric_zero_iqr_policy="unit")[0]
        std = raw_p95_and_score(normal, post, COLS, metric_zero_iqr_policy="std")[0]
        assert unit[0] == pytest.approx(6. * scale)
        assert std[0] == pytest.approx(2.)
    # Both-zero columns still use denominator 1 under std, so their raw
    # evidence depends on units until the fixed cap binds.
    normal, post = matrices()
    for factor, expected in ((1., 4.), (2., 8.), (1000., 10.)):
        shifted = post.copy()
        shifted[:, 2] *= factor
        std = raw_p95_and_score(normal, shifted, COLS, metric_zero_iqr_policy="std")[0]
        assert std[2] == pytest.approx(expected)


def test_small_std_can_inflate_std_policy_and_cap_depends_on_absolute_guard():
    # 99 zeros + one small value: IQR = 0, std = value * sqrt(99) / 100.
    normal = np.column_stack([np.r_[np.zeros(99), 1e-3], np.r_[np.zeros(99), 1e-5]])
    post = np.ones((20, 2))
    cols = ["tiny_cpu", "tinier_cpu"]
    tiny_std, tinier_std = np.std(normal, axis=0)
    assert tiny_std > 1e-6 >= tinier_std
    std = raw_p95_and_score(normal, post, cols, metric_zero_iqr_policy="std")[0]
    unit = raw_p95_and_score(normal, post, cols, metric_zero_iqr_policy="unit")[0]
    np.testing.assert_allclose(unit, [1., 1.])
    # Above the 1e-6 guard: no cap, so a tiny baseline spread inflates rz.
    assert std[0] == pytest.approx(1. / tiny_std)
    assert std[0] > 1000.
    # At or below the guard the existing fixed cap still applies.
    assert std[1] == pytest.approx(10.)


def public_dataset():
    """One synthetic case shared by every policy; names carry no labels."""
    rng = np.random.default_rng(20261022)
    sparse_normal = np.zeros(40)
    sparse_normal[[3, 13, 23, 33]] = 10.
    normal = np.column_stack([
        rng.uniform(0, 2, 40), rng.uniform(0, 2, 40), sparse_normal,
        np.zeros(40), rng.uniform(0, 2, 40),
    ])
    post = np.column_stack([
        rng.uniform(0, 2, 30) + 5., rng.uniform(0, 2, 30) + 1., np.full(30, 6.),
        np.full(30, 3.), rng.uniform(0, 2, 30),
    ])
    cols = ["svc-a_cpu", "svc-a_latency", "svc-b_mem", "svc-c_cpu", "svc-d_cpu"]
    data = pd.DataFrame(np.vstack([normal, post]), columns=cols)
    data.insert(0, "time", np.arange(len(data)))
    return data, normal, post, cols


def test_public_default_equals_explicit_unit():
    merit = importlib.import_module("RCAEval.e2e.merit")
    data, *_ = public_dataset()
    with patch.dict(os.environ, clean_env(), clear=True):
        default = merit.merit(data, inject_time=40, return_diagnostics=True)
        explicit = merit.merit(data, inject_time=40, metric_zero_iqr_policy="unit",
                               return_diagnostics=True)
    assert default["ranks"] == explicit["ranks"]
    for key in ("service_ranks", "metric_evidence", "service_banzhaf"):
        assert default["diagnostics"][key] == explicit["diagnostics"][key]
    for result in (default, explicit):
        assert result["diagnostics"]["configuration"]["metric_zero_iqr_policy"] == "unit"


def test_public_merit_routes_policy_and_records_diagnostics():
    merit = importlib.import_module("RCAEval.e2e.merit")
    data, normal, post, cols = public_dataset()
    evidence = {}
    with patch.dict(os.environ, clean_env(), clear=True):
        for policy in POLICIES:
            result = merit.merit(data, inject_time=40, metric_zero_iqr_policy=policy,
                                 return_diagnostics=True)
            diagnostics = result["diagnostics"]
            assert diagnostics["configuration"]["metric_zero_iqr_policy"] == policy
            expected = merit_evidence._metric_evidence(
                normal, post, cols,
                config=_GabanzhafConfig.from_kwargs({"metric_zero_iqr_policy": policy}),
            )
            assert diagnostics["metric_evidence"] == pytest.approx(expected, rel=0, abs=1e-12)
            evidence[policy] = diagnostics["metric_evidence"]
    assert evidence["std"] != evidence["unit"]
    assert evidence["exclude"] != evidence["unit"]
    assert evidence["exclude"]["svc-b_mem"] == evidence["exclude"]["svc-c_cpu"] == 0.0


def test_public_merit_rejects_invalid_policy_and_cusum_combination():
    merit = importlib.import_module("RCAEval.e2e.merit")
    data, *_ = public_dataset()
    with patch.dict(os.environ, clean_env(), clear=True):
        with pytest.raises(ValueError, match="metric_zero_iqr_policy"):
            merit.merit(data, inject_time=40, metric_zero_iqr_policy="median")
        with pytest.raises(ValueError, match="scorer"):
            merit.merit(data, inject_time=40, scorer="cusum", metric_zero_iqr_policy="std")
