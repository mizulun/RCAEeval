import numpy as np
import pandas as pd

from RCAEval.e2e._rcd_causal_learn import local_skeleton_discovery
from RCAEval.e2e.rcd import F_NODE, rcd, run_pc


def _discrete_sample(seed=7, rows=160):
    rng = np.random.default_rng(seed)
    cause = rng.integers(0, 3, size=rows)
    effect = (cause + rng.integers(0, 2, size=rows)) % 3
    f_node = (cause > 0).astype(int)
    return np.column_stack([cause, effect, f_node])


def test_localized_pc_keeps_rcd_bookkeeping_and_labels():
    data = _discrete_sample()
    labels = {0: "cause_cpu", 1: "effect_cpu", 2: F_NODE}

    graph = local_skeleton_discovery(
        data,
        local_node=2,
        alpha=0.01,
        indep_test="chisq",
        labels=labels,
    )
    graph.to_nx_graph()

    assert F_NODE in graph.nx_graph
    assert graph.no_ci_tests > 0
    assert graph.p_values.shape == (3, 3)
    assert graph.mi.shape == (3,)


def test_full_pc_path_accepts_causal_learn_013_api():
    data = pd.DataFrame(_discrete_sample(), columns=["cause_cpu", "effect_cpu", F_NODE])
    graph = run_pc(data, alpha=0.01, localized=False)

    assert F_NODE in graph.nx_graph
    assert graph.no_ci_tests > 0


def test_rcd_runs_in_shared_python310_environment():
    rng = np.random.default_rng(11)
    rows = 120
    data = pd.DataFrame(
        {
            "time": np.arange(rows),
            "carts_cpu": np.r_[rng.normal(0, 0.2, rows // 2), rng.normal(4, 0.2, rows // 2)],
            "orders_cpu": rng.normal(0, 0.2, rows),
        }
    )

    result = rcd(data, inject_time=rows // 2, dataset="re2-ss", seed=20260713)

    assert isinstance(result["ranks"], list)
