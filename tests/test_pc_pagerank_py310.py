import numpy as np
import pandas as pd
from unittest.mock import patch

from RCAEval.graph_construction.pc import (
    _NumericallyStableFisherZ,
    pc_numerically_stable,
)
from RCAEval.graph_heads.page_rank import page_rank


def test_pc_handles_perfectly_collinear_metrics_without_math_domain_error():
    x = np.linspace(-1.0, 1.0, 80)
    data = pd.DataFrame(
        {
            "service_a_error": x,
            "service_b_error": 2.0 * x + 3.0,
            "frontend_latency": np.sin(3.0 * x),
        }
    )

    graph = pc_numerically_stable(data, show_progress=False)

    assert graph.G.graph.shape == (3, 3)


def test_fisher_z_conservatively_retains_degenerate_conditional_edge():
    x = np.linspace(-1.0, 1.0, 80)
    values = np.column_stack((x, 2.0 * x + 3.0, np.sin(3.0 * x)))
    test = _NumericallyStableFisherZ(values)

    with patch(
        "RCAEval.graph_construction.pc.np.linalg.inv",
        side_effect=np.linalg.LinAlgError,
    ), patch(
        "RCAEval.graph_construction.pc.np.linalg.pinv",
        return_value=np.zeros((3, 3)),
    ):
        p_value = test(0, 2, [1])

    assert p_value == 0.0


def test_page_rank_uses_current_scikit_network_api_and_keeps_isolates():
    # causal-learn encoding for service_a -> service_b plus one isolated node
    adjacency = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )

    ranks = page_rank(
        adjacency,
        node_names=["service_a", "service_b", "isolated"],
    )

    assert {name for name, _score in ranks} == {
        "service_a",
        "service_b",
        "isolated",
    }
    assert ranks[0][0] == "service_a"
