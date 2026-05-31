import importlib

import numpy as np
import pandas as pd

baro_mod = importlib.import_module("RCAEval.e2e.baro")


def test_baro_reports_algorithm_time(monkeypatch):
    n = 20
    df = pd.DataFrame(
        {
            "time": np.arange(n),
            "svc-a_cpu": np.r_[np.zeros(10), np.ones(10)],
            "svc-b_cpu": np.zeros(n),
        }
    )
    ticks = iter([10.0, 10.75])
    monkeypatch.setattr(baro_mod, "perf_counter", lambda: next(ticks))

    result = baro_mod.baro(df, inject_time=10, dataset=None)

    assert result["algorithm_time_sec"] == 0.75
    assert "svc-a_cpu" in result["ranks"]
