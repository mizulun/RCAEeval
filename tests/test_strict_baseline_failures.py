import pytest

from RCAEval.e2e import rca


def _failing_method(data, **_kwargs):
    raise RuntimeError("baseline failure")


def test_rca_decorator_propagates_failures_in_formal_mode(monkeypatch):
    monkeypatch.setenv("RCAEVAL_STRICT_FAILURES", "1")

    with pytest.raises(RuntimeError, match="baseline failure"):
        rca(_failing_method)(object())
