import subprocess
import sys
import importlib
from pathlib import Path

import pytest


def test_importing_merit_does_not_eagerly_import_unlicensed_baselines():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import sys
import RCAEval.e2e
assert 'RCAEval.e2e.causalrca' not in sys.modules
assert 'RCAEval.e2e.tracerca' not in sys.modules
from RCAEval.e2e import merit
assert callable(merit)
assert 'RCAEval.e2e.merit' in sys.modules
assert 'RCAEval.e2e.causalrca' not in sys.modules
assert 'RCAEval.e2e.tracerca' not in sys.modules
"""

    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert process.returncode == 0, process.stderr


def test_lazy_exports_keep_existing_public_import_contract():
    from RCAEval.e2e import baro, microrank, pc_pagerank, rcd

    assert all(
        callable(method)
        for method in (baro, microrank, pc_pagerank, rcd)
    )

    package = importlib.import_module("RCAEval.e2e")
    source_root = Path(package.__file__).parent
    for name in ("causalrca", "tracerca"):
        if (source_root / f"{name}.py").exists():
            assert callable(getattr(package, name))
        else:
            with pytest.raises(ModuleNotFoundError):
                getattr(package, name)
