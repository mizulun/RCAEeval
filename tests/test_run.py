"""Tests."""
import os
import shutil
from os import path
import subprocess
import sys
import pytest

from typing import Callable
import numpy as np
import pandas as pd
import pytest
import tempfile


pytestmark = pytest.mark.skip(
    reason="legacy main-ase.py intentionally supports only Python 3.8/3.12"
)


def test_run():
    command = [sys.executable, "main-ase.py", "--method", "run", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"
