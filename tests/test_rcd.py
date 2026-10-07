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
    reason=(
        "legacy main-ase.py intentionally supports only Python 3.8/3.12; "
        "paper RCD on 3.10 is covered by test_rcd_py310.py"
    )
)


def test_rcd():
    command = [sys.executable, "main-ase.py", "--method", "rcd", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"


def test_ht():
    command = [sys.executable, "main-ase.py", "--method", "ht", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"


def test_e_diagnosis():
    command = [sys.executable, "main-ase.py", "--method", "e_diagnosis", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"
