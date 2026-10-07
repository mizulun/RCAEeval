from pathlib import Path

import pytest

from script.verify_environment_lock import audit_environment, parse_lock


def test_lock_parser_normalizes_names_and_ignores_editable_project(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "# frozen\n-e .\nExample_Package==1.2.3\nother.package==4\n",
        encoding="utf-8",
    )

    assert parse_lock(lock) == {"example-package": "1.2.3", "other-package": "4"}


def test_environment_audit_requires_exact_python_and_pin_versions(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("numpy==1.26.4\nscipy==1.15.3\n", encoding="utf-8")

    report = audit_environment(
        lock,
        "3.10.18",
        actual_packages={"NumPy": "1.26.4", "scipy": "1.15.2", "extra": "1"},
        actual_python="3.10.18",
    )

    assert report["ok"] is False
    assert report["missing"] == []
    assert report["version_mismatches"] == {
        "scipy": {"expected": "1.15.3", "actual": "1.15.2"}
    }
    assert report["extra_packages_allowed"] is True


def test_environment_audit_reports_missing_and_python_mismatch(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("numpy==1.26.4\n", encoding="utf-8")

    report = audit_environment(
        lock,
        "3.10.18",
        actual_packages={},
        actual_python="3.10.17",
    )

    assert report["ok"] is False
    assert report["missing"] == ["numpy"]
    assert report["python"]["matches"] is False


def test_lock_parser_rejects_non_exact_requirements(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("numpy>=1.26\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-exact requirement"):
        parse_lock(Path(lock))
