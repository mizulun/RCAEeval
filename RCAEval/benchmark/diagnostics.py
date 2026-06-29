import json
from pathlib import Path


def write_diagnostics_sidecar(result_dir, result_filename, diagnostics):
    """Write case diagnostics outside the evaluator's top-level JSON glob."""
    diagnostics_dir = Path(result_dir) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostics_dir / result_filename
    output_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
