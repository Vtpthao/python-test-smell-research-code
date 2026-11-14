"""Read-only preflight checks for the code-only release candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

LABELS = ("AR", "CTL", "MNT", "ET")
FROZEN_INPUTS = {
    "dataset/final/python_4label_test_smell_experimental_v2.csv": "ad1d9cd262f11b6ba29287e791c4cdd1be3d8dd3bbb36b949821b8bd807448da",
    "dataset/final/lopo_fold_assignments_v2.csv": "d3609f61bacdc655fe84e64273e0bf74ca801f2bbf16647693ded36b9e661faf",
    "dataset/final/lopo_fold_manifest_v2.csv": "51b37dc27182d254f9182ca01b703e3236c6ba308d426e8eeac2942248e0c8b3",
    "dataset/final/multi_label_experiment_protocol_v2.md": "bb45415df64276cf80a305d4c385b84215b99f46303c8915c0c1f340ea2b2a64",
}
RUNS = {
    "R1": ("experiments/v2/r1_tfidf", "r1_v2"),
    "R2": ("experiments/v2/r2_ast", "r2_v2"),
    "R3": ("experiments/v2/r3_unixcoder", "r3_v2"),
    "H12": ("experiments/v2/h12", "h12_v2"),
    "H12-B": ("experiments/v2/h12_balanced", "h12b_v2"),
}
ARTIFACT_SUFFIXES = {
    "predictions": "oop_predictions.csv",
    "pooled": "pooled_metrics.csv",
    "per_label": "per_label_metrics.csv",
    "per_project": "per_project_metrics.csv",
    "per_project_label": "per_project_label_metrics.csv",
}
R1_OUTPUT_FILES = (
    "r1_v2_oop_predictions.csv",
    "r1_v2_pooled_metrics.csv",
    "r1_v2_per_label_metrics.csv",
    "r1_v2_per_project_metrics.csv",
    "r1_v2_per_project_label_metrics.csv",
    "r1_v2_fold_vocabulary_audit.csv",
    "r1_v2_summary.md",
    "r1_v2_run_manifest.csv",
)


class PreflightError(Exception):
    """An expected input/output condition prevents a safe run."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(root: Path, relative: str) -> Path:
    path = root / relative
    _refuse_symlink_components(root, relative, "input")
    if not path.is_file():
        raise PreflightError(f"Required input is missing: {relative}")
    return path


def _refuse_symlink_components(root: Path, relative: str, kind: str) -> None:
    current = root
    if current.is_symlink():
        raise PreflightError(f"Refusing symlink candidate root: {root}")
    for component in Path(relative).parts:
        current = current / component
        if current.is_symlink():
            raise PreflightError(f"Refusing symlink {kind} path component: {current.relative_to(root)}")


def _verify_frozen_inputs(root: Path) -> None:
    for relative, expected in FROZEN_INPUTS.items():
        path = _regular_file(root, relative)
        actual = sha256(path)
        if actual != expected:
            raise PreflightError(
                f"Frozen input hash mismatch: {relative} (expected {expected}, observed {actual})"
            )


def _require_fresh_output(root: Path, relative: str) -> None:
    path = root / relative
    _refuse_symlink_components(root, relative, "output")
    if path.exists():
        if not path.is_dir():
            raise PreflightError(f"Output path exists and is not a directory: {relative}")
        if any(path.iterdir()):
            raise PreflightError(
                f"Output directory is not empty: {relative}. Preserve existing files and choose a fresh candidate copy."
            )
        raise PreflightError(
            f"Output directory already exists: {relative}. Remove or archive it yourself before running."
        )


def _read_manifest(path: Path, run_name: str) -> dict[str, str]:
    csv.field_size_limit(2**31 - 1)
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        raise PreflightError(f"Cannot read {run_name} run manifest: {path.name}: {exc}") from None
    if len(rows) != 1:
        raise PreflightError(f"{run_name} run manifest must contain exactly one row")
    return rows[0]


def _csv_row_count(path: Path, run_name: str, artifact_name: str) -> int:
    csv.field_size_limit(2**31 - 1)
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            next(reader, None)
            return sum(1 for _ in reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        raise PreflightError(
            f"Cannot parse {run_name} {artifact_name} CSV {path.name}: {exc}"
        ) from None


def _verify_run_manifest(root: Path, run_name: str, directory: str, prefix: str) -> None:
    manifest_relative = f"{directory}/{prefix}_run_manifest.csv"
    manifest_path = _regular_file(root, manifest_relative)
    row = _read_manifest(manifest_path, run_name)
    try:
        recorded_hashes = json.loads(row["artifact_sha256_json"])
        recorded_counts = json.loads(row["artifact_row_counts_json"])
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise PreflightError(f"{run_name} manifest has invalid artifact hashes/counts: {exc}") from None

    for frozen_field, expected_relative in (
        ("dataset_sha256", "dataset/final/python_4label_test_smell_experimental_v2.csv"),
        ("assignment_sha256", "dataset/final/lopo_fold_assignments_v2.csv"),
        ("fold_manifest_sha256", "dataset/final/lopo_fold_manifest_v2.csv"),
        ("protocol_sha256", "dataset/final/multi_label_experiment_protocol_v2.md"),
    ):
        expected = FROZEN_INPUTS[expected_relative]
        if row.get(frozen_field, "").lower() != expected:
            raise PreflightError(f"{run_name} manifest {frozen_field} does not match the frozen V2 input")

    for key, suffix in ARTIFACT_SUFFIXES.items():
        relative = f"{directory}/{prefix}_{suffix}"
        path = _regular_file(root, relative)
        expected_hash = recorded_hashes.get(key)
        if not isinstance(expected_hash, str) or sha256(path) != expected_hash:
            raise PreflightError(f"{run_name} manifest hash mismatch for {key}: {relative}")
        expected_rows = recorded_counts.get(key)
        if not isinstance(expected_rows, int) or _csv_row_count(path, run_name, key) != expected_rows:
            raise PreflightError(f"{run_name} manifest row-count mismatch for {key}: {relative}")


def preflight_r1(root: Path) -> None:
    """Verify all immutable R1 inputs and reject an existing output tree."""
    _verify_frozen_inputs(root)
    _require_fresh_output(root, "experiments/v2/r1_tfidf")


def preflight_analysis(root: Path) -> None:
    """Verify frozen inputs, five saved runs, and an unused analysis output path."""
    _verify_frozen_inputs(root)
    for run_name, (directory, prefix) in RUNS.items():
        _verify_run_manifest(root, run_name, directory, prefix)
    _require_fresh_output(root, "experiments/v2/final_comparison")


def _run_checked(check: Any, root: Path) -> int:
    try:
        check(root)
    except (PreflightError, OSError, ValueError, KeyError) as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2
    print("Preflight passed. No files were written and no model was trained.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only checks before selected frozen V2 runners")
    parser.add_argument("target", choices=("r1", "analysis"), help="Input contract to verify")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    check = preflight_r1 if args.target == "r1" else preflight_analysis
    return _run_checked(check, root)


def preflight_r1_or_exit(root: Path) -> None:
    if _run_checked(preflight_r1, root):
        raise SystemExit(2)


def preflight_analysis_or_exit(root: Path) -> None:
    if _run_checked(preflight_analysis, root):
        raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
