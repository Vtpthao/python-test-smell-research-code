"""Synthetic tests for safe, read-only release-candidate preflight."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.input_preflight import (
    PreflightError,
    _require_fresh_output,
    _verify_run_manifest,
    preflight_analysis,
    preflight_r1,
)


class FrozenInputPreflightTests(unittest.TestCase):
    def test_r1_fails_clearly_when_protected_inputs_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(PreflightError, "Required input is missing"):
                preflight_r1(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_analysis_fails_before_looking_for_results_without_frozen_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(PreflightError, "Required input is missing"):
                preflight_analysis(root)
            self.assertEqual(list(root.iterdir()), [])


class OutputPreflightTests(unittest.TestCase):
    def test_absent_output_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _require_fresh_output(Path(temporary), "run/output")

    def test_existing_output_directory_is_refused_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run/output"
            output.mkdir(parents=True)
            preserved = output / "keep.txt"
            preserved.write_text("synthetic", encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "not empty"):
                _require_fresh_output(root, "run/output")
            self.assertEqual(preserved.read_text(encoding="utf-8"), "synthetic")

    def test_existing_empty_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run/output").mkdir(parents=True)
            with self.assertRaisesRegex(PreflightError, "already exists"):
                _require_fresh_output(root, "run/output")

    def test_symlinked_output_parent_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, external = base / "candidate", base / "external"
            root.mkdir()
            external.mkdir()
            (root / "experiments").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(PreflightError, "symlink output path component"):
                _require_fresh_output(root, "experiments/run")
            self.assertEqual(list(external.iterdir()), [])


class RunManifestPreflightTests(unittest.TestCase):
    @staticmethod
    def write_csv(path: Path, header: str = "field\n") -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header, encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_valid_synthetic_run_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, prefix = "experiments/v2/test_run", "test_v2"
            artifacts = {
                "predictions": f"{prefix}_oop_predictions.csv",
                "pooled": f"{prefix}_pooled_metrics.csv",
                "per_label": f"{prefix}_per_label_metrics.csv",
                "per_project": f"{prefix}_per_project_metrics.csv",
                "per_project_label": f"{prefix}_per_project_label_metrics.csv",
            }
            hashes = {}
            counts = {}
            for key, filename in artifacts.items():
                path = root / directory / filename
                hashes[key] = self.write_csv(path)
                counts[key] = 0
            frozen = {
                "dataset_sha256": "ad1d9cd262f11b6ba29287e791c4cdd1be3d8dd3bbb36b949821b8bd807448da",
                "assignment_sha256": "d3609f61bacdc655fe84e64273e0bf74ca801f2bbf16647693ded36b9e661faf",
                "fold_manifest_sha256": "51b37dc27182d254f9182ca01b703e3236c6ba308d426e8eeac2942248e0c8b3",
                "protocol_sha256": "bb45415df64276cf80a305d4c385b84215b99f46303c8915c0c1f340ea2b2a64",
                "artifact_sha256_json": json.dumps(hashes),
                "artifact_row_counts_json": json.dumps(counts),
            }
            manifest = root / directory / f"{prefix}_run_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(frozen), lineterminator="\n")
                writer.writeheader()
                writer.writerow(frozen)
            _verify_run_manifest(root, "TEST", directory, prefix)

    def test_run_manifest_hash_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, prefix = "experiments/v2/test_run", "test_v2"
            artifacts = {
                "predictions": f"{prefix}_oop_predictions.csv",
                "pooled": f"{prefix}_pooled_metrics.csv",
                "per_label": f"{prefix}_per_label_metrics.csv",
                "per_project": f"{prefix}_per_project_metrics.csv",
                "per_project_label": f"{prefix}_per_project_label_metrics.csv",
            }
            hashes = {}
            counts = {}
            for key, filename in artifacts.items():
                path = root / directory / filename
                hashes[key] = self.write_csv(path)
                counts[key] = 0
            hashes["predictions"] = "0" * 64
            frozen = {
                "dataset_sha256": "ad1d9cd262f11b6ba29287e791c4cdd1be3d8dd3bbb36b949821b8bd807448da",
                "assignment_sha256": "d3609f61bacdc655fe84e64273e0bf74ca801f2bbf16647693ded36b9e661faf",
                "fold_manifest_sha256": "51b37dc27182d254f9182ca01b703e3236c6ba308d426e8eeac2942248e0c8b3",
                "protocol_sha256": "bb45415df64276cf80a305d4c385b84215b99f46303c8915c0c1f340ea2b2a64",
                "artifact_sha256_json": json.dumps(hashes),
                "artifact_row_counts_json": json.dumps(counts),
            }
            manifest = root / directory / f"{prefix}_run_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(frozen), lineterminator="\n")
                writer.writeheader()
                writer.writerow(frozen)
            with self.assertRaisesRegex(PreflightError, "manifest hash mismatch"):
                _verify_run_manifest(root, "TEST", directory, prefix)


if __name__ == "__main__":
    unittest.main()
