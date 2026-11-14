#!/usr/bin/env python3
"""Analysis-only final comparison of frozen V2 OOP predictions; never trains models."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import binomtest, rankdata
from input_preflight import preflight_analysis_or_exit

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "dataset/final"
V2 = ROOT / "experiments/v2"
OUTPUT = V2 / "final_comparison"
DATASET = FINAL / "python_4label_test_smell_experimental_v2.csv"
ASSIGNMENTS = FINAL / "lopo_fold_assignments_v2.csv"
PROTOCOL = FINAL / "multi_label_experiment_protocol_v2.md"
FOLD_MANIFEST = FINAL / "lopo_fold_manifest_v2.csv"
LABELS = ("AR", "CTL", "MNT", "ET")
PROJECTS = ("chainer", "clusterfuzz", "grimoirelab-perceval", "numba", "pika", "psutil", "pyftpdlib", "python-driver")
REPS = ("R1", "R2", "R3", "H12")
ALL_RUNS = (*REPS, "H12-B")
TOL = 1e-8
BOOTSTRAPS = 10_000
SEED = 20261006
EXPECTED_INPUT_HASHES = {
    DATASET: "ad1d9cd262f11b6ba29287e791c4cdd1be3d8dd3bbb36b949821b8bd807448da",
    ASSIGNMENTS: "d3609f61bacdc655fe84e64273e0bf74ca801f2bbf16647693ded36b9e661faf",
    PROTOCOL: "bb45415df64276cf80a305d4c385b84215b99f46303c8915c0c1f340ea2b2a64",
    FOLD_MANIFEST: "51b37dc27182d254f9182ca01b703e3236c6ba308d426e8eeac2942248e0c8b3",
}
RUNS = {
    "R1": {"directory": V2 / "r1_tfidf", "prefix": "r1_v2", "manifest": "r1_v2_run_manifest.csv"},
    "R2": {"directory": V2 / "r2_ast", "prefix": "r2_v2", "manifest": "r2_v2_run_manifest.csv"},
    "R3": {"directory": V2 / "r3_unixcoder", "prefix": "r3_v2", "manifest": "r3_v2_run_manifest.csv"},
    "H12": {"directory": V2 / "h12", "prefix": "h12_v2", "manifest": "h12_v2_run_manifest.csv"},
    "H12-B": {"directory": V2 / "h12_balanced", "prefix": "h12b_v2", "manifest": "h12b_v2_run_manifest.csv"},
}
ARTIFACT_SUFFIXES = {
    "predictions": "oop_predictions.csv", "pooled": "pooled_metrics.csv",
    "per_label": "per_label_metrics.csv", "per_project": "per_project_metrics.csv",
    "per_project_label": "per_project_label_metrics.csv",
}
OUT_FILES = {
    "01_primary_overall_comparison.csv": ["representation", "hamming_loss", "exact_match", "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1", "rank_hamming", "rank_exact_match", "rank_micro_f1", "rank_macro_f1"],
    "02_primary_per_label_comparison.csv": ["label", "representation", "support", "predicted_positive", "precision", "recall", "F1", "TP", "FP", "FN", "TN", "highest_observed_precision", "highest_observed_recall", "highest_observed_F1", "precision_highest_observed_representation", "recall_highest_observed_representation", "F1_highest_observed_representation"],
    "03_primary_per_project_comparison.csv": ["project", "representation", "hamming_loss", "exact_match", "micro_f1", "macro_f1", "MNT_support_status"],
    "04_primary_metric_ranks.csv": ["metric", "representation", "value", "rank", "direction"],
    "05_paired_bootstrap_overall.csv": ["comparison", "metric", "observed_delta", "bootstrap_mean_delta", "CI_low", "CI_high", "bootstrap_tail_p", "holm_adjusted_p", "reject_at_0_05", "delta_definition"],
    "06_paired_bootstrap_per_label_f1.csv": ["comparison", "label", "observed_delta", "CI_low", "CI_high", "bootstrap_tail_p", "holm_adjusted_p", "reject_at_0_05", "delta_definition"],
    "07_mcnemar_per_label.csv": ["comparison", "label", "n00_both_wrong", "n01_H12_wrong_baseline_correct", "n10_H12_correct_baseline_wrong", "n11_both_correct", "discordant_total", "exact_p", "holm_adjusted_p", "reject_at_0_05", "net_correct_gain", "net_correct_gain_rate", "interpretation_scope"],
    "08_project_robustness_summary.csv": ["comparison", "metric", "projects_H12_better", "projects_equal", "projects_H12_worse", "median_project_delta", "min_project_delta", "max_project_delta", "delta_definition", "test_used"],
    "09_project_metric_deltas.csv": ["comparison", "project", "metric", "H12_value", "baseline_value", "improvement_oriented_delta", "direction"],
    "10_h12b_sensitivity_overall.csv": ["metric", "H12", "H12_B", "delta_H12B_minus_H12", "metric_role"],
    "11_h12b_sensitivity_per_label.csv": ["label", "metric", "H12", "H12_B", "delta_H12B_minus_H12"],
    "12_h12b_et_tradeoff.csv": ["metric", "H12", "H12_B", "delta_H12B_minus_H12"],
    "13_h12b_paired_bootstrap.csv": ["analysis_family", "metric", "observed_delta", "bootstrap_mean_delta", "CI_low", "CI_high", "bootstrap_tail_p", "holm_adjusted_p", "reject_at_0_05", "delta_definition"],
    "14_h12b_prediction_change_validation.csv": ["label", "0_to_0", "1_to_1", "0_to_1", "1_to_0", "total_changed", "audit_matches_predictions"],
    "15_metric_reproduction_audit.csv": ["representation", "scope", "item", "metric", "stored_value", "recomputed_value", "absolute_difference", "passed", "note"],
    "16_input_integrity_audit.csv": ["item", "status", "expected", "observed", "sha256", "notes"],
    "17_statistical_analysis_manifest.csv": ["artifact_path", "format", "data_rows", "sha256", "bootstrap_replicates", "bootstrap_seed", "bootstrap_paired", "bootstrap_unit", "same_indices_for_pair", "primary_comparisons", "primary_overall_test_family", "primary_per_label_f1_test_family", "primary_mcnemar_test_family", "multiple_comparison_method", "mcnemar_method", "project_level_standard_ttest", "H12B_in_primary_ranking", "model_training", "threshold_tuning", "input_dataset_sha256", "input_assignments_sha256", "input_protocol_sha256"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    csv.field_size_limit(2**31 - 1)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"Invalid CSV schema: {path}")
    return rows, fields


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def confusion_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    require(y.shape == pred.shape and y.ndim == 2 and y.shape[1] == len(LABELS), "Invalid truth/prediction array shape")
    tp = np.logical_and(y == 1, pred == 1).sum(axis=0).astype(np.int64)
    fp = np.logical_and(y == 0, pred == 1).sum(axis=0).astype(np.int64)
    fn = np.logical_and(y == 1, pred == 0).sum(axis=0).astype(np.int64)
    tn = np.logical_and(y == 0, pred == 0).sum(axis=0).astype(np.int64)
    per_label: list[dict[str, Any]] = []
    for j, label in enumerate(LABELS):
        support = int(tp[j] + fn[j])
        predicted_positive = int(tp[j] + fp[j])
        precision = float(tp[j] / predicted_positive) if predicted_positive else 0.0
        recall = float(tp[j] / support) if support else 0.0
        f1 = float(2 * tp[j] / (2 * tp[j] + fp[j] + fn[j])) if (2 * tp[j] + fp[j] + fn[j]) else 0.0
        per_label.append({
            "label": label, "support": support, "predicted_positive": predicted_positive,
            "negative_support": int(fp[j] + tn[j]), "precision": precision, "recall": recall,
            "F1": f1, "TP": int(tp[j]), "FP": int(fp[j]), "FN": int(fn[j]), "TN": int(tn[j]),
            "support_status": "NO_POSITIVE_SUPPORT" if support == 0 else "SUPPORTED",
            "precision_presentation": "NA" if predicted_positive == 0 else precision,
            "recall_presentation": "NA" if support == 0 else recall,
            "f1_presentation": "NA" if support == 0 or predicted_positive == 0 or precision + recall == 0 else f1,
        })
    micro_tp, micro_fp, micro_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    micro_p = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = 2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn) if 2 * micro_tp + micro_fp + micro_fn else 0.0
    pooled = {
        "hamming_loss": float(np.not_equal(y, pred).mean()),
        "exact_match_ratio": float(np.all(y == pred, axis=1).mean()),
        "micro_precision": float(micro_p), "micro_recall": float(micro_r), "micro_f1": float(micro_f1),
        "macro_precision": float(np.mean([r["precision"] for r in per_label])),
        "macro_recall": float(np.mean([r["recall"] for r in per_label])),
        "macro_f1": float(np.mean([r["F1"] for r in per_label])),
    }
    return pooled, per_label


def check_numeric(stored: Any, expected: float, context: str, audit_rows: list[dict[str, Any]],
                  representation: str, scope: str, item: str, metric: str) -> None:
    try:
        value = float(stored)
        difference = abs(value - expected)
        passed = math.isfinite(value) and difference <= TOL
        note = "absolute tolerance 1e-8"
    except (TypeError, ValueError):
        value, difference, passed, note = stored, "NA", False, "stored value is not numeric"
    audit_rows.append({"representation": representation, "scope": scope, "item": item, "metric": metric,
                       "stored_value": value, "recomputed_value": expected,
                       "absolute_difference": difference, "passed": "YES" if passed else "NO", "note": note})
    require(passed, f"Metric reproduction failed: {context}; stored={stored}, recomputed={expected}")


def check_text(stored: Any, expected: str, context: str, audit_rows: list[dict[str, Any]],
               representation: str, scope: str, item: str, metric: str) -> None:
    passed = str(stored) == expected
    audit_rows.append({"representation": representation, "scope": scope, "item": item, "metric": metric,
                       "stored_value": stored, "recomputed_value": expected,
                       "absolute_difference": "NA", "passed": "YES" if passed else "NO", "note": "exact categorical match"})
    require(passed, f"Metric reproduction failed: {context}; stored={stored}, recomputed={expected}")


def metric_artifact_paths(run: dict[str, Any]) -> dict[str, Path]:
    directory, prefix = run["directory"], run["prefix"]
    paths = {key: directory / f"{prefix}_{suffix}" for key, suffix in ARTIFACT_SUFFIXES.items()}
    if run["prefix"] == "r3_v2":
        paths["predictions"] = directory / "r3_v2_oop_predictions.csv"
    return paths


def verify_manifest(run_name: str, run: dict[str, Any], paths: dict[str, Path], input_audit: list[dict[str, Any]]) -> dict[str, str]:
    manifest_path = run["directory"] / run["manifest"]
    rows, fields = read_csv(manifest_path)
    require(len(rows) == 1, f"{run_name} manifest must have one row")
    manifest = rows[0]
    dataset_hash = manifest.get("dataset_sha256", "")
    assignment_hash = manifest.get("assignment_sha256", "")
    protocol_hash = manifest.get("protocol_sha256", "")
    fold_hash = manifest.get("fold_manifest_sha256", "")
    require(dataset_hash == EXPECTED_INPUT_HASHES[DATASET].lower(), f"{run_name} manifest dataset hash mismatch")
    require(assignment_hash == EXPECTED_INPUT_HASHES[ASSIGNMENTS].lower(), f"{run_name} manifest assignment hash mismatch")
    require(protocol_hash == EXPECTED_INPUT_HASHES[PROTOCOL].lower(), f"{run_name} manifest protocol hash mismatch")
    require(fold_hash == EXPECTED_INPUT_HASHES[FOLD_MANIFEST].lower(), f"{run_name} manifest fold-manifest hash mismatch")
    hash_field = manifest.get("artifact_sha256_json")
    count_field = manifest.get("artifact_row_counts_json")
    require(bool(hash_field) and bool(count_field), f"{run_name} manifest lacks artifact hashes/counts")
    recorded_hashes, recorded_counts = json.loads(hash_field), json.loads(count_field)
    for key in ("predictions", "pooled", "per_label", "per_project", "per_project_label"):
        path = paths[key]
        actual = sha256(path)
        require(recorded_hashes.get(key) == actual, f"{run_name} manifest hash mismatch for {key}")
        rows_out, _ = read_csv(path)
        require(int(recorded_counts.get(key, -1)) == len(rows_out), f"{run_name} manifest count mismatch for {key}")
        input_audit.append({"item": f"{run_name}:{key}", "status": "PASS", "expected": recorded_hashes[key],
                            "observed": actual, "sha256": actual, "notes": f"manifest-authenticated frozen artifact; rows={len(rows_out)}"})
    actual_manifest_hash = sha256(manifest_path)
    input_audit.append({"item": f"{run_name}:run_manifest", "status": "PASS", "expected": "one-row manifest",
                        "observed": f"rows={len(rows)}; fields={len(fields)}", "sha256": actual_manifest_hash,
                        "notes": "manifest physical hash recorded for provenance"})
    return recorded_hashes


def paired_bootstrap_metrics(y: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Compute paired bootstrap metrics for all models from one shared sample-index stream."""
    rng = np.random.default_rng(SEED)
    result = {name: {metric: np.empty(BOOTSTRAPS, dtype=np.float64) for metric in
                     ("hamming_loss", "exact_match", "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")}
              | {"per_label_f1": np.empty((BOOTSTRAPS, len(LABELS)), dtype=np.float64)} for name in predictions}
    n = len(y)
    batch_size = 64
    for start in range(0, BOOTSTRAPS, batch_size):
        stop = min(start + batch_size, BOOTSTRAPS)
        indices = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
        truth = y[indices]
        for name, pred_all in predictions.items():
            pred = pred_all[indices]
            tp = np.logical_and(truth == 1, pred == 1).sum(axis=1, dtype=np.int64)
            fp = np.logical_and(truth == 0, pred == 1).sum(axis=1, dtype=np.int64)
            fn = np.logical_and(truth == 1, pred == 0).sum(axis=1, dtype=np.int64)
            denom_p, denom_r = tp + fp, tp + fn
            precision = np.divide(tp, denom_p, out=np.zeros(tp.shape, dtype=np.float64), where=denom_p != 0)
            recall = np.divide(tp, denom_r, out=np.zeros(tp.shape, dtype=np.float64), where=denom_r != 0)
            f1_denom = 2 * tp + fp + fn
            f1 = np.divide(2 * tp, f1_denom, out=np.zeros(tp.shape, dtype=np.float64), where=f1_denom != 0)
            micro_tp, micro_fp, micro_fn = tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1)
            micro_p = np.divide(micro_tp, micro_tp + micro_fp, out=np.zeros(stop-start), where=(micro_tp + micro_fp) != 0)
            micro_r = np.divide(micro_tp, micro_tp + micro_fn, out=np.zeros(stop-start), where=(micro_tp + micro_fn) != 0)
            micro_f = np.divide(2 * micro_tp, 2 * micro_tp + micro_fp + micro_fn,
                                out=np.zeros(stop-start), where=(2 * micro_tp + micro_fp + micro_fn) != 0)
            block = result[name]
            block["hamming_loss"][start:stop] = np.not_equal(truth, pred).mean(axis=(1, 2))
            block["exact_match"][start:stop] = np.all(truth == pred, axis=2).mean(axis=1)
            block["micro_precision"][start:stop] = micro_p
            block["micro_recall"][start:stop] = micro_r
            block["micro_f1"][start:stop] = micro_f
            block["macro_precision"][start:stop] = precision.mean(axis=1)
            block["macro_recall"][start:stop] = recall.mean(axis=1)
            block["macro_f1"][start:stop] = f1.mean(axis=1)
            block["per_label_f1"][start:stop] = f1
    return result


def bootstrap_tail_p(deltas: np.ndarray) -> float:
    return float(min(1.0, 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))))


def summarize_bootstrap(deltas: np.ndarray) -> tuple[float, float, float, float]:
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(low), float(high), bootstrap_tail_p(deltas)


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    result = np.empty_like(adjusted_sorted)
    result[order] = adjusted_sorted
    return result.tolist()


def main() -> None:
    preflight_analysis_or_exit(ROOT)
    if OUTPUT.exists():
        expected_partial = set(OUT_FILES) - {"17_statistical_analysis_manifest.csv"} | {"final_comparative_analysis.md"}
        current = {path.name for path in OUTPUT.iterdir() if path.is_file()}
        require(current == expected_partial,
                f"Refusing to overwrite an unexpected existing final-comparison directory: {sorted(current)}")
        # Only a prior incomplete output from this analysis-only runner may be
        # cleared for deterministic regeneration; frozen experiment directories
        # are never touched.
        for path in OUTPUT.iterdir():
            require(path.is_file(), f"Unexpected directory in partial comparison output: {path}")
            path.unlink()
        OUTPUT.rmdir()
    input_audit: list[dict[str, Any]] = []
    input_hashes: dict[Path, str] = {}
    for path, expected in EXPECTED_INPUT_HASHES.items():
        actual = sha256(path)
        input_hashes[path] = actual
        passed = actual == expected
        input_audit.append({"item": str(path.relative_to(ROOT)), "status": "PASS" if passed else "FAIL",
                            "expected": expected, "observed": actual, "sha256": actual, "notes": "authoritative frozen input SHA-256"})
        require(passed, f"Authoritative frozen input hash mismatch: {path}")

    dataset, dataset_fields = read_csv(DATASET)
    require(dataset_fields and len(dataset) == 14755, "V2 dataset row count failed")
    ids = [row["sample_id"] for row in dataset]
    require(len(set(ids)) == 14755, "V2 dataset sample_id uniqueness failed")
    require({"sample_id", "project_name", *LABELS} <= set(dataset_fields), "V2 dataset schema lacks required columns")
    require(all(row[label] in {"0", "1"} for row in dataset for label in LABELS), "V2 dataset labels are not binary")
    dataset_by_id = {row["sample_id"]: row for row in dataset}
    project_by_id = {row["sample_id"]: row["project_name"] for row in dataset}
    assignments, assignment_fields = read_csv(ASSIGNMENTS)
    require(assignment_fields == ["canonical_index", "sample_id", "project_name", "fold_id", "held_out_project"], "LOPO assignment schema mismatch")
    require(len(assignments) == 14755 and [row["sample_id"] for row in assignments] == ids
            and len({row["sample_id"] for row in assignments}) == 14755, "LOPO assignment membership/order failed")
    fold_by_id = {row["sample_id"]: row["fold_id"] for row in assignments}
    require(all(row["project_name"] == project_by_id[row["sample_id"]]
                and row["held_out_project"] == project_by_id[row["sample_id"]]
                for row in assignments), "LOPO project assignment mismatch")
    fold_manifest, fold_fields = read_csv(FOLD_MANIFEST)
    require(len(fold_manifest) == 8 and {row["held_out_project"] for row in fold_manifest} == set(PROJECTS), "LOPO fold manifest membership failed")
    require(all(row["fold_id"] == fold_by_id[row["sample_id"]] for row in assignments), "Assignment/fold manifest ID mismatch")
    input_audit.extend([
        {"item": "dataset_row_count_unique_ids", "status": "PASS", "expected": "14755/14755", "observed": "14755/14755", "sha256": input_hashes[DATASET], "notes": "exact frozen experimental population"},
        {"item": "LOPO_assignment_order", "status": "PASS", "expected": "dataset canonical relative order", "observed": "exact match", "sha256": input_hashes[ASSIGNMENTS], "notes": "fold_id and held-out project join verified by sample_id"},
        {"item": "LOPO_fold_manifest", "status": "PASS", "expected": "8 frozen projects", "observed": ",".join(PROJECTS), "sha256": sha256(FOLD_MANIFEST), "notes": f"fields={len(fold_fields)}"},
    ])

    frozen_snapshots: dict[Path, str] = {}
    run_paths: dict[str, dict[str, Path]] = {}
    prediction_rows: dict[str, list[dict[str, str]]] = {}
    truths: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    metric_paths_by_run: dict[str, dict[str, Path]] = {}
    manifest_hash_maps: dict[str, dict[str, str]] = {}
    run_metric_artifacts: dict[str, dict[str, list[dict[str, str]]]] = {}
    for name in ALL_RUNS:
        run = RUNS[name]
        paths = metric_artifact_paths(run)
        run_paths[name] = paths
        metric_paths_by_run[name] = paths
        for path in run["directory"].rglob("*"):
            if path.is_file():
                frozen_snapshots[path] = sha256(path)
        manifest_hash_maps[name] = verify_manifest(name, run, paths, input_audit)
        for key in ("pooled", "per_label", "per_project", "per_project_label"):
            run_metric_artifacts.setdefault(name, {})[key] = read_csv(paths[key])[0]
        rows, fields = read_csv(paths["predictions"])
        require(fields == ["sample_id", "project_name", "fold"] + [
            field for label in LABELS for field in (f"y_true_{label}", f"y_pred_{label}", f"y_score_{label}")
        ], f"{name} prediction schema mismatch")
        require(len(rows) == 14755 and [row["sample_id"] for row in rows] == ids
                and len({row["sample_id"] for row in rows}) == 14755,
                f"{name} OOP rows/unique membership/canonical relative order mismatch")
        y_true = np.zeros((14755, 4), dtype=np.int8)
        y_pred = np.zeros((14755, 4), dtype=np.int8)
        for i, row in enumerate(rows):
            sid = row["sample_id"]
            require(row["project_name"] == project_by_id[sid], f"{name} project mismatch for {sid}")
            require(row["fold"] == fold_by_id[sid], f"{name} held-out fold mismatch for {sid}")
            for j, label in enumerate(LABELS):
                require(row[f"y_true_{label}"] in {"0", "1"} and row[f"y_pred_{label}"] in {"0", "1"},
                        f"{name} non-binary truth/prediction at {sid}:{label}")
                score = float(row[f"y_score_{label}"])
                require(math.isfinite(score) and 0 <= score <= 1, f"{name} invalid probability at {sid}:{label}")
                require(int(row[f"y_true_{label}"]) == int(dataset_by_id[sid][label]),
                        f"{name} ground truth differs from frozen V2 labels for {sid}:{label}")
                y_true[i, j], y_pred[i, j] = int(row[f"y_true_{label}"]), int(row[f"y_pred_{label}"])
        prediction_rows[name], truths[name], preds[name] = rows, y_true, y_pred
    reference_truth = truths[REPS[0]]
    for name in ALL_RUNS[1:]:
        require(np.array_equal(truths[name], reference_truth), f"Ground truth differs between {REPS[0]} and {name}")
    require(all(np.array_equal(truths[name], np.asarray([[int(row[label]) for label in LABELS] for row in dataset], dtype=np.int8)) for name in ALL_RUNS),
            "Ground truth does not exactly match authoritative dataset")
    input_audit += [
        {"item": f"{name}_OOP_population_order_truth_project_fold", "status": "PASS", "expected": "14755 exact IDs/order/truth/project/fold", "observed": "PASS", "sha256": sha256(run_paths[name]["predictions"]), "notes": "validated by exact sample_id join; one OOP row per sample"}
        for name in ALL_RUNS
    ]
    input_audit.append({"item": "cross_representation_ground_truth_consistency", "status": "PASS", "expected": "identical y_true_AR/CTL/MNT/ET across all five", "observed": "exact match", "sha256": "", "notes": "all truth vectors also match the authoritative dataset"})

    metric_audit: list[dict[str, Any]] = []
    recomputed_pooled: dict[str, dict[str, float]] = {}
    recomputed_labels: dict[str, list[dict[str, Any]]] = {}
    recomputed_projects: dict[str, dict[str, dict[str, Any]]] = {}
    recomputed_project_labels: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for name in ALL_RUNS:
        pooled, per_label = confusion_metrics(truths[name], preds[name])
        recomputed_pooled[name], recomputed_labels[name] = pooled, per_label
        saved = run_metric_artifacts[name]
        pooled_rows, label_rows = saved["pooled"], saved["per_label"]
        require(len(pooled_rows) == 8 and {r["metric"] for r in pooled_rows} == set(pooled), f"{name} pooled metric artifact schema/content invalid")
        for row in pooled_rows:
            check_numeric(row["value"], pooled[row["metric"]], f"{name} pooled {row['metric']}", metric_audit, name, "pooled", "ALL", row["metric"])
        require(len(label_rows) == 4 and {r["label"] for r in label_rows} == set(LABELS), f"{name} per-label metrics invalid")
        label_by_name = {row["label"]: row for row in label_rows}
        for computed in per_label:
            label = computed["label"]
            stored = label_by_name[label]
            for metric, expected in (("support", computed["support"]), ("predicted_positive", computed["predicted_positive"]),
                                     ("negative_support", computed["negative_support"]), ("TP", computed["TP"]),
                                     ("FP", computed["FP"]), ("FN", computed["FN"]), ("TN", computed["TN"])):
                if metric == "predicted_positive" and metric not in stored:
                    actual = int(stored["TP"]) + int(stored["FP"])
                else:
                    actual = int(stored[metric])
                check_numeric(actual, float(expected), f"{name} {label} {metric}", metric_audit, name, "per_label", label, metric)
            for metric in ("precision", "recall", "f1"):
                check_numeric(stored[f"{metric}_computational"], float(computed["F1" if metric == "f1" else metric]),
                              f"{name} {label} {metric}", metric_audit, name, "per_label", label, f"{metric}_computational")
                presentation_expected = computed[f"{metric}_presentation"]
                stored_presentation = stored[f"{metric}_presentation"]
                if presentation_expected == "NA":
                    check_text(stored_presentation, "NA", f"{name} {label} {metric}_presentation", metric_audit, name, "per_label", label, f"{metric}_presentation")
                else:
                    check_numeric(stored_presentation, float(presentation_expected), f"{name} {label} {metric}_presentation", metric_audit, name, "per_label", label, f"{metric}_presentation")
            check_text(stored["support_status"], computed["support_status"], f"{name} {label} support_status", metric_audit, name, "per_label", label, "support_status")

        project_metric_map: dict[str, dict[str, Any]] = {}
        project_label_map: dict[tuple[str, str], dict[str, Any]] = {}
        stored_project_rows = saved["per_project"]
        stored_project_label_rows = saved["per_project_label"]
        require(len(stored_project_rows) == 8 and {r["held_out_project"] for r in stored_project_rows} == set(PROJECTS), f"{name} per-project table invalid")
        require(len(stored_project_label_rows) == 32 and {(r["held_out_project"], r["label"]) for r in stored_project_label_rows} == {(p, l) for p in PROJECTS for l in LABELS}, f"{name} per-project-label table invalid")
        for project in PROJECTS:
            indices = np.asarray([i for i, row in enumerate(prediction_rows[name]) if row["project_name"] == project], dtype=np.int64)
            pm, pl = confusion_metrics(truths[name][indices], preds[name][indices])
            project_metric_map[project] = {**pm, "test_n": len(indices)}
            stored_project = next(row for row in stored_project_rows if row["held_out_project"] == project)
            require(int(stored_project["test_n"]) == len(indices), f"{name} {project} project sample count mismatch")
            for metric in ("hamming_loss", "exact_match_ratio", "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"):
                field = metric if metric not in {"macro_precision", "macro_recall", "macro_f1"} else f"{metric}_computational"
                check_numeric(stored_project[field], pm[metric], f"{name} {project} {metric}", metric_audit, name, "per_project", project, metric)
            stored_pl_by_label = {r["label"]: r for r in stored_project_label_rows if r["held_out_project"] == project}
            for computed in pl:
                label = computed["label"]
                project_label_map[(project, label)] = computed
                stored_pl = stored_pl_by_label[label]
                for metric in ("support", "predicted_positive", "negative_support", "TP", "FP", "FN", "TN"):
                    actual = (int(stored_pl["TP"]) + int(stored_pl["FP"])) if metric == "predicted_positive" and metric not in stored_pl else int(stored_pl[metric])
                    check_numeric(actual, float(computed[metric]), f"{name} {project}:{label} {metric}", metric_audit, name, "per_project_label", f"{project}:{label}", metric)
                for metric in ("precision", "recall", "f1"):
                    check_numeric(stored_pl[f"{metric}_computational"], float(computed["F1" if metric == "f1" else metric]),
                                  f"{name} {project}:{label} {metric}", metric_audit, name, "per_project_label", f"{project}:{label}", f"{metric}_computational")
                    if computed[f"{metric}_presentation"] == "NA":
                        check_text(stored_pl[f"{metric}_presentation"], "NA", f"{name} {project}:{label} {metric}_presentation", metric_audit, name, "per_project_label", f"{project}:{label}", f"{metric}_presentation")
                    else:
                        check_numeric(stored_pl[f"{metric}_presentation"], float(computed[f"{metric}_presentation"]), f"{name} {project}:{label} {metric}_presentation", metric_audit, name, "per_project_label", f"{project}:{label}", f"{metric}_presentation")
                check_text(stored_pl["support_status"], computed["support_status"], f"{name} {project}:{label} support_status", metric_audit, name, "per_project_label", f"{project}:{label}", "support_status")
        recomputed_projects[name] = project_metric_map
        recomputed_project_labels[name] = project_label_map
    require(sorted({f"{p}:{l}" for (p, l), row in recomputed_project_labels["H12"].items() if row["support_status"] == "NO_POSITIVE_SUPPORT"}) == ["pyftpdlib:MNT"],
            "The required pyftpdlib:MNT zero-positive-support case is missing or changed")

    # Primary table and individual metric ranks: H12-B remains a separate sensitivity.
    overall_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    metric_columns = {"hamming_loss": "hamming_loss", "exact_match": "exact_match_ratio",
                      "micro_f1": "micro_f1", "macro_f1": "macro_f1"}
    rank_values: dict[str, dict[str, int]] = {}
    for metric, field in metric_columns.items():
        values = np.asarray([recomputed_pooled[name][field] for name in REPS], dtype=np.float64)
        ranks = rankdata(values if metric == "hamming_loss" else -values, method="min").astype(int)
        rank_values[metric] = dict(zip(REPS, ranks.tolist(), strict=True))
        for rep, value, rank in zip(REPS, values, ranks, strict=True):
            rank_rows.append({"metric": metric, "representation": rep, "value": float(value), "rank": int(rank),
                              "direction": "lower_is_better" if metric == "hamming_loss" else "higher_is_better"})
    for rep in REPS:
        m = recomputed_pooled[rep]
        overall_rows.append({"representation": rep, "hamming_loss": m["hamming_loss"], "exact_match": m["exact_match_ratio"],
                             "micro_precision": m["micro_precision"], "micro_recall": m["micro_recall"], "micro_f1": m["micro_f1"],
                             "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"],
                             "rank_hamming": rank_values["hamming_loss"][rep], "rank_exact_match": rank_values["exact_match"][rep],
                             "rank_micro_f1": rank_values["micro_f1"][rep], "rank_macro_f1": rank_values["macro_f1"][rep]})

    per_label_rows: list[dict[str, Any]] = []
    label_highest: dict[str, dict[str, str]] = {}
    for label in LABELS:
        metric_row_per_rep = {rep: next(row for row in recomputed_labels[rep] if row["label"] == label) for rep in REPS}
        highest = {metric: max(REPS, key=lambda rep: metric_row_per_rep[rep][metric]) for metric in ("precision", "recall", "F1")}
        label_highest[label] = highest
        for rep in REPS:
            r = metric_row_per_rep[rep]
            per_label_rows.append({"label": label, "representation": rep, "support": r["support"], "predicted_positive": r["predicted_positive"],
                                   "precision": r["precision"], "recall": r["recall"], "F1": r["F1"], "TP": r["TP"],
                                   "FP": r["FP"], "FN": r["FN"], "TN": r["TN"],
                                   "highest_observed_precision": "YES" if highest["precision"] == rep else "NO",
                                   "highest_observed_recall": "YES" if highest["recall"] == rep else "NO",
                                   "highest_observed_F1": "YES" if highest["F1"] == rep else "NO",
                                   "precision_highest_observed_representation": highest["precision"],
                                   "recall_highest_observed_representation": highest["recall"],
                                   "F1_highest_observed_representation": highest["F1"]})
    per_project_rows = []
    for project in PROJECTS:
        mnt = recomputed_project_labels["H12"][(project, "MNT")]
        status = mnt["support_status"]
        for rep in REPS:
            pm = recomputed_projects[rep][project]
            per_project_rows.append({"project": project, "representation": rep, "hamming_loss": pm["hamming_loss"],
                                     "exact_match": pm["exact_match_ratio"], "micro_f1": pm["micro_f1"],
                                     "macro_f1": pm["macro_f1"], "MNT_support_status": status})

    # Project-level descriptive robustness contrasts (no IID t-test or sign test).
    project_delta_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    for baseline in ("R1", "R2", "R3"):
        for metric, field in (("micro_f1", "micro_f1"), ("macro_f1", "macro_f1"),
                              ("exact_match", "exact_match_ratio"), ("hamming_improvement", "hamming_loss")):
            values = []
            for project in PROJECTS:
                h = recomputed_projects["H12"][project][field]
                b = recomputed_projects[baseline][project][field]
                delta = b - h if metric == "hamming_improvement" else h - b
                values.append(delta)
                project_delta_rows.append({"comparison": f"H12_vs_{baseline}", "project": project, "metric": metric,
                                           "H12_value": h, "baseline_value": b,
                                           "improvement_oriented_delta": delta,
                                           "direction": "positive_favors_H12"})
            robustness_rows.append({"comparison": f"H12_vs_{baseline}", "metric": metric,
                                    "projects_H12_better": sum(v > 0 for v in values),
                                    "projects_equal": sum(v == 0 for v in values),
                                    "projects_H12_worse": sum(v < 0 for v in values),
                                    "median_project_delta": float(statistics.median(values)),
                                    "min_project_delta": float(min(values)), "max_project_delta": float(max(values)),
                                    "delta_definition": "improvement-oriented; positive favors H12",
                                    "test_used": "NONE; descriptive project-level analysis"})

    # H12-B descriptive sensitivity, no ranking and no primary-family correction.
    sensitivity_overall = []
    for metric, key in (("hamming_loss", "hamming_loss"), ("exact_match", "exact_match_ratio"),
                        ("micro_precision", "micro_precision"), ("micro_recall", "micro_recall"), ("micro_f1", "micro_f1"),
                        ("macro_precision", "macro_precision"), ("macro_recall", "macro_recall"), ("macro_f1", "macro_f1")):
        h, b = recomputed_pooled["H12"][key], recomputed_pooled["H12-B"][key]
        sensitivity_overall.append({"metric": metric, "H12": h, "H12_B": b, "delta_H12B_minus_H12": b - h,
                                    "metric_role": "lower_is_better" if metric == "hamming_loss" else "higher_is_better"})
    sensitivity_labels = []
    for label in LABELS:
        h = next(row for row in recomputed_labels["H12"] if row["label"] == label)
        b = next(row for row in recomputed_labels["H12-B"] if row["label"] == label)
        for metric in ("precision", "recall", "F1", "predicted_positive", "TP", "FP", "FN", "TN"):
            sensitivity_labels.append({"label": label, "metric": metric, "H12": h[metric], "H12_B": b[metric], "delta_H12B_minus_H12": b[metric] - h[metric]})
    et_sensitivity = [{"metric": metric, "H12": sensitivity_labels_row["H12"], "H12_B": sensitivity_labels_row["H12_B"],
                      "delta_H12B_minus_H12": sensitivity_labels_row["delta_H12B_minus_H12"]}
                     for metric in ("predicted_positive", "precision", "recall", "F1", "TP", "FP", "FN", "TN")
                     for sensitivity_labels_row in sensitivity_labels
                     if sensitivity_labels_row["label"] == "ET" and sensitivity_labels_row["metric"] == metric]
    et_h = next(row for row in recomputed_labels["H12"] if row["label"] == "ET")
    et_b = next(row for row in recomputed_labels["H12-B"] if row["label"] == "ET")
    et_sensitivity.insert(0, {"metric": "support", "H12": et_h["support"], "H12_B": et_b["support"], "delta_H12B_minus_H12": et_b["support"] - et_h["support"]})

    change_artifact = RUNS["H12-B"]["directory"] / "h12b_v2_prediction_change_audit.csv"
    stored_changes, _ = read_csv(change_artifact)
    require(len(stored_changes) == 4 and {r["label"] for r in stored_changes} == set(LABELS), "H12-B stored change audit malformed")
    change_rows = []
    for label in LABELS:
        j = LABELS.index(label)
        transitions = {(0, 0): 0, (1, 1): 0, (0, 1): 0, (1, 0): 0}
        for i in range(14755):
            transitions[(int(preds["H12"][i, j]), int(preds["H12-B"][i, j]))] += 1
        expected_fields = {"unchanged_0_to_0": transitions[(0, 0)], "unchanged_1_to_1": transitions[(1, 1)],
                           "changed_0_to_1": transitions[(0, 1)], "changed_1_to_0": transitions[(1, 0)],
                           "total_changed": transitions[(0, 1)] + transitions[(1, 0)]}
        stored = next(row for row in stored_changes if row["label"] == label)
        for field, value in expected_fields.items():
            require(int(stored[field]) == value, f"H12-B prediction-change audit disagreement: {label} {field}")
        change_rows.append({"label": label, "0_to_0": transitions[(0, 0)], "1_to_1": transitions[(1, 1)],
                            "0_to_1": transitions[(0, 1)], "1_to_0": transitions[(1, 0)],
                            "total_changed": expected_fields["total_changed"], "audit_matches_predictions": "YES"})

    # One shared bootstrap index stream is applied to all five frozen predictors.
    bootstrap = paired_bootstrap_metrics(reference_truth, preds)
    overall_boot_rows: list[dict[str, Any]] = []
    primary_overall_raw: list[dict[str, Any]] = []
    overall_specs = (("hamming_loss", "hamming_loss", "baseline - H12"),
                     ("exact_match", "exact_match", "H12 - baseline"),
                     ("micro_f1", "micro_f1", "H12 - baseline"),
                     ("macro_f1", "macro_f1", "H12 - baseline"))
    for baseline in ("R1", "R2", "R3"):
        for metric, boot_key, definition in overall_specs:
            delta = bootstrap[baseline][boot_key] - bootstrap["H12"][boot_key] if metric == "hamming_loss" else bootstrap["H12"][boot_key] - bootstrap[baseline][boot_key]
            mean, low, high, tail = summarize_bootstrap(delta)
            primary_overall_raw.append({"comparison": f"H12_vs_{baseline}", "metric": metric,
                                        "observed_delta": (recomputed_pooled[baseline]["hamming_loss"] - recomputed_pooled["H12"]["hamming_loss"] if metric == "hamming_loss" else recomputed_pooled["H12"]["exact_match_ratio" if metric == "exact_match" else metric] - recomputed_pooled[baseline]["exact_match_ratio" if metric == "exact_match" else metric]),
                                        "bootstrap_mean_delta": mean, "CI_low": low, "CI_high": high,
                                        "bootstrap_tail_p": tail, "delta_definition": definition})
    overall_adjusted = holm_adjust([row["bootstrap_tail_p"] for row in primary_overall_raw])
    for row, adjusted in zip(primary_overall_raw, overall_adjusted, strict=True):
        overall_boot_rows.append({**row, "holm_adjusted_p": adjusted, "reject_at_0_05": "YES" if adjusted <= 0.05 else "NO"})

    per_label_boot_raw: list[dict[str, Any]] = []
    for baseline in ("R1", "R2", "R3"):
        for label_index, label in enumerate(LABELS):
            delta = bootstrap["H12"]["per_label_f1"][:, label_index] - bootstrap[baseline]["per_label_f1"][:, label_index]
            _, low, high, tail = summarize_bootstrap(delta)
            observed = next(r for r in recomputed_labels["H12"] if r["label"] == label)["F1"] - next(r for r in recomputed_labels[baseline] if r["label"] == label)["F1"]
            per_label_boot_raw.append({"comparison": f"H12_vs_{baseline}", "label": label, "observed_delta": observed,
                                       "CI_low": low, "CI_high": high, "bootstrap_tail_p": tail,
                                       "delta_definition": "H12 label F1 - baseline label F1"})
    label_adjusted = holm_adjust([row["bootstrap_tail_p"] for row in per_label_boot_raw])
    per_label_boot_rows = [{**row, "holm_adjusted_p": adjusted, "reject_at_0_05": "YES" if adjusted <= 0.05 else "NO"}
                           for row, adjusted in zip(per_label_boot_raw, label_adjusted, strict=True)]

    mcnemar_raw = []
    for baseline in ("R1", "R2", "R3"):
        for j, label in enumerate(LABELS):
            y = reference_truth[:, j]
            h_ok = preds["H12"][:, j] == y
            b_ok = preds[baseline][:, j] == y
            n00 = int(np.sum(~h_ok & ~b_ok))
            n01 = int(np.sum(~h_ok & b_ok))
            n10 = int(np.sum(h_ok & ~b_ok))
            n11 = int(np.sum(h_ok & b_ok))
            discordant = n01 + n10
            p = float(binomtest(n10, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
            net = n10 - n01
            mcnemar_raw.append({"comparison": f"H12_vs_{baseline}", "label": label, "n00_both_wrong": n00,
                                "n01_H12_wrong_baseline_correct": n01, "n10_H12_correct_baseline_wrong": n10,
                                "n11_both_correct": n11, "discordant_total": discordant, "exact_p": p,
                                "net_correct_gain": net, "net_correct_gain_rate": net / 14755,
                                "interpretation_scope": "paired binary correctness; not an F1 test"})
    mcnemar_adjusted = holm_adjust([row["exact_p"] for row in mcnemar_raw])
    mcnemar_rows = [{**row, "holm_adjusted_p": adjusted, "reject_at_0_05": "YES" if adjusted <= 0.05 else "NO"}
                    for row, adjusted in zip(mcnemar_raw, mcnemar_adjusted, strict=True)]

    h12b_boot_rows = []
    for metric, boot_key, definition in (("hamming_loss", "hamming_loss", "H12 - H12-B; positive favors H12-B"),
                                         ("exact_match", "exact_match", "H12-B - H12"),
                                         ("micro_f1", "micro_f1", "H12-B - H12"),
                                         ("macro_f1", "macro_f1", "H12-B - H12")):
        delta = bootstrap["H12"][boot_key] - bootstrap["H12-B"][boot_key] if metric == "hamming_loss" else bootstrap["H12-B"][boot_key] - bootstrap["H12"][boot_key]
        mean, low, high, tail = summarize_bootstrap(delta)
        field = "hamming_loss" if metric == "hamming_loss" else "exact_match_ratio" if metric == "exact_match" else metric
        observed = (recomputed_pooled["H12"][field] - recomputed_pooled["H12-B"][field] if metric == "hamming_loss" else recomputed_pooled["H12-B"][field] - recomputed_pooled["H12"][field])
        h12b_boot_rows.append({"analysis_family": "SECONDARY_SENSITIVITY", "metric": metric, "observed_delta": observed,
                               "bootstrap_mean_delta": mean, "CI_low": low, "CI_high": high,
                               "bootstrap_tail_p": tail, "holm_adjusted_p": "NOT_APPLICABLE",
                               "reject_at_0_05": "NOT_APPLICABLE", "delta_definition": definition})

    # All primary metric/source checks have passed. Preserve every frozen V2 run file hash.
    for path, before in frozen_snapshots.items():
        require(sha256(path) == before, f"Frozen experiment artifact changed during analysis: {path}")
    for path, before in input_hashes.items():
        require(sha256(path) == before, f"Frozen input changed during analysis: {path}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    table_rows: dict[str, list[dict[str, Any]]] = {
        "01_primary_overall_comparison.csv": overall_rows,
        "02_primary_per_label_comparison.csv": per_label_rows,
        "03_primary_per_project_comparison.csv": per_project_rows,
        "04_primary_metric_ranks.csv": rank_rows,
        "05_paired_bootstrap_overall.csv": overall_boot_rows,
        "06_paired_bootstrap_per_label_f1.csv": per_label_boot_rows,
        "07_mcnemar_per_label.csv": mcnemar_rows,
        "08_project_robustness_summary.csv": robustness_rows,
        "09_project_metric_deltas.csv": project_delta_rows,
        "10_h12b_sensitivity_overall.csv": sensitivity_overall,
        "11_h12b_sensitivity_per_label.csv": sensitivity_labels,
        "12_h12b_et_tradeoff.csv": et_sensitivity,
        "13_h12b_paired_bootstrap.csv": h12b_boot_rows,
        "14_h12b_prediction_change_validation.csv": change_rows,
        "15_metric_reproduction_audit.csv": metric_audit,
        "16_input_integrity_audit.csv": input_audit,
    }
    for filename, rows in table_rows.items():
        write_csv(OUTPUT / filename, rows, OUT_FILES[filename])

    overall_by_rep = {row["representation"]: row for row in overall_rows}
    primary_best = {
        "hamming_loss": min(REPS, key=lambda rep: overall_by_rep[rep]["hamming_loss"]),
        "exact_match": max(REPS, key=lambda rep: overall_by_rep[rep]["exact_match"]),
        "micro_f1": max(REPS, key=lambda rep: overall_by_rep[rep]["micro_f1"]),
        "macro_f1": max(REPS, key=lambda rep: overall_by_rep[rep]["macro_f1"]),
    }
    report: list[str] = [
        "# V2 — Final Comparative & Statistical Analysis", "",
        "## 1. Experimental population", "",
        "The frozen V2 population contains 14,755 unique methods from eight selected Python projects. Each method is evaluated exactly once out-of-project under the frozen eight-fold LOPO protocol.", "",
        "## 2–3. Compared representations and integrity", "",
        "Primary representations: R1 TF-IDF, R2 AST, R3 UniXcoder, and H12 Hybrid. H12-B is reported separately as a class-imbalance sensitivity and is not a fifth primary representation.",
        "All five prediction files passed exact sample-ID membership/order, binary ground truth, project, and held-out fold checks. Ground truth was identical across models and matched the authoritative V2 dataset.", "",
        "## 4. Integrity and metric-reproduction audit", "",
        f"Independent recomputation from frozen OOP predictions matched all pooled, per-label, per-project, and per-project-per-label metric artifacts at absolute tolerance {TOL:g}. See `15_metric_reproduction_audit.csv` and `16_input_integrity_audit.csv`.",
        "The `pyftpdlib:MNT` per-project label cell is retained as `NO_POSITIVE_SUPPORT`; undefined presentation metrics remain NA and were not imputed.", "",
        "## 5. Primary pooled comparison", "",
        "| Representation | Hamming Loss | Exact Match | Micro Precision | Micro Recall | Micro F1 | Macro Precision | Macro Recall | Macro F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        report.append(f"| {row['representation']} | {row['hamming_loss']:.6f} | {row['exact_match']:.6f} | {row['micro_precision']:.6f} | {row['micro_recall']:.6f} | {row['micro_f1']:.6f} | {row['macro_precision']:.6f} | {row['macro_recall']:.6f} | {row['macro_f1']:.6f} |")
    report += ["", "Highest observed among the four primary representations (descriptive; not a universal or statistical winner): " + "; ".join(f"{metric}: {rep}" for metric, rep in primary_best.items()) + ".", "",
               "## 6. Primary per-label comparison", "",
               "Highest observed precision/recall/F1 are descriptive maxima across R1/R2/R3/H12; they are not claims of statistical superiority.", ""]
    for label in LABELS:
        report += [f"### {label}", "", "| Representation | Support | Predicted positive | Precision | Recall | F1 | TP | FP | FN | TN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in (r for r in per_label_rows if r["label"] == label):
            report.append(f"| {row['representation']} | {row['support']} | {row['predicted_positive']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['F1']:.6f} | {row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} |")
        high = label_highest[label]
        report.append(f"Highest observed: precision {high['precision']}; recall {high['recall']}; F1 {high['F1']}.")
        report.append("")
    report += ["## 7. Per-project comparison", "", "| Project | Representation | Hamming Loss | Exact Match | Micro F1 | Macro F1 | MNT support status |", "|---|---|---:|---:|---:|---:|---|"]
    for row in per_project_rows:
        report.append(f"| {row['project']} | {row['representation']} | {row['hamming_loss']:.6f} | {row['exact_match']:.6f} | {row['micro_f1']:.6f} | {row['macro_f1']:.6f} | {row['MNT_support_status']} |")
    report += ["", "## 8. Paired bootstrap methodology", "",
               f"For each of B={BOOTSTRAPS} replicates (seed {SEED}), one vector of 14,755 sample indices was drawn with replacement and applied to all five systems, preserving pairing. Percentile 95% intervals use the 2.5th and 97.5th percentiles. The reported `bootstrap_tail_p` is the two-sided sign-tail measure 2×min(P(Δ≤0), P(Δ≥0)), capped at 1; it is not an exact parametric p-value. Positive deltas are improvement-oriented for H12 in primary comparisons.",
               "## 9. Primary overall paired-bootstrap results", "", "| Comparison | Metric | Observed improvement | Bootstrap mean | 95% CI | Tail p | Holm p | Reject .05 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in overall_boot_rows:
        report.append(f"| {row['comparison']} | {row['metric']} | {row['observed_delta']:.6f} | {row['bootstrap_mean_delta']:.6f} | [{row['CI_low']:.6f}, {row['CI_high']:.6f}] | {row['bootstrap_tail_p']:.6g} | {row['holm_adjusted_p']:.6g} | {row['reject_at_0_05']} |")
    report += ["", "Holm correction was applied across the 12 primary overall tests. Confidence intervals remain central; statistical significance alone is not evidence of practical importance.",
               "## 10. Per-label F1 paired bootstrap", "", "| Comparison | Label | Observed ΔF1 | 95% CI | Tail p | Holm p | Reject .05 |", "|---|---|---:|---:|---:|---:|---|"]
    for row in per_label_boot_rows:
        report.append(f"| {row['comparison']} | {row['label']} | {row['observed_delta']:.6f} | [{row['CI_low']:.6f}, {row['CI_high']:.6f}] | {row['bootstrap_tail_p']:.6g} | {row['holm_adjusted_p']:.6g} | {row['reject_at_0_05']} |")
    report += ["", "Holm correction is within the 12-test per-label F1 family.",
               "## 11. McNemar paired-error analysis", "", "McNemar tests paired binary correctness, not equality of F1. Exact two-sided binomial p-values use the discordant pairs; Holm correction is across the 12 primary tests.", "",
               "| Comparison | Label | n00 | n01 | n10 | n11 | Discordant | Exact p | Holm p | Reject .05 | Net gain | Gain rate |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for row in mcnemar_rows:
        report.append(f"| {row['comparison']} | {row['label']} | {row['n00_both_wrong']} | {row['n01_H12_wrong_baseline_correct']} | {row['n10_H12_correct_baseline_wrong']} | {row['n11_both_correct']} | {row['discordant_total']} | {row['exact_p']:.6g} | {row['holm_adjusted_p']:.6g} | {row['reject_at_0_05']} | {row['net_correct_gain']} | {row['net_correct_gain_rate']:.6f} |")
    report += ["", "## 12. Project-aware robustness", "", "These are descriptive project-level contrasts; no standard t-test treats the eight projects as IID.", "",
               "| Comparison | Metric | Better | Equal | Worse | Median Δ | Range |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in robustness_rows:
        report.append(f"| {row['comparison']} | {row['metric']} | {row['projects_H12_better']} | {row['projects_equal']} | {row['projects_H12_worse']} | {row['median_project_delta']:.6f} | [{row['min_project_delta']:.6f}, {row['max_project_delta']:.6f}] |")
    report += ["", "## 13. H12-B secondary sensitivity", "", "H12-B is not ranked with the primary representations. Positive/negative directions depend on the metric; differences below are H12-B minus H12.", "",
               "| Metric | H12 | H12-B | Δ |", "|---|---:|---:|---:|"]
    for row in sensitivity_overall:
        report.append(f"| {row['metric']} | {row['H12']:.6f} | {row['H12_B']:.6f} | {row['delta_H12B_minus_H12']:+.6f} |")
    report += ["", "### H12-B per-label sensitivity", "", "| Label | Metric | H12 | H12-B | Δ |", "|---|---|---:|---:|---:|"]
    for row in sensitivity_labels:
        report.append(f"| {row['label']} | {row['metric']} | {row['H12']:.6f} | {row['H12_B']:.6f} | {row['delta_H12B_minus_H12']:+.6f} |")
    report += ["", "## 14. ET-specific sensitivity", "", "| Metric | H12 | H12-B | Δ |", "|---|---:|---:|---:|"]
    for row in et_sensitivity:
        report.append(f"| {row['metric']} | {row['H12']} | {row['H12_B']} | {row['delta_H12B_minus_H12']} |")
    report += ["", "## 15. H12-B paired bootstrap", "", "This is a separate `SECONDARY_SENSITIVITY` family; its p-values are not combined with primary Holm families.", "",
               "| Metric | Observed improvement | Bootstrap mean | 95% CI | Tail p |", "|---|---:|---:|---:|---:|"]
    for row in h12b_boot_rows:
        report.append(f"| {row['metric']} | {row['observed_delta']:.6f} | {row['bootstrap_mean_delta']:.6f} | [{row['CI_low']:.6f}, {row['CI_high']:.6f}] | {row['bootstrap_tail_p']:.6g} |")
    report += ["", "## H12-B prediction-change validation", "", "Physical exact-ID comparison against H12 was checked against the existing H12-B change audit.", "",
               "| Label | 0→0 | 1→1 | 0→1 | 1→0 | Total changed | Audit matches |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in change_rows:
        report.append(f"| {row['label']} | {row['0_to_0']} | {row['1_to_1']} | {row['0_to_1']} | {row['1_to_0']} | {row['total_changed']} | {row['audit_matches_predictions']} |")
    report += ["", "## 16. Limitations", "",
               "- Evaluation covers eight selected open-source Python projects; LOPO estimates cross-project generalization only within this collection.",
               "- AR/CTL/MNT are frozen deterministic static silver labels; ET is based on the frozen semantic adjudication/reference process.",
               "- Ten ET-unresolved canonical samples were excluded from the 14,755-row experimental dataset.",
               "- Label prevalence differs substantially; pyftpdlib has zero positive MNT support, retained without imputation or project removal.",
               "- Sample-level paired bootstrap captures uncertainty over sampled test methods but does not make the eight projects IID; repeated or structurally related tests within projects may reduce strict sample independence.",
               "- Statistical significance does not imply practical importance. No causal interpretation is made.",
               "- H12-B is a class-imbalance sensitivity analysis, not a primary representation.",
               "", "## 17. Final evidence-based conclusions", "",
               f"Among the four primary frozen V2 representations, the highest observed pooled results were: Hamming Loss {primary_best['hamming_loss']}, Exact Match {primary_best['exact_match']}, Micro-F1 {primary_best['micro_f1']}, and Macro-F1 {primary_best['macro_f1']}. These pooled summaries do not establish a universal winner; per-label maxima and project-level contrasts are reported separately above. H12-B is interpreted only as a secondary sensitivity.",
               "", "## Statistical implementation audit", "",
               f"B={BOOTSTRAPS}; seed={SEED}; paired=YES; unit=SAMPLE; same indices for pair=YES; primary comparisons=3; primary overall family=12; primary per-label F1 family=12; primary McNemar family=12; multiple comparison=HOLM; McNemar=EXACT_TWO_SIDED_BINOMIAL; project-level standard t-test=NO; H12-B in primary ranking=NO; model training=NO; threshold tuning=NO; predictions regenerated=NO.", ""]
    report_path = OUTPUT / "final_comparative_analysis.md"
    with report_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(report))

    # Physically reopen all 16 tables and markdown before creating the manifest.
    expected_rows = {
        "01_primary_overall_comparison.csv": 4, "02_primary_per_label_comparison.csv": 16,
        "03_primary_per_project_comparison.csv": 32, "04_primary_metric_ranks.csv": 16,
        "05_paired_bootstrap_overall.csv": 12, "06_paired_bootstrap_per_label_f1.csv": 12,
        "07_mcnemar_per_label.csv": 12, "08_project_robustness_summary.csv": 12,
        "09_project_metric_deltas.csv": 96, "10_h12b_sensitivity_overall.csv": 8,
        "11_h12b_sensitivity_per_label.csv": 32, "12_h12b_et_tradeoff.csv": 9,
        "13_h12b_paired_bootstrap.csv": 4, "14_h12b_prediction_change_validation.csv": 4,
        "15_metric_reproduction_audit.csv": len(metric_audit), "16_input_integrity_audit.csv": len(input_audit),
    }
    physical_outputs: dict[str, tuple[int, str]] = {}
    for filename, schema in OUT_FILES.items():
        if filename in {"17_statistical_analysis_manifest.csv"}:
            continue
        rows, fields = read_csv(OUTPUT / filename)
        require(fields == schema and len(rows) == expected_rows[filename], f"Output schema/count failed: {filename}")
        physical_outputs[filename] = (len(rows), sha256(OUTPUT / filename))
    report_content = report_path.read_text(encoding="utf-8")
    require(report_content == "\n".join(report), "Markdown physical readback mismatch")
    physical_outputs[report_path.name] = (len(report_content.splitlines()), sha256(report_path))
    require(len({(r["comparison"], r["metric"]) for r in overall_boot_rows}) == 12
            and len({(r["comparison"], r["label"]) for r in per_label_boot_rows}) == 12
            and len({(r["comparison"], r["label"]) for r in mcnemar_rows}) == 12
            and len(h12b_boot_rows) == 4, "Statistical family row-count/uniqueness check failed")
    for filename, rows in table_rows.items():
        require(len(rows) == expected_rows[filename], f"Required analysis table count failed: {filename}")
    for collection, keys in ((overall_boot_rows, ("CI_low", "CI_high", "bootstrap_tail_p", "holm_adjusted_p")),
                             (per_label_boot_rows, ("CI_low", "CI_high", "bootstrap_tail_p", "holm_adjusted_p")),
                             (mcnemar_rows, ("exact_p", "holm_adjusted_p")),
                             (h12b_boot_rows, ("CI_low", "CI_high", "bootstrap_tail_p"))):
        for row in collection:
            for key in keys:
                val = float(row[key])
                require(math.isfinite(val) and 0 <= val <= 1 if "p" in key else math.isfinite(val), f"Invalid statistical value {key}: {row}")
            if "CI_low" in row:
                require(float(row["CI_low"]) <= float(row["CI_high"]), f"Invalid confidence interval: {row}")
    require(all(row["passed"] == "YES" for row in metric_audit), "Metric reproduction audit contains a failure")
    require(len([r for r in per_project_rows if r["MNT_support_status"] == "NO_POSITIVE_SUPPORT"]) == 4,
            "Per-project table failed to retain the pyftpdlib:MNT note across primary systems")
    require(all(sha256(path) == before for path, before in frozen_snapshots.items()), "A frozen model artifact changed during analysis")
    require(all(sha256(path) == before for path, before in input_hashes.items()), "A frozen protocol input changed during analysis")

    manifest_rows = []
    for artifact, (count, digest) in physical_outputs.items():
        manifest_rows.append({
            "artifact_path": artifact, "format": "CSV" if artifact.endswith(".csv") else "Markdown",
            "data_rows": count, "sha256": digest, "bootstrap_replicates": BOOTSTRAPS,
            "bootstrap_seed": SEED, "bootstrap_paired": "YES", "bootstrap_unit": "SAMPLE",
            "same_indices_for_pair": "YES", "primary_comparisons": 3,
            "primary_overall_test_family": 12, "primary_per_label_f1_test_family": 12,
            "primary_mcnemar_test_family": 12, "multiple_comparison_method": "HOLM",
            "mcnemar_method": "EXACT_TWO_SIDED_BINOMIAL", "project_level_standard_ttest": "NO",
            "H12B_in_primary_ranking": "NO", "model_training": "NO", "threshold_tuning": "NO",
            "input_dataset_sha256": input_hashes[DATASET], "input_assignments_sha256": input_hashes[ASSIGNMENTS],
            "input_protocol_sha256": input_hashes[PROTOCOL],
        })
    manifest_path = OUTPUT / "17_statistical_analysis_manifest.csv"
    write_csv(manifest_path, manifest_rows, OUT_FILES[manifest_path.name])
    manifest_reopened, manifest_fields = read_csv(manifest_path)
    require(manifest_fields == OUT_FILES[manifest_path.name] and len(manifest_reopened) == len(physical_outputs),
            "Statistical manifest schema/row count failed")
    for row in manifest_reopened:
        target = OUTPUT / row["artifact_path"]
        require(target.is_file() and sha256(target) == row["sha256"], f"Manifest output hash mismatch: {target}")
        if row["format"] == "CSV":
            parsed, _ = read_csv(target)
            require(len(parsed) == int(row["data_rows"]), f"Manifest output row count mismatch: {target}")
    require(all(sha256(path) == before for path, before in frozen_snapshots.items()), "Frozen experiment artifacts modified")
    require(all(sha256(path) == before for path, before in input_hashes.items()), "Frozen input files modified")

    output_hashes = {p.name: sha256(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}
    console = {
        "final": "PASS", "population": 14755, "primary_representations": list(REPS),
        "secondary_sensitivity": "H12-B", "input_integrity": "PASS", "metric_reproduction": "PASS",
        "primary_overall": overall_rows, "observed_best": primary_best,
        "primary_bootstrap": overall_boot_rows, "mcnemar_test_count": len(mcnemar_rows),
        "project_robustness": robustness_rows, "h12b_sensitivity": sensitivity_overall,
        "et_sensitivity": et_sensitivity, "h12b_prediction_changes": change_rows,
        "statistical_audit": {"B": BOOTSTRAPS, "seed": SEED, "paired": "YES", "Holm": "YES",
                              "McNemar": "EXACT_TWO_SIDED_BINOMIAL", "models_trained": 0, "predictions_regenerated": 0},
        "outputs": [{"path": f"experiments/v2/final_comparison/{name}",
                 "rows": len(manifest_reopened) if name == "17_statistical_analysis_manifest.csv" else physical_outputs[name][0],
                 "sha256": digest}
                    for name, digest in output_hashes.items()],
        "frozen_experiment_artifacts_modified": "NO", "figures_created": "NO", "commit": "NO",
    }
    print(json.dumps(console, indent=2))


if __name__ == "__main__":
    main()
