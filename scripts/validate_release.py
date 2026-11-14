"""Validate source-free metadata, folds and frozen metrics without model training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("AR", "CTL", "MNT", "ET")
FIELDS = ["sample_id", "project_name", "file_path", "class_name", "method_name",
          "definition_line", *LABELS, "label_count", "AR_provenance", "CTL_provenance",
          "MNT_provenance", "ET_provenance"]
RUNS = {"R1": ("r1_tfidf", "r1"), "R2": ("r2_ast", "r2"),
        "R3": ("r3_unixcoder", "r3"), "H12": ("h12", "h12"),
        "H12-B": ("h12_balanced", "h12b")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_csv(path):
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        return reader.fieldnames, list(reader)


def metric_bundle(predictions):
    per_label = {}
    for label in LABELS:
        counts = Counter((int(row[f"y_true_{label}"]), int(row[f"y_pred_{label}"])) for row in predictions)
        true_positive = counts[(1, 1)]
        false_positive = counts[(0, 1)]
        false_negative = counts[(1, 0)]
        true_negative = counts[(0, 0)]
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        f1_denominator = 2 * true_positive + false_positive + false_negative
        per_label[label] = {
            "TP": true_positive, "FP": false_positive, "FN": false_negative, "TN": true_negative,
            "support": recall_denominator, "negative_support": false_positive + true_negative,
            "precision_computational": true_positive / precision_denominator if precision_denominator else 0,
            "recall_computational": true_positive / recall_denominator if recall_denominator else 0,
            "f1_computational": 2 * true_positive / f1_denominator if f1_denominator else 0,
        }
    true_positive = sum(row["TP"] for row in per_label.values())
    false_positive = sum(row["FP"] for row in per_label.values())
    false_negative = sum(row["FN"] for row in per_label.values())
    pooled = {
        "hamming_loss": (false_positive + false_negative) / (len(predictions) * len(LABELS)),
        "exact_match_ratio": sum(all(row[f"y_true_{label}"] == row[f"y_pred_{label}"] for label in LABELS) for row in predictions) / len(predictions),
        "micro_precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0,
        "micro_recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0,
        "micro_f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative) if 2 * true_positive + false_positive + false_negative else 0,
    }
    for metric in ("precision", "recall", "f1"):
        pooled[f"macro_{metric}"] = sum(row[f"{metric}_computational"] for row in per_label.values()) / len(LABELS)
    return pooled, per_label


def validate():
    contract = json.loads((ROOT / "configs/metadata_contract.json").read_text())
    fields, metadata = read_csv(ROOT / "dataset/final/python_4label_test_smell_metadata_v2.csv")
    require(fields == FIELDS, "Unexpected metadata schema or source-bearing column")
    require(len(metadata) == len({row["sample_id"] for row in metadata}) == 14755, "Sample population or unique IDs changed")
    require(sha256(ROOT / contract["public_metadata_path"]) == contract["public_metadata_sha256"], "Public metadata hash mismatch")
    by_id = {row["sample_id"]: row for row in metadata}
    projects = sorted({row["project_name"] for row in metadata})
    require(len(projects) == 8, "Expected eight projects")
    sensitive_patterns = [r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                          r"https?://", r"/(?:Users|home)/", r"-----BEGIN .*PRIVATE KEY",
                          r"gh[pousr]_[A-Za-z0-9]{30,}", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"]
    for row in metadata:
        require(all(row[label] in ("0", "1") for label in LABELS), "Non-binary label")
        require(int(row["label_count"]) == sum(int(row[label]) for label in LABELS), "Label count changed")
        path = Path(row["file_path"])
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe source file identifier")
        expected_id = "::".join([row["project_name"], row["file_path"], row["class_name"], row["method_name"], "L" + row["definition_line"]])
        require(row["sample_id"] == expected_id, "Sample ID fields disagree")
        require(not any(re.search(pattern, value) for pattern in sensitive_patterns for value in row.values()), "Sensitive metadata pattern detected")
    _, assignments = read_csv(ROOT / "dataset/final/lopo_fold_assignments_v2.csv")
    _, folds = read_csv(ROOT / "dataset/final/lopo_fold_manifest_v2.csv")
    require([row["sample_id"] for row in assignments] == list(by_id), "Fold order or population changed")
    for row in assignments:
        require(row["project_name"] == row["held_out_project"] == by_id[row["sample_id"]]["project_name"], "Fold project mismatch")
        require(row["fold_id"] == f"LOPO_{projects.index(row['project_name']) + 1:02d}", "Fold ID mismatch")
    require(len(folds) == 8, "Fold manifest count mismatch")
    for fold in folds:
        for split in ("train", "test"):
            selected = [row for row in metadata if (row["project_name"] == fold["held_out_project"]) == (split == "test")]
            require(len(selected) == int(fold[f"{split}_n"]), "Fold size mismatch")
            for label in LABELS:
                positive = sum(int(row[label]) for row in selected)
                require(positive == int(fold[f"{split}_{label}_pos"]), "Positive fold support changed")
                require(len(selected) - positive == int(fold[f"{split}_{label}_neg"]), "Negative fold support changed")
    checked_metrics = 0
    results = {}
    for name, (directory, prefix) in RUNS.items():
        folder = ROOT / "experiments/v2" / directory
        _, predictions = read_csv(folder / f"{prefix}_v2_oop_predictions.csv")
        require(len(predictions) == len({row["sample_id"] for row in predictions}) == len(metadata), "Prediction count mismatch")
        require({row["sample_id"] for row in predictions} == set(by_id), "Prediction coverage mismatch")
        for row in predictions:
            require(row["project_name"] == by_id[row["sample_id"]]["project_name"], "Prediction project changed")
            require(row["fold"] == f"LOPO_{projects.index(row['project_name']) + 1:02d}", "Prediction fold changed")
            require(all(row[f"y_true_{label}"] == by_id[row["sample_id"]][label] and row[f"y_pred_{label}"] in ("0", "1") for label in LABELS), "Prediction truth/labels mismatch")
        pooled, _ = metric_bundle(predictions)
        _, stored = read_csv(folder / f"{prefix}_v2_pooled_metrics.csv")
        for row in stored:
            require(abs(float(row["value"]) - pooled[row["metric"]]) < 1e-12, f"Pooled metric mismatch: {name}")
            checked_metrics += 1
        for suffix, project_scope, label_scope in (("per_label_metrics", False, True), ("per_project_metrics", True, False), ("per_project_label_metrics", True, True)):
            _, stored = read_csv(folder / f"{prefix}_v2_{suffix}.csv")
            for row in stored:
                subset = [prediction for prediction in predictions if prediction["project_name"] == row["held_out_project"]] if project_scope else predictions
                computed, labels = metric_bundle(subset)
                expected = labels[row["label"]] if label_scope else computed
                for key, value in expected.items():
                    saved_key = key if key in row else f"{key}_computational"
                    if saved_key in row:
                        require(abs(float(row[saved_key]) - value) < 1e-12, f"Metric mismatch: {name}/{suffix}/{key}")
                        checked_metrics += 1
        results[name] = pooled
    manifest = ROOT / "FILE_MANIFEST.csv"
    require(manifest.is_file(), "FILE_MANIFEST.csv missing")
    _, files = read_csv(manifest)
    for row in files:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest path")
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), "Missing file or symlink")
        require(path.stat().st_size == int(row["bytes"]) and sha256(path) == row["sha256"], "Manifest hash/size mismatch")
    return {"status": "PASS", "samples": len(metadata), "projects": len(projects),
            "source_code_included": False, "metadata_sensitive_pattern_hits": 0,
            "metric_comparisons": checked_metrics, "historical_metrics": results,
            "model_training": False, "full_training_reproducibility": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(validate(), indent=2))


if __name__ == "__main__":
    main()