#!/usr/bin/env python3
"""Run the isolated secondary H12-B class_weight='balanced' sensitivity."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Reuse only the frozen primary H12 representation and metric helpers. Importing
# this module has no execution side effects (its run is guarded by __main__).
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_h12_v2_lopo as h12  # noqa: E402

ROOT = h12.ROOT
FINAL = h12.FINAL
PRIMARY_DIR = ROOT / "experiments/v2/h12"
OUTPUT = ROOT / "experiments/v2/h12_balanced"
DATASET = h12.DATASET
ASSIGNMENTS = h12.ASSIGNMENTS
FOLD_MANIFEST = h12.FOLD_MANIFEST
PROTOCOL = h12.PROTOCOL
CANONICAL = h12.CANONICAL
PRIMARY_PRED = PRIMARY_DIR / "h12_v2_oop_predictions.csv"
PRIMARY_POOLED = PRIMARY_DIR / "h12_v2_pooled_metrics.csv"
PRIMARY_LABEL = PRIMARY_DIR / "h12_v2_per_label_metrics.csv"
LABELS = h12.LABELS
PROJECTS = h12.PROJECTS
R1_DIMENSIONS = h12.R1_DIMENSIONS
R2_DIMENSIONS = h12.R2_DIMENSIONS
H12_DIMENSIONS = {project: R1_DIMENSIONS[project] + R2_DIMENSIONS[project] for project in PROJECTS}
EXPECTED_HASHES = {
    DATASET: "ad1d9cd262f11b6ba29287e791c4cdd1be3d8dd3bbb36b949821b8bd807448da",
    ASSIGNMENTS: "d3609f61bacdc655fe84e64273e0bf74ca801f2bbf16647693ded36b9e661faf",
    FOLD_MANIFEST: "51b37dc27182d254f9182ca01b703e3236c6ba308d426e8eeac2942248e0c8b3",
    PROTOCOL: "bb45415df64276cf80a305d4c385b84215b99f46303c8915c0c1f340ea2b2a64",
    CANONICAL: "61495f2dcd54530d4afb36ea9ac85a45b605b620502ead91ffd8ea4cbbff1092",
    PRIMARY_PRED: "c56c8c2f9d48daf1ce78cb4ebeb93bcc86e66eb9dc37cd561a2946430da67e94",
    PRIMARY_POOLED: "d337f09f5ff2476e92efd719790415bfb2a8a3b1e0afe8a9a6ea5deb27cfc9d4",
    PRIMARY_LABEL: "9fe7c50410b65275b1b634fede6c8dc1452b2f4e58d6af71d063c9046a5d4acf",
}
OUTPUT_FILES = {
    "predictions": "h12b_v2_oop_predictions.csv",
    "pooled": "h12b_v2_pooled_metrics.csv",
    "per_label": "h12b_v2_per_label_metrics.csv",
    "per_project": "h12b_v2_per_project_metrics.csv",
    "per_project_label": "h12b_v2_per_project_label_metrics.csv",
    "feature_audit": "h12b_v2_fold_feature_audit.csv",
    "comparison": "h12b_v2_vs_h12_comparison.csv",
    "et_sensitivity": "h12b_v2_et_sensitivity.csv",
    "prediction_changes": "h12b_v2_prediction_change_audit.csv",
    "summary": "h12b_v2_summary.md",
    "manifest": "h12b_v2_run_manifest.csv",
}
BLIND_TEMP = ".h12b_v2_oop_predictions_blind.tmp.csv"
PRED_FIELDS = ["sample_id", "project_name", "fold"] + [
    field for label in LABELS for field in (f"y_true_{label}", f"y_pred_{label}", f"y_score_{label}")
]
POOLED_FIELDS = ["metric", "value", "aggregation", "zero_division_policy"]
METRIC_FIELDS = [
    "label", "support", "predicted_positive", "negative_support", "TP", "FP", "FN", "TN",
    "precision_computational", "precision_presentation", "recall_computational",
    "recall_presentation", "f1_computational", "f1_presentation", "support_status",
]
PER_PROJECT_FIELDS = [
    "held_out_project", "test_n", "hamming_loss", "exact_match_ratio",
    "micro_precision", "micro_recall", "micro_f1", "macro_precision_computational",
    "macro_recall_computational", "macro_f1_computational",
]
PER_PROJECT_LABEL_FIELDS = ["held_out_project", *METRIC_FIELDS]
FEATURE_FIELDS = [
    "fold", "held_out_project", "train_rows", "test_rows", "R1_dimension", "R2_dimension",
    "H12_dimension", "R1_expected_dimension", "R2_expected_dimension", "H12_expected_dimension",
    "train_matrix_rows", "train_matrix_cols", "test_matrix_rows", "test_matrix_cols", "train_nnz", "test_nnz",
    "R1_fit_scope", "R2_fit_scope", "R2_log1p", "R2_row_L2", "additional_block_weighting",
    "whole_H12_normalization", "class_weight", "AST_source_files", "AST_parse_failures",
    "AST_target_resolution_failures", "feature_extraction_failures", "test_project_in_R1_fit",
    "test_project_in_R2_fit",
]
COMPARISON_FIELDS = ["metric", "H12_primary", "H12B_balanced", "delta_H12B_minus_H12"]
ET_FIELDS = ["metric", "H12_primary", "H12B_balanced", "delta_H12B_minus_H12"]
CHANGE_FIELDS = ["label", "unchanged_0_to_0", "unchanged_1_to_1", "changed_0_to_1", "changed_1_to_0", "total_changed"]
MANIFEST_FIELDS = [
    "dataset_sha256", "assignment_sha256", "fold_manifest_sha256", "protocol_sha256",
    "canonical_identity_metadata_sha256", "primary_predictions_sha256", "primary_pooled_sha256",
    "primary_per_label_sha256", "primary_h12_all_file_hashes_json", "dataset_rows",
    "prediction_rows", "fold_count", "labels", "label_order", "python_version",
    "scikit_learn_version", "numpy_version", "scipy_version", "random_state",
    "representation_config_json", "primary_classifier_config_json", "H12B_classifier_config_json",
    "score_definition", "artifact_row_counts_json", "artifact_sha256_json",
    "R1_fit_scope", "R2_fit_scope", "full_data_fitting", "test_labels_used_in_training",
    "resampling", "threshold_tuning", "hyperparameter_tuning", "V1_folds_used",
    "excluded_unresolved_samples_used", "leakage_violations", "R1_rerun", "R2_rerun",
    "R3_rerun", "primary_H12_rerun", "H12B_only", "commit",
]
PRIMARY_H12_CONFIG = dict(h12.CLASSIFIER_CONFIG)
H12B_CLASSIFIER_CONFIG = {**PRIMARY_H12_CONFIG, "class_weight": "balanced"}
if ({key: value for key, value in H12B_CLASSIFIER_CONFIG.items() if key != "class_weight"}
        != {key: value for key, value in PRIMARY_H12_CONFIG.items() if key != "class_weight"}
        or PRIMARY_H12_CONFIG["class_weight"] is not None
        or H12B_CLASSIFIER_CONFIG["class_weight"] != "balanced"):
    raise ValueError("H12-B classifier config must differ from primary H12 only by class_weight")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    return h12.read_csv(path)


def require(condition: bool, message: str) -> None:
    h12.require(condition, message)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def per_label_row(y_true, y_pred, column: int, label: str, held_out: str | None = None) -> dict[str, Any]:
    base = h12.label_metric_row(y_true, y_pred, column, label, held_out_project=held_out)
    base["predicted_positive"] = base["TP"] + base["FP"]
    keys = PER_PROJECT_LABEL_FIELDS if held_out else METRIC_FIELDS
    return {key: base[key] for key in keys}


def finalize_existing_h12b_predictions() -> None:
    """Complete artifacts from an already-fitted run without fitting models again."""
    allowed_partial = {
        BLIND_TEMP, OUTPUT_FILES["predictions"], OUTPUT_FILES["pooled"],
        OUTPUT_FILES["per_label"], OUTPUT_FILES["per_project"],
        OUTPUT_FILES["per_project_label"], OUTPUT_FILES["feature_audit"],
    }
    present = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    require(present == allowed_partial, f"Refusing unexpected/incomplete H12-B recovery state: {sorted(present)}")
    for path, expected in EXPECTED_HASHES.items():
        require(sha(path) == expected, f"Frozen input changed before H12-B output recovery: {path}")
    primary_snapshot = {p: sha(p) for p in sorted(PRIMARY_DIR.rglob("*")) if p.is_file()}
    require(primary_snapshot.get(PRIMARY_PRED) == EXPECTED_HASHES[PRIMARY_PRED]
            and primary_snapshot.get(PRIMARY_POOLED) == EXPECTED_HASHES[PRIMARY_POOLED]
            and primary_snapshot.get(PRIMARY_LABEL) == EXPECTED_HASHES[PRIMARY_LABEL],
            "Primary H12 references changed before H12-B recovery")

    data, _ = read_csv(DATASET)
    ids = [row["sample_id"] for row in data]
    require(len(ids) == 14755 and len(set(ids)) == 14755, "Frozen V2 IDs are invalid during recovery")
    blind_path = OUTPUT / BLIND_TEMP
    blind, blind_fields = read_csv(blind_path)
    expected_blind_fields = ["sample_id", "project_name", "fold"] + [
        f"{prefix}_{label}" for label in LABELS for prefix in ("y_pred", "y_score")
    ]
    require(blind_fields == expected_blind_fields and len(blind) == 14755
            and [row["sample_id"] for row in blind] == ids, "Saved blind H12-B predictions failed recovery gate")
    pred_rows, pred_fields = read_csv(OUTPUT / OUTPUT_FILES["predictions"])
    require(pred_fields == h12.PRED_FIELDS and len(pred_rows) == 14755
            and [row["sample_id"] for row in pred_rows] == ids, "Saved H12-B predictions failed recovery gate")
    data_by_id = {row["sample_id"]: row for row in data}
    blind_by_id = {row["sample_id"]: row for row in blind}
    prediction_by_id = {row["sample_id"]: row for row in pred_rows}
    for sid in ids:
        source, blind_row, pred = data_by_id[sid], blind_by_id[sid], prediction_by_id[sid]
        require(all(int(pred[f"y_true_{label}"]) == int(source[label])
                    and int(pred[f"y_pred_{label}"]) == int(blind_row[f"y_pred_{label}"])
                    and float(pred[f"y_score_{label}"]) == float(blind_row[f"y_score_{label}"])
                    for label in LABELS), f"Saved H12-B output and blind predictions disagree for {sid}")

    yt = np.asarray([[int(row[f"y_true_{label}"]) for label in LABELS] for row in pred_rows], dtype=np.int8)
    yp = np.asarray([[int(row[f"y_pred_{label}"]) for label in LABELS] for row in pred_rows], dtype=np.int8)
    pooled = h12.metrics(yt, yp)
    pooled_saved, pooled_fields = read_csv(OUTPUT / OUTPUT_FILES["pooled"])
    require(pooled_fields == h12.POOLED_FIELDS and len(pooled_saved) == 8
            and {row["metric"]: float(row["value"]) for row in pooled_saved} == pooled,
            "Saved H12-B pooled metrics do not reproduce from predictions")
    per_label = [per_label_row(yt, yp, index, label) for index, label in enumerate(LABELS)]
    labels_saved, labels_fields = read_csv(OUTPUT / OUTPUT_FILES["per_label"])
    require(labels_fields == METRIC_FIELDS and len(labels_saved) == 4
            and [row["label"] for row in labels_saved] == list(LABELS),
            "Saved H12-B per-label metrics have invalid schema or labels")
    for expected, stored in zip(per_label, labels_saved, strict=True):
        for key in METRIC_FIELDS:
            if key in {"precision_presentation", "recall_presentation", "f1_presentation", "support_status"}:
                require(str(expected[key]) == stored[key], f"Saved H12-B label metric mismatch: {expected['label']} {key}")
            elif key == "label":
                require(expected[key] == stored[key], "Saved H12-B label order mismatch")
            elif key.endswith("_computational"):
                require(np.isclose(float(expected[key]), float(stored[key]), rtol=0, atol=1e-15),
                        f"Saved H12-B metric mismatch: {expected['label']} {key}")
            else:
                require(int(expected[key]) == int(stored[key]), f"Saved H12-B count mismatch: {expected['label']} {key}")

    primary_rows, _ = read_csv(PRIMARY_PRED)
    primary_by_id = {row["sample_id"]: row for row in primary_rows}
    primary_pooled = {row["metric"]: float(row["value"]) for row in read_csv(PRIMARY_POOLED)[0]}
    primary_labels = {row["label"]: row for row in read_csv(PRIMARY_LABEL)[0]}
    comparison = []
    for metric in ("hamming_loss", "exact_match_ratio", "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"):
        comparison.append({"metric": metric, "H12_primary": primary_pooled[metric],
                           "H12B_balanced": pooled[metric],
                           "delta_H12B_minus_H12": pooled[metric] - primary_pooled[metric]})
    for label in LABELS:
        primary = primary_labels[label]
        balanced = per_label[LABELS.index(label)]
        for metric in ("precision", "recall", "f1"):
            old, new = float(primary[f"{metric}_computational"]), float(balanced[f"{metric}_computational"])
            comparison.append({"metric": f"{label}_{metric}", "H12_primary": old,
                               "H12B_balanced": new, "delta_H12B_minus_H12": new - old})
        old_positive = int(primary["TP"]) + int(primary["FP"])
        new_positive = int(balanced["predicted_positive"])
        comparison.append({"metric": f"{label}_predicted_positive", "H12_primary": old_positive,
                           "H12B_balanced": new_positive, "delta_H12B_minus_H12": new_positive - old_positive})

    et_sensitivity = []
    et_primary = primary_labels["ET"]
    et_balanced = per_label[LABELS.index("ET")]
    for metric in ("support", "predicted_positive", "precision", "recall", "f1", "TP", "FP", "FN", "TN"):
        old = (int(et_primary["support"]) if metric == "support" else
               int(et_primary["TP"]) + int(et_primary["FP"]) if metric == "predicted_positive" else
               float(et_primary[f"{metric}_computational"]) if metric in {"precision", "recall", "f1"} else
               int(et_primary[metric]))
        new = (int(et_balanced[metric]) if metric in {"support", "predicted_positive", "TP", "FP", "FN", "TN"}
               else float(et_balanced[f"{metric}_computational"]))
        delta = new - old
        et_sensitivity.append({"metric": metric, "H12_primary": old, "H12B_balanced": new,
                               "delta_H12B_minus_H12": delta})

    changes = []
    for label in LABELS:
        transitions = Counter()
        for sid in ids:
            old = int(primary_by_id[sid][f"y_pred_{label}"])
            new = int(blind_by_id[sid][f"y_pred_{label}"])
            transitions[f"{old}->{new}"] += 1
        row = {"label": label, "unchanged_0_to_0": transitions["0->0"],
               "unchanged_1_to_1": transitions["1->1"], "changed_0_to_1": transitions["0->1"],
               "changed_1_to_0": transitions["1->0"],
               "total_changed": transitions["0->1"] + transitions["1->0"]}
        require(sum(row[key] for key in ("unchanged_0_to_0", "unchanged_1_to_1", "changed_0_to_1", "changed_1_to_0")) == 14755,
                f"Prediction transition count mismatch for {label}")
        changes.append(row)

    project_metrics, project_fields = read_csv(OUTPUT / OUTPUT_FILES["per_project"])
    project_label_metrics, project_label_fields = read_csv(OUTPUT / OUTPUT_FILES["per_project_label"])
    require(project_fields == PER_PROJECT_FIELDS and len(project_metrics) == 8
            and project_label_fields == PER_PROJECT_LABEL_FIELDS and len(project_label_metrics) == 32,
            "Saved H12-B project metrics failed recovery schema/count check")
    zero_support = sorted({f"{row['held_out_project']}:{row['label']}" for row in project_label_metrics
                           if row["support_status"] == "NO_POSITIVE_SUPPORT"})
    require(zero_support == ["pyftpdlib:MNT"], f"Unexpected H12-B zero-support labels: {zero_support}")

    # The H12-B representation is identical to H12. Reuse its physical fold
    # feature audit, changing only the recorded classifier class_weight.
    primary_audit, primary_audit_fields = read_csv(PRIMARY_DIR / "h12_v2_fold_feature_audit.csv")
    require(primary_audit_fields == h12.FEATURE_AUDIT_FIELDS and len(primary_audit) == 8,
            "Frozen primary H12 feature audit cannot support H12-B recovery")
    feature_audit = []
    for row in primary_audit:
        item = {key: row[key] for key in FEATURE_FIELDS if key in row}
        item["H12_expected_dimension"] = H12_DIMENSIONS[row["held_out_project"]]
        item["class_weight"] = "balanced"
        feature_audit.append(item)
    feature_path = OUTPUT / OUTPUT_FILES["feature_audit"]
    stale_feature_rows, stale_feature_fields = read_csv(feature_path)
    require(not stale_feature_rows, "Refusing to replace a nonempty partial H12-B feature audit")
    feature_path.unlink()
    write_csv(feature_path, feature_audit, FEATURE_FIELDS)

    write_csv(OUTPUT / OUTPUT_FILES["comparison"], comparison, COMPARISON_FIELDS)
    write_csv(OUTPUT / OUTPUT_FILES["et_sensitivity"], et_sensitivity, ET_FIELDS)
    write_csv(OUTPUT / OUTPUT_FILES["prediction_changes"], changes, CHANGE_FIELDS)
    blind_path.unlink()

    summary = [
        "# V2 — H12-B Balanced Sensitivity", "",
        "Secondary sensitivity analysis only; it does not replace primary H12. The representation and all classifier parameters are identical to H12 except `class_weight=None` is changed to `class_weight='balanced'`.",
        "", "- Dataset: 14,755 rows; frozen V2 LOPO: 8 folds; 14,755 pooled OOP predictions.",
        "- Representation: [R1 TF-IDF | row-wise L2(log1p(R2 AST))]; no additional weighting or whole-H12 normalization.",
        "- Binary Relevance; default threshold; no resampling, threshold tuning, or hyperparameter tuning.",
        "", "## Pooled metrics", "", "| Metric | H12-B balanced |", "|---|---:|",
    ]
    for row in pooled_saved:
        summary.append(f"| {row['metric']} | {float(row['value']):.8f} |")
    summary += ["", "## Per-label metrics", "", "| Label | Support | Predicted positive | Precision | Recall | F1 | TP | FP | FN | TN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in per_label:
        summary.append(f"| {row['label']} | {row['support']} | {row['predicted_positive']} | {row['precision_presentation']} | {row['recall_presentation']} | {row['f1_presentation']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} |")
    summary += ["", "## H12-B minus primary H12", "", "| Metric | H12 | H12-B | Delta |", "|---|---:|---:|---:|"]
    for row in comparison:
        summary.append(f"| {row['metric']} | {float(row['H12_primary']):.8f} | {float(row['H12B_balanced']):.8f} | {float(row['delta_H12B_minus_H12']):+.8f} |")
    summary += ["", "## ET sensitivity", "", "| Metric | H12 | H12-B | Delta |", "|---|---:|---:|---:|"]
    for row in et_sensitivity:
        summary.append(f"| {row['metric']} | {float(row['H12_primary']):.8f} | {float(row['H12B_balanced']):.8f} | {float(row['delta_H12B_minus_H12']):+.8f} |")
    summary += ["", "## Prediction changes", "", "| Label | 0→0 | 1→1 | 0→1 | 1→0 | Changed |", "|---|---:|---:|---:|---:|---:|"]
    for row in changes:
        summary.append(f"| {row['label']} | {row['unchanged_0_to_0']} | {row['unchanged_1_to_1']} | {row['changed_0_to_1']} | {row['changed_1_to_0']} | {row['total_changed']} |")
    summary += ["", "## Fold and leakage audit", "", "- All eight H12 dimensions reused and verified; AST failures: 0.",
                "- R1/R2 fitted TRAIN_ONLY_PER_FOLD; no test labels used in training, resampling, threshold tuning, or hyperparameter tuning.",
                "- Unresolved samples and V1 folds excluded; leakage violations: 0.",
                f"- Zero-positive-support combinations: {', '.join(zero_support)} (retained).",
                "- Primary H12 artifacts unchanged; no R1/R2/R3 or primary H12 models rerun. H12-B only; no statistical testing or commit.", ""]
    summary_path = OUTPUT / OUTPUT_FILES["summary"]
    with summary_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(summary))

    artifact_keys = [key for key in OUTPUT_FILES if key != "manifest"]
    artifact_hashes = {key: sha(OUTPUT / OUTPUT_FILES[key]) for key in artifact_keys}
    artifact_counts = {key: (len(read_csv(OUTPUT / OUTPUT_FILES[key])[0])
                             if OUTPUT_FILES[key].endswith(".csv") else
                             len((OUTPUT / OUTPUT_FILES[key]).read_text().splitlines()))
                       for key in artifact_keys}
    manifest_row = {
        "dataset_sha256": EXPECTED_HASHES[DATASET], "assignment_sha256": EXPECTED_HASHES[ASSIGNMENTS],
        "fold_manifest_sha256": EXPECTED_HASHES[FOLD_MANIFEST], "protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "canonical_identity_metadata_sha256": EXPECTED_HASHES[CANONICAL],
        "primary_predictions_sha256": EXPECTED_HASHES[PRIMARY_PRED],
        "primary_pooled_sha256": EXPECTED_HASHES[PRIMARY_POOLED],
        "primary_per_label_sha256": EXPECTED_HASHES[PRIMARY_LABEL],
        "primary_h12_all_file_hashes_json": json.dumps({p.name: primary_snapshot[p] for p in primary_snapshot}, sort_keys=True, separators=(",", ":")),
        "dataset_rows": 14755, "prediction_rows": 14755, "fold_count": 8,
        "labels": ",".join(LABELS), "label_order": ",".join(LABELS),
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "random_state": 42,
        "representation_config_json": json.dumps(h12.H12_CONFIG, sort_keys=True, separators=(",", ":")),
        "primary_classifier_config_json": json.dumps(PRIMARY_H12_CONFIG, sort_keys=True, separators=(",", ":")),
        "H12B_classifier_config_json": json.dumps(H12B_CLASSIFIER_CONFIG, sort_keys=True, separators=(",", ":")),
        "score_definition": "LogisticRegression.predict_proba(X_test)[:,1]",
        "artifact_row_counts_json": json.dumps(artifact_counts, sort_keys=True, separators=(",", ":")),
        "artifact_sha256_json": json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")),
        "R1_fit_scope": "TRAIN_ONLY_PER_FOLD", "R2_fit_scope": "TRAIN_ONLY_PER_FOLD",
        "full_data_fitting": "NO", "test_labels_used_in_training": "NO", "resampling": "NO",
        "threshold_tuning": "NO", "hyperparameter_tuning": "NO", "V1_folds_used": "NO",
        "excluded_unresolved_samples_used": "NO", "leakage_violations": 0,
        "R1_rerun": "NO", "R2_rerun": "NO", "R3_rerun": "NO", "primary_H12_rerun": "NO",
        "H12B_only": "YES", "commit": "NO",
    }
    write_csv(OUTPUT / OUTPUT_FILES["manifest"], [manifest_row], MANIFEST_FIELDS)
    require({p: sha(p) for p in sorted(PRIMARY_DIR.rglob("*")) if p.is_file()} == primary_snapshot,
            "Primary H12 directory changed during H12-B recovery")
    for path, expected in EXPECTED_HASHES.items():
        require(sha(path) == expected, f"Frozen H12-B input changed during recovery: {path}")
    print(json.dumps({"final": "PASS", "recovery_mode": "NO_MODEL_REFIT",
                      "pooled_metrics": pooled, "per_label": per_label,
                      "comparisons": comparison, "ET_sensitivity": et_sensitivity,
                      "prediction_changes": changes, "zero_support": zero_support,
                      "primary_unchanged": True, "artifact_hashes": artifact_hashes,
                      "manifest_sha256": sha(OUTPUT / OUTPUT_FILES["manifest"])}, indent=2))


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        finalize_existing_h12b_predictions()
        return
    # Required authoritative inputs, primary H12 reference files, and every file
    # currently under the primary directory are treated as immutable.
    for path, expected in EXPECTED_HASHES.items():
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"Frozen H12-B prerequisite hash mismatch for {path}: {actual}")
    primary_snapshot = {p: sha(p) for p in sorted(PRIMARY_DIR.rglob("*")) if p.is_file()}
    require(bool(primary_snapshot), "Primary H12 directory contains no files")
    require(primary_snapshot.get(PRIMARY_PRED) == EXPECTED_HASHES[PRIMARY_PRED]
            and primary_snapshot.get(PRIMARY_POOLED) == EXPECTED_HASHES[PRIMARY_POOLED]
            and primary_snapshot.get(PRIMARY_LABEL) == EXPECTED_HASHES[PRIMARY_LABEL],
            "Primary H12 hashes are not intact")

    data, data_fields = read_csv(DATASET)
    canonical, canonical_fields = read_csv(CANONICAL)
    assignments, assignment_fields = read_csv(ASSIGNMENTS)
    fold_manifest, _ = read_csv(FOLD_MANIFEST)
    primary_rows, primary_fields = read_csv(PRIMARY_PRED)
    primary_pooled_rows, _ = read_csv(PRIMARY_POOLED)
    primary_label_rows, _ = read_csv(PRIMARY_LABEL)
    require(len(data) == 14755 and len({r["sample_id"] for r in data}) == 14755, "V2 dataset integrity failed")
    require(len(canonical) == 14765 and len({r["sample_id"] for r in canonical}) == 14765, "Canonical identity metadata integrity failed")
    require({"sample_id", "start_line", "end_line", "repository_path", "file_path", "class_name", "method_name", "definition_line"} <= set(canonical_fields),
            "Canonical identity metadata lacks full-file AST resolver fields")
    require({"sample_id", "project_name", "file_path", "class_name", "method_name", "definition_line", "source_code", *LABELS} <= set(data_fields),
            "V2 dataset is missing required H12-B fields")
    require(all(row[label] in {"0", "1"} for row in data for label in LABELS), "Non-binary V2 labels")
    require([row["sample_id"] for row in canonical if row["sample_id"] in {r["sample_id"] for r in data}] ==
            [r["sample_id"] for r in data], "V2 order is not the canonical relative order")
    require(len({r["sample_id"] for r in canonical} - {r["sample_id"] for r in data}) == 10,
                "Expected exactly 10 unresolved canonical samples excluded from V2")
    require(assignment_fields == ["canonical_index", "sample_id", "project_name", "fold_id", "held_out_project"]
            and len(assignments) == 14755 and [r["sample_id"] for r in assignments] == [r["sample_id"] for r in data],
            "Frozen LOPO assignments schema/membership/order failed")
    require(len(fold_manifest) == 8 and {r["held_out_project"] for r in fold_manifest} == set(PROJECTS),
            "Frozen fold manifest is invalid")
    fold_by_project = {r["held_out_project"]: r for r in fold_manifest}
    require(primary_fields == h12.PRED_FIELDS and len(primary_rows) == 14755
            and [r["sample_id"] for r in primary_rows] == [r["sample_id"] for r in data]
            and len({r["sample_id"] for r in primary_rows}) == 14755,
            "Primary H12 predictions schema/membership/order failed")
    primary_by_id = {r["sample_id"]: r for r in primary_rows}
    data_ids = [r["sample_id"] for r in data]
    primary_labels = {r["label"]: r for r in primary_label_rows}
    primary_pooled = {r["metric"]: float(r["value"]) for r in primary_pooled_rows}

    # Importing the H12 module has no execution effect. Reuse its frozen lexical,
    # AST, row-normalization, CSR and metric implementation for exact comparability.
    lexical = [h12.lexical_tokens(row["source_code"]) for row in data]
    canonical_by_id = {r["sample_id"]: r for r in canonical}
    file_trees: dict[Path, ast.Module] = {}
    source_file_counts: Counter[tuple[str, str]] = Counter()
    ast_parse_failures: list[dict[str, str]] = []
    target_failures: list[str] = []
    nodes: list[ast.AST | None] = [None] * len(data)
    for i, row in enumerate(data):
        meta = canonical_by_id[row["sample_id"]]
        for field in ("project_name", "file_path", "class_name", "method_name", "definition_line", "source_code"):
            require(row[field] == meta[field], f"Canonical/V2 identity or source mismatch: {row['sample_id']} {field}")
        source_path = ROOT / "repositories" / row["project_name"] / row["file_path"]
        try:
            if source_path not in file_trees:
                file_trees[source_path] = ast.parse(source_path.read_bytes().decode("utf-8"), filename=str(source_path))
            tree = file_trees[source_path]
        except Exception as exc:
            ast_parse_failures.append({"sample_id": row["sample_id"], "file_path": row["file_path"],
                                       "error_type": type(exc).__name__, "error": str(exc)})
            continue
        candidates = [
            method for class_node in tree.body
            if isinstance(class_node, ast.ClassDef) and class_node.name == meta["class_name"]
            for method in class_node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name == meta["method_name"]
            and method.lineno == int(meta["definition_line"])
            and min([d.lineno for d in method.decorator_list] or [method.lineno]) == int(meta["start_line"])
            and (method.end_lineno or method.lineno) == int(meta["end_line"])
        ]
        if len(candidates) != 1:
            target_failures.append(row["sample_id"])
            continue
        nodes[i] = candidates[0]
        source_file_counts[(row["project_name"], row["file_path"])] += 1
    if ast_parse_failures or target_failures:
        raise ValueError(f"H12-B AST identity gate failure: parse={ast_parse_failures[:20]} target={target_failures[:100]}")
    feature_dicts = []
    extraction_failures = []
    for row, node in zip(data, nodes, strict=True):
        try:
            require(node is not None, f"Missing AST node for {row['sample_id']}")
            feature_dicts.append(h12.ast_features(node))
        except Exception as exc:
            extraction_failures.append(row["sample_id"])
            feature_dicts.append({})
    if extraction_failures:
        raise ValueError(f"H12-B AST feature extraction failed for IDs: {extraction_failures[:100]}")

    assignment_by_id = {r["sample_id"]: r["held_out_project"] for r in assignments}
    require(set(assignment_by_id) == set(data_ids), "LOPO assignments do not cover exactly the V2 IDs")
    require(all(r["project_name"] == row["project_name"] and r["held_out_project"] == row["project_name"]
                for r, row in zip(assignments, data, strict=True)), "LOPO assignment project mismatch")
    require(all(r["fold_id"] == fold_by_project[r["held_out_project"]]["fold_id"] for r in assignments),
            "LOPO assignment fold IDs disagree with frozen fold manifest")
    require(all(sum(r["held_out_project"] == held for r in assignments) == int(fold_by_project[held]["test_n"])
                for held in PROJECTS), "LOPO held-out counts disagree with frozen fold manifest")
    # Preflight all fold-specific dimensions before H12-B classifier fitting.
    for held in PROJECTS:
        train_idx = [i for i, sid in enumerate(data_ids) if assignment_by_id[sid] != held]
        r1 = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), tokenizer=str.split, preprocessor=None,
            token_pattern=None, lowercase=False, min_df=1, max_df=1.0, max_features=None,
            norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False,
            strip_accents=None, binary=False, dtype=np.float64,
        ).fit([lexical[i] for i in train_idx])
        r2 = DictVectorizer(sparse=True, sort=True).fit([feature_dicts[i] for i in train_idx])
        actual_dimensions = (len(r1.vocabulary_), len(r2.feature_names_))
        expected_dimensions = (h12.R1_DIMENSIONS[held], h12.R2_DIMENSIONS[held])
        require(actual_dimensions == expected_dimensions,
                f"H12-B representation dimension reproduction failed for {held}: {actual_dimensions} != {expected_dimensions}")

    blind_by_id = {}
    feature_audit = []
    for fold_no, held in enumerate(PROJECTS, 1):
        fold = f"LOPO_{fold_no:02d}"
        train_idx = [i for i, sid in enumerate(data_ids) if assignment_by_id[sid] != held]
        test_idx = [i for i, sid in enumerate(data_ids) if assignment_by_id[sid] == held]
        train_rows, test_rows = [data[i] for i in train_idx], [data[i] for i in test_idx]
        require({r["project_name"] for r in test_rows} == {held}
                and held not in {r["project_name"] for r in train_rows}, f"LOPO partition failure: {fold}")
        r1v = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), tokenizer=str.split, preprocessor=None,
            token_pattern=None, lowercase=False, min_df=1, max_df=1.0, max_features=None,
            norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False,
            strip_accents=None, binary=False, dtype=np.float64,
        )
        r1_train = r1v.fit_transform([lexical[i] for i in train_idx])
        r1_test = r1v.transform([lexical[i] for i in test_idx])
        r2v = DictVectorizer(sparse=True, sort=True)
        r2_train_raw = r2v.fit_transform([feature_dicts[i] for i in train_idx])
        r2_test_raw = r2v.transform([feature_dicts[i] for i in test_idx])
        d1, d2 = r1_train.shape[1], r2_train_raw.shape[1]
        require((d1, d2) == (h12.R1_DIMENSIONS[held], h12.R2_DIMENSIONS[held]),
                f"In-fold dimension mismatch for {held}")
        r2_train, r2_test = h12.normalize_r2_block(r2_train_raw), h12.normalize_r2_block(r2_test_raw)
        x_train = h12.sklearn_compatible_csr(hstack((r1_train, r2_train), format="csr"))
        x_test = h12.sklearn_compatible_csr(hstack((r1_test, r2_test), format="csr"))
        dh = d1 + d2
        require(dh == H12_DIMENSIONS[held] and x_train.shape == (len(train_rows), dh)
                and x_test.shape == (len(test_rows), dh), f"H12 dimension mismatch in {fold}")
        yt_train = np.asarray([[int(r[l]) for l in LABELS] for r in train_rows], dtype=np.int8)
        yp = np.zeros((len(test_rows), len(LABELS)), dtype=np.int8)
        ys = np.zeros((len(test_rows), len(LABELS)), dtype=np.float64)
        for j, label in enumerate(LABELS):
            model = LogisticRegression(**H12B_CLASSIFIER_CONFIG)
            model.fit(x_train, yt_train[:, j])
            require(np.array_equal(model.classes_, np.array([0, 1])), f"Binary classes missing {fold} {label}")
            yp[:, j], ys[:, j] = model.predict(x_test), model.predict_proba(x_test)[:, 1]
        for local, index in enumerate(test_idx):
            row = data[index]
            item = {"sample_id": row["sample_id"], "project_name": row["project_name"], "fold": fold}
            for j, label in enumerate(LABELS):
                require(np.isfinite(ys[local, j]) and 0 <= ys[local, j] <= 1, f"Bad score {row['sample_id']} {label}")
                item[f"y_pred_{label}"] = int(yp[local, j])
                item[f"y_score_{label}"] = format(float(ys[local, j]), ".17g")
            blind_by_id[row["sample_id"]] = item
        train_n2 = np.sqrt(np.asarray(r2_train.multiply(r2_train).sum(axis=1)).ravel())
        test_n2 = np.sqrt(np.asarray(r2_test.multiply(r2_test).sum(axis=1)).ravel())
        require(np.allclose(train_n2, 1.0, rtol=1e-7, atol=1e-9)
                and np.allclose(test_n2, 1.0, rtol=1e-7, atol=1e-9), f"R2 row normalization failed {fold}")
        feature_audit.append({
            "fold": fold, "held_out_project": held, "train_rows": len(train_rows), "test_rows": len(test_rows),
            "R1_dimension": d1, "R2_dimension": d2, "H12_dimension": dh,
            "R1_expected_dimension": R1_DIMENSIONS[held], "R2_expected_dimension": R2_DIMENSIONS[held],
            "H12_expected_dimension": H12_DIMENSIONS[held],
            "train_matrix_rows": x_train.shape[0], "train_matrix_cols": x_train.shape[1],
            "test_matrix_rows": x_test.shape[0], "test_matrix_cols": x_test.shape[1],
            "train_nnz": int(x_train.nnz), "test_nnz": int(x_test.nnz),
            "R1_fit_scope": "TRAIN_ONLY", "R2_fit_scope": "TRAIN_ONLY", "R2_log1p": "YES",
            "R2_row_L2": "YES", "additional_block_weighting": "NO", "whole_H12_normalization": "NO",
            "class_weight": "balanced", "AST_source_files": len(source_file_counts),
            "AST_parse_failures": 0, "AST_target_resolution_failures": 0, "feature_extraction_failures": 0,
            "test_project_in_R1_fit": "NO", "test_project_in_R2_fit": "NO",
        })

    require(len(blind_by_id) == 14755 and set(blind_by_id) == set(data_ids), "H12-B OOP coverage failure")
    blind_rows = [blind_by_id[sid] for sid in data_ids]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    require(not any((OUTPUT / name).exists() for name in OUTPUT_FILES.values()), "H12-B output file exists; refusing overwrite")
    def write_out(name, rows, fields):
        with (OUTPUT / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
            writer.writeheader(); writer.writerows(rows)
    blind_fields = ["sample_id", "project_name", "fold"] + [f"{p}_{l}" for l in LABELS for p in ("y_pred", "y_score")]
    blind_temp_path = OUTPUT / BLIND_TEMP
    require(not blind_temp_path.exists(), "H12-B blind temporary file already exists; refusing overwrite")
    write_out(BLIND_TEMP, blind_rows, blind_fields)
    # Test labels are evaluated only after the complete blind prediction table is physically frozen.
    blind_reopened, blind_schema = read_csv(blind_temp_path)
    require(blind_schema == blind_fields and len(blind_reopened) == 14755
            and [r["sample_id"] for r in blind_reopened] == data_ids
            and len({r["sample_id"] for r in blind_reopened}) == 14755, "Blind H12-B prediction freeze failed")
    blind_by_id_reopened = {r["sample_id"]: r for r in blind_reopened}
    row_by_id = {r["sample_id"]: r for r in data}
    predictions = []
    for b in blind_reopened:
        src = row_by_id[b["sample_id"]]
        predictions.append({"sample_id": b["sample_id"], "project_name": b["project_name"], "fold": b["fold"],
                            **{f"y_true_{l}": int(src[l]) for l in LABELS},
                            **{f"y_pred_{l}": int(b[f"y_pred_{l}"]) for l in LABELS},
                            **{f"y_score_{l}": b[f"y_score_{l}"] for l in LABELS}})
    yt = np.asarray([[r[f"y_true_{l}"] for l in LABELS] for r in predictions], dtype=np.int8)
    yp = np.asarray([[r[f"y_pred_{l}"] for l in LABELS] for r in predictions], dtype=np.int8)
    pooled = h12.metrics(yt, yp)
    per_label = []
    for j, label in enumerate(LABELS):
        per_label.append(per_label_row(yt, yp, j, label))
    pooled_rows = [{"metric": k, "value": pooled[k], "aggregation": "pooled OOP",
                    "zero_division_policy": "not applicable" if k in {"hamming_loss", "exact_match_ratio"} else "zero_division=0; presentation NA recorded per label"}
                   for k in ("hamming_loss", "exact_match_ratio", "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1")]
    project_metrics, project_label_metrics = [], []
    for held in PROJECTS:
        idx = [i for i, r in enumerate(predictions) if r["project_name"] == held]
        myt, myp = yt[idx], yp[idx]
        mb = h12.metrics(myt, myp)
        project_metrics.append({"held_out_project": held, "test_n": len(idx),
                                "hamming_loss": mb["hamming_loss"], "exact_match_ratio": mb["exact_match_ratio"],
                                "micro_precision": mb["micro_precision"], "micro_recall": mb["micro_recall"],
                                "micro_f1": mb["micro_f1"], "macro_precision_computational": mb["macro_precision"],
                                "macro_recall_computational": mb["macro_recall"], "macro_f1_computational": mb["macro_f1"]})
        for j, label in enumerate(LABELS):
            project_label_metrics.append(per_label_row(myt, myp, j, label, held_out=held))

    # Exact-ID primary-H12 comparison and prediction changes.
    primary_rows = h12.read_csv(PRIMARY_PRED)[0]
    require(len(primary_rows) == 14755 and [r["sample_id"] for r in primary_rows] == data_ids, "Primary H12 prediction alignment invalid")
    primary_by_id = {r["sample_id"]: r for r in primary_rows}
    primary_pooled = {r["metric"]: float(r["value"]) for r in h12.read_csv(PRIMARY_POOLED)[0]}
    primary_label_rows = {r["label"]: r for r in h12.read_csv(PRIMARY_LABEL)[0]}
    comparison = []
    for metric in ("hamming_loss", "exact_match_ratio", "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"):
        comparison.append({"metric": metric, "H12_primary": primary_pooled[metric],
                           "H12B_balanced": pooled[metric],
                           "delta_H12B_minus_H12": pooled[metric] - primary_pooled[metric]})
    for label in LABELS:
        primary_label = primary_label_rows[label]
        balanced_label = next(row for row in per_label if row["label"] == label)
        for metric in ("precision", "recall", "f1"):
            primary_value = float(primary_label[f"{metric}_computational"])
            balanced_value = float(balanced_label[f"{metric}_computational"])
            comparison.append({"metric": f"{label}_{metric}", "H12_primary": primary_value,
                               "H12B_balanced": balanced_value,
                               "delta_H12B_minus_H12": balanced_value - primary_value})
        primary_positive = int(primary_label["TP"]) + int(primary_label["FP"])
        balanced_positive = int(balanced_label["predicted_positive"])
        comparison.append({"metric": f"{label}_predicted_positive", "H12_primary": primary_positive,
                           "H12B_balanced": balanced_positive,
                           "delta_H12B_minus_H12": balanced_positive - primary_positive})
    et_sensitivity = []
    primary_et_columns = {
        "support": "support", "precision": "precision_computational",
        "recall": "recall_computational", "f1": "f1_computational",
        "TP": "TP", "FP": "FP", "FN": "FN", "TN": "TN",
    }
    for metric in ("support", "predicted_positive", "precision", "recall", "f1", "TP", "FP", "FN", "TN"):
        pr = primary_label_rows["ET"]
        pvalue = (int(pr["support"]) if metric == "support" else
              int(pr["TP"]) + int(pr["FP"]) if metric == "predicted_positive" else
              float(pr[primary_et_columns[metric]]))
        b = per_label[3]
        bvalue = int(b[metric]) if metric in {"support", "predicted_positive", "TP", "FP", "FN", "TN"} else float(b[f"{metric}_computational"])
        pvalue_display = pvalue
        delta = None if pvalue_display is None or bvalue is None else bvalue - pvalue_display
        et_sensitivity.append({"metric": metric, "H12_primary": "NA" if pvalue_display is None else pvalue_display,
                               "H12B_balanced": "NA" if bvalue is None else bvalue,
                               "delta_H12B_minus_H12": "NA" if delta is None else delta})
    change_rows = []
    for label in LABELS:
        trans = Counter()
        for sid in data_ids:
            old = int(primary_by_id[sid][f"y_pred_{label}"])
            new = int(blind_by_id_reopened[sid][f"y_pred_{label}"])
            trans[f"{old}->{new}"] += 1
        change_rows.append({"label": label, "unchanged_0_to_0": trans["0->0"], "unchanged_1_to_1": trans["1->1"],
                            "changed_0_to_1": trans["0->1"], "changed_1_to_0": trans["1->0"],
                            "total_changed": trans["0->1"] + trans["1->0"]})
    require(all(sum(change[key] for key in ("unchanged_0_to_0", "unchanged_1_to_1", "changed_0_to_1", "changed_1_to_0")) == 14755
                for change in change_rows), "Prediction transition counts do not cover every sample")
    zero_support = sorted({f"{row['held_out_project']}:{row['label']}" for row in project_label_metrics
                           if row["support_status"] == "NO_POSITIVE_SUPPORT"})
    require(zero_support == ["pyftpdlib:MNT"], f"Unexpected zero-positive-support combinations: {zero_support}")

    # Write metrics and analyses; prediction artifact is blind-freeze counterpart plus truth for scoring.
    write_out(OUTPUT_FILES["predictions"], predictions, h12.PRED_FIELDS)
    write_out(OUTPUT_FILES["pooled"], pooled_rows, h12.POOLED_FIELDS)
    write_out(OUTPUT_FILES["per_label"], per_label, METRIC_FIELDS)
    write_out(OUTPUT_FILES["per_project"], project_metrics, PER_PROJECT_FIELDS)
    write_out(OUTPUT_FILES["per_project_label"], project_label_metrics, PER_PROJECT_LABEL_FIELDS)
    write_out(OUTPUT_FILES["feature_audit"], feature_audit, FEATURE_FIELDS)
    write_out(OUTPUT_FILES["comparison"], comparison, COMPARISON_FIELDS)
    write_out(OUTPUT_FILES["et_sensitivity"], et_sensitivity, ET_FIELDS)
    write_out(OUTPUT_FILES["prediction_changes"], change_rows, CHANGE_FIELDS)

    # Reopen/validate each table physically.
    expected_rows = {"predictions": 14755, "pooled": 8,
                     "per_label": 4, "per_project": 8, "per_project_label": 32,
                     "feature_audit": 8, "comparison": 24, "et_sensitivity": 9, "prediction_changes": 4}
    schemas = {"predictions": h12.PRED_FIELDS, "pooled": h12.POOLED_FIELDS,
               "per_label": METRIC_FIELDS, "per_project": PER_PROJECT_FIELDS,
               "per_project_label": PER_PROJECT_LABEL_FIELDS, "feature_audit": FEATURE_FIELDS,
               "comparison": COMPARISON_FIELDS, "et_sensitivity": ET_FIELDS, "prediction_changes": CHANGE_FIELDS}
    physical = {}
    for key, schema in schemas.items():
        physical[key], fields = read_csv(OUTPUT / OUTPUT_FILES[key])
        require(fields == schema and len(physical[key]) == expected_rows[key], f"Physical output validation failed: {key}")
    require([r["sample_id"] for r in physical["predictions"]] == data_ids
            and len({r["sample_id"] for r in physical["predictions"]}) == 14755,
            "Physical H12-B predictions order/membership/unique IDs failed")
    for row in physical["predictions"]:
        require(row["fold"] == fold_by_project[row["project_name"]]["fold_id"],
            f"Prediction fold mismatch for {row['sample_id']}")
        for label in LABELS:
            require(row[f"y_true_{label}"] in {"0", "1"} and row[f"y_pred_{label}"] in {"0", "1"},
                f"Invalid physical H12-B label for {row['sample_id']}:{label}")
            score = float(row[f"y_score_{label}"])
            require(np.isfinite(score) and 0 <= score <= 1,
                f"Invalid physical H12-B score for {row['sample_id']}:{label}")
    yt_reopened = np.asarray([[int(r[f"y_true_{l}"]) for l in LABELS] for r in physical["predictions"]], dtype=np.int8)
    yp_reopened = np.asarray([[int(r[f"y_pred_{l}"]) for l in LABELS] for r in physical["predictions"]], dtype=np.int8)
    require(h12.metrics(yt_reopened, yp_reopened) == pooled,
            "Pooled H12-B metrics do not reproduce from physically reopened predictions")
    primary_files_before_end = {p: sha(p) for p in sorted(PRIMARY_DIR.rglob("*")) if p.is_file()}
    require(primary_files_before_end == primary_snapshot, "Primary H12 artifact files changed during H12-B run")

    summary = [
        "# V2 — H12-B Balanced Sensitivity", "",
        "Secondary sensitivity analysis only. It does not replace the primary H12 result. The representation and all classifier parameters are identical to H12 except `class_weight=None` is changed to `class_weight='balanced'`.",
        "", f"- Dataset: {len(data)} rows; exact frozen V2 LOPO folds: 8; pooled OOP predictions: {len(predictions)}.",
        "- Representation: H12 = [R1 TF-IDF | row-wise L2(log1p(R2 AST))]. No feature weighting, whole-H12 normalization, resampling, or SMOTE.",
        "- Classifier: Binary Relevance; only intended change is `class_weight='balanced'`; default threshold; no tuning.",
        "", "## Pooled metrics", "", "| Metric | H12-B balanced |", "|---|---:|",
    ]
    for r in pooled_rows:
        summary.append(f"| {r['metric']} | {float(r['value']):.8f} |")
    summary += ["", "## Per-label metrics", "", "| Label | Support | Predicted positive | Precision | Recall | F1 | TP | FP | FN | TN | Status |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in per_label:
        summary.append(f"| {r['label']} | {r['support']} | {r['predicted_positive']} | {r['precision_presentation']} | {r['recall_presentation']} | {r['f1_presentation']} | {r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} | {r['support_status']} |")
    summary += ["", "## H12-B minus primary H12", "", "| Metric | H12 primary | H12-B | Delta |", "|---|---:|---:|---:|"]
    for r in comparison:
        summary.append(f"| {r['metric']} | {float(r['H12_primary']):.8f} | {float(r['H12B_balanced']):.8f} | {float(r['delta_H12B_minus_H12']):+.8f} |")
    summary += ["", "### ET sensitivity", "", "| ET metric | H12 | H12-B | Delta |", "|---|---:|---:|---:|"]
    for r in et_sensitivity:
        def show(v):
            try: return f"{float(v):.8f}"
            except (ValueError, TypeError): return str(v)
        summary.append(f"| {r['metric']} | {show(r['H12_primary'])} | {show(r['H12B_balanced'])} | {show(r['delta_H12B_minus_H12'])} |")
    summary += ["", "## Prediction-change audit", "", "| Label | 0→0 | 1→1 | 0→1 | 1→0 | Total changed |", "|---|---:|---:|---:|---:|---:|"]
    for r in change_rows:
        summary.append(f"| {r['label']} | {r['unchanged_0_to_0']} | {r['unchanged_1_to_1']} | {r['changed_0_to_1']} | {r['changed_1_to_0']} | {r['total_changed']} |")
    summary += ["", "## Per-project metrics", "", "| Project | N | Hamming | Exact | Micro P | Micro R | Micro F1 | Macro F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in project_metrics:
        summary.append(f"| {r['held_out_project']} | {r['test_n']} | {r['hamming_loss']:.8f} | {r['exact_match_ratio']:.8f} | {r['micro_precision']:.8f} | {r['micro_recall']:.8f} | {r['micro_f1']:.8f} | {r['macro_f1_computational']:.8f} |")
    summary += ["", "## Feature and leakage audit", "", "Dimension reproduction: PASS. R1/R2 fitting is train-only; R2 log1p and row L2 are YES; additional weighting and whole-H12 normalization are NO. AST parse/target/feature failures: 0.", f"- Zero support: {', '.join(zero_support)}, NO_POSITIVE_SUPPORT (fold retained).", "- Full-data feature fitting/test labels in training/resampling/threshold tuning/hyperparameter tuning: NO/NO/NO/NO/NO.", "- V1 folds / unresolved samples used: NO / NO. Leakage violations: 0.", "- Primary H12 outputs unchanged: YES. H12-B is secondary; it does not replace primary H12. No causal claim is made.", "- R1/R2/R3 baselines and primary H12 were not rerun. Only H12-B was trained. Statistical testing: NO. Commit: NO.", ""]
    # Remove the blind temporary only after the evaluated OOP prediction output
    # has been written and validated; it is not a persistent experiment artifact.
    blind_temp_path.unlink()

    # Manifest hashes the summary; summary intentionally points to manifest as an external run record.
    summary_path = OUTPUT / OUTPUT_FILES["summary"]
    with summary_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(summary))
    artifact_keys = [key for key in OUTPUT_FILES if key not in {"manifest"}]
    artifact_hashes = {key: sha(OUTPUT / OUTPUT_FILES[key]) for key in artifact_keys}
    artifact_counts = {key: (len(read_csv(OUTPUT / OUTPUT_FILES[key])[0]) if OUTPUT_FILES[key].endswith(".csv") else len((OUTPUT / OUTPUT_FILES[key]).read_text().splitlines())) for key in artifact_keys}
    manifest_row = {
        "dataset_sha256": EXPECTED_HASHES[DATASET], "assignment_sha256": EXPECTED_HASHES[ASSIGNMENTS],
        "fold_manifest_sha256": EXPECTED_HASHES[FOLD_MANIFEST], "protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "canonical_identity_metadata_sha256": EXPECTED_HASHES[CANONICAL],
        "primary_predictions_sha256": EXPECTED_HASHES[PRIMARY_PRED], "primary_pooled_sha256": EXPECTED_HASHES[PRIMARY_POOLED],
        "primary_per_label_sha256": EXPECTED_HASHES[PRIMARY_LABEL],
        "primary_h12_all_file_hashes_json": json.dumps({p.name: primary_snapshot[p] for p in primary_snapshot}, sort_keys=True, separators=(",", ":")),
        "dataset_rows": 14755, "prediction_rows": 14755, "fold_count": 8, "labels": ",".join(LABELS),
        "label_order": ",".join(LABELS), "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "random_state": 42, "representation_config_json": json.dumps(h12.H12_CONFIG, sort_keys=True, separators=(",", ":")),
        "primary_classifier_config_json": json.dumps(PRIMARY_H12_CONFIG, sort_keys=True, separators=(",", ":")),
        "H12B_classifier_config_json": json.dumps(H12B_CLASSIFIER_CONFIG, sort_keys=True, separators=(",", ":")),
        "score_definition": "LogisticRegression.predict_proba(X_test)[:,1]",
        "artifact_row_counts_json": json.dumps(artifact_counts, sort_keys=True, separators=(",", ":")),
        "artifact_sha256_json": json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")),
        "R1_fit_scope": "TRAIN_ONLY_PER_FOLD", "R2_fit_scope": "TRAIN_ONLY_PER_FOLD",
        "full_data_fitting": "NO", "test_labels_used_in_training": "NO", "resampling": "NO",
        "threshold_tuning": "NO", "hyperparameter_tuning": "NO", "V1_folds_used": "NO",
        "excluded_unresolved_samples_used": "NO", "leakage_violations": 0,
        "R1_rerun": "NO", "R2_rerun": "NO", "R3_rerun": "NO", "primary_H12_rerun": "NO",
        "H12B_only": "YES", "commit": "NO",
    }
    write_csv(OUTPUT / OUTPUT_FILES["manifest"], [manifest_row], MANIFEST_FIELDS)
    final_manifest, manifest_fields = read_csv(OUTPUT / OUTPUT_FILES["manifest"])
    require(manifest_fields == MANIFEST_FIELDS and len(final_manifest) == 1, "H12-B manifest validation failed")
    mf = final_manifest[0]
    require(json.loads(mf["artifact_sha256_json"]) == artifact_hashes
            and json.loads(mf["artifact_row_counts_json"]) == artifact_counts, "H12-B manifest hashes/counts mismatch")
    require({p: sha(p) for p in sorted(PRIMARY_DIR.rglob("*")) if p.is_file()} == primary_snapshot,
            "Primary H12 directory changed during H12-B")
    for path, expected in EXPECTED_HASHES.items():
        require(sha(path) == expected, f"Frozen H12-B prerequisite changed: {path}")
    print(json.dumps({"final": "PASS", "pooled_metrics": pooled, "per_label": per_label,
                      "comparisons": comparison, "ET_sensitivity": et_sensitivity,
                      "prediction_changes": change_rows, "zero_support": zero_support,
                      "primary_unchanged": True, "artifact_hashes": artifact_hashes,
                      "manifest_sha256": sha(OUTPUT / OUTPUT_FILES["manifest"])}, indent=2))


if __name__ == "__main__":
    main()
