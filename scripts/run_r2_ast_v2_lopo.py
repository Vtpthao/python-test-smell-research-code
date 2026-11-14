#!/usr/bin/env python3
"""Run only the frozen R2 AST structural V2 LOPO baseline."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "dataset/final"
DATASET = FINAL / "python_4label_test_smell_experimental_v2.csv"
ASSIGNMENTS = FINAL / "lopo_fold_assignments_v2.csv"
FOLD_MANIFEST = FINAL / "lopo_fold_manifest_v2.csv"
PROTOCOL = FINAL / "multi_label_experiment_protocol_v2.md"
CANONICAL = FINAL / "python_multilabel_test_smell_dataset_v1.csv"
R1_DIR = ROOT / "experiments/v2/r1_tfidf"
R1_MANIFEST = R1_DIR / "r1_v2_run_manifest.csv"
R1_POOLED = R1_DIR / "r1_v2_pooled_metrics.csv"
R1_LABEL = R1_DIR / "r1_v2_per_label_metrics.csv"
OUTPUT = ROOT / "experiments/v2/r2_ast"
LABELS = ("AR", "CTL", "MNT", "ET")
PROJECTS = (
    "chainer", "clusterfuzz", "grimoirelab-perceval", "numba",
    "pika", "psutil", "pyftpdlib", "python-driver",
)
EXPECTED_HASHES = {
    DATASET: "ad1d9cd262f11b6ba29287e791c4cdd1be3d8dd3bbb36b949821b8bd807448da",
    ASSIGNMENTS: "d3609f61bacdc655fe84e64273e0bf74ca801f2bbf16647693ded36b9e661faf",
    FOLD_MANIFEST: "51b37dc27182d254f9182ca01b703e3236c6ba308d426e8eeac2942248e0c8b3",
    PROTOCOL: "bb45415df64276cf80a305d4c385b84215b99f46303c8915c0c1f340ea2b2a64",
    CANONICAL: "61495f2dcd54530d4afb36ea9ac85a45b605b620502ead91ffd8ea4cbbff1092",
}
CLASSIFIER_CONFIG = {
    "C": 1.0, "penalty": "l2", "solver": "liblinear", "class_weight": None,
    "max_iter": 2000, "random_state": 42,
}
FEATURE_CONFIG = {
    "policy": "FULL_FILE_TARGET_METHOD_AST",
    "node_counts": "generic count of every AST node type",
    "parent_child_counts": "generic ParentType->ChildType relation counts",
    "metrics": ["total AST node count", "maximum AST depth"],
    "vectorizer": "DictVectorizer(sparse=True, sort=True)",
    "sparse_index_storage": "CSR indices/indptr cast to int32 for scikit-learn 1.9 sparse-input compatibility; feature values/order unchanged",
    "scaling": "none (standalone R2; no log1p or L2 block transformation)",
}
OUTPUT_FILES = {
    "predictions": "r2_v2_oop_predictions.csv",
    "pooled": "r2_v2_pooled_metrics.csv",
    "per_label": "r2_v2_per_label_metrics.csv",
    "per_project": "r2_v2_per_project_metrics.csv",
    "per_project_label": "r2_v2_per_project_label_metrics.csv",
    "feature_audit": "r2_v2_fold_feature_audit.csv",
    "source_manifest": "r2_v2_source_file_manifest.csv",
    "summary": "r2_v2_summary.md",
    "manifest": "r2_v2_run_manifest.csv",
}
PREDICTION_FIELDS = ["sample_id", "project_name", "fold"] + [
    field for label in LABELS for field in (f"y_true_{label}", f"y_pred_{label}", f"y_score_{label}")
]
POOLED_FIELDS = ["metric", "value", "aggregation", "zero_division_policy"]
PER_LABEL_FIELDS = [
    "label", "support", "negative_support", "TP", "FP", "FN", "TN",
    "precision_computational", "precision_presentation", "recall_computational",
    "recall_presentation", "f1_computational", "f1_presentation", "support_status",
]
PER_PROJECT_FIELDS = [
    "held_out_project", "test_n", "hamming_loss", "exact_match_ratio",
    "micro_precision", "micro_recall", "micro_f1", "macro_precision_computational",
    "macro_recall_computational", "macro_f1_computational",
]
PER_PROJECT_LABEL_FIELDS = ["held_out_project", *PER_LABEL_FIELDS]
FEATURE_AUDIT_FIELDS = [
    "fold", "held_out_project", "train_rows", "test_rows", "feature_dimension",
    "train_matrix_rows", "train_matrix_cols", "test_matrix_rows", "test_matrix_cols",
    "dictvectorizer_fit_rows", "fit_scope", "test_project_in_fit", "full_data_fit",
    "ast_source_files", "ast_parse_failures", "target_method_resolution_failures",
    "feature_extraction_failures",
]
SOURCE_MANIFEST_FIELDS = ["project_name", "file_path", "file_sha256", "target_method_count"]
MANIFEST_FIELDS = [
    "dataset_sha256", "assignment_sha256", "fold_manifest_sha256", "protocol_sha256",
    "canonical_identity_metadata_sha256", "r1_pooled_metrics_sha256", "r1_per_label_metrics_sha256",
    "dataset_rows", "prediction_rows", "fold_count", "labels", "label_order",
    "python_version", "scikit_learn_version", "numpy_version", "scipy_version", "random_state",
    "feature_config_json", "classifier_config_json", "score_definition",
    "zero_division_policy", "ast_parse_failure_count", "target_method_resolution_failure_count",
    "feature_extraction_failure_count", "artifact_row_counts_json", "artifact_sha256_json",
    "dictvectorizer_fit_scope", "full_data_vectorizer_fit", "test_project_in_vectorizer_fit",
    "test_labels_used_in_training", "resampling", "threshold_tuning", "hyperparameter_tuning",
    "v1_folds_used", "excluded_unresolved_samples_used", "leakage_violations",
    "r1_executed_this_task", "r3_executed", "h12_executed", "h12b_executed", "commit",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def ast_maximum_depth(node: ast.AST) -> int:
    """Return full recursive depth including node, as in the frozen V1 R2 code."""
    return 1 + max((ast_maximum_depth(child) for child in ast.iter_child_nodes(node)), default=0)


def ast_features(node: ast.AST) -> dict[str, float]:
    """Frozen generic AST node/relation counts plus two structural metrics."""
    features: Counter[str] = Counter()
    for parent in ast.walk(node):
        features[f"node::{type(parent).__name__}"] += 1
        for child in ast.iter_child_nodes(parent):
            features[f"edge::{type(parent).__name__}->{type(child).__name__}"] += 1
    features["metric::ast_node_count"] = sum(1 for _ in ast.walk(node))
    features["metric::ast_max_depth"] = ast_maximum_depth(node)
    return dict(features)


def sklearn_compatible_csr(matrix):
    """Keep DictVectorizer values unchanged; downcast only sparse index storage.

    This environment's scikit-learn requires 32-bit CSR indices. The V2
    structural matrices are far below the int32 capacity; the conversion
    changes no shape, feature values, vocabulary, or sample membership.
    """
    matrix = matrix.tocsr(copy=False)
    int32_max = np.iinfo(np.int32).max
    require(matrix.shape[0] <= int32_max and matrix.shape[1] <= int32_max
            and matrix.nnz <= int32_max, "CSR dimensions exceed 32-bit index capacity")
    if matrix.indices.dtype != np.int32:
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
    if matrix.indptr.dtype != np.int32:
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_ratio": float(accuracy_score(y_true, y_pred)),
        "micro_precision": float(micro_p), "micro_recall": float(micro_r), "micro_f1": float(micro_f),
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f),
    }


def label_metric_row(y_true: np.ndarray, y_pred: np.ndarray, column: int, label: str,
                     held_out_project: str | None = None) -> dict[str, Any]:
    truth = y_true[:, column]
    pred = y_pred[:, column]
    tp = int(np.logical_and(truth == 1, pred == 1).sum())
    fp = int(np.logical_and(truth == 0, pred == 1).sum())
    fn = int(np.logical_and(truth == 1, pred == 0).sum())
    tn = int(np.logical_and(truth == 0, pred == 0).sum())
    support = tp + fn
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if support == 0 else tp / support
    f1 = None if support == 0 or precision is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    row = {
        "label": label, "support": support, "negative_support": fp + tn,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision_computational": 0.0 if precision is None else precision,
        "precision_presentation": "NA" if precision is None else precision,
        "recall_computational": 0.0 if recall is None else recall,
        "recall_presentation": "NA" if recall is None else recall,
        "f1_computational": 0.0 if f1 is None else f1,
        "f1_presentation": "NA" if f1 is None else f1,
        "support_status": "NO_POSITIVE_SUPPORT" if support == 0 else "SUPPORTED",
    }
    if held_out_project is not None:
        row["held_out_project"] = held_out_project
    return row


def main() -> None:
    # Immutable required inputs are checksum-verified before any parsing/fitting.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing R2 V2 artifacts: {OUTPUT}")
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen input SHA mismatch for {path}: {actual}")

    rows, fields = read_csv(DATASET)
    canonical, canonical_fields = read_csv(CANONICAL)
    require(len(rows) == 14755 and len({r["sample_id"] for r in rows}) == 14755,
            "V2 experimental dataset must contain 14,755 unique samples")
    require(len(canonical) == 14765 and len({r["sample_id"] for r in canonical}) == 14765,
            "Canonical identity metadata row count/unique IDs failed")
    require(set(LABELS) <= set(fields) and {"sample_id", "source_code", "project_name", "file_path", "class_name", "method_name", "definition_line"} <= set(fields),
            "V2 dataset is missing structural extraction inputs")
    require({"sample_id", "file_path", "class_name", "method_name", "definition_line", "start_line", "end_line", "repository_path"} <= set(canonical_fields),
            "Canonical identity metadata lacks full-file resolver fields")
    require(all(row[label] in {"0", "1"} for row in rows for label in LABELS), "V2 labels are not binary")
    ids = [row["sample_id"] for row in rows]
    projects = sorted({row["project_name"] for row in rows})
    require(projects == list(PROJECTS), f"Unexpected V2 projects: {projects}")
    canonical_by_id = {row["sample_id"]: row for row in canonical}
    for row in rows:
        original = canonical_by_id.get(row["sample_id"])
        require(original is not None, f"V2 sample lacks canonical identity row: {row['sample_id']}")
        for field in ("project_name", "file_path", "class_name", "method_name", "definition_line", "source_code"):
            require(row[field] == original[field], f"Canonical identity/source mismatch for {row['sample_id']} field={field}")

    assignment_rows, assignment_fields = read_csv(ASSIGNMENTS)
    manifest_rows, manifest_fields = read_csv(FOLD_MANIFEST)
    require(assignment_fields == ["canonical_index", "sample_id", "project_name", "fold_id", "held_out_project"],
            "Unexpected frozen fold-assignment schema")
    require([row["sample_id"] for row in assignment_rows] == ids
            and len({row["sample_id"] for row in assignment_rows}) == 14755,
            "Frozen fold assignment membership/order failed")
    require(len(manifest_rows) == 8 and {row["held_out_project"] for row in manifest_rows} == set(PROJECTS),
            "Frozen fold manifest project coverage failed")
    assignment_by_id = {row["sample_id"]: row for row in assignment_rows}
    fold_by_project = {row["held_out_project"]: row for row in manifest_rows}
    for row in rows:
        assignment = assignment_by_id[row["sample_id"]]
        require(assignment["project_name"] == row["project_name"]
                and assignment["held_out_project"] == row["project_name"],
                f"Assignment project mismatch for {row['sample_id']}")
    for held in PROJECTS:
        mf = fold_by_project[held]
        train = [row for row in rows if assignment_by_id[row["sample_id"]]["held_out_project"] != held]
        test = [row for row in rows if assignment_by_id[row["sample_id"]]["held_out_project"] == held]
        require(len(train) + len(test) == len(rows)
                and {r["sample_id"] for r in train}.isdisjoint({r["sample_id"] for r in test}),
                f"Frozen fold partition failure for {held}")
        require(len(train) == int(mf["train_n"]) and len(test) == int(mf["test_n"]),
                f"Frozen fold size mismatch for {held}")
        for label in LABELS:
            for split, split_rows in (("train", train), ("test", test)):
                pos = sum(r[label] == "1" for r in split_rows)
                require(int(mf[f"{split}_{label}_pos"]) == pos
                        and int(mf[f"{split}_{label}_neg"]) == len(split_rows) - pos,
                        f"Frozen fold label support mismatch: {held} {split} {label}")

    # Resolve every target to its enclosing full repository file and exact method
    # identity, following the established V1 FULL_FILE_TARGET_METHOD_AST resolver.
    file_trees: dict[Path, ast.Module] = {}
    source_file_counts: Counter[tuple[str, str]] = Counter()
    source_file_hashes: dict[tuple[str, str], str] = {}
    nodes: list[ast.AST | None] = [None] * len(rows)
    parse_failures: list[dict[str, str]] = []
    method_failures: list[str] = []
    for index, row in enumerate(rows):
        meta = canonical_by_id[row["sample_id"]]
        source_path = ROOT / "repositories" / row["project_name"] / row["file_path"]
        key = (row["project_name"], row["file_path"])
        try:
            if source_path not in file_trees:
                raw = source_path.read_bytes()
                source_file_hashes[key] = hashlib.sha256(raw).hexdigest()
                file_trees[source_path] = ast.parse(raw.decode("utf-8"), filename=str(source_path))
            tree = file_trees[source_path]
        except Exception as exc:
            parse_failures.append({"sample_id": row["sample_id"], "file_path": row["file_path"],
                                   "error_type": type(exc).__name__, "error": str(exc)})
            continue
        candidates = [
            method
            for class_node in tree.body
            if isinstance(class_node, ast.ClassDef) and class_node.name == meta["class_name"]
            for method in class_node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name == meta["method_name"]
            and method.lineno == int(meta["definition_line"])
            and min([decorator.lineno for decorator in method.decorator_list] or [method.lineno]) == int(meta["start_line"])
            and (method.end_lineno or method.lineno) == int(meta["end_line"])
        ]
        if len(candidates) != 1:
            method_failures.append(row["sample_id"])
            continue
        nodes[index] = candidates[0]
        source_file_counts[key] += 1
    if parse_failures or method_failures:
        parse_detail = json.dumps(parse_failures[:25], ensure_ascii=False)
        raise ValueError(
            f"AST resolution gate failed before training: parse_failures={len(parse_failures)}, "
            f"target_method_resolution_failures={len(method_failures)}, "
            f"parse_details_first25={parse_detail}, method_failure_sample_ids_first100={method_failures[:100]}"
        )

    structural_features: list[dict[str, float]] = []
    feature_failures: list[str] = []
    for row, node in zip(rows, nodes, strict=True):
        try:
            if node is None:
                raise ValueError("Target AST node missing after resolver gate")
            structural_features.append(ast_features(node))
        except Exception:
            feature_failures.append(row["sample_id"])
            structural_features.append({})
    if feature_failures:
        raise ValueError(f"R2 feature extraction failures={len(feature_failures)} sample_ids_first100={feature_failures[:100]}")
    require(len(structural_features) == 14755, "Structural feature extraction count mismatch")

    # Freeze OOP predictions once all fold predictions are generated, prior to
    # reading the R1 metrics used solely for descriptive R2-minus-R1 reporting.
    prediction_by_id: dict[str, dict[str, Any]] = {}
    feature_audit: list[dict[str, Any]] = []
    per_project_metrics: list[dict[str, Any]] = []
    per_project_label_metrics: list[dict[str, Any]] = []
    for fold_number, held in enumerate(PROJECTS, start=1):
        fold = f"LOPO_{fold_number:02d}"
        train_indices = [i for i, row in enumerate(rows) if assignment_by_id[row["sample_id"]]["held_out_project"] != held]
        test_indices = [i for i, row in enumerate(rows) if assignment_by_id[row["sample_id"]]["held_out_project"] == held]
        train = [rows[i] for i in train_indices]
        test = [rows[i] for i in test_indices]
        require({r["project_name"] for r in test} == {held}
                and held not in {r["project_name"] for r in train}, f"LOPO project leakage for {fold}")
        train_x_dicts = [structural_features[i] for i in train_indices]
        test_x_dicts = [structural_features[i] for i in test_indices]
        vectorizer = DictVectorizer(sparse=True, sort=True)
        train_x = sklearn_compatible_csr(vectorizer.fit_transform(train_x_dicts))
        test_x = sklearn_compatible_csr(vectorizer.transform(test_x_dicts))
        feature_dim = len(vectorizer.feature_names_)
        require(train_x.shape == (len(train), feature_dim)
                and test_x.shape == (len(test), feature_dim), f"DictVectorizer shape mismatch for {fold}")
        require(int(fold_by_project[held]["train_n"]) == len(train)
                and int(fold_by_project[held]["test_n"]) == len(test), f"Frozen fold size mismatch for {fold}")

        y_train = np.asarray([[int(row[label]) for label in LABELS] for row in train], dtype=np.int8)
        y_test = np.asarray([[int(row[label]) for label in LABELS] for row in test], dtype=np.int8)
        y_pred = np.zeros_like(y_test)
        y_score = np.zeros(y_test.shape, dtype=np.float64)
        for column, label in enumerate(LABELS):
            model = LogisticRegression(**CLASSIFIER_CONFIG)
            model.fit(train_x, y_train[:, column])
            require(np.array_equal(model.classes_, np.asarray([0, 1])), f"Binary classes unavailable for {fold} {label}")
            y_pred[:, column] = model.predict(test_x)
            y_score[:, column] = model.predict_proba(test_x)[:, 1]

        for local_i, data_i in enumerate(test_indices):
            source = rows[data_i]
            sid = source["sample_id"]
            require(sid not in prediction_by_id, f"Duplicate OOP prediction for {sid}")
            prediction = {"sample_id": sid, "project_name": source["project_name"], "fold": fold}
            for column, label in enumerate(LABELS):
                score = float(y_score[local_i, column])
                require(np.isfinite(score) and 0 <= score <= 1, f"Invalid probability score for {sid} {label}")
                prediction[f"y_true_{label}"] = int(y_test[local_i, column])
                prediction[f"y_pred_{label}"] = int(y_pred[local_i, column])
                prediction[f"y_score_{label}"] = format(score, ".17g")
            prediction_by_id[sid] = prediction

        project_metrics = metrics(y_test, y_pred)
        per_project_metrics.append({
            "held_out_project": held, "test_n": len(test),
            "hamming_loss": project_metrics["hamming_loss"],
            "exact_match_ratio": project_metrics["exact_match_ratio"],
            "micro_precision": project_metrics["micro_precision"],
            "micro_recall": project_metrics["micro_recall"],
            "micro_f1": project_metrics["micro_f1"],
            "macro_precision_computational": project_metrics["macro_precision"],
            "macro_recall_computational": project_metrics["macro_recall"],
            "macro_f1_computational": project_metrics["macro_f1"],
        })
        for column, label in enumerate(LABELS):
            per_project_label_metrics.append(label_metric_row(y_test, y_pred, column, label, held_out_project=held))
        feature_audit.append({
            "fold": fold, "held_out_project": held,
            "train_rows": len(train), "test_rows": len(test), "feature_dimension": feature_dim,
            "train_matrix_rows": train_x.shape[0], "train_matrix_cols": train_x.shape[1],
            "test_matrix_rows": test_x.shape[0], "test_matrix_cols": test_x.shape[1],
            "dictvectorizer_fit_rows": len(train_x_dicts), "fit_scope": "TRAIN_ONLY",
            "test_project_in_fit": "NO", "full_data_fit": "NO",
            "ast_source_files": len(source_file_hashes), "ast_parse_failures": 0,
            "target_method_resolution_failures": 0, "feature_extraction_failures": 0,
        })
        print(f"Completed {fold} ({held}): train={len(train)} test={len(test)} dimension={feature_dim}")

    require(len(prediction_by_id) == len(rows) == 14755 and set(prediction_by_id) == set(ids),
            "Every V2 sample must receive exactly one OOP prediction")
    predictions = [prediction_by_id[sid] for sid in ids]
    y_true = np.asarray([[int(row[f"y_true_{label}"]) for label in LABELS] for row in predictions], dtype=np.int8)
    y_pred = np.asarray([[int(row[f"y_pred_{label}"]) for label in LABELS] for row in predictions], dtype=np.int8)
    pooled = metrics(y_true, y_pred)
    per_label = [label_metric_row(y_true, y_pred, column, label) for column, label in enumerate(LABELS)]
    pooled_rows = [
        {"metric": key, "value": pooled[key], "aggregation": "pooled OOP",
         "zero_division_policy": "not applicable" if key in {"hamming_loss", "exact_match_ratio"}
         else "zero_division=0 for sklearn-compatible computational aggregation"}
        for key in ("hamming_loss", "exact_match_ratio", "micro_precision", "micro_recall", "micro_f1",
                    "macro_precision", "macro_recall", "macro_f1")
    ]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_paths = {key: OUTPUT / name for key, name in OUTPUT_FILES.items()}
    require(not any(path.exists() for path in output_paths.values()), "Refusing to overwrite existing R2 V2 output")
    write_csv(output_paths["predictions"], predictions, PREDICTION_FIELDS)
    reopened_predictions, pred_fields = read_csv(output_paths["predictions"])
    require(pred_fields == PREDICTION_FIELDS and len(reopened_predictions) == 14755
            and len({row["sample_id"] for row in reopened_predictions}) == 14755,
            "Frozen R2 OOP prediction artifact validation failed")
    require([row["sample_id"] for row in reopened_predictions] == ids,
            "Frozen R2 prediction artifact is not in canonical experimental order")

    # Read frozen R1 performance only after the physical R2 prediction artifact is frozen.
    r1_input_hashes = {
        R1_MANIFEST: sha256(R1_MANIFEST), R1_POOLED: sha256(R1_POOLED), R1_LABEL: sha256(R1_LABEL),
    }
    r1_manifest, _ = read_csv(R1_MANIFEST)
    require(len(r1_manifest) == 1, "Frozen R1 run manifest row count failed")
    r1_hashes_recorded = json.loads(r1_manifest[0]["artifact_sha256_json"])
    require(r1_hashes_recorded.get("pooled") == r1_input_hashes[R1_POOLED]
            and r1_hashes_recorded.get("per_label") == r1_input_hashes[R1_LABEL],
            "Frozen R1 reference metric artifact hashes do not match its run manifest")
    r1_pooled_rows, _ = read_csv(R1_POOLED)
    r1_pooled = {row["metric"]: float(row["value"]) for row in r1_pooled_rows}
    r1_label_rows, _ = read_csv(R1_LABEL)
    r1_label_f1 = {row["label"]: float(row["f1_computational"]) for row in r1_label_rows}
    comparisons = {
        "hamming_loss": pooled["hamming_loss"] - r1_pooled["hamming_loss"],
        "exact_match_ratio": pooled["exact_match_ratio"] - r1_pooled["exact_match_ratio"],
        "micro_f1": pooled["micro_f1"] - r1_pooled["micro_f1"],
        "macro_f1": pooled["macro_f1"] - r1_pooled["macro_f1"],
          **{f"{label}_f1": float(next(row["f1_computational"] for row in per_label if row["label"] == label))
              - r1_label_f1[label] for label in LABELS},
    }

    # AST-file provenance is part of the run record, using the same full-file method resolver.
    source_manifest_rows = [
        {"project_name": project, "file_path": file_path, "file_sha256": digest,
         "target_method_count": source_file_counts[(project, file_path)]}
        for (project, file_path), digest in sorted(source_file_hashes.items())
    ]

    write_csv(output_paths["pooled"], pooled_rows, POOLED_FIELDS)
    write_csv(output_paths["per_label"], per_label, PER_LABEL_FIELDS)
    write_csv(output_paths["per_project"], per_project_metrics, PER_PROJECT_FIELDS)
    write_csv(output_paths["per_project_label"], per_project_label_metrics, PER_PROJECT_LABEL_FIELDS)
    write_csv(output_paths["feature_audit"], feature_audit, FEATURE_AUDIT_FIELDS)
    write_csv(output_paths["source_manifest"], source_manifest_rows, SOURCE_MANIFEST_FIELDS)

    rows_for_artifact = {
        "predictions": len(predictions), "pooled": len(pooled_rows), "per_label": len(per_label),
        "per_project": len(per_project_metrics), "per_project_label": len(per_project_label_metrics),
        "feature_audit": len(feature_audit), "source_manifest": len(source_manifest_rows),
    }
    output_schemas = {
        "predictions": PREDICTION_FIELDS, "pooled": POOLED_FIELDS, "per_label": PER_LABEL_FIELDS,
        "per_project": PER_PROJECT_FIELDS, "per_project_label": PER_PROJECT_LABEL_FIELDS,
        "feature_audit": FEATURE_AUDIT_FIELDS, "source_manifest": SOURCE_MANIFEST_FIELDS,
    }
    reopened: dict[str, list[dict[str, str]]] = {}
    for key in output_schemas:
        reopened[key], out_fields = read_csv(output_paths[key])
        require(out_fields == output_schemas[key] and len(reopened[key]) == rows_for_artifact[key],
                f"Physical output schema/count mismatch: {key}")
    require([row["sample_id"] for row in reopened["predictions"]] == ids
            and len({row["sample_id"] for row in reopened["predictions"]}) == 14755,
            "Physical prediction order/uniqueness failed")
    for row in reopened["predictions"]:
        require(row["fold"] == f"LOPO_{PROJECTS.index(row['project_name']) + 1:02d}",
                f"Incorrect OOP fold label: {row['sample_id']}")
        for label in LABELS:
            require(row[f"y_true_{label}"] in {"0", "1"} and row[f"y_pred_{label}"] in {"0", "1"},
                    f"Non-binary prediction artifact value: {row['sample_id']} {label}")
            score = float(row[f"y_score_{label}"])
            require(np.isfinite(score) and 0.0 <= score <= 1.0,
                    f"Invalid physical probability score: {row['sample_id']} {label}")
    require(all(row["fit_scope"] == "TRAIN_ONLY"
                and row["dictvectorizer_fit_rows"] == row["train_rows"]
                and row["test_project_in_fit"] == "NO" and row["full_data_fit"] == "NO"
                and row["ast_parse_failures"] == "0"
                and row["target_method_resolution_failures"] == "0"
                and row["feature_extraction_failures"] == "0"
                for row in reopened["feature_audit"]), "Feature audit indicates leakage/failures")
    y_true_reopened = np.asarray([[int(row[f"y_true_{label}"]) for label in LABELS]
                                  for row in reopened["predictions"]], dtype=np.int8)
    y_pred_reopened = np.asarray([[int(row[f"y_pred_{label}"]) for label in LABELS]
                                  for row in reopened["predictions"]], dtype=np.int8)
    require(metrics(y_true_reopened, y_pred_reopened) == pooled,
            "Pooled metrics do not reproduce from physically reopened OOP predictions")

    # User-facing summary with only descriptive R2-minus-R1 differences.
    warning_pairs = sorted({f"{r['held_out_project']}:{r['label']}" for r in per_project_label_metrics
                            if r["support_status"] == "NO_POSITIVE_SUPPORT"})
    report_lines = [
        "# V2 — R2 AST Structural LOPO Baseline", "",
        "## Run scope", "",
        "Only the frozen R2 AST structural baseline was run. R1 outputs were consulted only after the R2 OOP prediction CSV was physically frozen, for descriptive differences. No R3, H12, or H12-B representation was run; no statistical comparison was performed.",
        f"- V2 dataset rows: {len(rows)}; unique sample IDs: {len(ids)}.",
        f"- Pooled out-of-project predictions: {len(reopened_predictions)}.",
        "- Frozen V2 LOPO assignments and fold manifest were used exactly; predictions preserve canonical experimental order.",
        "- AST policy: FULL_FILE_TARGET_METHOD_AST using the established V1 identity resolver matched to canonical metadata; all feature values derive from the target method AST node.",
        "- Features: generic counts for all AST node types, all parent-child node-type relations, total AST node count, maximum AST depth.",
        "- DictVectorizer(sparse=True, sort=True) fitted separately on TRAIN only in each fold. No standalone R2 scaling was applied (no H12 log1p/L2 transformation).",
        "- Binary Relevance: four independent Logistic Regression models; C=1.0, L2, liblinear, class_weight=None, max_iter=2000, random_state=42. Predictions use the default threshold; saved scores are predict_proba positive-class probabilities.", "",
        "## AST extraction audit", "",
        f"- Target methods resolved: {len(nodes)} / {len(rows)}.",
        f"- AST source files parsed: {len(source_file_hashes)}.",
        f"- AST parse failures: {len(parse_failures)}.",
        f"- Target method resolution failures: {len(method_failures)}.",
        f"- Feature extraction failures: {len(feature_failures)}.",
        "- Per-file source SHA-256 values and method counts are recorded in `r2_v2_source_file_manifest.csv`.", "",
        "## Fold feature dimensions", "",
        "| Fold | Held-out project | Train rows | Test rows | Feature dimension | DictVectorizer fit rows | Fit scope |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in feature_audit:
        report_lines.append(
            f"| {row['fold']} | {row['held_out_project']} | {row['train_rows']} | {row['test_rows']} | "
            f"{row['feature_dimension']} | {row['dictvectorizer_fit_rows']} | {row['fit_scope']} |"
        )
    report_lines += [
        "", "## Pooled metrics", "",
        f"- Hamming Loss: {pooled['hamming_loss']:.8f}",
        f"- Exact Match Ratio: {pooled['exact_match_ratio']:.8f}",
        f"- Micro Precision / Recall / F1: {pooled['micro_precision']:.8f} / {pooled['micro_recall']:.8f} / {pooled['micro_f1']:.8f}",
        f"- Macro Precision / Recall / F1 (computational, zero_division=0): {pooled['macro_precision']:.8f} / {pooled['macro_recall']:.8f} / {pooled['macro_f1']:.8f}",
        "", "## Pooled per-label metrics", "",
        "| Label | Support | Precision | Recall | F1 | TP | FP | FN | TN | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in per_label:
        report_lines.append(
            f"| {row['label']} | {row['support']} | {row['precision_presentation']} | {row['recall_presentation']} | "
            f"{row['f1_presentation']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} | {row['support_status']} |"
        )
    report_lines += [
        "", "Undefined metric policy: presentation precision is NA when no positive predictions exist. Presentation recall/F1 are NA when positive support is zero; computational values use zero_division=0 for pooled macro aggregation.",
        "", "## Pooled R2 minus frozen R1 differences (descriptive only)", "",
        "| Metric | R1 | R2 | R2 − R1 |", "|---|---:|---:|---:|",
    ]
    for metric, r1_key, r2_key in (
        ("Hamming Loss", "hamming_loss", "hamming_loss"),
        ("Exact Match Ratio", "exact_match_ratio", "exact_match_ratio"),
        ("Micro F1", "micro_f1", "micro_f1"),
        ("Macro F1", "macro_f1", "macro_f1"),
    ):
        report_lines.append(f"| {metric} | {r1_pooled[r1_key]:.8f} | {pooled[r2_key]:.8f} | {comparisons[r2_key]:+.8f} |")
    report_lines += ["", "### Per-label F1 differences", "", "| Label | R1 F1 | R2 F1 | R2 − R1 |", "|---|---:|---:|---:|"]
    for label in LABELS:
        r1_f1 = r1_label_f1[label]
        r2_f1 = float(next(row["f1_computational"] for row in per_label if row["label"] == label))
        report_lines.append(f"| {label} | {r1_f1:.8f} | {r2_f1:.8f} | {r2_f1-r1_f1:+.8f} |")
    report_lines += ["", "R1 is a frozen reference only. Differences are descriptive; neither representation is selected as a winner and no inferential/model-comparison test was performed.", "", "## Per-project metrics", "", "| Held-out project | N | Hamming Loss | Exact Match | Micro F1 | Macro F1 (computational, zero_division=0) |", "|---|---:|---:|---:|---:|---:|"]
    for row in per_project_metrics:
        report_lines.append(
            f"| {row['held_out_project']} | {row['test_n']} | {row['hamming_loss']:.8f} | {row['exact_match_ratio']:.8f} | "
            f"{row['micro_f1']:.8f} | {row['macro_f1_computational']:.8f} |"
        )
    report_lines += [
        "", "Per-project/per-label precision, recall, F1, support, and confusion counts are in `r2_v2_per_project_label_metrics.csv`.",
        "", "## Zero-support and parse warnings", "",
        "- pyftpdlib:MNT: support=0; support_status=NO_POSITIVE_SUPPORT; recall/F1 presentation is NA. The fold is retained unchanged.",
        "- NO_POSITIVE_SUPPORT pairs: " + (", ".join(warning_pairs) if warning_pairs else "none") + ".",
        "- AST parse failures: 0; target-method resolution failures: 0; feature extraction failures: 0.", "",
        "## Leakage audit", "",
        "- DictVectorizer fitted train-only: PASS (one fold-local vectorizer per held-out project).",
        "- Full-data fitted vectorizer: NO.", "- Test project included in its vectorizer fit: NO.",
        "- Test labels used during feature/model fitting: NO.", "- Resampling / class balancing: NO / NO.",
        "- Threshold tuning / hyperparameter tuning: NO / NO.", "- V1 folds used: NO.",
        "- Excluded unresolved ET samples used: NO.", "- Leakage violations: 0.",
        "- R1 rerun / R3 / H12 / H12-B executed: NO / NO / NO / NO.", "",
        "## Artifact hashes", "",
        "| Artifact | Rows | SHA-256 |", "|---|---:|---|",
    ]

    # Hash tabular artifacts, write report with their hashes, then hash the report.
    def write_csv_exclusive(path: Path, data: list[dict[str, Any]], fields: list[str]) -> None:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
            writer.writeheader()
            writer.writerows(data)

    artifact_keys = [key for key in output_paths if key not in {"summary", "manifest"}]
    physical_hashes = {key: sha256(output_paths[key]) for key in artifact_keys}
    physical_rows = {key: len(read_csv(output_paths[key])[0]) for key in artifact_keys}
    for key in artifact_keys:
        report_lines.append(f"| {output_paths[key].name} | {physical_rows[key]} | `{physical_hashes[key]}` |")
    report_lines.append("")
    report_path = output_paths["summary"]
    report_text = "\n".join(report_lines)
    with report_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(report_text)
    physical_hashes["summary"] = sha256(report_path)
    physical_rows["summary"] = len(report_text.splitlines())
    manifest_row = {
        "dataset_sha256": EXPECTED_HASHES[DATASET],
        "assignment_sha256": EXPECTED_HASHES[ASSIGNMENTS],
        "fold_manifest_sha256": EXPECTED_HASHES[FOLD_MANIFEST],
        "protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "canonical_identity_metadata_sha256": EXPECTED_HASHES[CANONICAL],
        "r1_pooled_metrics_sha256": r1_input_hashes[R1_POOLED],
        "r1_per_label_metrics_sha256": r1_input_hashes[R1_LABEL],
        "dataset_rows": len(rows), "prediction_rows": len(reopened_predictions), "fold_count": len(PROJECTS),
        "labels": ",".join(LABELS), "label_order": ",".join(LABELS),
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "random_state": CLASSIFIER_CONFIG["random_state"],
        "feature_config_json": json.dumps(FEATURE_CONFIG, sort_keys=True, separators=(",", ":")),
        "classifier_config_json": json.dumps(CLASSIFIER_CONFIG, sort_keys=True, separators=(",", ":")),
        "score_definition": "LogisticRegression.predict_proba(X_test)[:, 1]",
        "zero_division_policy": "presentation NA for undefined precision; support-zero recall/F1 NA; computational macro zero_division=0",
        "ast_parse_failure_count": len(parse_failures),
        "target_method_resolution_failure_count": len(method_failures),
        "feature_extraction_failure_count": len(feature_failures),
        "artifact_row_counts_json": json.dumps(physical_rows, sort_keys=True, separators=(",", ":")),
        "artifact_sha256_json": json.dumps(physical_hashes, sort_keys=True, separators=(",", ":")),
        "dictvectorizer_fit_scope": "TRAIN_ONLY_PER_FOLD", "full_data_vectorizer_fit": "NO",
        "test_project_in_vectorizer_fit": "NO", "test_labels_used_in_training": "NO",
        "resampling": "NO", "threshold_tuning": "NO", "hyperparameter_tuning": "NO",
        "v1_folds_used": "NO", "excluded_unresolved_samples_used": "NO", "leakage_violations": 0,
        "r1_executed_this_task": "NO", "r3_executed": "NO", "h12_executed": "NO",
        "h12b_executed": "NO", "commit": "NO",
    }
    write_csv_exclusive(output_paths["manifest"], [manifest_row], MANIFEST_FIELDS)
    final_manifest, actual_fields = read_csv(output_paths["manifest"])
    require(actual_fields == MANIFEST_FIELDS and len(final_manifest) == 1, "Run manifest physical validation failed")
    manifest_record = final_manifest[0]
    require(json.loads(manifest_record["artifact_sha256_json"]) == physical_hashes,
            "Run manifest physical hashes do not match artifacts")
    require(json.loads(manifest_record["artifact_row_counts_json"]) == physical_rows,
            "Run manifest row counts do not match artifacts")
    for key, expected_hash in EXPECTED_HASHES.items():
        require(sha256(key) == expected_hash, f"Frozen input changed during R2 run: {key}")

    print(json.dumps({
        "final": "PASS", "dataset_rows": len(rows), "prediction_rows": len(reopened_predictions),
        "pooled_metrics": pooled, "per_label_metrics": per_label,
        "per_project_metrics": per_project_metrics, "feature_audit": feature_audit,
        "ast_source_files": len(source_file_hashes), "ast_parse_failures": len(parse_failures),
        "target_method_resolution_failures": len(method_failures),
        "feature_extraction_failures": len(feature_failures),
        "r2_minus_r1": comparisons, "lopo_zero_support": warning_pairs,
        "artifacts": {key: {"path": str(output_paths[key].relative_to(ROOT)),
                             "rows": physical_rows.get(key, 1), "sha256": sha256(path)}
                      for key, path in output_paths.items()},
        "frozen_inputs_unchanged": True, "leakage_violations": 0,
        "r3_h12_h12b": "NOT_RUN", "commit": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
