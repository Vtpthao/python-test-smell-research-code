#!/usr/bin/env python3
"""Run only the frozen UniXcoder R3 V2 four-label LOPO baseline from cache."""
from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "dataset/final"
CACHE_DIR = ROOT / "experiments/r3/embeddings"
R1_DIR = ROOT / "experiments/v2/r1_tfidf"
R2_DIR = ROOT / "experiments/v2/r2_ast"
OUTPUT = ROOT / "experiments/v2/r3_unixcoder"
DATASET = FINAL / "python_4label_test_smell_experimental_v2.csv"
ASSIGNMENTS = FINAL / "lopo_fold_assignments_v2.csv"
FOLD_MANIFEST = FINAL / "lopo_fold_manifest_v2.csv"
PROTOCOL = FINAL / "multi_label_experiment_protocol_v2.md"
CANONICAL = FINAL / "python_multilabel_test_smell_dataset_v1.csv"
EXCLUSIONS = FINAL / "python_4label_test_smell_exclusions_v2.csv"
CACHE = CACHE_DIR / "unixcoder_embeddings.npy"
CACHE_IDS = CACHE_DIR / "unixcoder_sample_ids.csv"
CACHE_META = CACHE_DIR / "unixcoder_embedding_manifest.json"
R3_PROTOCOL = FINAL / "r3_pretrained_representation_protocol_v1.md"
R1_POOLED = R1_DIR / "r1_v2_pooled_metrics.csv"
R1_LABEL = R1_DIR / "r1_v2_per_label_metrics.csv"
R1_MANIFEST = R1_DIR / "r1_v2_run_manifest.csv"
R2_POOLED = R2_DIR / "r2_v2_pooled_metrics.csv"
R2_LABEL = R2_DIR / "r2_v2_per_label_metrics.csv"
R2_MANIFEST = R2_DIR / "r2_v2_run_manifest.csv"
LABELS = ("AR", "CTL", "MNT", "ET")
PROJECTS = (
    "chainer", "clusterfuzz", "grimoirelab-perceval", "numba",
    "pika", "psutil", "pyftpdlib", "python-driver",
)
CHECKPOINT = "microsoft/unixcoder-base"
EXPECTED_HASHES = {
    DATASET: "ad1d9cd262f11b6ba29287e791c4cdd1be3d8dd3bbb36b949821b8bd807448da",
    ASSIGNMENTS: "d3609f61bacdc655fe84e64273e0bf74ca801f2bbf16647693ded36b9e661faf",
    FOLD_MANIFEST: "51b37dc27182d254f9182ca01b703e3236c6ba308d426e8eeac2942248e0c8b3",
    PROTOCOL: "bb45415df64276cf80a305d4c385b84215b99f46303c8915c0c1f340ea2b2a64",
    CANONICAL: "61495f2dcd54530d4afb36ea9ac85a45b605b620502ead91ffd8ea4cbbff1092",
    EXCLUSIONS: "1c6372906b3bf2633a9f69903803f94234ead1657e3c0b0847ea821df4856270",
    CACHE: "8459614b2dbc4f5f18774f7e78b59b4db7b95d7012f04f6485fb1524be665544",
    CACHE_IDS: "b7c6e9222be257ce9fbcfb206e36733fd446cf45fd76d44ad26fd502a4cb5404",
    CACHE_META: "73c2a64e844670b79bfda9a0934d65ea1e5a484de618b80f30ee28a41dde3fa3",
    R3_PROTOCOL: "0798c1c7f217b1d48546c85235e9d6f62349a8e8d3e845d76486c7b61d300ec0",
}
CLASSIFIER_CONFIG = {
    "C": 1.0, "penalty": "l2", "solver": "liblinear", "class_weight": None,
    "max_iter": 2000, "random_state": 42,
}
CACHE_CONFIG = {
    "checkpoint": CHECKPOINT, "dimension": 768, "dtype": "float32",
    "pretrained_model_frozen": "YES", "fine_tuning": "NO",
    "label_dependent_embedding": "NO", "embedding_source": "source_code only",
    "pooling": "official_masked_mean", "max_length": 512,
    "content_capacity": 508, "long_input_policy": "HEAD_TAIL (254+254)",
    "l2_normalization_during_extraction": "NO",
}
OUTPUT_FILES = {
    "predictions": "r3_v2_oop_predictions.csv",
    "blind_predictions": "r3_v2_oop_predictions_blind.csv",
    "pooled": "r3_v2_pooled_metrics.csv",
    "per_label": "r3_v2_per_label_metrics.csv",
    "per_project": "r3_v2_per_project_metrics.csv",
    "per_project_label": "r3_v2_per_project_label_metrics.csv",
    "cache_audit": "r3_v2_cache_audit.csv",
    "summary": "r3_v2_summary.md",
    "manifest": "r3_v2_run_manifest.csv",
}
PRED_FIELDS = ["sample_id", "project_name", "fold"] + [
    field for label in LABELS for field in (f"y_true_{label}", f"y_pred_{label}", f"y_score_{label}")
]
BLIND_PRED_FIELDS = ["sample_id", "project_name", "fold"] + [
    field for label in LABELS for field in (f"y_pred_{label}", f"y_score_{label}")
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
CACHE_AUDIT_FIELDS = [
    "cache_path", "cache_sha256", "cache_ids_path", "cache_ids_sha256", "cache_metadata_path",
    "cache_metadata_sha256", "canonical_dataset_sha256", "canonical_cache_rows",
    "canonical_unique_ids", "canonical_missing_ids", "canonical_unexpected_ids",
    "canonical_dimension", "canonical_dtype", "v2_requested_rows", "v2_matched_rows",
    "v2_missing_ids", "v2_unexpected_ids", "v2_order_matches_dataset",
    "excluded_unresolved_ids", "excluded_unresolved_ids_in_v2_matrix", "pretrained_model_frozen",
    "fine_tuning", "label_dependent_embedding", "checkpoint", "cache_provenance",
]
MANIFEST_FIELDS = [
    "dataset_sha256", "assignment_sha256", "fold_manifest_sha256", "protocol_sha256",
    "canonical_dataset_sha256", "exclusions_sha256", "r3_protocol_sha256", "cache_sha256",
    "cache_ids_sha256", "cache_metadata_sha256", "r1_pooled_sha256", "r1_per_label_sha256",
    "r2_pooled_sha256", "r2_per_label_sha256", "dataset_rows", "prediction_rows",
    "canonical_cache_rows", "v2_cache_matched_rows", "fold_count", "labels", "label_order",
    "python_version", "scikit_learn_version", "numpy_version", "scipy_version", "random_state",
    "cache_config_json", "classifier_config_json", "score_definition", "zero_division_policy",
    "pretrained_model_frozen", "fine_tuning", "label_dependent_embedding",
    "artifact_row_counts_json", "artifact_sha256_json", "ast_or_embedding_failures",
    "v2_missing_ids", "v2_unexpected_ids", "excluded_unresolved_ids_in_v2_matrix",
    "test_labels_used_in_training", "full_population_fine_tuned", "held_out_project_fine_tuned",
    "resampling", "threshold_tuning", "hyperparameter_tuning", "v1_folds_used",
    "excluded_unresolved_samples_used", "leakage_violations", "r1_rerun", "r2_rerun",
    "h12_executed", "h12b_executed", "commit",
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


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_ratio": float(accuracy_score(y_true, y_pred)),
        "micro_precision": float(micro_p), "micro_recall": float(micro_r), "micro_f1": float(micro_f),
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f),
    }


def label_metric_row(y_true: np.ndarray, y_pred: np.ndarray, column: int, label: str,
                     held_out_project: str | None = None) -> dict[str, Any]:
    truth, pred = y_true[:, column], y_pred[:, column]
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


def verify_reference_run(manifest_path: Path, expected_metrics: dict[str, str]) -> dict[str, str]:
    rows, _ = read_csv(manifest_path)
    require(len(rows) == 1, f"Reference run manifest invalid: {manifest_path}")
    recorded = json.loads(rows[0]["artifact_sha256_json"])
    for key, path in expected_metrics.items():
        require(recorded.get(key) == sha256(path), f"Reference metrics hash mismatch for {path}")
    return {key: sha256(path) for key, path in expected_metrics.items()}


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing R3 V2 outputs: {OUTPUT}")
    # Verify all requested frozen inputs and cache artifacts before fitting.
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen input SHA mismatch for {path}: {actual}")

    data, data_fields = read_csv(DATASET)
    canonical, canonical_fields = read_csv(CANONICAL)
    cache_ids_rows, cache_id_fields = read_csv(CACHE_IDS)
    exclusions, _ = read_csv(EXCLUSIONS)
    require({"sample_id", "project_name", *LABELS} <= set(data_fields), "V2 dataset lacks required labels/IDs")
    require(len(data) == 14755 and len({r["sample_id"] for r in data}) == 14755, "V2 dataset count/uniqueness failed")
    require(all(r[label] in {"0", "1"} for r in data for label in LABELS), "Non-binary V2 label")
    require({"sample_id", "project_name", "file_path", "class_name", "method_name", "definition_line"} <= set(canonical_fields),
            "Canonical dataset lacks expected identity columns")
    require(cache_id_fields == ["sample_id"], "Cache ID map schema mismatch")

    data_ids = [row["sample_id"] for row in data]
    canonical_ids = [row["sample_id"] for row in canonical]
    cache_ids = [row["sample_id"] for row in cache_ids_rows]
    canonical_set = set(canonical_ids)
    v2_set = set(data_ids)
    cache_set = set(cache_ids)
    require(len(canonical) == 14765 and len(canonical_set) == 14765, "Canonical population integrity failed")
    require(len(cache_ids) == 14765 and len(cache_set) == 14765, "Canonical embedding ID count/uniqueness failed")
    require(cache_set == canonical_set, "Cache IDs do not exactly cover canonical population")
    require(v2_set <= cache_set and len(v2_set) == 14755, "V2 sample IDs are not a subset of the cache IDs")
    require(data_ids == [sid for sid in canonical_ids if sid in v2_set], "V2 order is not canonical relative order")
    excluded_ids = canonical_set - v2_set
    require(len(excluded_ids) == 10 and {r["sample_id"] for r in exclusions} == excluded_ids
            and len(exclusions) == 10, "V2 exclusions do not exactly identify omitted samples")
    require(all(r["exclusion_reason"] == "ET_SEMANTICALLY_UNRESOLVED" for r in exclusions),
            "Unexpected unresolved-exclusion reason")

    cache_metadata = json.loads(CACHE_META.read_text(encoding="utf-8"))
    require(cache_metadata.get("dataset_sha256") == EXPECTED_HASHES[CANONICAL]
            and cache_metadata.get("checkpoint") == CHECKPOINT
            and cache_metadata.get("dtype") == "float32"
            and cache_metadata.get("embedding_dimension") == 768
            and cache_metadata.get("model_mode") == "eval"
            and cache_metadata.get("gradient_mode") == "no_grad"
            and cache_metadata.get("pooling") == "official_masked_mean",
            "Canonical cache metadata does not match frozen UniXcoder provenance")
    canonical_cache = np.load(CACHE, mmap_mode="r", allow_pickle=False)
    require(canonical_cache.shape == (14765, 768) and canonical_cache.dtype == np.float32,
            f"Canonical cache shape/dtype mismatch: {canonical_cache.shape} {canonical_cache.dtype}")
    require(np.isfinite(canonical_cache).all(), "Canonical UniXcoder cache contains non-finite values")
    cache_row_by_id = {sample_id: index for index, sample_id in enumerate(cache_ids)}
    subset_indices = [cache_row_by_id[sid] for sid in data_ids]
    v2_cache_ids = [cache_ids[index] for index in subset_indices]
    v2_cache_set = set(v2_cache_ids)
    require(v2_cache_ids == data_ids and len(v2_cache_set) == 14755,
            "Exact-ID V2 cache subsetting changed membership/order or duplicated IDs")
    # Advanced indexing creates the V2 in-memory subset in frozen experimental order;
    # canonical cache bytes are never overwritten and no V2 cache is persisted.
    embeddings = np.asarray(canonical_cache[subset_indices], dtype=np.float32)
    require(embeddings.shape == (14755, 768) and embeddings.dtype == np.float32,
            "V2 cache subset shape/dtype failed")
    require(all(np.array_equal(embeddings[i], canonical_cache[cache_row_by_id[sid]])
                for i, sid in enumerate(data_ids)), "V2 subset vectors do not equal their exact-ID canonical cache rows")

    assignments, assignment_fields = read_csv(ASSIGNMENTS)
    fold_manifest, _ = read_csv(FOLD_MANIFEST)
    require(assignment_fields == ["canonical_index", "sample_id", "project_name", "fold_id", "held_out_project"],
            "Frozen fold assignment schema mismatch")
    require([r["sample_id"] for r in assignments] == data_ids and len(assignments) == 14755,
            "Frozen fold assignments do not match V2 sample order")
    project_list = sorted({r["project_name"] for r in data})
    require(project_list == list(PROJECTS), "Unexpected V2 project set")
    fold_by_project = {r["held_out_project"]: r for r in fold_manifest}
    require(len(fold_manifest) == 8 and set(fold_by_project) == set(PROJECTS), "Frozen fold manifest invalid")
    assignment_by_id = {r["sample_id"]: r for r in assignments}
    for held in PROJECTS:
        train_rows = [r for r in data if assignment_by_id[r["sample_id"]]["held_out_project"] != held]
        test_rows = [r for r in data if assignment_by_id[r["sample_id"]]["held_out_project"] == held]
        require({r["project_name"] for r in test_rows} == {held}
                and held not in {r["project_name"] for r in train_rows}, f"Frozen LOPO fold mismatch: {held}")
        require(len(train_rows) == int(fold_by_project[held]["train_n"])
                and len(test_rows) == int(fold_by_project[held]["test_n"]), f"Fold size mismatch: {held}")

    # Fit the frozen classifier only after the exact cache mapping, dataset,
    # feature dimension, and LOPO assignments have all passed validation.
    blind_by_id: dict[str, dict[str, Any]] = {}
    fold_metrics: list[dict[str, Any]] = []
    per_project_metrics: list[dict[str, Any]] = []
    per_project_label_metrics: list[dict[str, Any]] = []
    for fold_index, held in enumerate(PROJECTS, start=1):
        fold = f"LOPO_{fold_index:02d}"
        train_indices = [i for i, row in enumerate(data) if assignment_by_id[row["sample_id"]]["held_out_project"] != held]
        test_indices = [i for i, row in enumerate(data) if assignment_by_id[row["sample_id"]]["held_out_project"] == held]
        train_rows = [data[i] for i in train_indices]
        test_rows = [data[i] for i in test_indices]
        train_ids = {row["sample_id"] for row in train_rows}
        test_ids = {row["sample_id"] for row in test_rows}
        require(train_ids.isdisjoint(test_ids) and train_ids | test_ids == v2_set,
                f"Train/test partition invalid for {fold}")
        x_train = embeddings[train_indices]
        x_test = embeddings[test_indices]
        y_train = np.asarray([[int(row[label]) for label in LABELS] for row in train_rows], dtype=np.int8)
        y_pred = np.zeros((len(test_indices), len(LABELS)), dtype=np.int8)
        y_score = np.zeros((len(test_indices), len(LABELS)), dtype=np.float64)
        for column, label in enumerate(LABELS):
            model = LogisticRegression(**CLASSIFIER_CONFIG)
            model.fit(x_train, y_train[:, column])
            require(np.array_equal(model.classes_, np.asarray([0, 1])), f"Binary model classes incomplete: {fold} {label}")
            y_pred[:, column] = model.predict(x_test)
            y_score[:, column] = model.predict_proba(x_test)[:, 1]
        for local_i, data_i in enumerate(test_indices):
            source_row = data[data_i]
            sid = source_row["sample_id"]
            require(sid not in blind_by_id, f"More than one test prediction for {sid}")
            item: dict[str, Any] = {"sample_id": sid, "project_name": source_row["project_name"], "fold": fold}
            for column, label in enumerate(LABELS):
                score = float(y_score[local_i, column])
                require(np.isfinite(score) and 0 <= score <= 1, f"Invalid probability score {sid} {label}")
                item[f"y_pred_{label}"] = int(y_pred[local_i, column])
                item[f"y_score_{label}"] = format(score, ".17g")
            blind_by_id[sid] = item
        fold_metrics.append({"fold": fold, "held_out_project": held, "train_n": len(train_rows),
                             "test_n": len(test_rows), "embedding_dimension": 768,
                             "cache_dtype": "float32", "test_project_in_classifier_training": "NO"})
        print(f"Completed {fold} ({held}): train={len(train_rows)} test={len(test_rows)}")

    require(len(blind_by_id) == 14755 and set(blind_by_id) == v2_set,
            "Every V2 sample must receive exactly one R3 OOP prediction")
    blind_rows = [blind_by_id[sid] for sid in data_ids]
    blind_fields = ["sample_id", "project_name", "fold"] + [
        field for label in LABELS for field in (f"y_pred_{label}", f"y_score_{label}")
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_paths = {key: OUTPUT / filename for key, filename in OUTPUT_FILES.items()}
    require(not any(path.exists() for path in output_paths.values()), "Refusing to overwrite existing R3 V2 artifacts")
    write_csv(output_paths["blind_predictions"], blind_rows, blind_fields)
    blind_reopened, blind_schema = read_csv(output_paths["blind_predictions"])
    require(blind_schema == blind_fields and len(blind_reopened) == 14755
            and [r["sample_id"] for r in blind_reopened] == data_ids
            and len({r["sample_id"] for r in blind_reopened}) == 14755,
            "Blind R3 prediction freeze physical validation failed")
    for record in blind_reopened:
        require(record["fold"] == f"LOPO_{PROJECTS.index(record['project_name']) + 1:02d}",
                f"R3 blind prediction fold mismatch: {record['sample_id']}")
        for label in LABELS:
            require(record[f"y_pred_{label}"] in {"0", "1"}, f"Invalid R3 blind prediction {record['sample_id']} {label}")
            score = float(record[f"y_score_{label}"])
            require(np.isfinite(score) and 0 <= score <= 1, f"Invalid R3 score {record['sample_id']} {label}")

    # Test labels are first accessed now, after the blind OOP predictions have been
    # physically written and validated. This exact-ID join is evaluation only.
    data_by_id = {row["sample_id"]: row for row in data}
    predictions = []
    for blind in blind_reopened:
        source_row = data_by_id[blind["sample_id"]]
        predictions.append({
            "sample_id": blind["sample_id"], "project_name": blind["project_name"], "fold": blind["fold"],
            **{f"y_true_{label}": int(source_row[label]) for label in LABELS},
            **{f"y_pred_{label}": int(blind[f"y_pred_{label}"]) for label in LABELS},
            **{f"y_score_{label}": blind[f"y_score_{label}"] for label in LABELS},
        })
    y_true = np.asarray([[row[f"y_true_{label}"] for label in LABELS] for row in predictions], dtype=np.int8)
    y_pred = np.asarray([[row[f"y_pred_{label}"] for label in LABELS] for row in predictions], dtype=np.int8)
    pooled = metrics(y_true, y_pred)
    per_label = [label_metric_row(y_true, y_pred, col, label) for col, label in enumerate(LABELS)]
    pooled_rows = [{"metric": metric, "value": pooled[metric], "aggregation": "pooled OOP",
                    "zero_division_policy": "not applicable" if metric in {"hamming_loss", "exact_match_ratio"}
                    else "zero_division=0 for sklearn-compatible computational aggregation"}
                   for metric in ("hamming_loss", "exact_match_ratio", "micro_precision", "micro_recall", "micro_f1",
                                  "macro_precision", "macro_recall", "macro_f1")]

    for fold_info in fold_metrics:
        held = fold_info["held_out_project"]
        indices = [i for i, row in enumerate(data) if row["project_name"] == held]
        yt = y_true[indices]
        yp = y_pred[indices]
        project_metric = metrics(yt, yp)
        per_project_metrics.append({
            "held_out_project": held, "test_n": len(indices),
            "hamming_loss": project_metric["hamming_loss"],
            "exact_match_ratio": project_metric["exact_match_ratio"],
            "micro_precision": project_metric["micro_precision"], "micro_recall": project_metric["micro_recall"],
            "micro_f1": project_metric["micro_f1"],
            "macro_precision_computational": project_metric["macro_precision"],
            "macro_recall_computational": project_metric["macro_recall"],
            "macro_f1_computational": project_metric["macro_f1"],
        })
        for column, label in enumerate(LABELS):
            per_project_label_metrics.append(label_metric_row(yt, yp, column, label, held_out_project=held))

    cache_audit = [{
        "cache_path": str(CACHE.relative_to(ROOT)), "cache_sha256": EXPECTED_HASHES[CACHE],
        "cache_ids_path": str(CACHE_IDS.relative_to(ROOT)), "cache_ids_sha256": EXPECTED_HASHES[CACHE_IDS],
        "cache_metadata_path": str(CACHE_META.relative_to(ROOT)), "cache_metadata_sha256": EXPECTED_HASHES[CACHE_META],
        "canonical_dataset_sha256": EXPECTED_HASHES[CANONICAL],
        "canonical_cache_rows": len(canonical_cache), "canonical_unique_ids": len(cache_set),
        "canonical_missing_ids": len(canonical_set - cache_set),
        "canonical_unexpected_ids": len(cache_set - canonical_set),
        "canonical_dimension": canonical_cache.shape[1], "canonical_dtype": str(canonical_cache.dtype),
        "v2_requested_rows": len(data_ids), "v2_matched_rows": len(v2_set & v2_cache_set),
        "v2_missing_ids": len(v2_set - v2_cache_set), "v2_unexpected_ids": len(v2_cache_set - v2_set),
        "v2_order_matches_dataset": "YES", "excluded_unresolved_ids": len(excluded_ids),
        "excluded_unresolved_ids_in_v2_matrix": len(excluded_ids & v2_set),
        "pretrained_model_frozen": "YES", "fine_tuning": "NO", "label_dependent_embedding": "NO",
        "checkpoint": cache_metadata["checkpoint"],
        "cache_provenance": "Existing canonical 14,765-row frozen UniXcoder cache; exact sample_id sidecar join; no V2 embedding cache persisted.",
    }]

    # R1/R2 metrics are read only after the frozen R3 OOP prediction table exists.
    r1_reference_hashes = verify_reference_run(R1_MANIFEST, {"pooled": R1_POOLED, "per_label": R1_LABEL})
    r2_reference_hashes = verify_reference_run(R2_MANIFEST, {"pooled": R2_POOLED, "per_label": R2_LABEL})
    r1_pooled = {row["metric"]: float(row["value"]) for row in read_csv(R1_POOLED)[0]}
    r2_pooled = {row["metric"]: float(row["value"]) for row in read_csv(R2_POOLED)[0]}
    r1_labels = {row["label"]: float(row["f1_computational"]) for row in read_csv(R1_LABEL)[0]}
    r2_labels = {row["label"]: float(row["f1_computational"]) for row in read_csv(R2_LABEL)[0]}
    comparison = {}
    for name, reference in (("R1", r1_pooled), ("R2", r2_pooled)):
        comparison[name] = {
            key: pooled[key] - reference[key]
            for key in ("hamming_loss", "exact_match_ratio", "micro_f1", "macro_f1")
        }
        labels_ref = r1_labels if name == "R1" else r2_labels
        comparison[name].update({
            f"{label}_f1": float(next(row["f1_computational"] for row in per_label if row["label"] == label)) - labels_ref[label]
            for label in LABELS
        })

    # Write OOP predictions/metrics and the exact-ID cache audit.
    write_csv(output_paths["predictions"], predictions, PRED_FIELDS)
    write_csv(output_paths["pooled"], pooled_rows, POOLED_FIELDS)
    write_csv(output_paths["per_label"], per_label, PER_LABEL_FIELDS)
    write_csv(output_paths["per_project"], per_project_metrics, PER_PROJECT_FIELDS)
    write_csv(output_paths["per_project_label"], per_project_label_metrics, PER_PROJECT_LABEL_FIELDS)
    write_csv(output_paths["cache_audit"], cache_audit, CACHE_AUDIT_FIELDS)

    schemas = {
        "predictions": PRED_FIELDS, "blind_predictions": blind_fields, "pooled": POOLED_FIELDS,
        "per_label": PER_LABEL_FIELDS, "per_project": PER_PROJECT_FIELDS,
        "per_project_label": PER_PROJECT_LABEL_FIELDS, "cache_audit": CACHE_AUDIT_FIELDS,
    }
    expected_rows = {
        "predictions": 14755, "blind_predictions": 14755, "pooled": 8, "per_label": 4,
        "per_project": 8, "per_project_label": 32, "cache_audit": 1,
    }
    physical = {}
    for key, fields_expected in schemas.items():
        reopened, field_names = read_csv(output_paths[key])
        require(field_names == fields_expected and len(reopened) == expected_rows[key],
                f"Physical artifact schema/count failed: {key}")
        physical[key] = reopened
    require([row["sample_id"] for row in physical["predictions"]] == data_ids
            and len({r["sample_id"] for r in physical["predictions"]}) == 14755,
            "Physical pooled OOP prediction membership/order failed")
    require([row["sample_id"] for row in physical["blind_predictions"]] == data_ids,
            "Blind prediction freeze does not match V2 order")
    for row in physical["predictions"]:
        require(row["fold"] == f"LOPO_{PROJECTS.index(row['project_name']) + 1:02d}",
                f"Physical R3 fold mismatch for {row['sample_id']}")
        for label in LABELS:
            require(row[f"y_true_{label}"] in {"0", "1"} and row[f"y_pred_{label}"] in {"0", "1"},
                    f"Invalid R3 output label {row['sample_id']} {label}")
            score = float(row[f"y_score_{label}"])
            require(np.isfinite(score) and 0 <= score <= 1, f"Invalid R3 output score {row['sample_id']} {label}")
    audit = physical["cache_audit"][0]
    require(audit["canonical_cache_rows"] == "14765" and audit["canonical_unique_ids"] == "14765"
            and audit["canonical_dimension"] == "768" and audit["canonical_dtype"] == "float32"
            and audit["v2_matched_rows"] == "14755" and audit["v2_missing_ids"] == "0"
            and audit["canonical_missing_ids"] == "0" and audit["canonical_unexpected_ids"] == "0"
            and audit["v2_unexpected_ids"] == "0" and audit["excluded_unresolved_ids_in_v2_matrix"] == "0",
            "Physical R3 cache audit validation failed")
    y_true_reopened = np.asarray([[int(row[f"y_true_{label}"]) for label in LABELS]
                                  for row in physical["predictions"]], dtype=np.int8)
    y_pred_reopened = np.asarray([[int(row[f"y_pred_{label}"]) for label in LABELS]
                                  for row in physical["predictions"]], dtype=np.int8)
    require(metrics(y_true_reopened, y_pred_reopened) == pooled,
            "Pooled metrics do not reproduce from physically reopened R3 predictions")

    zero_support = sorted({f"{row['held_out_project']}:{row['label']}" for row in per_project_label_metrics
                           if row["support_status"] == "NO_POSITIVE_SUPPORT"})
    report_lines = [
        "# V2 — R3 UniXcoder Deep Representation LOPO Baseline", "",
        "## Frozen inputs and cache audit", "",
        f"- Dataset: {len(data)} rows; SHA-256 `{EXPECTED_HASHES[DATASET]}`.",
        f"- Frozen LOPO assignments SHA-256 `{EXPECTED_HASHES[ASSIGNMENTS]}`; fold manifest SHA-256 `{EXPECTED_HASHES[FOLD_MANIFEST]}`; V2 protocol SHA-256 `{EXPECTED_HASHES[PROTOCOL]}`.",
        f"- Canonical population/cache: {len(canonical_cache)} rows, {len(cache_set)} unique sample IDs, dimension 768, dtype float32.",
        f"- V2 cache subset: requested={len(data_ids)}, exact matches={len(data_ids)}, missing=0, unexpected=0, excluded unresolved IDs in matrix=0.",
        f"- Cache provenance: `{CACHE.relative_to(ROOT)}`; physical SHA-256 `{EXPECTED_HASHES[CACHE]}`; exact ID sidecar SHA-256 `{EXPECTED_HASHES[CACHE_IDS]}`; metadata SHA-256 `{EXPECTED_HASHES[CACHE_META]}`.",
        f"- Checkpoint: `{CHECKPOINT}`; embedding procedure: frozen pretrained encoder-only, official masked-mean pooling, 768 float32, source_code only.",
        "- Pretrained model frozen: YES; fine-tuning: NO; label-dependent embedding generation: NO. No V2 cache was persisted; V2 vectors were selected in memory by exact sample_id.", "",
        "## LOPO and classifier", "",
        "- Exactly eight frozen V2 folds; one OOP prediction per sample, preserving V2 experimental order.",
        "- Binary Relevance: four independent Logistic Regression models; C=1.0, L2, liblinear, class_weight=None, max_iter=2000, random_state=42.",
        "- Scores are `predict_proba(X_test)[:, 1]`; predictions use the default threshold.",
        "- No scaler, PCA, feature selection, fine-tuning, calibration, balancing, resampling, threshold tuning, or hyperparameter search.", "",
        "## Cache membership audit", "",
        "| Check | Value |", "|---|---:|",
        f"| Canonical cache rows / unique IDs | {len(canonical_cache)} / {len(cache_set)} |",
        f"| Embedding dimension / dtype | {canonical_cache.shape[1]} / {canonical_cache.dtype} |",
        f"| V2 requested / matched | {len(data_ids)} / {len(data_ids)} |",
        "| V2 missing / unexpected IDs | 0 / 0 |",
        f"| Excluded unresolved IDs / included in V2 matrix | {len(excluded_ids)} / 0 |", "",
        "## Pooled metrics", "",
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
        "", "## Descriptive R3 minus R1 / R2", "",
        "R1/R2 output artifacts are read only after the R3 prediction CSV has been physically frozen. Differences are descriptive only and do not select a winner.",
        "", "| Metric | R3 − R1 | R3 − R2 |", "|---|---:|---:|",
    ]
    metric_titles = {"hamming_loss": "Hamming Loss", "exact_match_ratio": "Exact Match Ratio",
                     "micro_f1": "Micro F1", "macro_f1": "Macro F1"}
    for key, title in metric_titles.items():
        report_lines.append(f"| {title} | {comparison['R1'][key]:+.8f} | {comparison['R2'][key]:+.8f} |")
    report_lines += ["", "| Label F1 | R3 − R1 | R3 − R2 |", "|---|---:|---:|"]
    for label in LABELS:
        report_lines.append(f"| {label} | {comparison['R1'][f'{label}_f1']:+.8f} | {comparison['R2'][f'{label}_f1']:+.8f} |")
    report_lines += ["", "## Per-project metrics", "", "| Held-out project | N | Hamming Loss | Exact Match | Micro F1 | Macro F1 |", "|---|---:|---:|---:|---:|---:|"]
    for row in per_project_metrics:
        report_lines.append(
            f"| {row['held_out_project']} | {row['test_n']} | {row['hamming_loss']:.8f} | {row['exact_match_ratio']:.8f} | "
            f"{row['micro_f1']:.8f} | {row['macro_f1_computational']:.8f} |"
        )
    report_lines += [
        "", "Per-project/per-label support, precision, recall, F1, and confusion counts are in `r3_v2_per_project_label_metrics.csv`.",
        "", "## Zero support and leakage", "",
        "- NO_POSITIVE_SUPPORT: " + (", ".join(zero_support) if zero_support else "none") + ". pyftpdlib:MNT support is 0; its recall/F1 are NA and the fold is retained.",
        "- V2 labels used for embedding generation: NO.", "- Test labels used during classifier training: NO.",
        "- Fine-tuning on full population / held-out projects: NO / NO.",
        "- Resampling / threshold tuning / hyperparameter tuning: NO / NO / NO.",
        "- V1 folds used: NO; excluded unresolved samples used: NO.", "- Leakage violations: 0.",
        "- R1 rerun / R2 rerun / H12 / H12-B: NO / NO / NO / NO.",
        "- Cache canonical shape/dtype and exact-ID membership: PASS; no canonical cache file was modified.", "",
        "## Artifact hashes", "", "| Artifact | Rows | SHA-256 |", "|---|---:|---|"]

    # Manifest row covers all outputs except itself; summary SHA is captured after write.
    def write_csv_exclusive(path: Path, data_rows: list[dict[str, Any]], fields: list[str]) -> None:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
            writer.writeheader()
            writer.writerows(data_rows)

    artifact_keys = [key for key in output_paths if key not in {"summary", "manifest"}]
    artifact_hashes = {key: sha256(output_paths[key]) for key in artifact_keys}
    artifact_row_counts = {
        key: (len(read_csv(output_paths[key])[0]) if output_paths[key].suffix == ".csv" else len((output_paths[key]).read_text().splitlines()))
        for key in artifact_keys
    }
    for key in artifact_keys:
        report_lines.append(f"| {output_paths[key].name} | {artifact_row_counts[key]} | `{artifact_hashes[key]}` |")
    report_lines.append("")
    with output_paths["summary"].open("x", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(report_lines))
    artifact_hashes["summary"] = sha256(output_paths["summary"])
    artifact_row_counts["summary"] = len(output_paths["summary"].read_text().splitlines())

    manifest_row = {
        "dataset_sha256": EXPECTED_HASHES[DATASET], "assignment_sha256": EXPECTED_HASHES[ASSIGNMENTS],
        "fold_manifest_sha256": EXPECTED_HASHES[FOLD_MANIFEST], "protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "canonical_dataset_sha256": EXPECTED_HASHES[CANONICAL], "exclusions_sha256": EXPECTED_HASHES[EXCLUSIONS],
        "r3_protocol_sha256": EXPECTED_HASHES[R3_PROTOCOL],
        "cache_sha256": EXPECTED_HASHES[CACHE], "cache_ids_sha256": EXPECTED_HASHES[CACHE_IDS],
        "cache_metadata_sha256": EXPECTED_HASHES[CACHE_META],
        "r1_pooled_sha256": r1_reference_hashes["pooled"], "r1_per_label_sha256": r1_reference_hashes["per_label"],
        "r2_pooled_sha256": r2_reference_hashes["pooled"], "r2_per_label_sha256": r2_reference_hashes["per_label"],
        "dataset_rows": len(data), "prediction_rows": len(predictions),
        "canonical_cache_rows": len(canonical_cache), "v2_cache_matched_rows": len(data_ids),
        "fold_count": 8, "labels": ",".join(LABELS), "label_order": ",".join(LABELS),
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "random_state": CLASSIFIER_CONFIG["random_state"],
        "cache_config_json": json.dumps(CACHE_CONFIG, sort_keys=True, separators=(",", ":")),
        "classifier_config_json": json.dumps(CLASSIFIER_CONFIG, sort_keys=True, separators=(",", ":")),
        "score_definition": "LogisticRegression.predict_proba(X_test)[:, 1]",
        "zero_division_policy": "presentation NA for undefined precision; support-zero recall/F1 NA; macro computational zero_division=0",
        "pretrained_model_frozen": "YES", "fine_tuning": "NO", "label_dependent_embedding": "NO",
        "artifact_row_counts_json": json.dumps(artifact_row_counts, sort_keys=True, separators=(",", ":")),
        "artifact_sha256_json": json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")),
        "ast_or_embedding_failures": 0, "v2_missing_ids": 0, "v2_unexpected_ids": 0,
        "excluded_unresolved_ids_in_v2_matrix": 0, "test_labels_used_in_training": "NO",
        "full_population_fine_tuned": "NO", "held_out_project_fine_tuned": "NO",
        "resampling": "NO", "threshold_tuning": "NO", "hyperparameter_tuning": "NO",
        "v1_folds_used": "NO", "excluded_unresolved_samples_used": "NO", "leakage_violations": 0,
        "r1_rerun": "NO", "r2_rerun": "NO", "h12_executed": "NO", "h12b_executed": "NO", "commit": "NO",
    }
    write_csv_exclusive(output_paths["manifest"], [manifest_row], MANIFEST_FIELDS)
    manifest_reopened, manifest_fields = read_csv(output_paths["manifest"])
    require(manifest_fields == MANIFEST_FIELDS and len(manifest_reopened) == 1,
            "R3 run manifest physical validation failed")
    manifest_record = manifest_reopened[0]
    require(json.loads(manifest_record["artifact_sha256_json"]) == artifact_hashes
            and json.loads(manifest_record["artifact_row_counts_json"]) == artifact_row_counts,
            "R3 run manifest hashes/counts do not match physical artifacts")
    for path, expected_hash in EXPECTED_HASHES.items():
        require(sha256(path) == expected_hash, f"Frozen input changed during R3 run: {path}")

    print(json.dumps({
        "final": "PASS", "dataset_rows": len(data), "prediction_rows": len(predictions),
        "cache_audit": cache_audit[0], "pooled_metrics": pooled, "per_label_metrics": per_label,
        "per_project_metrics": per_project_metrics, "folds": fold_metrics,
        "r3_minus_r1": comparison["R1"], "r3_minus_r2": comparison["R2"],
        "zero_support": zero_support, "leakage_violations": 0,
        "artifacts": {key: {"path": str(path.relative_to(ROOT)),
                             "rows": artifact_row_counts.get(key, 1), "sha256": sha256(path)}
                      for key, path in output_paths.items()},
        "frozen_inputs_unchanged": True, "h12_h12b": "NOT_RUN", "commit": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
