#!/usr/bin/env python3
"""Run only the frozen R1 TF-IDF V2 four-label LOPO baseline."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import tokenize
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from input_preflight import preflight_r1_or_exit

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "dataset/final"
DATASET = FINAL / "python_4label_test_smell_experimental_v2.csv"
ASSIGNMENTS = FINAL / "lopo_fold_assignments_v2.csv"
FOLD_MANIFEST = FINAL / "lopo_fold_manifest_v2.csv"
PROTOCOL = FINAL / "multi_label_experiment_protocol_v2.md"
OUTPUT = ROOT / "experiments/v2/r1_tfidf"
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
}
IGNORED_TOKEN_TYPES = {
    tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL,
    tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT,
}
VECTORIZER_CONFIG = {
    "analyzer": "word", "ngram_range": (1, 2), "tokenizer": "str.split",
    "preprocessor": None, "token_pattern": None, "lowercase": False,
    "min_df": 1, "max_df": 1.0, "max_features": None, "norm": "l2",
    "use_idf": True, "smooth_idf": True, "sublinear_tf": False,
    "strip_accents": None, "binary": False, "dtype": "float64",
}
CLASSIFIER_CONFIG = {
    "C": 1.0, "penalty": "l2", "solver": "liblinear", "class_weight": None,
    "max_iter": 2000, "random_state": 42,
}
OUTPUT_FILES = {
    "predictions": "r1_v2_oop_predictions.csv",
    "pooled": "r1_v2_pooled_metrics.csv",
    "per_label": "r1_v2_per_label_metrics.csv",
    "per_project": "r1_v2_per_project_metrics.csv",
    "per_project_label": "r1_v2_per_project_label_metrics.csv",
    "vocabulary": "r1_v2_fold_vocabulary_audit.csv",
    "summary": "r1_v2_summary.md",
    "manifest": "r1_v2_run_manifest.csv",
}
PREDICTION_FIELDS = ["sample_id", "project_name", "fold"] + [
    field for label in LABELS for field in (f"y_true_{label}", f"y_pred_{label}", f"y_score_{label}")
]
POOLED_FIELDS = ["metric", "value", "aggregation", "zero_division_policy"]
PER_LABEL_FIELDS = [
    "label", "support", "negative_support", "TP", "FP", "FN", "TN",
    "precision_computational", "precision_presentation", "recall_computational",
    "recall_presentation", "f1_computational", "f1_presentation",
    "support_status",
]
PER_PROJECT_FIELDS = ["held_out_project", "test_n", "hamming_loss", "exact_match_ratio", "micro_f1", "macro_f1"]
PER_PROJECT_LABEL_FIELDS = [
    "held_out_project", "label", "support", "negative_support", "TP", "FP", "FN", "TN",
    "precision_computational", "precision_presentation", "recall_computational",
    "recall_presentation", "f1_computational", "f1_presentation", "support_status",
]
VOCAB_FIELDS = [
    "fold", "held_out_project", "train_rows", "test_rows", "vocabulary_size",
    "train_matrix_rows", "train_matrix_cols", "test_matrix_rows", "test_matrix_cols",
    "train_nnz", "test_nnz", "fit_scope", "test_project_in_fit", "full_data_fit",
]
MANIFEST_FIELDS = [
    "dataset_sha256", "assignment_sha256", "fold_manifest_sha256", "protocol_sha256",
    "dataset_rows", "prediction_rows", "fold_count", "labels", "label_order",
    "python_version", "scikit_learn_version", "numpy_version", "random_state",
    "representation_config_json", "classifier_config_json", "score_definition",
    "zero_division_policy", "artifact_row_counts_json", "artifact_sha256_json",
    "tfidf_fit_scope", "full_data_tfidf_fit", "test_project_in_tfidf_fit",
    "test_labels_used_in_training", "resampling", "threshold_tuning", "hyperparameter_tuning",
    "v1_folds_used", "excluded_unresolved_samples_used", "leakage_violations",
    "r2_executed", "r3_executed", "h12_executed", "commit",
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


def lexical_tokens(source: str) -> str:
    """Historical R1 policy: omit layout/comments and replace each string token."""
    output = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in IGNORED_TOKEN_TYPES:
            continue
        output.append("<STR>" if token.type == tokenize.STRING else token.string)
    return " ".join(output)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, column: int) -> tuple[int, int, int, int]:
    truth = y_true[:, column]
    pred = y_pred[:, column]
    tp = int(np.logical_and(truth == 1, pred == 1).sum())
    fp = int(np.logical_and(truth == 0, pred == 1).sum())
    fn = int(np.logical_and(truth == 1, pred == 0).sum())
    tn = int(np.logical_and(truth == 0, pred == 0).sum())
    return tp, fp, fn, tn


def label_metric_row(y_true: np.ndarray, y_pred: np.ndarray, column: int, label: str, project: str | None = None) -> dict[str, Any]:
    tp, fp, fn, tn = confusion(y_true, y_pred, column)
    support = tp + fn
    precision_comp = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall_comp = 0.0 if support == 0 else tp / support
    f1_denominator = 2 * tp + fp + fn
    f1_comp = 0.0 if f1_denominator == 0 else 2 * tp / f1_denominator
    support_status = "NO_POSITIVE_SUPPORT" if support == 0 else "SUPPORTED"
    # Presentation follows the experiment protocol: undefined positive-class
    # precision is NA when no positive prediction exists; zero-support recall/F1
    # are NA. Computational sklearn zero_division=0 values remain recorded.
    precision_show = None if tp + fp == 0 else precision_comp
    recall_show = None if support == 0 else recall_comp
    f1_show = None if support == 0 or precision_show is None or f1_denominator == 0 else f1_comp
    row = {
        "label": label, "support": support, "negative_support": fp + tn,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision_computational": precision_comp,
        "precision_presentation": "NA" if precision_show is None else precision_show,
        "recall_computational": recall_comp,
        "recall_presentation": "NA" if recall_show is None else recall_show,
        "f1_computational": f1_comp,
        "f1_presentation": "NA" if f1_show is None else f1_show,
        "support_status": support_status,
    }
    if project is not None:
        row["held_out_project"] = project
    return row


def fmt(value: float | int) -> str:
    return format(float(value), ".12g")


def main() -> None:
    preflight_r1_or_exit(ROOT)
    # Verify every frozen input before any model fitting begins.
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen input SHA mismatch for {path}: {actual}")

    rows, fields = read_csv(DATASET)
    require(len(rows) == 14755 and len({r["sample_id"] for r in rows}) == 14755,
            "V2 dataset must contain exactly 14,755 unique samples")
    require(set(LABELS) <= set(fields) and "source_code" in fields and "project_name" in fields,
            "V2 dataset is missing source/project/label fields")
    require(all(r[label] in {"0", "1"} for r in rows for label in LABELS), "Non-binary or missing label")
    ids = [r["sample_id"] for r in rows]
    projects = sorted({r["project_name"] for r in rows})
    require(projects == list(PROJECTS), f"Unexpected project set/order: {projects}")

    assignment_rows, assignment_fields = read_csv(ASSIGNMENTS)
    fold_manifest_rows, fold_manifest_fields = read_csv(FOLD_MANIFEST)
    require(assignment_fields == ["canonical_index", "sample_id", "project_name", "fold_id", "held_out_project"],
            "Unexpected frozen assignment schema")
    require([r["sample_id"] for r in assignment_rows] == ids, "Frozen fold IDs/order differ from V2 dataset")
    require(len(assignment_rows) == 14755 and len({r["sample_id"] for r in assignment_rows}) == 14755,
            "Frozen assignment count/uniqueness failed")
    require(fold_manifest_fields[0:4] == ["fold_id", "held_out_project", "train_n", "test_n"]
            and len(fold_manifest_rows) == 8, "Frozen fold manifest schema/count failed")
    assignment_by_id = {r["sample_id"]: r for r in assignment_rows}
    fold_by_project = {r["held_out_project"]: r for r in fold_manifest_rows}
    require(set(fold_by_project) == set(PROJECTS), "Fold manifest project set mismatch")

    # Re-derive and verify only the exact partitions specified by frozen artifacts.
    for project in PROJECTS:
        manifest = fold_by_project[project]
        test_rows = [r for r in rows if assignment_by_id[r["sample_id"]]["held_out_project"] == project]
        train_rows = [r for r in rows if assignment_by_id[r["sample_id"]]["held_out_project"] != project]
        require(all(r["project_name"] == project for r in test_rows), f"Non-held-out sample assigned to TEST: {project}")
        require(all(r["project_name"] != project for r in train_rows), f"Held-out project leaked into TRAIN: {project}")
        require(len(test_rows) == int(manifest["test_n"]) and len(train_rows) == int(manifest["train_n"]),
                f"Frozen fold sizes disagree for {project}")
        require({r["sample_id"] for r in train_rows}.isdisjoint({r["sample_id"] for r in test_rows}),
                f"Train/test overlap for {project}")
        require({r["sample_id"] for r in train_rows} | {r["sample_id"] for r in test_rows} == set(ids),
                f"Train/test union does not cover V2 data for {project}")
        for label in LABELS:
            for split, split_rows in (("train", train_rows), ("test", test_rows)):
                pos = sum(row[label] == "1" for row in split_rows)
                neg = len(split_rows) - pos
                require(int(manifest[f"{split}_{label}_pos"]) == pos
                        and int(manifest[f"{split}_{label}_neg"]) == neg,
                        f"Frozen fold label support mismatch: {project} {split} {label}")

    # Source-only lexical preprocessing is stateless; the vectorizer is fit anew
    # inside each fold on TRAIN only. Held-out labels do not enter .fit().
    token_texts = [lexical_tokens(row["source_code"]) for row in rows]
    pred_by_id: dict[str, dict[str, Any]] = {}
    vocabulary_audit: list[dict[str, Any]] = []
    per_project_metrics: list[dict[str, Any]] = []
    per_project_label_metrics: list[dict[str, Any]] = []
    for fold_number, held in enumerate(PROJECTS, start=1):
        fold_id = f"LOPO_{fold_number:02d}"
        train_indices = [i for i, row in enumerate(rows) if assignment_by_id[row["sample_id"]]["held_out_project"] != held]
        test_indices = [i for i, row in enumerate(rows) if assignment_by_id[row["sample_id"]]["held_out_project"] == held]
        train = [rows[i] for i in train_indices]
        test = [rows[i] for i in test_indices]
        require(len(train) == int(fold_by_project[held]["train_n"])
                and len(test) == int(fold_by_project[held]["test_n"]), f"Unexpected fold rows in {fold_id}")
        require(not ({r["project_name"] for r in train} & {r["project_name"] for r in test}),
                f"Project leakage in {fold_id}")
        require({r["project_name"] for r in test} == {held}, f"Unexpected TEST project in {fold_id}")

        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), tokenizer=str.split,
            preprocessor=None, token_pattern=None, lowercase=False,
            min_df=1, max_df=1.0, max_features=None, norm="l2",
            use_idf=True, smooth_idf=True, sublinear_tf=False,
            strip_accents=None, binary=False, dtype=np.float64,
        )
        train_text = [token_texts[i] for i in train_indices]
        test_text = [token_texts[i] for i in test_indices]
        train_x = vectorizer.fit_transform(train_text)
        test_x = vectorizer.transform(test_text)
        require(train_x.shape == (len(train), len(vectorizer.vocabulary_)),
                f"TRAIN TF-IDF shape mismatch in {fold_id}")
        require(test_x.shape == (len(test), len(vectorizer.vocabulary_)),
                f"TEST TF-IDF shape mismatch in {fold_id}")
        require(held not in {row["project_name"] for row in train},
                f"Held-out project influenced vectorizer fit in {fold_id}")

        y_train = np.asarray([[int(row[label]) for label in LABELS] for row in train], dtype=np.int8)
        y_test = np.asarray([[int(row[label]) for label in LABELS] for row in test], dtype=np.int8)
        y_pred = np.zeros_like(y_test)
        y_score = np.zeros(y_test.shape, dtype=np.float64)
        for column, label in enumerate(LABELS):
            model = LogisticRegression(**CLASSIFIER_CONFIG)
            model.fit(train_x, y_train[:, column])
            require(np.array_equal(model.classes_, np.asarray([0, 1])),
                    f"Binary model classes incomplete in {fold_id} for {label}")
            y_pred[:, column] = model.predict(test_x)
            y_score[:, column] = model.predict_proba(test_x)[:, 1]

        for local_index, dataset_index in enumerate(test_indices):
            source = rows[dataset_index]
            sid = source["sample_id"]
            require(sid not in pred_by_id, f"More than one OOP prediction for {sid}")
            record: dict[str, Any] = {
                "sample_id": sid, "project_name": source["project_name"], "fold": fold_id,
            }
            for j, label in enumerate(LABELS):
                score = float(y_score[local_index, j])
                require(np.isfinite(score) and 0.0 <= score <= 1.0,
                        f"Invalid probability score for {sid} {label}: {score}")
                record[f"y_true_{label}"] = int(y_test[local_index, j])
                record[f"y_pred_{label}"] = int(y_pred[local_index, j])
                record[f"y_score_{label}"] = format(score, ".17g")
            pred_by_id[sid] = record

        fold_audit = {
            "fold": fold_id, "held_out_project": held, "train_rows": len(train),
            "test_rows": len(test), "vocabulary_size": len(vectorizer.vocabulary_),
            "train_matrix_rows": train_x.shape[0], "train_matrix_cols": train_x.shape[1],
            "test_matrix_rows": test_x.shape[0], "test_matrix_cols": test_x.shape[1],
            "train_nnz": int(train_x.nnz), "test_nnz": int(test_x.nnz),
            "fit_scope": "TRAIN_ONLY", "test_project_in_fit": "NO", "full_data_fit": "NO",
        }
        vocabulary_audit.append(fold_audit)

        project_pred = y_pred
        project_bundle = metric_values(y_test, project_pred)
        per_project_metrics.append({
            "held_out_project": held, "test_n": len(test),
            "hamming_loss": project_bundle["hamming_loss"],
            "exact_match_ratio": project_bundle["exact_match_ratio"],
            "micro_precision": project_bundle["micro_precision"],
            "micro_recall": project_bundle["micro_recall"],
            "micro_f1": project_bundle["micro_f1"],
            "macro_precision_computational": project_bundle["macro_precision"],
            "macro_recall_computational": project_bundle["macro_recall"],
            "macro_f1_computational": project_bundle["macro_f1"],
        })
        for column, label in enumerate(LABELS):
            per_project_label_metrics.append(label_metric_row(y_test, y_pred, column, label, project=held))
        print(f"Completed {fold_id} ({held}): train={len(train)} test={len(test)} vocab={len(vectorizer.vocabulary_)}")

    require(len(pred_by_id) == 14755 and set(pred_by_id) == set(ids),
            "Expected exactly one OOP prediction per V2 sample")
    predictions = [pred_by_id[sid] for sid in ids]
    require([r["sample_id"] for r in predictions] == ids, "Pooled prediction order is not canonical experimental order")
    y_true = np.asarray([[int(row[f"y_true_{label}"]) for label in LABELS] for row in predictions], dtype=np.int8)
    y_pred = np.asarray([[int(row[f"y_pred_{label}"]) for label in LABELS] for row in predictions], dtype=np.int8)
    pooled = metric_values(y_true, y_pred)
    per_label = [label_metric_row(y_true, y_pred, j, label) for j, label in enumerate(LABELS)]
    pooled_metrics = [
        {"metric": "hamming_loss", "value": pooled["hamming_loss"], "aggregation": "pooled OOP", "zero_division_policy": "not applicable"},
        {"metric": "exact_match_ratio", "value": pooled["exact_match_ratio"], "aggregation": "pooled OOP", "zero_division_policy": "not applicable"},
    ]
    for average in ("micro", "macro"):
        for metric in ("precision", "recall", "f1"):
            pooled_metrics.append({
                "metric": f"{average}_{metric}", "value": pooled[f"{average}_{metric}"],
                "aggregation": "pooled OOP", "zero_division_policy": "zero_division=0 for sklearn-compatible computational aggregation",
            })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_paths = {key: OUTPUT / filename for key, filename in OUTPUT_FILES.items()}
    require(not any(path.exists() for path in output_paths.values()),
            "Refusing to overwrite existing R1 V2 run artifacts")
    write_csv(output_paths["predictions"], predictions, PREDICTION_FIELDS)
    write_csv(output_paths["pooled"], pooled_metrics, POOLED_FIELDS)
    write_csv(output_paths["per_label"], per_label, PER_LABEL_FIELDS)
    write_csv(output_paths["per_project"], per_project_metrics, list(per_project_metrics[0]))
    write_csv(output_paths["per_project_label"], per_project_label_metrics, PER_PROJECT_LABEL_FIELDS)
    write_csv(output_paths["vocabulary"], vocabulary_audit, VOCAB_FIELDS)

    # Physical reopen of all model output tables before computing artifact digests.
    reopened: dict[str, list[dict[str, str]]] = {}
    schemas = {
        "predictions": PREDICTION_FIELDS, "pooled": POOLED_FIELDS,
        "per_label": PER_LABEL_FIELDS, "per_project": list(per_project_metrics[0]),
        "per_project_label": PER_PROJECT_LABEL_FIELDS, "vocabulary": VOCAB_FIELDS,
    }
    expected_counts = {
        "predictions": 14755, "pooled": len(pooled_metrics), "per_label": 4,
        "per_project": 8, "per_project_label": 32, "vocabulary": 8,
    }
    for key in schemas:
        reopened[key], actual_fields = read_csv(output_paths[key])
        require(actual_fields == schemas[key], f"Physical schema mismatch for {key}")
        require(len(reopened[key]) == expected_counts[key], f"Physical row count mismatch for {key}")
    reopened_predictions = reopened["predictions"]
    require(len({r["sample_id"] for r in reopened_predictions}) == 14755
            and [r["sample_id"] for r in reopened_predictions] == ids,
            "Physical predictions do not have unique canonical-order membership")
    for row in reopened_predictions:
        require(row["fold"] == fold_by_project_for_row(row["project_name"], PROJECTS),
                f"Prediction assigned to incorrect fold: {row['sample_id']}")
        for label in LABELS:
            require(row[f"y_true_{label}"] in {"0", "1"} and row[f"y_pred_{label}"] in {"0", "1"},
                    f"Invalid physical prediction value: {row['sample_id']} {label}")
            score = float(row[f"y_score_{label}"])
            require(np.isfinite(score) and 0.0 <= score <= 1.0,
                    f"Invalid physical probability score: {row['sample_id']} {label}")
    require(all(r["fit_scope"] == "TRAIN_ONLY" and r["test_project_in_fit"] == "NO"
                and r["full_data_fit"] == "NO" for r in reopened["vocabulary"]),
            "Physical leakage vocabulary audit failed")
    for key, count in expected_counts.items():
        if key in {"predictions", "per_label", "per_project", "per_project_label", "vocabulary"}:
            require(len(reopened[key]) == count, f"Post-write row count mismatch for {key}")

    # Canonical target-label checks from exact-ID joined source rows; predictions
    # preserve these truths, and no held-out labels entered fitting.
    source_by_id = {row["sample_id"]: row for row in rows}
    for pred in reopened_predictions:
        truth_source = source_by_id[pred["sample_id"]]
        require(all(int(pred[f"y_true_{label}"]) == int(truth_source[label]) for label in LABELS),
                f"Pooled prediction truth mismatch: {pred['sample_id']}")

    score_definition = "LogisticRegression.predict_proba(X_test)[:, 1]"
    representation_config = {
        **VECTORIZER_CONFIG,
        "fit_scope": "separately on seven TRAIN projects in each LOPO fold",
        "string_literal_normalization": "<STR>",
        "source_field": "source_code",
    }
    confusion_policy = "presentation NA for undefined positive-class precision and zero-positive-support recall/F1; macro computational values use sklearn zero_division=0"
    preliminary_hashes = {key: sha256(path) for key, path in output_paths.items() if key not in {"summary", "manifest"}}
    preliminary_rows = {key: expected_counts[key] for key in preliminary_hashes}
    report_lines = [
        "# V2 — R1 TF-IDF LOPO Baseline", "",
        "## Run scope", "",
        "Only the frozen R1 lexical TF-IDF baseline was run. No R2, R3, H12, or H12-B representations were generated or evaluated. No model comparison or statistical test was performed.",
        f"- V2 dataset rows: {len(rows)}; unique samples: {len(set(ids))}.",
        f"- Pooled out-of-project predictions: {len(reopened_predictions)}.",
        "- Frozen V2 LOPO assignments were used without reconstruction; output predictions preserve dataset relative order.",
        "- Lexical normalization: tokenize Python source; omit layout/comments; replace each string token with `<STR>`; keep identifiers, keywords, operators, punctuation, and numeric literals.",
        "- TF-IDF was fit separately on TRAIN only in each fold. The held-out project did not contribute vocabulary or IDF.",
        "- Binary Relevance: four independent Logistic Regression models; C=1.0, L2, liblinear, class_weight=None, max_iter=2000, random_state=42. Predictions use the default decision threshold; saved scores are positive-class probabilities.", "",
        "## Fold vocabulary audit", "",
        "| Fold | Held-out project | Train rows | Test rows | Vocabulary | Train matrix | Test matrix | Fit scope |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for fold in vocabulary_audit:
        report_lines.append(
            f"| {fold['fold']} | {fold['held_out_project']} | {fold['train_rows']} | {fold['test_rows']} | {fold['vocabulary_size']} | "
            f"{fold['train_matrix_rows']}×{fold['train_matrix_cols']} | {fold['test_matrix_rows']}×{fold['test_matrix_cols']} | {fold['fit_scope']} |"
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
    for metric in per_label:
        report_lines.append(
            f"| {metric['label']} | {metric['support']} | {metric['precision_presentation']} | "
            f"{metric['recall_presentation']} | {metric['f1_presentation']} | {metric['TP']} | {metric['FP']} | {metric['FN']} | {metric['TN']} | {metric['support_status']} |"
        )
    report_lines += [
        "", "Undefined metric policy: presentation precision is NA when there are no positive predictions. Presentation recall/F1 are NA when true positive support is zero; computational precision/recall/F1 are separately recorded, using zero_division=0 for sklearn-compatible macro aggregation.",
        "", "## Per-project metrics", "",
        "| Held-out project | N | Hamming Loss | Exact Match | Micro F1 | Macro F1 (computational, zero_division=0) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in per_project_metrics:
        report_lines.append(
            f"| {metric['held_out_project']} | {metric['test_n']} | {metric['hamming_loss']:.8f} | "
            f"{metric['exact_match_ratio']:.8f} | {metric['micro_f1']:.8f} | {metric['macro_f1_computational']:.8f} |"
        )
    report_lines += ["", "Per-project per-label precision, recall, F1, support, and confusion counts are in `r1_v2_per_project_label_metrics.csv`.", "", "## Zero-support note", "", "pyftpdlib:MNT has support=0 and status=NO_POSITIVE_SUPPORT. Recall and F1 are presented as NA and must not be interpreted as measured performance. The fold is retained unchanged.", "", "## Leakage audit", "", "- TF-IDF fit train-only: PASS; eight separate training-fold vectorizers.", "- Full-data TF-IDF fit: NO.", "- Held-out project included in its TF-IDF fit: NO.", "- Test labels used during training: NO.", "- Resampling / class balancing: NO / NO.", "- Threshold tuning / hyperparameter tuning on test: NO / NO.", "- V1 folds used: NO.", "- Excluded unresolved ET samples used: NO.", "- Leakage violations: 0.", "- R2 / R3 / H12 / H12-B executed: NO / NO / NO / NO.", "", "## Artifact hashes", "", "| Artifact | Rows | SHA-256 |", "|---|---:|---|"]
    for key, digest in preliminary_hashes.items():
        report_lines.append(f"| {output_paths[key].name} | {preliminary_rows[key]} | `{digest}` |")
    with output_paths["summary"].open("x", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(report_lines) + "\n")

    artifact_hashes = {key: sha256(path) for key, path in output_paths.items() if key != "manifest"}
    artifact_rows = {key: (len(read_csv(path)[0]) if path.suffix == ".csv" else len(report_lines))
                     for key, path in output_paths.items() if key != "manifest"}
    run_manifest = [{
        "dataset_sha256": EXPECTED_HASHES[DATASET],
        "assignment_sha256": EXPECTED_HASHES[ASSIGNMENTS],
        "fold_manifest_sha256": EXPECTED_HASHES[FOLD_MANIFEST],
        "protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "dataset_rows": len(rows), "prediction_rows": len(reopened_predictions), "fold_count": len(PROJECTS),
        "labels": ",".join(LABELS), "label_order": ",".join(LABELS),
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__, "random_state": CLASSIFIER_CONFIG["random_state"],
        "representation_config_json": json.dumps(representation_config, sort_keys=True, separators=(",", ":")),
        "classifier_config_json": json.dumps(CLASSIFIER_CONFIG, sort_keys=True, separators=(",", ":")),
        "score_definition": score_definition, "zero_division_policy": confusion_policy,
        "artifact_row_counts_json": json.dumps(artifact_rows, sort_keys=True, separators=(",", ":")),
        "artifact_sha256_json": json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")),
        "tfidf_fit_scope": "TRAIN_ONLY_PER_FOLD", "full_data_tfidf_fit": "NO",
        "test_project_in_tfidf_fit": "NO", "test_labels_used_in_training": "NO",
        "resampling": "NO", "threshold_tuning": "NO", "hyperparameter_tuning": "NO",
        "v1_folds_used": "NO", "excluded_unresolved_samples_used": "NO",
        "leakage_violations": 0, "r2_executed": "NO", "r3_executed": "NO",
        "h12_executed": "NO", "commit": "NO",
    }]
    write_csv(output_paths["manifest"], run_manifest, MANIFEST_FIELDS)
    manifest_rows_reopened, manifest_fields = read_csv(output_paths["manifest"])
    require(manifest_fields == MANIFEST_FIELDS and len(manifest_rows_reopened) == 1,
            "Run manifest physical schema/count failed")
    manifest_record = manifest_rows_reopened[0]
    require(json.loads(manifest_record["artifact_sha256_json"]) == artifact_hashes,
            "Run manifest artifact hashes do not match physical output files")
    require(json.loads(manifest_record["artifact_row_counts_json"]) == artifact_rows,
            "Run manifest artifact row counts do not match physical files")

    # Recompute physical input hashes after training and output creation.
    post_hashes = {path: sha256(path) for path in EXPECTED_HASHES}
    require(post_hashes == EXPECTED_HASHES, "A frozen input changed during R1 run")
    require(sha256(output_paths["predictions"]) == artifact_hashes["predictions"],
            "Prediction artifact physical hash changed after manifest creation")

    print(json.dumps({
        "final": "PASS", "dataset_rows": len(rows), "predictions": len(predictions),
        "pooled_metrics": pooled,
        "per_label": per_label,
        "per_project": per_project_metrics,
        "fold_vocabulary": vocabulary_audit,
        "output_hashes": {key: sha256(path) for key, path in output_paths.items()},
        "frozen_inputs_unchanged": True,
        "leakage_violations": 0, "r2_r3_h12_h12b": "NOT_RUN", "commit": False,
    }, indent=2, ensure_ascii=False))


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def fold_by_project_for_row(project: str, projects: tuple[str, ...]) -> str:
    return f"LOPO_{projects.index(project) + 1:02d}"


if __name__ == "__main__":
    main()
