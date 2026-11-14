#!/usr/bin/env python3
"""Run the primary frozen H12 = [R1 TF-IDF | L2(log1p(R2 AST))] V2 LOPO baseline."""
from __future__ import annotations

import ast
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
import scipy
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "dataset/final"
R1_DIR = ROOT / "experiments/v2/r1_tfidf"
R2_DIR = ROOT / "experiments/v2/r2_ast"
R3_DIR = ROOT / "experiments/v2/r3_unixcoder"
OUTPUT = ROOT / "experiments/v2/h12"
DATASET = FINAL / "python_4label_test_smell_experimental_v2.csv"
ASSIGNMENTS = FINAL / "lopo_fold_assignments_v2.csv"
FOLD_MANIFEST = FINAL / "lopo_fold_manifest_v2.csv"
PROTOCOL = FINAL / "multi_label_experiment_protocol_v2.md"
CANONICAL = FINAL / "python_multilabel_test_smell_dataset_v1.csv"
R1_POOLED = R1_DIR / "r1_v2_pooled_metrics.csv"
R1_LABEL = R1_DIR / "r1_v2_per_label_metrics.csv"
R1_MANIFEST = R1_DIR / "r1_v2_run_manifest.csv"
R2_POOLED = R2_DIR / "r2_v2_pooled_metrics.csv"
R2_LABEL = R2_DIR / "r2_v2_per_label_metrics.csv"
R2_MANIFEST = R2_DIR / "r2_v2_run_manifest.csv"
R3_POOLED = R3_DIR / "r3_v2_pooled_metrics.csv"
R3_LABEL = R3_DIR / "r3_v2_per_label_metrics.csv"
R3_MANIFEST = R3_DIR / "r3_v2_run_manifest.csv"
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
R1_DIMENSIONS = {
    "chainer": 124836, "clusterfuzz": 124734, "grimoirelab-perceval": 131434,
    "numba": 74387, "pika": 138351, "psutil": 131098,
    "pyftpdlib": 137246, "python-driver": 120144,
}
R2_DIMENSIONS = {
    "chainer": 685, "clusterfuzz": 688, "grimoirelab-perceval": 690,
    "numba": 592, "pika": 691, "psutil": 679, "pyftpdlib": 689,
    "python-driver": 678,
}
IGNORED_TOKEN_TYPES = {
    tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL,
    tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT,
}
CLASSIFIER_CONFIG = {
    "C": 1.0, "penalty": "l2", "solver": "liblinear", "class_weight": None,
    "max_iter": 2000, "random_state": 42,
}
R1_CONFIG = {
    "analyzer": "word", "ngram_range": (1, 2), "tokenizer": "str.split",
    "preprocessor": None, "token_pattern": None, "lowercase": False,
    "min_df": 1, "max_df": 1.0, "max_features": None, "norm": "l2",
    "use_idf": True, "smooth_idf": True, "sublinear_tf": False,
    "strip_accents": None, "binary": False, "dtype": "float64",
    "string_literals": "<STR>", "source_field": "source_code",
}
H12_CONFIG = {
    "definition": "sparse_hstack(R1_TFIDF, row_l2_normalize(log1p(R2_AST)))",
    "r1": R1_CONFIG,
    "r2": "FULL_FILE_TARGET_METHOD_AST generic node/edge counts plus node count/depth; DictVectorizer(sparse=True, sort=True)",
    "r2_transform_order": ["elementwise log1p", "row-wise L2 normalization"],
    "whole_h12_normalization": "NO",
    "additional_block_weighting": "NO",
    "r2_standalone_scaling": "NO",
}
OUTPUT_FILES = {
    "predictions": "h12_v2_oop_predictions.csv",
    "blind_predictions": "h12_v2_oop_predictions_blind.csv",
    "pooled": "h12_v2_pooled_metrics.csv",
    "per_label": "h12_v2_per_label_metrics.csv",
    "per_project": "h12_v2_per_project_metrics.csv",
    "per_project_label": "h12_v2_per_project_label_metrics.csv",
    "feature_audit": "h12_v2_fold_feature_audit.csv",
    "summary": "h12_v2_summary.md",
    "manifest": "h12_v2_run_manifest.csv",
}
PRED_FIELDS = ["sample_id", "project_name", "fold"] + [
    field for label in LABELS for field in (f"y_true_{label}", f"y_pred_{label}", f"y_score_{label}")
]
BLIND_FIELDS = ["sample_id", "project_name", "fold"] + [
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
FEATURE_AUDIT_FIELDS = [
    "fold", "held_out_project", "train_rows", "test_rows", "R1_dimension", "R2_dimension",
    "R1_expected_dimension", "R2_expected_dimension", "H12_dimension", "train_matrix_rows", "train_matrix_cols", "test_matrix_rows", "test_matrix_cols",
    "train_nnz", "test_nnz", "R1_fit_scope", "R2_fit_scope", "R2_log1p", "R2_row_L2",
    "additional_block_weighting", "whole_H12_normalization", "AST_source_files",
    "AST_parse_failures", "AST_target_resolution_failures", "feature_extraction_failures",
    "test_project_in_R1_fit", "test_project_in_R2_fit",
]
MANIFEST_FIELDS = [
    "dataset_sha256", "assignment_sha256", "fold_manifest_sha256", "protocol_sha256",
    "canonical_identity_metadata_sha256", "R1_pooled_sha256", "R1_per_label_sha256",
    "R2_pooled_sha256", "R2_per_label_sha256", "R3_pooled_sha256", "R3_per_label_sha256",
    "dataset_rows", "prediction_rows", "fold_count", "labels", "label_order",
    "python_version", "scikit_learn_version", "numpy_version", "scipy_version", "random_state",
    "R1_config_json", "R2_config_json", "H12_config_json", "classifier_config_json",
    "score_definition", "zero_division_policy", "AST_parse_failure_count",
    "AST_target_resolution_failure_count", "feature_extraction_failure_count",
    "dimension_reproduction", "artifact_row_counts_json", "artifact_sha256_json",
    "R1_fit_scope", "R2_fit_scope", "full_data_fitting", "test_labels_used_in_training",
    "resampling", "threshold_tuning", "hyperparameter_tuning", "V1_folds_used",
    "excluded_unresolved_samples_used", "R3_used_in_H12", "H12B_executed",
    "statistical_testing", "leakage_violations", "commit",
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
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in IGNORED_TOKEN_TYPES:
            continue
        tokens.append("<STR>" if token.type == tokenize.STRING else token.string)
    return " ".join(tokens)


def ast_maximum_depth(node: ast.AST) -> int:
    return 1 + max((ast_maximum_depth(child) for child in ast.iter_child_nodes(node)), default=0)


def ast_features(node: ast.AST) -> dict[str, float]:
    features: Counter[str] = Counter()
    for parent in ast.walk(node):
        features[f"node::{type(parent).__name__}"] += 1
        for child in ast.iter_child_nodes(parent):
            features[f"edge::{type(parent).__name__}->{type(child).__name__}"] += 1
    features["metric::ast_node_count"] = sum(1 for _ in ast.walk(node))
    features["metric::ast_max_depth"] = ast_maximum_depth(node)
    return dict(features)


def sklearn_compatible_csr(matrix):
    """Downcast only sparse indices to meet scikit-learn 1.9 CSR requirements."""
    matrix = matrix.tocsr(copy=False)
    maximum = np.iinfo(np.int32).max
    require(matrix.shape[0] <= maximum and matrix.shape[1] <= maximum and matrix.nnz <= maximum,
            "CSR matrix exceeds int32 index capacity")
    if matrix.indices.dtype != np.int32:
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
    if matrix.indptr.dtype != np.int32:
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def normalize_r2_block(matrix):
    require(matrix.nnz == 0 or np.all(matrix.data >= 0), "R2 AST matrix contains a negative structural value")
    matrix = matrix.tocsr(copy=True)
    matrix.data = np.log1p(matrix.data)
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    return sklearn_compatible_csr(matrix)


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


def verify_reference_run(manifest_path: Path, expected_metrics: dict[str, Path]) -> dict[str, str]:
    manifest, _ = read_csv(manifest_path)
    require(len(manifest) == 1, f"Reference run manifest invalid: {manifest_path}")
    recorded = json.loads(manifest[0]["artifact_sha256_json"])
    hashes = {key: sha256(path) for key, path in expected_metrics.items()}
    for key, actual in hashes.items():
        require(recorded.get(key) == actual, f"Reference metric artifact hash mismatch: {path if (path := expected_metrics[key]) else key}")
    return hashes


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing H12 outputs: {OUTPUT}")
    # Required frozen dataset/fold/protocol hashes are checked before feature work.
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen input SHA mismatch: {path}: {actual}")
    rows, fields = read_csv(DATASET)
    canonical, canonical_fields = read_csv(CANONICAL)
    require(len(rows) == 14755 and len({r["sample_id"] for r in rows}) == 14755,
            "V2 dataset count/unique-ID integrity failed")
    require({"sample_id", "project_name", "file_path", "class_name", "method_name", "definition_line", "source_code", *LABELS} <= set(fields),
            "V2 dataset lacks H12 required columns")
    require(all(row[label] in {"0", "1"} for row in rows for label in LABELS), "Non-binary V2 labels")
    ids = [row["sample_id"] for row in rows]
    require(len(canonical) == 14765 and len({r["sample_id"] for r in canonical}) == 14765,
            "Canonical identity metadata integrity failed")
    require({"sample_id", "start_line", "end_line", "repository_path", "file_path", "class_name", "method_name", "definition_line"} <= set(canonical_fields),
            "Canonical metadata lacks FULL_FILE_TARGET_METHOD_AST resolver fields")
    canonical_by_id = {row["sample_id"]: row for row in canonical}
    for row in rows:
        meta = canonical_by_id.get(row["sample_id"])
        require(meta is not None, f"V2 ID missing in canonical identity metadata: {row['sample_id']}")
        for field in ("project_name", "file_path", "class_name", "method_name", "definition_line", "source_code"):
            require(row[field] == meta[field], f"Canonical/V2 identity or source mismatch: {row['sample_id']} {field}")
    projects = sorted({row["project_name"] for row in rows})
    require(projects == list(PROJECTS), f"Unexpected V2 project list: {projects}")

    assignment_rows, assignment_fields = read_csv(ASSIGNMENTS)
    fold_manifest_rows, _ = read_csv(FOLD_MANIFEST)
    require(assignment_fields == ["canonical_index", "sample_id", "project_name", "fold_id", "held_out_project"],
            "Frozen assignment schema mismatch")
    require([r["sample_id"] for r in assignment_rows] == ids and len({r["sample_id"] for r in assignment_rows}) == 14755,
            "Frozen assignment membership/order mismatch")
    assignment_by_id = {row["sample_id"]: row for row in assignment_rows}
    fold_by_project = {row["held_out_project"]: row for row in fold_manifest_rows}
    require(len(fold_manifest_rows) == 8 and set(fold_by_project) == set(PROJECTS), "Frozen LOPO manifest invalid")

    # Stateless lexical normalization for every sample; no fitted statistic is
    # derived here. The TF-IDF vectorizer is fit independently within each fold.
    lexical = [lexical_tokens(row["source_code"]) for row in rows]

    # Resolve each V2 method against its full checked-out source file using the
    # exact V1 resolver (project/file/class/method/definition/start/end identity).
    trees: dict[Path, ast.Module] = {}
    source_file_counts: Counter[tuple[str, str]] = Counter()
    parse_failures: list[dict[str, str]] = []
    resolution_failures: list[str] = []
    nodes: list[ast.AST | None] = [None] * len(rows)
    for i, row in enumerate(rows):
        meta = canonical_by_id[row["sample_id"]]
        path = ROOT / "repositories" / row["project_name"] / row["file_path"]
        try:
            if path not in trees:
                raw = path.read_bytes()
                trees[path] = ast.parse(raw.decode("utf-8"), filename=str(path))
            tree = trees[path]
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
            resolution_failures.append(row["sample_id"])
            continue
        nodes[i] = candidates[0]
        source_file_counts[(row["project_name"], row["file_path"])] += 1
    if parse_failures or resolution_failures:
        raise ValueError(
            f"H12 AST resolution STOP before fitting: parse failures={len(parse_failures)} "
            f"sample/file details={parse_failures[:25]}, target-resolution failures={len(resolution_failures)} "
            f"sample IDs={resolution_failures[:100]}"
        )

    ast_dicts: list[dict[str, float]] = []
    feature_failures: list[str] = []
    for row, node in zip(rows, nodes, strict=True):
        try:
            if node is None:
                raise ValueError("Target AST node missing")
            ast_dicts.append(ast_features(node))
        except Exception:
            feature_failures.append(row["sample_id"])
            ast_dicts.append({})
    if feature_failures:
        raise ValueError(f"H12 AST feature extraction failures={len(feature_failures)} IDs={feature_failures[:100]}")

    fold_by_id = {row["sample_id"]: row["held_out_project"] for row in assignment_rows}

    # Dimension reproduction is checked across every fold BEFORE fitting any LR.
    # These fits are feature-vocabulary checks only; no labels are accessed.
    dimension_audit: dict[str, tuple[int, int]] = {}
    for held in PROJECTS:
        train_indices = [i for i, sid in enumerate(ids) if fold_by_id[sid] != held]
        r1_probe = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), tokenizer=str.split, preprocessor=None,
            token_pattern=None, lowercase=False, min_df=1, max_df=1.0, max_features=None,
            norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False,
            dtype=np.float64, strip_accents=None, binary=False,
        )
        r1_probe.fit([lexical[i] for i in train_indices])
        r2_probe = DictVectorizer(sparse=True, sort=True)
        r2_probe.fit([ast_dicts[i] for i in train_indices])
        d1, d2 = len(r1_probe.vocabulary_), len(r2_probe.feature_names_)
        require(d1 == R1_DIMENSIONS[held], f"R1 vocabulary reproduction FAIL for {held}: expected {R1_DIMENSIONS[held]} got {d1}")
        require(d2 == R2_DIMENSIONS[held], f"R2 dimension reproduction FAIL for {held}: expected {R2_DIMENSIONS[held]} got {d2}")
        dimension_audit[held] = (d1, d2)
        del r1_probe, r2_probe
    require(len(dimension_audit) == 8, "Feature dimension preflight did not cover all folds")

    # Fit each block's vocabulary on TRAIN only, apply the frozen H12 structural
    # sample-wise transformation, sparse-hstack blocks, and fit four LRs.
    blind_by_id: dict[str, dict[str, Any]] = {}
    feature_audit: list[dict[str, Any]] = []
    for fold_index, held in enumerate(PROJECTS, start=1):
        fold = f"LOPO_{fold_index:02d}"
        train_indices = [i for i, sid in enumerate(ids) if fold_by_id[sid] != held]
        test_indices = [i for i, sid in enumerate(ids) if fold_by_id[sid] == held]
        train_rows = [rows[i] for i in train_indices]
        test_rows = [rows[i] for i in test_indices]
        require({r["project_name"] for r in test_rows} == {held}
                and held not in {r["project_name"] for r in train_rows}, f"LOPO leakage in {fold}")
        require(not ({r["sample_id"] for r in train_rows} & {r["sample_id"] for r in test_rows}), f"Train/test overlap in {fold}")

        r1_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), tokenizer=str.split, preprocessor=None,
            token_pattern=None, lowercase=False, min_df=1, max_df=1.0, max_features=None,
            norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False,
            dtype=np.float64, strip_accents=None, binary=False,
        )
        r1_train = r1_vectorizer.fit_transform([lexical[i] for i in train_indices])
        r1_test = r1_vectorizer.transform([lexical[i] for i in test_indices])
        r2_vectorizer = DictVectorizer(sparse=True, sort=True)
        r2_train_raw = r2_vectorizer.fit_transform([ast_dicts[i] for i in train_indices])
        r2_test_raw = r2_vectorizer.transform([ast_dicts[i] for i in test_indices])
        d1, d2 = r1_train.shape[1], r2_train_raw.shape[1]
        require((d1, d2) == dimension_audit[held], f"Training dimensions differ from preflight in {fold}")
        r2_train = normalize_r2_block(r2_train_raw)
        r2_test = normalize_r2_block(r2_test_raw)
        x_train = sklearn_compatible_csr(hstack((r1_train, r2_train), format="csr"))
        x_test = sklearn_compatible_csr(hstack((r1_test, r2_test), format="csr"))
        d_h12 = d1 + d2
        require(x_train.shape == (len(train_rows), d_h12) and x_test.shape == (len(test_rows), d_h12),
                f"H12 matrix dimensions invalid in {fold}")
        require(d_h12 == d1 + d2, f"H12 concatenated dimension mismatch in {fold}")
        # Confirm structural normalization is row-wise unit L2 for nonempty rows.
        train_norms = np.sqrt(np.asarray(r2_train.multiply(r2_train).sum(axis=1)).ravel())
        test_norms = np.sqrt(np.asarray(r2_test.multiply(r2_test).sum(axis=1)).ravel())
        require(np.allclose(train_norms, 1.0, rtol=1e-7, atol=1e-9)
                and np.allclose(test_norms, 1.0, rtol=1e-7, atol=1e-9),
                f"R2 row-L2 normalization failed in {fold}")

        y_train = np.asarray([[int(row[label]) for label in LABELS] for row in train_rows], dtype=np.int8)
        y_pred = np.zeros((len(test_rows), len(LABELS)), dtype=np.int8)
        y_score = np.zeros((len(test_rows), len(LABELS)), dtype=np.float64)
        for column, label in enumerate(LABELS):
            model = LogisticRegression(**CLASSIFIER_CONFIG)
            model.fit(x_train, y_train[:, column])
            require(np.array_equal(model.classes_, np.asarray([0, 1])), f"Binary classes incomplete: {fold} {label}")
            y_pred[:, column] = model.predict(x_test)
            y_score[:, column] = model.predict_proba(x_test)[:, 1]

        for local_i, data_i in enumerate(test_indices):
            row = rows[data_i]
            sid = row["sample_id"]
            require(sid not in blind_by_id, f"Duplicate H12 OOP prediction: {sid}")
            item = {"sample_id": sid, "project_name": row["project_name"], "fold": fold}
            for column, label in enumerate(LABELS):
                score = float(y_score[local_i, column])
                require(np.isfinite(score) and 0 <= score <= 1, f"Invalid H12 score {sid} {label}")
                item[f"y_pred_{label}"] = int(y_pred[local_i, column])
                item[f"y_score_{label}"] = format(score, ".17g")
            blind_by_id[sid] = item

        feature_audit.append({
            "fold": fold, "held_out_project": held, "train_rows": len(train_rows), "test_rows": len(test_rows),
            "R1_dimension": d1, "R2_dimension": d2,
            "R1_expected_dimension": R1_DIMENSIONS[held], "R2_expected_dimension": R2_DIMENSIONS[held],
            "H12_dimension": d_h12,
            "train_matrix_rows": x_train.shape[0], "train_matrix_cols": x_train.shape[1],
            "test_matrix_rows": x_test.shape[0], "test_matrix_cols": x_test.shape[1],
            "train_nnz": int(x_train.nnz), "test_nnz": int(x_test.nnz),
            "R1_fit_scope": "TRAIN_ONLY", "R2_fit_scope": "TRAIN_ONLY",
            "R2_log1p": "YES", "R2_row_L2": "YES", "additional_block_weighting": "NO",
            "whole_H12_normalization": "NO", "AST_source_files": len(source_file_counts),
            "AST_parse_failures": 0, "AST_target_resolution_failures": 0,
            "feature_extraction_failures": 0, "test_project_in_R1_fit": "NO",
            "test_project_in_R2_fit": "NO",
        })
        print(f"Completed {fold} ({held}): train={len(train_rows)} test={len(test_rows)} dims=({d1},{d2},{d_h12})")

    require(len(blind_by_id) == 14755 and set(blind_by_id) == set(ids),
            "Every H12 sample must receive exactly one OOP prediction")
    blind_rows = [blind_by_id[sid] for sid in ids]
    blind_fields = ["sample_id", "project_name", "fold"] + [
        field for label in LABELS for field in (f"y_pred_{label}", f"y_score_{label}")
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_paths = {key: OUTPUT / filename for key, filename in OUTPUT_FILES.items()}
    require(not any(path.exists() for path in output_paths.values()), "Refusing to overwrite existing H12 outputs")
    def write_csv(path: Path, data_rows: list[dict[str, Any]], fields: list[str]) -> None:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
            writer.writeheader()
            writer.writerows(data_rows)

    write_csv(output_paths["blind_predictions"], blind_rows, blind_fields)
    blind_reopened, blind_schema = read_csv(output_paths["blind_predictions"])
    require(blind_schema == blind_fields and len(blind_reopened) == 14755
            and [r["sample_id"] for r in blind_reopened] == ids
            and len({r["sample_id"] for r in blind_reopened}) == 14755,
            "Blind H12 prediction freeze physical validation failed")
    for row in blind_reopened:
        for label in LABELS:
            require(row[f"y_pred_{label}"] in {"0", "1"}, f"Invalid frozen H12 prediction {row['sample_id']} {label}")
            score = float(row[f"y_score_{label}"])
            require(np.isfinite(score) and 0 <= score <= 1, f"Invalid frozen H12 score {row['sample_id']} {label}")

    # Test labels are used for evaluation only after blind OOP predictions have
    # been physically frozen. They never enter preprocessing or model fitting.
    row_by_id = {row["sample_id"]: row for row in rows}
    predictions = []
    for blind in blind_reopened:
        source = row_by_id[blind["sample_id"]]
        predictions.append({
            "sample_id": blind["sample_id"], "project_name": blind["project_name"], "fold": blind["fold"],
            **{f"y_true_{label}": int(source[label]) for label in LABELS},
            **{f"y_pred_{label}": int(blind[f"y_pred_{label}"]) for label in LABELS},
            **{f"y_score_{label}": blind[f"y_score_{label}"] for label in LABELS},
        })
    y_true = np.asarray([[row[f"y_true_{label}"] for label in LABELS] for row in predictions], dtype=np.int8)
    y_pred = np.asarray([[row[f"y_pred_{label}"] for label in LABELS] for row in predictions], dtype=np.int8)
    pooled = metrics(y_true, y_pred)
    per_label = [label_metric_row(y_true, y_pred, column, label) for column, label in enumerate(LABELS)]
    # Only after the complete blind OOP prediction artifact is physically frozen
    # are test labels accessed for pooled and per-project evaluation.
    per_project_metrics: list[dict[str, Any]] = []
    per_project_label_metrics: list[dict[str, Any]] = []
    for held in PROJECTS:
        indices = [i for i, row in enumerate(predictions) if row["project_name"] == held]
        yt, yp = y_true[indices], y_pred[indices]
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
            per_project_label_metrics.append(
                label_metric_row(yt, yp, column, label, held_out_project=held)
            )
    pooled_rows = [{"metric": metric, "value": pooled[metric], "aggregation": "pooled OOP",
                    "zero_division_policy": "not applicable" if metric in {"hamming_loss", "exact_match_ratio"}
                    else "zero_division=0 for sklearn-compatible computational aggregation"}
                   for metric in ("hamming_loss", "exact_match_ratio", "micro_precision", "micro_recall", "micro_f1",
                                  "macro_precision", "macro_recall", "macro_f1")]

    # R1/R2/R3 metrics are frozen comparison references only; read only after
    # H12 OOP predictions have been physically frozen.
    r1_refs = verify_reference_run(R1_MANIFEST, {"pooled": R1_POOLED, "per_label": R1_LABEL})
    r2_refs = verify_reference_run(R2_MANIFEST, {"pooled": R2_POOLED, "per_label": R2_LABEL})
    r3_refs = verify_reference_run(R3_MANIFEST, {"pooled": R3_POOLED, "per_label": R3_LABEL})
    r1_pooled = {r["metric"]: float(r["value"]) for r in read_csv(R1_POOLED)[0]}
    r2_pooled = {r["metric"]: float(r["value"]) for r in read_csv(R2_POOLED)[0]}
    r3_pooled = {r["metric"]: float(r["value"]) for r in read_csv(R3_POOLED)[0]}
    r1_labels = {r["label"]: float(r["f1_computational"]) for r in read_csv(R1_LABEL)[0]}
    r2_labels = {r["label"]: float(r["f1_computational"]) for r in read_csv(R2_LABEL)[0]}
    r3_labels = {r["label"]: float(r["f1_computational"]) for r in read_csv(R3_LABEL)[0]}
    references = {"R1": (r1_pooled, r1_labels), "R2": (r2_pooled, r2_labels), "R3": (r3_pooled, r3_labels)}
    comparisons = {}
    for name, (reference, reference_labels) in references.items():
        comparisons[name] = {
            metric: pooled[metric] - reference[metric]
            for metric in ("hamming_loss", "exact_match_ratio", "micro_f1", "macro_f1")
        }
        comparisons[name].update({
            f"{label}_f1": float(next(row["f1_computational"] for row in per_label if row["label"] == label)) - reference_labels[label]
            for label in LABELS
        })

    write_csv(output_paths["predictions"], predictions, PRED_FIELDS)
    write_csv(output_paths["pooled"], pooled_rows, POOLED_FIELDS)
    write_csv(output_paths["per_label"], per_label, PER_LABEL_FIELDS)
    write_csv(output_paths["per_project"], per_project_metrics, PER_PROJECT_FIELDS)
    write_csv(output_paths["per_project_label"], per_project_label_metrics, PER_PROJECT_LABEL_FIELDS)
    write_csv(output_paths["feature_audit"], feature_audit, FEATURE_AUDIT_FIELDS)

    expected_counts = {
        "predictions": 14755, "blind_predictions": 14755, "pooled": 8, "per_label": 4,
        "per_project": 8, "per_project_label": 32, "feature_audit": 8,
    }
    schemas = {
        "predictions": PRED_FIELDS, "blind_predictions": blind_fields, "pooled": POOLED_FIELDS,
        "per_label": PER_LABEL_FIELDS, "per_project": PER_PROJECT_FIELDS,
        "per_project_label": PER_PROJECT_LABEL_FIELDS, "feature_audit": FEATURE_AUDIT_FIELDS,
    }
    physical = {}
    for key, expected_schema in schemas.items():
        physical[key], output_schema = read_csv(output_paths[key])
        require(output_schema == expected_schema and len(physical[key]) == expected_counts[key],
                f"H12 output schema/count mismatch: {key}")
    require([r["sample_id"] for r in physical["predictions"]] == ids
            and len({r["sample_id"] for r in physical["predictions"]}) == 14755,
            "Physical H12 prediction coverage/order/uniqueness failed")
    for row in physical["predictions"]:
        require(row["fold"] == f"LOPO_{PROJECTS.index(row['project_name']) + 1:02d}",
                f"Prediction fold mismatch {row['sample_id']}")
        for label in LABELS:
            require(row[f"y_true_{label}"] in {"0", "1"} and row[f"y_pred_{label}"] in {"0", "1"},
                    f"Prediction label invalid {row['sample_id']} {label}")
            score = float(row[f"y_score_{label}"])
            require(np.isfinite(score) and 0 <= score <= 1, f"Prediction score invalid {row['sample_id']} {label}")
    require(all(int(row["H12_dimension"]) == int(row["R1_dimension"]) + int(row["R2_dimension"])
                and row["R1_fit_scope"] == "TRAIN_ONLY" and row["R2_fit_scope"] == "TRAIN_ONLY"
                and row["R2_log1p"] == "YES" and row["R2_row_L2"] == "YES"
                and row["additional_block_weighting"] == "NO" and row["whole_H12_normalization"] == "NO"
                and row["AST_parse_failures"] == row["AST_target_resolution_failures"] == row["feature_extraction_failures"] == "0"
                for row in physical["feature_audit"]), "H12 feature/leakage audit failed")
    y_true_reopened = np.asarray([[int(row[f"y_true_{label}"]) for label in LABELS]
                                  for row in physical["predictions"]], dtype=np.int8)
    y_pred_reopened = np.asarray([[int(row[f"y_pred_{label}"]) for label in LABELS]
                                  for row in physical["predictions"]], dtype=np.int8)
    require(metrics(y_true_reopened, y_pred_reopened) == pooled,
            "Pooled H12 metrics do not reproduce from physically reopened predictions")

    zero_support = sorted({f"{row['held_out_project']}:{row['label']}" for row in per_project_label_metrics
                           if row["support_status"] == "NO_POSITIVE_SUPPORT"})
    summary_lines = [
        "# V2 — H12 Hybrid R1 + R2 LOPO — Primary", "",
        "H12 is the primary hybrid representation; results are an exploratory follow-up after R1/R2/R3 outcomes were observed. This task ran H12 only. H12-B balanced sensitivity was not run.",
        f"- Dataset: {len(rows)} samples; SHA-256 `{EXPECTED_HASHES[DATASET]}`.",
        "- Frozen eight-fold V2 LOPO; 14,755 pooled OOP predictions in canonical V2 relative order.",
        "- H12 = sparse_hstack([R1_TFIDF, row-wise L2(log1p(R2_AST))]); no block weighting and no normalization of the concatenated vector.",
        "- R1 uses frozen lexical tokenization and fold-local word unigram/bigram TF-IDF fitted on TRAIN only.",
        "- R2 uses FULL_FILE_TARGET_METHOD_AST generic node/edge counts, node count, and maximum depth; fold-local DictVectorizer fitted on TRAIN only.",
        "- Binary Relevance, four Logistic Regression models: C=1.0, L2, liblinear, class_weight=None, max_iter=2000, random_state=42.",
        "- R1/R2/R3 metric artifacts were read only after the H12 OOP prediction file was frozen. Differences below are descriptive only.", "",
        "## Feature-dimension reproduction", "",
        "| Fold | Held-out project | R1 dimension | Expected R1 | R2 dimension | Expected R2 | H12 dimension | Expected sum | Train n | Test n | R1/R2 fit | R2 log1p / row L2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in feature_audit:
        summary_lines.append(
            f"| {row['fold']} | {row['held_out_project']} | {row['R1_dimension']} | {row['R1_expected_dimension']} | "
            f"{row['R2_dimension']} | {row['R2_expected_dimension']} | {row['H12_dimension']} | "
            f"{row['R1_dimension'] + row['R2_dimension']} | {row['train_rows']} | {row['test_rows']} | TRAIN_ONLY / TRAIN_ONLY | YES / YES |"
        )
    summary_lines += [
        "", "Dimension reproduction: PASS for all eight frozen R1 vocabulary and R2 AST dimensions.",
        f"AST parse failures: {len(parse_failures)}; target resolution failures: {len(resolution_failures)}; feature extraction failures: {len(feature_failures)}.",
        "", "## Pooled metrics", "",
        f"- Hamming Loss: {pooled['hamming_loss']:.8f}", f"- Exact Match Ratio: {pooled['exact_match_ratio']:.8f}",
        f"- Micro Precision / Recall / F1: {pooled['micro_precision']:.8f} / {pooled['micro_recall']:.8f} / {pooled['micro_f1']:.8f}",
        f"- Macro Precision / Recall / F1 (computational, zero_division=0): {pooled['macro_precision']:.8f} / {pooled['macro_recall']:.8f} / {pooled['macro_f1']:.8f}",
        "", "## Pooled per-label metrics", "", "| Label | Support | Precision | Recall | F1 | TP | FP | FN | TN | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in per_label:
        summary_lines.append(
            f"| {row['label']} | {row['support']} | {row['precision_presentation']} | {row['recall_presentation']} | "
            f"{row['f1_presentation']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} | {row['support_status']} |"
        )
    summary_lines += ["", "## H12 minus frozen baselines (descriptive only)", "", "| Metric | H12 − R1 | H12 − R2 | H12 − R3 |", "|---|---:|---:|---:|"]
    for key, title in (("hamming_loss", "Hamming Loss"), ("exact_match_ratio", "Exact Match Ratio"),
                       ("micro_f1", "Micro F1"), ("macro_f1", "Macro F1")):
        summary_lines.append(f"| {title} | {comparisons['R1'][key]:+.8f} | {comparisons['R2'][key]:+.8f} | {comparisons['R3'][key]:+.8f} |")
    summary_lines += ["", "| Per-label F1 | H12 − R1 | H12 − R2 | H12 − R3 |", "|---|---:|---:|---:|"]
    for label in LABELS:
        summary_lines.append(f"| {label} | {comparisons['R1'][f'{label}_f1']:+.8f} | {comparisons['R2'][f'{label}_f1']:+.8f} | {comparisons['R3'][f'{label}_f1']:+.8f} |")
    summary_lines += ["", "No statistical significance testing was performed and no final representation winner was selected.", "", "## Per-project metrics", "", "| Held-out project | N | Hamming Loss | Exact Match | Micro F1 | Macro F1 |", "|---|---:|---:|---:|---:|---:|"]
    for row in per_project_metrics:
        summary_lines.append(f"| {row['held_out_project']} | {row['test_n']} | {row['hamming_loss']:.8f} | {row['exact_match_ratio']:.8f} | {row['micro_f1']:.8f} | {row['macro_f1_computational']:.8f} |")
    summary_lines += [
        "", "Per-project/per-label precision, recall, F1, and support are in `h12_v2_per_project_label_metrics.csv`.",
        "", "## Zero support and leakage", "",
        "- NO_POSITIVE_SUPPORT: " + (", ".join(zero_support) if zero_support else "none") + ". pyftpdlib:MNT is retained with recall/F1 presentation NA.",
        "- R1 TF-IDF fit TRAIN_ONLY: PASS; R2 DictVectorizer fit TRAIN_ONLY: PASS.",
        "- Full-data fitting: NO; test labels in training: NO; resampling/threshold tuning/hyperparameter tuning: NO/NO/NO.",
        "- V1 folds: NO; excluded unresolved samples: NO; R3 used in H12: NO; H12-B: NOT RUN; leakage violations: 0.",
        "- H12 additional block weighting: NO; whole-H12 normalization: NO.", "", "## Artifact hashes", "", "| Artifact | Rows | SHA-256 |", "|---|---:|---|",
    ]

    # All outputs remain exclusive and hash-verified; summary hash is recorded in manifest.
    artifact_keys = [key for key in output_paths if key not in {"summary", "manifest"}]
    artifact_hashes = {key: sha256(output_paths[key]) for key in artifact_keys}
    artifact_rows = {key: len(read_csv(output_paths[key])[0]) for key in artifact_keys}
    for key in artifact_keys:
        summary_lines.append(f"| {output_paths[key].name} | {artifact_rows[key]} | `{artifact_hashes[key]}` |")
    summary_lines.append("")
    with output_paths["summary"].open("x", encoding="utf-8", newline="") as stream:
        stream.write("\n".join(summary_lines))
    artifact_hashes["summary"] = sha256(output_paths["summary"])
    artifact_rows["summary"] = len(output_paths["summary"].read_text().splitlines())

    manifest_row = {
        "dataset_sha256": EXPECTED_HASHES[DATASET], "assignment_sha256": EXPECTED_HASHES[ASSIGNMENTS],
        "fold_manifest_sha256": EXPECTED_HASHES[FOLD_MANIFEST], "protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "canonical_identity_metadata_sha256": EXPECTED_HASHES[CANONICAL],
        "R1_pooled_sha256": r1_refs["pooled"], "R1_per_label_sha256": r1_refs["per_label"],
        "R2_pooled_sha256": r2_refs["pooled"], "R2_per_label_sha256": r2_refs["per_label"],
        "R3_pooled_sha256": r3_refs["pooled"], "R3_per_label_sha256": r3_refs["per_label"],
        "dataset_rows": len(rows), "prediction_rows": len(predictions), "fold_count": 8,
        "labels": ",".join(LABELS), "label_order": ",".join(LABELS),
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "random_state": CLASSIFIER_CONFIG["random_state"],
        "R1_config_json": json.dumps(R1_CONFIG, sort_keys=True, separators=(",", ":")),
        "R2_config_json": json.dumps(H12_CONFIG["r2"], sort_keys=True, separators=(",", ":")),
        "H12_config_json": json.dumps(H12_CONFIG, sort_keys=True, separators=(",", ":")),
        "classifier_config_json": json.dumps(CLASSIFIER_CONFIG, sort_keys=True, separators=(",", ":")),
        "score_definition": "LogisticRegression.predict_proba(X_test)[:, 1]",
        "zero_division_policy": "presentation NA for undefined precision; support-zero recall/F1 NA; macro computational zero_division=0",
        "AST_parse_failure_count": len(parse_failures), "AST_target_resolution_failure_count": len(resolution_failures),
        "feature_extraction_failure_count": len(feature_failures), "dimension_reproduction": "PASS",
        "artifact_row_counts_json": json.dumps(artifact_rows, sort_keys=True, separators=(",", ":")),
        "artifact_sha256_json": json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")),
        "R1_fit_scope": "TRAIN_ONLY_PER_FOLD", "R2_fit_scope": "TRAIN_ONLY_PER_FOLD",
        "full_data_fitting": "NO", "test_labels_used_in_training": "NO",
        "resampling": "NO", "threshold_tuning": "NO", "hyperparameter_tuning": "NO",
        "V1_folds_used": "NO", "excluded_unresolved_samples_used": "NO", "R3_used_in_H12": "NO",
        "H12B_executed": "NO", "statistical_testing": "NO", "leakage_violations": 0, "commit": "NO",
    }
    write_csv(output_paths["manifest"], [manifest_row], MANIFEST_FIELDS)
    manifest_reopened, manifest_fields = read_csv(output_paths["manifest"])
    require(manifest_fields == MANIFEST_FIELDS and len(manifest_reopened) == 1,
            "H12 run manifest physical validation failed")
    recorded = manifest_reopened[0]
    require(json.loads(recorded["artifact_sha256_json"]) == artifact_hashes
            and json.loads(recorded["artifact_row_counts_json"]) == artifact_rows,
            "H12 run manifest physical hashes/counts mismatch")
    for path, expected in EXPECTED_HASHES.items():
        require(sha256(path) == expected, f"Frozen H12 input changed: {path}")

    print(json.dumps({
        "final": "PASS", "dataset_rows": len(rows), "prediction_rows": len(predictions),
        "pooled_metrics": pooled, "per_label_metrics": per_label,
        "per_project_metrics": per_project_metrics, "dimension_reproduction": "PASS",
        "feature_audit": feature_audit, "ast_parse_failures": len(parse_failures),
        "ast_target_resolution_failures": len(resolution_failures), "feature_extraction_failures": len(feature_failures),
        "comparisons": comparisons, "zero_support": zero_support,
        "leakage_violations": 0,
        "artifacts": {key: {"path": str(path.relative_to(ROOT)),
                             "rows": artifact_rows.get(key, 1), "sha256": sha256(path)}
                      for key, path in output_paths.items()},
        "inputs_unchanged": True, "H12B": "NOT_RUN", "commit": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
