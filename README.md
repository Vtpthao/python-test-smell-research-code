# Python Multi-Label Test Smell Detection

Public, source-free research artifact accompanying the Python multi-label test
smell detection study. This repository provides metadata, original experiment
scripts, frozen predictions and evaluation outputs. It does not contain the
original third-party source code or support full training from public inputs.
Repository availability does not imply acceptance or publication of the
FJCAI 2027 paper.

## Dataset and Provenance

dataset/final/python_4label_test_smell_metadata_v2.csv is an explicitly named
**metadata projection**, not the original source-containing experimental CSV.
It contains all 14,755 samples, in original order, with unchanged IDs, project
assignments, labels and retained metadata. Only source_code is omitted.
The original scientific dataset remains byte-for-byte unchanged with SHA-256
ad1d9cd262f11b6ba29287e791c4cdd1be3d8dd3bbb36b949821b8bd807448da.
The public CSV has its own distinct checksum in configs/metadata_contract.json.

The 15 public fields are sample_id, project_name, file_path, class_name,
method_name, definition_line, AR, CTL, MNT, ET, label_count, AR_provenance,
CTL_provenance, MNT_provenance and ET_provenance. file_path is a relative factual
upstream location; no method bodies, source snippets, reviewer logs, absolute
machine paths, credential literals or copied source-context fields are included.

| Project | Samples |
|---|---:|
| chainer | 3185 |
| clusterfuzz | 1750 |
| grimoirelab-perceval | 1012 |
| numba | 6238 |
| pika | 78 |
| psutil | 703 |
| pyftpdlib | 198 |
| python-driver | 1591 |

| Label | Meaning | Positive | Negative |
|---|---|---:|---:|
| AR | Assertion Roulette | 4575 | 10180 |
| CTL | Conditional Test Logic | 3773 | 10982 |
| MNT | Magic Number Test | 1844 | 12911 |
| ET | Eager Test | 1280 | 13475 |

### Annotation workflow and review scope

AR, CTL and MNT reference labels were generated using deterministic static
rules and are treated as silver labels. PyNose was used for comparison with
the custom detector on selected static-rule labels; it was not the source
of the ET annotation workflow.

ET labels followed a separate staged annotation process. After the initial
analysis, 2,752 low-confidence or ambiguous ET cases proceeded to a second
stage involving human review. Within those 2,752 cases, 737 remained difficult
after the second stage; AI-assisted semantic analysis was used as supporting
evidence for these cases, while humans made the final labeling decisions.
AI assistance was not an autonomous source of final ET labels.

An additional review batch comprised 48 cases with differing outputs from
PyNose and the custom detector, plus 500 randomly selected cases used to
check labeling accuracy. The 500-case random sample was selected separately
and did not overlap the 2,752 ET cases. The documented review workload totals
3,300 review cases (2,752 + 48 + 500); this is a count of review cases,
not a claim that 3,300 distinct methods have a fully recoverable, expert-gold
method-level adjudication record. The 737 difficult ET cases are a subset of
the 2,752, not an additional batch. These figures describe the historical
annotation workflow and are not independently reconstructed from the
source-free public metadata.

Ten unresolved ET methods were excluded historically from the 14,765-method
canonical population; eight of those ten belonged to the 737 difficult ET
cases. The released 14,755-method dataset makes no additional exclusions.
The public labels remain research reference labels rather than independently
expert-validated ground truth for the entire population. The repository does
not include the full method-level review ledger or paired detector outputs,
so it cannot independently reproduce inter-detector agreement, Cohen's kappa,
or the exact count of labels corrected during review.

All 16 label combinations occur; 7,318 methods have no positive label.

The source-free privacy check inspects the exact retained schema, every retained
cell, relative-path/ID structure and sensitive patterns. No email, URL, home-path,
private-key, provider-token or credential-assignment pattern was found in the
public CSV. This is a bounded retained-field screen, not a guarantee about all
possible personal-information interpretations of arbitrary factual identifiers.

## Code and Verified Historical Results

The original R1, R2, R3, H12 and H12-B scripts under scripts/ are byte-identical
copies, with no altered models, feature extraction, labels, thresholds or hashes.
Their data-loading/feature/LOPO/metric helpers are inside these files; H12-B's
required first-party H12 module is included. No labeling, construction or PyNose
code, third-party source tree, pretrained weights or embedding matrix is included.
The first-party code is provided under the scoped license described below.

experiments/v2/ contains the final frozen predictions, pooled metrics, per-label,
per-project and project-label metrics plus original run manifests for all five
representations. These are **historical results**, not newly trained results.
R1 is TF-IDF; R2 is full-file AST counts; R3 is frozen UniXcoder; H12 combines
TF-IDF with row-L2-normalized log1p AST counts; H12-B is secondary balanced-class
sensitivity. Eight project-held-out LOPO folds use Binary Relevance/LR, seed 42.
Final assignments and fold manifest are included unchanged. Computational
zero_division is 0; pyftpdlib:MNT has no positive test support.

| Model | Pooled Micro-F1 | Pooled Macro-F1 |
|---|---:|---:|
| R1 | 0.6893096791984558 | 0.5326503629105769 |
| R2 | 0.6842005497109279 | 0.5450996918210858 |
| R3 | 0.621457918590368 | 0.5397608675070849 |
| H12 | 0.7641726551124889 | 0.5955059458654159 |
| H12-B, secondary | 0.7183032321111875 | 0.6607479797944884 |

## Setup and Runnable Validation

Metadata/fold/saved-metric validation uses only the standard library and does
not train models, download source/models or write experiment outputs:

```sh
python -B scripts/validate_release.py
```

It checks 14,755 IDs and projects, every fold support, manifest hashes, saved
prediction truth labels and 1,980 pooled/label/project metric cells. This new
release-support validator is not an invented historical experiment implementation.

The optional dependencies for importing the historical runners are recorded
in requirements.txt; the matching existing research environment was Python
3.12.14. For a new environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

This validation command does not train models or reconstruct omitted inputs.

## Training Limitations and Historical Commands

**The public metadata CSV cannot execute any original training runner.** All
five require the exact original dataset SHA and/or source-dependent features.
The renamed metadata CSV intentionally does not masquerade as that input.
No checksum gate has been weakened, and no source code is reconstructed here.

| Run | Omitted Runtime Requirements |
|---|---|
| R1 | Exact original CSV with source_code for lexical features |
| R2 | Original CSV, canonical V1 identity/source CSV, 1016 full upstream files |
| R3 | Original CSV, canonical V1, exclusions, exact embedding matrix/IDs/metadata and R3 protocol |
| H12 | Original CSV, canonical V1 and full-file AST sources; reference artifacts/features as required |
| H12-B | Same H12 inputs plus primary H12 feature audit and exact reference artifacts |

The canonical V1, source-containing exclusions, cache and cache-generation/model
implementation are not bundled. Original source redistribution, especially the
six DSE-origin methods, remains a separate issue if source is ever acquired or
conveyed. Their IDs/labels remain here; this metadata-only release does not
convey those method bodies. GPL/Numba/DSE source terms are not waived or claimed
cleared. No automatic downloader or full repository clone is provided.

These are the **historical commands only**, for a separately rights-cleared,
disposable complete research layout, not runnable commands for this public CSV:

```sh
python -B scripts/run_r1_tfidf_v2_lopo.py
python -B scripts/run_r2_ast_v2_lopo.py
python -B scripts/run_r3_unixcoder_v2_lopo.py
python -B scripts/run_h12_v2_lopo.py
python -B scripts/run_h12b_v2_balanced.py
```

Do not invoke them here: they fail on missing/changed original inputs and their
historical outputs overlap bundled frozen references. Some H12-B finalization
paths can remove existing results. The expected scientific outputs have prefixes
r1_v2, r2_v2, r3_v2, h12_v2 and h12b_v2 under experiments/v2/, including
*_oop_predictions.csv, *_pooled_metrics.csv and *_run_manifest.csv.

R3's microsoft/unixcoder-base identifier is known, but its immutable historical
checkpoint revision is not recorded. Merely generating new 768-dimensional
embeddings cannot guarantee the exact historical cache hash. No claim of full
retraining, fresh embedding or bootstrap reproducibility is made. The saved
numeric metrics can be independently checked without third-party source using
the included validator.

## License Scopes and Attribution

- LICENSE: MIT, Copyright (c) 2026 Vo Thi Phuong Thao, for the owner's original
  first-party code only. No copied third-party implementation is licensed by it.
- LICENSE-DATA.md, linking to the canonical CC BY legal code: CC BY 4.0 for
  original annotations and metadata/compilation contributions **to the extent
  the owner holds the rights**.
  Third-party annotations/source, names/facts as new exclusive property, model
  assets, trademarks and privacy rights are not included in that grant.
- No third-party project source text, extracted method bodies or model code is
  redistributed. Referring to source identifiers is not a blanket grant to
  acquire or redistribute the omitted source. Dependencies retain their licenses.

Code and data licenses are separate, not interchangeable dual licenses for every
file. Accuracy, label quality and third-party title are not warranted. Owner
scope authorization is not independent authentication of every possible outside
right. No unrelated institutional or co-owner consent is invented.

## Repository-Only Citation

Use CITATION.cff: Vo Thi Phuong Thao (2026), *Python Multi-Label Test Smell
Detection: Research Dataset and Experimental Code*,
https://github.com/Vtpthao/python-test-smell-research-code.
There is no DOI or claim that the FJCAI paper has been accepted or published.
Internal audit reports and the original method-level annotation ledger are
not distributed in this repository.