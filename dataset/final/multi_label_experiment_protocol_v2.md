# V2 Four-Label Experiment Protocol and LOPO Fold Freeze

## Frozen population and unit

- Authoritative experimental population: 14755 methods in `python_4label_test_smell_experimental_v2.csv`.
- Dataset SHA-256: `ad1d9cd262f11b6ba29287e791c4cdd1be3d8dd3bbb36b949821b8bd807448da`.
- Labels, fixed order: AR, CTL, MNT, ET.
- Unit: one Python test method; grouping field: `project_name`.
- Projects: chainer, clusterfuzz, grimoirelab-perceval, numba, pika, psutil, pyftpdlib, python-driver. No merging, exclusion, or random split.
- Labels are binary integer 0/1. AR/CTL/MNT are deterministic static silver labels; ET is from the assistant-semantic ET reference V2.

## Frozen LOPO folds

Fold order is alphabetical by stored project name; fold IDs are LOPO_01 through LOPO_08. For each fold, TEST is every sample in exactly the held-out project and TRAIN is every sample in the other seven projects. The assignment CSV is in experimental dataset order; `fold_id` identifies the unique fold in which that row is held out. The fold manifest records train/test size and per-label positive/negative support.

| Fold | Held-out project | Train n | Test n | Test AR +/− | Test CTL +/− | Test MNT +/− | Test ET +/− |
|---|---|---:|---:|---:|---:|---:|---:|
| LOPO_01 | chainer | 11570 | 3185 | 547 / 2638 | 337 / 2848 | 233 / 2952 | 207 / 2978 |
| LOPO_02 | clusterfuzz | 13005 | 1750 | 634 / 1116 | 110 / 1640 | 345 / 1405 | 85 / 1665 |
| LOPO_03 | grimoirelab-perceval | 13743 | 1012 | 618 / 394 | 423 / 589 | 357 / 655 | 40 / 972 |
| LOPO_04 | numba | 8517 | 6238 | 1597 / 4641 | 2207 / 4031 | 568 / 5670 | 616 / 5622 |
| LOPO_05 | pika | 14677 | 78 | 35 / 43 | 7 / 71 | 28 / 50 | 7 / 71 |
| LOPO_06 | psutil | 14052 | 703 | 334 / 369 | 270 / 433 | 2 / 701 | 73 / 630 |
| LOPO_07 | pyftpdlib | 14557 | 198 | 56 / 142 | 42 / 156 | 0 / 198 | 20 / 178 |
| LOPO_08 | python-driver | 13164 | 1591 | 754 / 837 | 377 / 1214 | 311 / 1280 | 232 / 1359 |

Each sample appears once as TEST and seven times as TRAIN over the eight folds. Within every fold, train and test are disjoint and their union is the complete 14,755-sample population. Fold assignments are common to all future representations.

## Label distribution in frozen population

| Label | Positive | Negative |
|---|---:|---:|
| AR | 4575 | 10180 |
| CTL | 3773 | 10982 |
| MNT | 1844 | 12911 |
| ET | 1280 | 13475 |

## LOPO support warning

Zero positive support is flagged for experiment interpretation only. It does not change folds or membership; do not resample or merge projects.
- NO_POSITIVE_SUPPORT: pyftpdlib:MNT.

## Leakage and execution boundary

This freeze creates identifiers and project-held-out assignments only. It does not generate representations, fit TF-IDF, fit classifiers, train models, tune parameters, or calculate predictive performance. For subsequent experiments, fit learned preprocessing and representations on each fold's TRAIN samples only; transform that fold's TEST samples without fitting on test data. Do not use held-out labels for fitting, model selection, or threshold selection. Do not use or reuse V1 fold assignments: V1 represents a different population.

## Frozen artifacts

- `lopo_fold_assignments_v2.csv`: one row per V2 sample in frozen dataset order, naming its one test fold.
- `lopo_fold_manifest_v2.csv`: eight fold sizes and per-label train/test positive/negative support.
- `lopo_v2_freeze_manifest.csv`: physical hashes for the dataset and both fold artifacts and this protocol.

**FREEZE: PASS.** No model was trained and no predictions or predictive metrics were produced.

HARD STOP — do not train models as part of this fold-freeze task.
