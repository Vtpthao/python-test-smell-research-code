# Python Multi-Label Test Smell Detection — Research Code

This repository provides a **limited code-only release** associated with research on multi-label test smell detection in Python tests. It is **not** a complete reproduction package.

## Research scope

The study examines Assertion Roulette (AR), Conditional Test Logic (CTL), Magic Number Test (MNT), and Eager Test (ET) across 14,755 Python test methods from eight open-source projects, using a Leave-One-Project-Out (LOPO) evaluation protocol.

## Included

- `scripts/run_r1_tfidf_v2_lopo.py`: R1 TF-IDF/Logistic Regression baseline runner
- `scripts/run_final_comparative_analysis.py`: analysis of saved results for R1, R2, R3, H12, and H12-B
- `scripts/input_preflight.py`: checks required frozen inputs and saved run artifacts before running
- `tests/`: synthetic preflight tests
- `requirements.txt`: recorded dependencies, not a complete lockfile

## Environment and tests

The target recorded environment is CPython 3.12.14. Package metadata indicates the dependency versions support Python 3.12, but installation, numerical equivalence, and runtime tests on **3.12.14 have not been verified** in this release preparation. The synthetic suite passed on Python 3.13.5.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Required inputs (not distributed)

The R1 runner requires separately authorized frozen V2 dataset, fold assignments, fold manifest, and protocol files with exact expected SHA-256 hashes. Run `python scripts/input_preflight.py r1` before `python scripts/run_r1_tfidf_v2_lopo.py`. The runner fits models and generates new results.

The comparative-analysis script additionally requires complete saved prediction, metric, and run-manifest artifacts for **all five** runs (R1, R2, R3, H12, H12-B). Run `python scripts/input_preflight.py analysis` before `python scripts/run_final_comparative_analysis.py`. It does not train models.

Both scripts refuse to proceed when required frozen inputs are missing or inconsistent. They do not provide complete reproduction without separately approved inputs.

## Data and provenance

No dataset, upstream source code, PyNose-derived detector, frozen predictions, embeddings, pretrained model weights, or private Git history is included. Dataset redistribution requires a separate rights/provenance review. The research labels are not presented as independently established human gold-standard labels.

## License

The original code and candidate support files in this release are made available under the MIT License (see `LICENSE`). This license does **not** apply to excluded third-party datasets, source code, software packages, or models. Dependencies retain their own licenses.

## Citation

A final paper citation will be added after publication details are confirmed.

## Contact

Vo Thi Phuong Thao — vtpthao061192@gmail.com
