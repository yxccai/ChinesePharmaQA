# Release Verification

This document records the validation performed before packaging
`ChinesePharmaQA-Code-v1.0.1`.

Validation date: 2026-08-24

## Environment

| Component | Version |
| --- | --- |
| Operating system | Windows 11, build 22631 |
| Python | 3.13.1 |
| NumPy | 2.3.3 |
| FAISS CPU | 1.12.0 |
| PyTorch | 2.6.0+cu118 |
| CUDA runtime reported by PyTorch | 11.8 |
| Sentence Transformers | 5.1.2 |
| DashScope | 1.25.2 |
| Matplotlib | 3.10.8 |

All Python commands were run with the project's specified virtual-environment interpreter.
Wall-clock times below are observations from this machine and are not reproducibility
targets.

## Dataset Checks

Both public dataset scripts completed successfully:

```bash
python ../ChinesePharmaQA/scripts/validate.py
python ../ChinesePharmaQA/scripts/rebuild_qrels.py
```

Observed counts in the current release-update check:

| Item | Count |
| --- | ---: |
| Corpus records | 5,138 |
| Fusion queries | 1,065 |
| Single-drug questions | 200 |
| Relevance judgments | 48,335 |

`rebuild_qrels.py` reproduced all 48,335 bundled judgments exactly.
The current runs took 19.4 seconds for `validate.py` and 15.3 seconds for
`rebuild_qrels.py` on the validation machine.

## Script Validation

### `common.py`

`common.py` has no standalone CLI. Its dataset loading, BM25, structured indexes, parser,
metrics, hashing, and I/O functions were exercised by all evaluation scripts below.

### `evaluate_classic.py`

Status: passed. The original full release run included FullCorpusScan and controlled
ablations; the current update reran all indexed methods and controlled ablations with
`--skip-full-scan` because the full-scan implementation and frozen corpus are unchanged.

Observed wall time: 91.7 seconds for the original full run and approximately 61.4 seconds
for the current indexed-method recheck.

| Result | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| BM25-Original, K=50 | 0.2260 | 0.4591 | 0.2557 | 0.0000 |
| BM25-FullFields, K=50 | 0.2336 | 0.4739 | 0.2646 | 0.0000 |
| E2E-Rule (supplementary diagnostic) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| FullCorpusScan (supplementary diagnostic) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Observed internal query times were 27.8 ms for E2E-Rule and 15,912.5 ms for
FullCorpusScan. These values are hardware-dependent and are retained as supplementary
execution diagnostics, not as current headline methods.

Additional runs with different `PYTHONHASHSEED` values produced identical per-question
BM25 metrics.

### `evaluate_semantic.py`

Status: passed as the current task-specific baseline using the pinned BGE model and the
released global constraint vocabulary. The parser performs intent-gated semantic candidate
ranking followed by the shared structured set executor. It uses no additional similarity
threshold and returns an empty set if no category gate activates. At prediction time it
does not read the current query's standard constraints or answer set.

Observed wall time for the current full rerun, including local model loading: approximately
49.6 seconds.

| Result | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| Semantic-Parser | 0.6732 | 0.8185 | 0.6665 | 0.5577 |

Per-type F1 was 1.0000 for A, 0.9333 for B, 0.9225 for C, and 0.6413 for D.
The run used the local `BAAI/bge-large-zh-v1.5` model at revision
`79e7739b6ab944e86d6171e44d24c997fc1e0116` with the project-specified Python environment.
The overall, per-type, constraint-summary, input-hash, and all 1,065 per-question fields
matched the sanitized frozen `semantic_results.json` exactly.

### `evaluate_bge.py`

Status: passed through both a full re-encoding run and a validated-cache run.

- Full encoding of 5,138 records and 1,065 queries: 323.6 seconds.
- Second run using matching cache metadata: approximately 2.0 seconds.
- Cache-hit output explicitly reported that a validated cache was loaded.

| Result | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| BGE, K=200 | 0.0388 | 0.2095 | 0.0523 |

The two runs produced identical retrieval metrics. The final code also checks that the
public corpus and frozen evaluation view contain the same record-name set, and fingerprints
the model source, local-model file manifest, revision, corpus, serialized documents, and
query text before reusing arrays.

### `evaluate_qwen.py`

Status: passed in offline score-only mode.

Observed wall time in the current check: 0.3 seconds.

| Result | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Released sanitized Qwen-Plus predictions | 0.0125 | 0.0109 | 0.0084 |

All 1,065 released rows passed the strict query-ID, corpus-name, duplicate-name, and
non-negative `unmapped_count` checks.

Negative tests also passed:

- Existing generation outputs are protected unless `--overwrite` is explicit.
- Prediction, score-output, and raw-output paths cannot collide.
- A negative `unmapped_count` is rejected.

The chargeable online DashScope generation branch was not used during release validation;
no external API requests were sent. Its local argument validation and overwrite protection
were tested.

### `evaluate_paraphrases.py`

Status: passed.

Observed wall time in the current check: 0.8 seconds.

- Sampled source questions: 100.
- Generated programmatic paraphrases: 200.
- Exact constraint recovery: 200/200.
- Precision/Recall/F1/EM: 1.0000/1.0000/1.0000/1.0000.

This is a program-template paraphrase test, not an evaluation on unrestricted human
paraphrases, and it is retained only as a supplementary diagnostic.

### `make_figures.py`

Status: passed.

Observed wall time in the current check: 6.0 seconds.

Generated and visually inspected:

- `fig02_answer_distribution.pdf`
- `fig02_answer_distribution.png`
- `fig03_topk_curves.pdf`
- `fig03_topk_curves.png`

Chinese labels, annotations, axes, and legends rendered correctly in the validation
environment. The script reads the frozen `paper_summary.json`; it does not merge newly run
evaluation outputs.

## Frozen Artifact Hashes

Hashes below are computed from the LF-normalized Git repository blobs.

```text
eval_corpus.json       4F17F313FB9C8E66578BD1F8E4237A9C10F39A21BE77E66B36D5FD017C602F27
qwen_predictions.jsonl 74FD460D794B032850F4C774E1C35FE52996E9EB63BABFFC487B1D526AA45125
paper_summary.json     E0B0C1E3AA228C3564D9B0AA618F66B69A8089A0072A5E48BAB45E3EC6ABFE36
semantic_results.json  59147BC78EBA9D1D4CDE86CED2C3A7C6EAE014E39AEA1238BF5371F4957780DE
```

The machine-readable counterpart of this report is
`benchmark/results/verification_summary.json`.
