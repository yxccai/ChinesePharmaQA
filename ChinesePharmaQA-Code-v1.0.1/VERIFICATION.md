# Release Verification

This document records the validation performed before packaging
`ChinesePharmaQA-Code-v1.0.1`.

Validation date: 2026-08-08

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

Observed counts:

| Item | Count |
| --- | ---: |
| Corpus records | 5,138 |
| Fusion queries | 1,065 |
| Single-drug questions | 200 |
| Relevance judgments | 48,335 |

`rebuild_qrels.py` reproduced all 48,335 bundled judgments exactly.

## Script Validation

### `common.py`

`common.py` has no standalone CLI. Its dataset loading, BM25, structured indexes, parser,
metrics, hashing, and I/O functions were exercised by all evaluation scripts below.

### `evaluate_classic.py`

Status: passed, full run including FullCorpusScan and controlled ablations.

Observed wall time: 91.7 seconds.

| Result | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| BM25-Original, K=50 | 0.2260 | 0.4591 | 0.2557 | 0.0000 |
| BM25-FullFields, K=50 | 0.2336 | 0.4739 | 0.2646 | 0.0000 |
| E2E-Rule | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| FullCorpusScan | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Observed internal query times were 27.8 ms for E2E-Rule and 15,912.5 ms for
FullCorpusScan. These values are hardware-dependent; the metrics match the frozen paper
results.

Additional runs with different `PYTHONHASHSEED` values produced identical per-question
BM25 metrics.

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

Observed wall time: 1.1 seconds.

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

Observed wall time: 1.4 seconds.

- Sampled source questions: 100.
- Generated programmatic paraphrases: 200.
- Exact constraint recovery: 200/200.
- Precision/Recall/F1/EM: 1.0000/1.0000/1.0000/1.0000.

This is a program-template paraphrase test, not an evaluation on unrestricted human
paraphrases.

### `make_figures.py`

Status: passed.

Observed wall time: 3.2 seconds.

Generated and visually inspected:

- `fig02_answer_distribution.pdf`
- `fig02_answer_distribution.png`
- `fig03_topk_curves.pdf`
- `fig03_topk_curves.png`

Chinese labels, annotations, axes, and legends rendered correctly in the validation
environment. The script reads the frozen `paper_summary.json`; it does not merge newly run
evaluation outputs.

## Frozen Artifact Hashes

```text
eval_corpus.json       837471973343F864F605891312A8B50ECB872962271F9CC4A332DAE801087693
qwen_predictions.jsonl 74FD460D794B032850F4C774E1C35FE52996E9EB63BABFFC487B1D526AA45125
paper_summary.json     146EDE86B0B149F9318468CE94ECC9A68B2468FB2DA726372F5535854A91EAC1
```

The machine-readable counterpart of this report is
`benchmark/results/verification_summary.json`.
