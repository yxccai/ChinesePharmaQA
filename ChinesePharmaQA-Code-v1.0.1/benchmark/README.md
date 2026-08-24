# Benchmark Command Reference

This directory contains the executable evaluation code and frozen artifacts used to
reproduce the ChinesePharmaQA paper results. For the project overview, quick start, expected
metrics, and evaluation protocol, begin with the root `README.md` or `README_zh-CN.md`.

## Script and Artifact Map

| Path | Command-line program | Inputs | Default output | External dependencies |
| --- | --- | --- | --- | --- |
| `common.py` | No | Public dataset and frozen records through importing scripts | None | Standard library |
| `semantic_parser.py` | No | Candidate vocabulary, semantic scores, and structured indexes through `evaluate_semantic.py` | None | Standard library |
| `evaluate_classic.py` | Yes | Public dataset, `resources/eval_corpus.json` | `results/classic_results.json` | Standard library |
| `evaluate_bge.py` | Yes | Public dataset, frozen corpus, BGE model | `results/bge_results.json` | NumPy, FAISS, PyTorch, Sentence Transformers |
| `evaluate_semantic.py` | Yes | Public dataset, frozen corpus, local BGE model | `results/semantic_results.json` | PyTorch, Sentence Transformers |
| `evaluate_qwen.py` | Yes | Public dataset and normalized prediction JSONL | `results/qwen_results.json` | Standard library for scoring; DashScope only for `--generate` |
| `evaluate_paraphrases.py` | Yes | Public dataset and frozen corpus | `results/paraphrase_results.json` | Standard library |
| `make_figures.py` | Yes | `results/paper_summary.json` | `figures/*.pdf` and `figures/*.png` | Matplotlib |
| `resources/eval_corpus.json` | No | Frozen 5,138-record evaluation view | None | None |
| `results/paper_summary.json` | No | Frozen paper metrics and plotting data | None | None |
| `results/qwen_predictions.jsonl` | No | Sanitized released Qwen predictions | None | None |
| `results/semantic_results.json` | No | Frozen Semantic-Parser protocol and per-query results | None | None |
| `results/verification_summary.json` | No | Release validation record | None | None |

## Dataset Placement

The default paths assume the extracted dataset and code directories are siblings:

```text
parent/
├── ChinesePharmaQA/
└── ChinesePharmaQA-Code-v1.0.1/
```

Every evaluation command accepts `--dataset-root`, so another layout is also supported.

## Recommended Run Order

From the code-package root:

```bash
python ../ChinesePharmaQA/scripts/validate.py
python ../ChinesePharmaQA/scripts/rebuild_qrels.py

python benchmark/evaluate_classic.py \
  --dataset-root ../ChinesePharmaQA \
  --output reproduced/classic_results.json

python benchmark/evaluate_paraphrases.py \
  --dataset-root ../ChinesePharmaQA \
  --output reproduced/paraphrase_results.json

python benchmark/evaluate_qwen.py \
  --dataset-root ../ChinesePharmaQA \
  --predictions benchmark/results/qwen_predictions.jsonl \
  --output reproduced/qwen_results.json

python benchmark/evaluate_bge.py \
  --dataset-root ../ChinesePharmaQA \
  --cache-dir reproduced/bge_cache \
  --output reproduced/bge_results.json

python benchmark/evaluate_semantic.py \
  --dataset-root ../ChinesePharmaQA \
  --model BAAI/bge-large-zh-v1.5 \
  --revision 79e7739b6ab944e86d6171e44d24c997fc1e0116 \
  --output reproduced/semantic_results.json

python benchmark/make_figures.py \
  --summary benchmark/results/paper_summary.json \
  --output-dir reproduced/paper_figures
```

The first three evaluation commands are deterministic and can be run without network
access. BGE needs the model locally or downloads the pinned revision. Figure generation
uses the frozen paper summary rather than newly produced result JSON files.

## `common.py`

Shared functions used by the other scripts:

- Read the public JSONL/TSV release and join qrels to corpus titles.
- Load and verify the 5,138-record frozen evaluation view.
- Build the released constraint vocabulary.
- Tokenize Chinese text into characters, adjacent character bigrams, and alphanumeric tokens.
- Build and search the two BM25 indexes.
- Build reagent, instrument/method, property, and volume indexes.
- Parse explicit constraint terms from natural-language questions.
- Execute structured intersections and full-corpus scans.
- Compute per-question and macro-averaged set metrics.

It is a library module and should not be executed directly.

## `evaluate_classic.py`

Runs all standard-library baselines and controlled analyses:

- `Random_K10`, `Random_K50`
- `BM25_Original_K{10,20,50,100,200,500,1000}`
- `BM25_FullFields_K{10,20,50,100,200,500,1000}`
- `E2E_Rule`
- `FullCorpusScan`
- `OracleStructured`
- `BM25_Filter_K{200,500,1000}`
- `first_only`, `or_logic`, `no_reagent`, `no_instrument`, `no_property`, and `full_structured` ablations

Important options:

| Option | Meaning |
| --- | --- |
| `--dataset-root` | Root of the public `ChinesePharmaQA` dataset. |
| `--eval-corpus` | Frozen paper evaluation view; defaults to `resources/eval_corpus.json`. |
| `--output` | Destination JSON file. Parent directories are created automatically. |
| `--skip-full-scan` | Skip only the slow FullCorpusScan efficiency reference. All other methods still run. |

`classic_results.json` contains `summary`, `parser_accuracy`, `ablation`, and `per_item`.

## `evaluate_bge.py`

Serializes each frozen record with `json.dumps(..., ensure_ascii=False)`, encodes records and
questions with BGE, applies L2 normalization through Sentence Transformers, and performs
FAISS inner-product retrieval at K = 10, 50, 100, 200, 500, and 1000.

The default paper configuration is:

```text
model:    BAAI/bge-large-zh-v1.5
revision: 79e7739b6ab944e86d6171e44d24c997fc1e0116
```

Important options:

| Option | Meaning |
| --- | --- |
| `--dataset-root` | Root of the public `ChinesePharmaQA` dataset. |
| `--eval-corpus` | Frozen 5,138-record evaluation view. |
| `--model` | Hugging Face model identifier or local model directory. |
| `--revision` | Model revision used when `--model` is not a local path. |
| `--batch-size` | Encoding batch size; CPU runs are capped internally at 8. |
| `--cache-dir` | Directory for document/query arrays and cache metadata. |
| `--rebuild-cache` | Force re-encoding even if a valid cache exists. |
| `--output` | Destination `bge_results.json`. |

The cache metadata fingerprints the model source, revision, local-model file manifest,
frozen-corpus SHA-256, serialized document text, and query text. Existing arrays are reused
only when all values match. Model weights and embedding arrays are not distributed.

## `evaluate_semantic.py`

Evaluates a semantic constraint parser followed by the same structured set executor.
The parser encodes the question and category-specific descriptions of the released global
constraint vocabulary with the pinned BGE model. Category intent gates prevent method
questions from activating reagent candidates, while explicit multi-reagent expressions
allow two reagent candidates. Unambiguous ``第一部``/``第二部`` expressions are normalized
before the volume candidate is selected. The parser never reads the current query's gold
constraints or answer set. The output records the candidate and query hashes, parser
protocol, per-type metrics, and error-type summaries for reproducibility.
No additional similarity threshold is used. If no category gate activates, no candidate is
selected and the executor returns an empty set.

The paper run uses:

```text
model:    BAAI/bge-large-zh-v1.5
revision: 79e7739b6ab944e86d6171e44d24c997fc1e0116
device:   local PyTorch runtime (CPU or CUDA)
```

Important options:

| Option | Meaning |
| --- | --- |
| `--dataset-root` | Root of the public `ChinesePharmaQA` dataset. |
| `--eval-corpus` | Frozen 5,138-record evaluation view. |
| `--model` | Hugging Face model identifier or local model directory. |
| `--revision` | Model revision used when `--model` is not a local path. |
| `--batch-size` | Encoding batch size; CPU runs are capped internally at 8. |
| `--output` | Destination `semantic_results.json`. |

The released `semantic_results.json` identifies the public model and pinned revision while
omitting the local filesystem path used to load the validated model snapshot.

## `evaluate_qwen.py`

### Offline score-only mode

The default release workflow reads `results/qwen_predictions.jsonl` and recomputes metrics.
It does not import DashScope, access the network, or send API requests.

```bash
python benchmark/evaluate_qwen.py \
  --dataset-root ../ChinesePharmaQA \
  --predictions benchmark/results/qwen_predictions.jsonl \
  --output reproduced/qwen_results.json
```

The scorer validates that every fusion query appears exactly once, names exist in the
public corpus, predictions contain no duplicates, and `unmapped_count` is a non-negative
integer.

### Optional generation mode

`--generate` sends all 1,065 questions to DashScope. It is a fresh, chargeable experiment
and may produce different responses from the paper run.

```bash
python benchmark/evaluate_qwen.py \
  --dataset-root ../ChinesePharmaQA \
  --generate \
  --model qwen-plus \
  --predictions reproduced/qwen_new_predictions.jsonl \
  --output reproduced/qwen_new_results.json
```

Set `DASHSCOPE_API_KEY` in the local environment. Prediction, score-output, and raw-output
paths must be distinct. Existing generation outputs are protected; use new paths or pass
`--overwrite` explicitly. `--raw-output` is optional and is the only way to save raw model
responses.

## `evaluate_paraphrases.py`

The script fixes `random.seed(42)`, samples A/B/C/D questions in counts 10/5/10/75, and
creates two paraphrases per sampled question. It reports:

- Exact, partial, or failed constraint recovery.
- Precision, recall, F1, Jaccard, and exact match of the final structured answer set.
- All 200 generated paraphrases for inspection.

The D8 three-constraint template is selected before the two-reagent D1 template so that
volume information is retained.

## `make_figures.py`

Reads `results/paper_summary.json` and creates:

- `fig02_answer_distribution.pdf` and `.png`
- `fig03_topk_curves.pdf` and `.png`

The script does not create Figure 1 and does not merge current evaluation outputs. To plot
new experiment results, first construct a summary with the same schema as
`paper_summary.json`, then pass it through `--summary`.

## Frozen Artifacts

SHA-256 values in this section are computed from the LF-normalized Git repository blobs.

### `resources/eval_corpus.json`

The public `data/corpus.jsonl` is the authoritative dataset format. The frozen evaluation
view preserves the paper experiment's field names, layout, and placeholder values. The two
files contain the same record-name set, but their serialized retrieval text is different.

SHA-256:

```text
4F17F313FB9C8E66578BD1F8E4237A9C10F39A21BE77E66B36D5FD017C602F27
```

### `results/qwen_predictions.jsonl`

Each line contains exactly:

```json
{"query_id":"cpqa-fq-00001","pred_names":["..."],"unmapped_count":0}
```

No raw answer text is included. SHA-256:

```text
74FD460D794B032850F4C774E1C35FE52996E9EB63BABFFC487B1D526AA45125
```

### `results/paper_summary.json`

Contains the frozen dataset statistics, retrieval curves, headline methods, controlled
results, ablations, and supplementary diagnostic records. `make_figures.py` uses its answer
distribution and retrieval-curve sections. The file is not overwritten by evaluation commands.

SHA-256: `E0B0C1E3AA228C3564D9B0AA618F66B69A8089A0072A5E48BAB45E3EC6ABFE36`.

### `results/semantic_results.json`

Contains the frozen Semantic-Parser protocol, pinned public model identifier, input hashes,
overall and per-type metrics, error summaries, and 1,065 per-question records.

SHA-256: `59147BC78EBA9D1D4CDE86CED2C3A7C6EAE014E39AEA1238BF5371F4957780DE`.

## Expected Headline Results

| Script/result key | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| `classic_results.summary.BM25_Original_K50` | 0.2260 | 0.4591 | 0.2557 | 0.0000 |
| `classic_results.summary.BM25_FullFields_K50` | 0.2336 | 0.4739 | 0.2646 | 0.0000 |
| `bge_results.summary["K=200"].overall` | 0.0388 | 0.2095 | 0.0523 | 0.0000 |
| `qwen_results.overall` | 0.0125 | 0.0109 | 0.0084 | 0.0000 |
| `semantic_results.overall` | 0.6732 | 0.8185 | 0.6665 | 0.5577 |

Only methods that do not read the current question's gold constraints at prediction time
belong in the natural-language method comparison. OracleStructured, BM25+Filter, and all
component ablations are controlled analyses.

`classic_results.summary.E2E_Rule`, FullCorpusScan, and
`paraphrase_results.e2e_metrics` are retained as supplementary diagnostics. They are not
headline methods in the current paper comparison.

Run any script with `--help` for its complete option list.
