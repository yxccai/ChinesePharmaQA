# ChinesePharmaQA Reproducibility Code

[中文说明](README_zh-CN.md)

This package provides the evaluation and figure-reproduction code for
**ChinesePharmaQA: A Benchmark Dataset for Fusion Knowledge Retrieval over the Chinese Pharmacopoeia**.

ChinesePharmaQA evaluates set-valued retrieval: given a natural-language question such as
"Which drugs use a specified reagent?", a system must return the complete set of drug
monographs satisfying all stated constraints. The accompanying dataset release contains
5,138 structured records, 1,065 fusion queries, 200 single-drug questions, and 48,335
query-record relevance judgments.

This code release reproduces the paper's benchmark experiments over the 1,065 fusion
queries. The 200 single-drug questions are distributed with the dataset as a control task,
but are not evaluated by the scripts in this package.

## What This Package Reproduces

- Random, BM25-Original, and BM25-FullFields retrieval baselines.
- BGE-large-zh-v1.5 dense retrieval with FAISS.
- Offline rescoring of the released, sanitized Qwen-Plus predictions.
- Semantic-Parser, which performs intent-gated semantic constraint parsing followed by field-level set execution.
- Structured oracle, BM25+Filter, and controlled component ablations.
- Released-lexicon rule parsing, full-corpus scan, and programmatic paraphrases as supplementary diagnostics.
- An optional fresh Qwen-Plus API run.
- Paper Figures 2 and 3 from the frozen paper summary.

The package is intentionally scoped to benchmark evaluation and figure reproduction. It
does not distribute the original pharmacopoeia PDFs, OCR intermediates, model weights, or
embedding indexes. Figure 1 is a manually prepared pipeline diagram and is not generated
by `make_figures.py`.

## Release Layout

```text
ChinesePharmaQA-Code-v1.0.1/
├── README.md
├── README_zh-CN.md
├── VERIFICATION.md
├── LICENSE
└── benchmark/
    ├── README.md
    ├── requirements.txt
    ├── common.py
    ├── semantic_parser.py
    ├── evaluate_classic.py
    ├── evaluate_bge.py
    ├── evaluate_semantic.py
    ├── evaluate_qwen.py
    ├── evaluate_paraphrases.py
    ├── make_figures.py
    ├── resources/
    │   └── eval_corpus.json
    └── results/
        ├── paper_summary.json
        ├── qwen_predictions.jsonl
        ├── semantic_results.json
        └── verification_summary.json
```

## File Reference

| File | Purpose |
| --- | --- |
| `benchmark/common.py` | Shared dataset loading, tokenization, BM25, structured indexes, query parsing, metrics, hashing, and JSON I/O. It is imported by the evaluation scripts and has no command-line entry point. |
| `benchmark/semantic_parser.py` | Candidate descriptions, intent gates, candidate selection, volume normalization, and field-level set execution for Semantic-Parser. |
| `benchmark/evaluate_classic.py` | Runs Random, both BM25 variants, released-lexicon E2E-Rule, FullCorpusScan, OracleStructured, BM25+Filter, parser analysis, and structured-component ablations. |
| `benchmark/evaluate_bge.py` | Encodes the frozen evaluation corpus and queries with BGE, builds a FAISS inner-product index, and evaluates Top-K retrieval. Cache metadata binds vectors to the model, revision, corpus, and queries. |
| `benchmark/evaluate_semantic.py` | Encodes questions and category-specific constraint descriptions with the pinned BGE model, performs intent-gated semantic parsing, and evaluates structured set execution. |
| `benchmark/evaluate_qwen.py` | By default, scores the released sanitized predictions without network access. With `--generate`, calls DashScope to create a new Qwen run. |
| `benchmark/evaluate_paraphrases.py` | Uses seed 42 to sample 100 fusion questions, generates two paraphrases per question, and evaluates constraint parsing and final answer sets. |
| `benchmark/make_figures.py` | Reads the frozen `paper_summary.json` and creates paper Figures 2 and 3 as PDF and PNG. It does not aggregate newly generated result files. |
| `benchmark/resources/eval_corpus.json` | Frozen 5,138-record field view used by the paper baselines. It preserves the exact serialization used in the experiments. |
| `benchmark/results/paper_summary.json` | Frozen paper metrics, ablations, efficiency values, and plotting data. It is a release artifact, not an automatically merged output of the evaluation scripts. |
| `benchmark/results/qwen_predictions.jsonl` | 1,065 sanitized prediction rows containing only `query_id`, `pred_names`, and `unmapped_count`; raw model responses are not included. |
| `benchmark/results/verification_summary.json` | Machine-readable record of the release validation run and its headline results. |
| `benchmark/results/semantic_results.json` | Reproducible Semantic-Parser run with per-type metrics, hashes, and parser error types. |
| `VERIFICATION.md` | Human-readable validation record showing which scripts were run and what they reproduced. |

## Quick Start: Offline Reproduction

Extract the dataset and code packages as sibling directories:

```text
parent/
├── ChinesePharmaQA/
└── ChinesePharmaQA-Code-v1.0.1/
```

Run the following commands from the code-package root. These steps use no API and, except
for figure rendering, require only the Python standard library.

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
```

Generate the two data-driven paper figures from the frozen summary:

```bash
python benchmark/make_figures.py \
  --summary benchmark/results/paper_summary.json \
  --output-dir reproduced/paper_figures
```

`make_figures.py` intentionally uses the frozen paper summary. It does not read
`reproduced/classic_results.json`, `bge_results.json`, or `qwen_results.json`.

## Environment and BGE-Based Reproduction

The release was validated with Python 3.13.1. Classic evaluation, offline Qwen scoring,
and paraphrase evaluation use only the standard library. Figure generation requires
Matplotlib. BGE retrieval additionally requires NumPy and FAISS; both BGE retrieval and
Semantic-Parser require Sentence Transformers and PyTorch.

Install a CPU or CUDA PyTorch build appropriate for the machine first, then install the
remaining pinned packages:

```bash
pip install -r benchmark/requirements.txt
```

The paper environment used:

- `numpy==2.3.3`
- `sentence-transformers==5.1.2`
- `faiss-cpu==1.12.0`
- `torch==2.6.0+cu118`
- `dashscope==1.25.2`
- `matplotlib==3.10.8`

Run BGE with the paper model and revision:

```bash
python benchmark/evaluate_bge.py \
  --dataset-root ../ChinesePharmaQA \
  --model BAAI/bge-large-zh-v1.5 \
  --revision 79e7739b6ab944e86d6171e44d24c997fc1e0116 \
  --cache-dir reproduced/bge_cache \
  --output reproduced/bge_results.json
```

The first run downloads or loads the model and creates embeddings. The cache includes a
metadata fingerprint; changing the model, revision, frozen corpus, or query text triggers
a rebuild. Use `--rebuild-cache` to force re-encoding.

Run Semantic-Parser with the same pinned model:

```bash
python benchmark/evaluate_semantic.py \
  --dataset-root ../ChinesePharmaQA \
  --model BAAI/bge-large-zh-v1.5 \
  --revision 79e7739b6ab944e86d6171e44d24c997fc1e0116 \
  --output reproduced/semantic_results.json
```

## Optional Fresh Qwen-Plus Run

The normal Qwen command above is offline: it only scores the released predictions. A new
API run is a separate experiment, incurs API cost, and may not reproduce historical text.

Set `DASHSCOPE_API_KEY` locally and always choose new output paths:

```bash
python benchmark/evaluate_qwen.py \
  --dataset-root ../ChinesePharmaQA \
  --generate \
  --model qwen-plus \
  --predictions reproduced/qwen_new_predictions.jsonl \
  --output reproduced/qwen_new_results.json
```

The script requires prediction, score-output, and optional raw-output paths to be distinct,
and refuses to replace existing generation outputs unless `--overwrite` is explicitly
supplied. Raw answers are not saved unless `--raw-output` is requested.

## Expected Results

The main table below reports macro-averaged set metrics over all 1,065 fusion queries.

| Method | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| BM25-Original, K=50 | 0.2260 | 0.4591 | 0.2557 | 0.0000 |
| BM25-FullFields, K=50 | 0.2336 | 0.4739 | 0.2646 | 0.0000 |
| BGE, K=200 | 0.0388 | 0.2095 | 0.0523 | 0.0000 |
| Direct Qwen-Plus | 0.0125 | 0.0109 | 0.0084 | 0.0000 |
| Semantic-Parser | 0.6732 | 0.8185 | 0.6665 | 0.5577 |

Semantic-Parser receives only the natural-language question for each prediction. It uses a
global released constraint vocabulary, but does not read the current question's annotated
constraints or answer set.

Controlled results that read released gold constraints must be interpreted separately:

| Controlled setting | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| BM25+Filter, K=200 | 0.9991 | 0.7995 | 0.8591 | 0.3878 |
| BM25+Filter, K=500 | 0.9991 | 0.9262 | 0.9504 | 0.6939 |
| BM25+Filter, K=1000 | 1.0000 | 0.9772 | 0.9856 | 0.8789 |
| OracleStructured | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The release also preserves the following supplementary diagnostics for auditability:

| Supplementary diagnostic | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| E2E-Rule | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| FullCorpusScan | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Programmatic paraphrases | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The paraphrase diagnostic covers 200 generated rewrites from 100 sampled questions and
retains the standard constraint terms.

## Evaluation Protocol

- Predictions are normalized to sets of drug names.
- Precision, recall, F1, Jaccard, and exact match are computed per question and then macro-averaged.
- BM25, BGE, Direct Qwen, and Semantic-Parser receive natural-language question text at prediction time.
- Semantic-Parser activates constraint categories with predefined lexical and sentence-pattern gates, selects Top-1 within each active category or Top-2 for an explicit two-reagent expression, normalizes explicit volume expressions, and applies no additional similarity threshold. If no category is activated, it returns an empty set.
- OracleStructured, BM25+Filter, and the component ablations read each question's released gold constraints and are therefore controlled analyses, not directly comparable end-to-end systems.
- E2E-Rule uses exact matching against the released constraint inventory. FullCorpusScan uses the same parsed constraints without indexes, and the programmatic-paraphrase test retains the standard constraint terms. These are supplementary diagnostics, not current headline methods.

The supplementary efficiency values (47.3 ms for E2E-Rule and 20,296.1 ms for
FullCorpusScan over all queries, approximately 429.1x) are hardware-dependent and are not
expected to match exactly on another machine.

## Frozen Evaluation View

The public dataset's `data/corpus.jsonl` is the authoritative release format.
`benchmark/resources/eval_corpus.json` contains the same 5,138 record names but preserves
the field layout and placeholder values used by the paper's BM25 and BGE experiments.
It is included only to reproduce the reported rankings and metrics exactly.

Repository-artifact SHA-256 values, computed from the LF-normalized Git blobs:

```text
eval_corpus.json       4F17F313FB9C8E66578BD1F8E4237A9C10F39A21BE77E66B36D5FD017C602F27
qwen_predictions.jsonl 74FD460D794B032850F4C774E1C35FE52996E9EB63BABFFC487B1D526AA45125
paper_summary.json     E0B0C1E3AA228C3564D9B0AA618F66B69A8089A0072A5E48BAB45E3EC6ABFE36
semantic_results.json  59147BC78EBA9D1D4CDE86CED2C3A7C6EAE014E39AEA1238BF5371F4957780DE
```

## Generated Outputs

| Output | Contents |
| --- | --- |
| `classic_results.json` | Overall and per-type metrics, per-question records, parser analysis, controlled results, ablations, and timing. |
| `bge_results.json` | BGE metrics for each Top-K value and per-question results. |
| `semantic_results.json` | Semantic-Parser protocol, aggregate and per-type metrics, hashes, parser errors, and per-question results. |
| `qwen_results.json` | Overall, per-type, and per-question metrics for the normalized Qwen predictions. |
| `paraphrase_results.json` | Generated paraphrases, constraint-match status, and end-to-end metrics. |
| `fig02_answer_distribution.pdf/png` | Distribution of fusion-query answer-set sizes. |
| `fig03_topk_curves.pdf/png` | Precision, recall, and F1 curves for BM25-FullFields and BGE. |

## Troubleshooting

- **Dataset not found:** keep the dataset and code directories as siblings or pass an absolute `--dataset-root`.
- **BGE model cannot be downloaded:** download the pinned model separately and pass its local directory through `--model`.
- **BGE cache is rebuilt:** this is expected when the model, revision, corpus, or queries differ from the cache metadata.
- **CUDA/PyTorch installation fails:** install the PyTorch build recommended for the local CUDA driver, then install `requirements.txt`.
- **Qwen generation refuses to start:** choose a new `--predictions` path or explicitly pass `--overwrite`.
- **Chinese labels use a fallback font:** install Microsoft YaHei, SimHei, or Noto Sans CJK SC before rendering figures.

## Verification

Every executable script in this release was run with the specified project interpreter
before packaging. See `VERIFICATION.md` and
`benchmark/results/verification_summary.json` for commands, versions, exit status, runtime,
and reproduced metrics.

## License and Citation

See `LICENSE`. The current release applies CC BY-NC 4.0 to the distributed benchmark code,
annotations, organization, and accompanying artifacts. Third-party source publications and
model weights retain their original rights and licenses.

Paper metadata is not finalized in this draft release. Before public archival, add the final
authors, year, venue, repository URL, and DOI or preprint identifier here and in a
`CITATION.cff` file.
