# Reproducing the paper

## Reported results

Use the released predictions to reproduce the paper without API calls:

```bash
python src/verify.py
python src/evaluation.py --split test
python src/evaluation.py --split dev
python src/rebuild_references.py
```

`verify.py` checks the fixed splits, reference sets, qrels, reported scores, Top-K scores, and deterministic execution of the saved model outputs. `results/answer_size_results.json` contains the Test answer-size breakdown. `results/topk_results.json` contains Development retrieval-depth results.

## Fresh inference

Install a suitable PyTorch build and the packages in `requirements.txt`. The release was checked with NumPy 2.3.3, sentence-transformers 5.1.2, PyTorch 2.6.0, and DashScope 1.25.2. Set `DASHSCOPE_API_KEY` as an environment variable for LLM methods. The default endpoint is Alibaba Cloud DashScope; `DASHSCOPE_COMPATIBLE_BASE_URL` can override its base URL.

```bash
python src/build_indexes.py --device cuda
```

This builds section-level retrieval chunks and field-level evidence records, using the pinned BGE revision in `configs/paper.json`. Use `--model-path /path/to/bge-large-zh-v1.5` for a local copy of that revision, or `--device cpu` for CPU execution. Generated indexes and embeddings are stored in `cache/`.

For BM25 alone, text indexes suffice:

```bash
python src/build_indexes.py --text-only
python src/run.py --method BM25 --split test --output outputs/bm25_test.json
```

The method choices are `Random-K`, `BM25`, `BGE`, `RAG-Filter`, `Direct LLM`, `LLM-Parser`, `RAG+QD`, and `Ours`. For example:

```bash
python src/run.py --method RAG+QD --split test --device cuda --output outputs/qd_test.json
python src/evaluation.py --split test --predictions outputs/qd_test.json
```

All LLM methods use `qwen3.7-flash`, temperature 0, and top-p 1. BM25, BGE, and Random-K use K = 50, 100, and 600. RAG-Filter uses 600 candidates in batches of 50. RAG+QD retrieves with the original question and extracted condition spans, merges by maximum cosine similarity, and retains ten passages without a reranker.

LLM-Parser, RAG+QD, and all proposed-method variants share a fallback: an empty core result is supplemented by RAG-Filter candidates intersected with any validated predicted conditions.

Each run writes a prediction file for evaluation and a `.core.json` file containing the corresponding core model outputs. A new output path is required for each run. Use `--limit 2` for a small smoke run; partial files are not accepted as full-split evaluations.

## Reuse parsing and filtering

LLM-Parser and Ours share their first parsing stage. Save it once and reuse its `.core.json` output. The same applies to RAG-Filter:

```bash
python src/run.py --method RAG-Filter --split dev --device cuda --output outputs/filter_dev.json
python src/run.py --method LLM-Parser --split dev --rag-predictions outputs/filter_dev.core.json --output outputs/parser_dev.json
python src/run.py --method Ours --split dev --device cuda --parser-predictions outputs/parser_dev.core.json --rag-predictions outputs/filter_dev.core.json --output outputs/ours_dev.json
```

The third command reuses the initial parser predictions and invokes evidence repair only where needed. The saved outputs in `results/` reproduce the reported scores; these commands perform fresh inference.

## Development comparisons

| Command value for `--method` | Change |
|---|---|
| `w/o Evidence Agent` | Use the shared first-pass parser without repair |
| `w/o Evidence Agent (+ Feedback-only Repair)` | Replace evidence repair with one validation-feedback repair call per unresolved query |
| `w/o Field-aware Retrieval` | Retrieve evidence from all fields |
| `w/o Re-grounding` | Use only the initial evidence and repair decision |

Use these methods with `--split dev` and the same parser and RAG-Filter outputs as above. Feedback-only Repair retains valid first-pass constraints and repairs unresolved ones without evidence or a canonical-value menu in its repair prompt.

| Configuration | Exact | EM |
|---|---:|---:|
| Full Adaptive Evidence PharmaParser | 142/214 | 0.6636 |
| w/o Evidence Agent | 104/214 | 0.4860 |
| w/o Evidence Agent (+ Feedback-only Repair) | 122/214 | 0.5701 |
| w/o Field-aware Retrieval | 119/214 | 0.5561 |
| w/o Re-grounding | 138/214 | 0.6449 |

For retrieval-depth sensitivity, run `Ours` on Development with `--evidence-k 1`, `3`, `5`, or `10`. The default is five. The published prompts are retained for all four settings.

## Random-K averaging

Random-K writes `.seeds.json` with separate predictions for seeds 11, 22, 33, 44, and 55. Score each seed and average its micro P/R/F1 and EM, as in the paper. The default Test evaluation command performs this averaging for the released predictions.

```bash
python src/run.py --method Random-K --split test --output outputs/random_test.json
python src/evaluation.py --split test --seeds outputs/random_test.seeds.json
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests exercise empty-set scoring, the parser input schema, and replay of 640 saved Development/Test repair outputs without network access.
