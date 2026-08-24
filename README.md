# ChinesePharmaQA

ChinesePharmaQA is a benchmark dataset and reproducibility toolkit for fusion knowledge retrieval over the Chinese Pharmacopoeia. Given a natural-language query, a system must return the complete set of drug monographs satisfying all stated constraints.

[中文说明](README_zh-CN.md)

Project repository: <https://github.com/yxccai/ChinesePharmaQA>

## Repository contents

```text
ChinesePharmaQA/
├── README.md
├── README_zh-CN.md
├── NOTICE.md
├── RELEASE_MANIFEST.md
├── .gitignore
├── ChinesePharmaQA/
│   ├── data/
│   └── scripts/
└── ChinesePharmaQA-Code-v1.0.1/
    └── benchmark/
```

- `ChinesePharmaQA/` contains the frozen evaluation dataset (v1.0.0): 5,138 corpus records, 1,065 fusion queries, 200 single-drug questions, and 48,335 relevance judgments.
- `ChinesePharmaQA-Code-v1.0.1/` contains the benchmark evaluation and figure-reproduction code, including Semantic-Parser, frozen results, and verification records.
- The original Chinese Pharmacopoeia PDFs, OCR intermediates, model weights, vector caches, and raw API responses are not included.

## Dataset

The dataset is evaluation-only and does not define a training split. The main task is set-valued retrieval: a system receives a natural-language query and returns all records satisfying the stated reagent, instrument/method, property, or volume constraints.

Run the dataset checks from the repository root:

```bash
python ChinesePharmaQA/scripts/validate.py
python ChinesePharmaQA/scripts/rebuild_qrels.py
```

The second command deterministically reconstructs the released qrels from the frozen corpus and query constraints.

## Reproduce the benchmark

The code package README contains the complete command reference. The classic evaluation can be launched as follows:

```bash
python ChinesePharmaQA-Code-v1.0.1/benchmark/evaluate_classic.py \
  --dataset-root ChinesePharmaQA \
  --output reproduced/classic_results.json
```

After installing the listed BGE dependencies and an appropriate PyTorch build, reproduce the task-specific method with:

```bash
python ChinesePharmaQA-Code-v1.0.1/benchmark/evaluate_semantic.py \
  --dataset-root ChinesePharmaQA \
  --model BAAI/bge-large-zh-v1.5 \
  --revision 79e7739b6ab944e86d6171e44d24c997fc1e0116 \
  --output reproduced/semantic_results.json
```

Offline scoring of the released Qwen predictions and the supplementary programmatic-paraphrase diagnostic do not require network access. BGE-based evaluation requires the dependencies listed in `ChinesePharmaQA-Code-v1.0.1/benchmark/requirements.txt`. A fresh Qwen API run is optional and incurs external API cost; the released offline predictions are the reproducibility target.

## Reported headline results

The frozen paper evaluation covers all 1,065 fusion queries. The released verification record reports the following macro-averaged results:

| Method | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| BM25-Original, K=50 | 0.2260 | 0.4591 | 0.2557 | 0.0000 |
| BM25-FullFields, K=50 | 0.2336 | 0.4739 | 0.2646 | 0.0000 |
| BGE, K=200 | 0.0388 | 0.2095 | 0.0523 | 0.0000 |
| Direct Qwen-Plus | 0.0125 | 0.0109 | 0.0084 | 0.0000 |
| Semantic-Parser | 0.6732 | 0.8185 | 0.6665 | 0.5577 |

Controlled analyses that read released gold constraints are reported separately in the code package README and should not be interpreted as ordinary end-to-end comparisons. E2E-Rule, FullCorpusScan, and programmatic paraphrases remain available only as supplementary diagnostics and are not part of the current paper's main method comparison.

## Versioning

- Dataset: `v1.0.0`
- Reproduction code: `v1.0.1`
- Combined release archive: `v1.0.1`

The dataset version is unchanged while the code package contains later reproducibility fixes and verification records.
The complete current file inventory and authoritative artifact hashes are listed in [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md).

## License and source notice

See the package-level `LICENSE` files and [NOTICE.md](NOTICE.md). The project licenses cover the project-authored annotations, organization, scripts, and documentation within their stated scope. Rights in third-party source publications remain with their respective rights holders.

## Citation

The final author, venue, DOI, and citation metadata will be added when the paper version and public archival record are finalized.
