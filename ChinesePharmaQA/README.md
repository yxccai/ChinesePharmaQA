# ChinesePharmaQA

[中文说明](README_zh-CN.md)

ChinesePharmaQA is an evaluation benchmark for set-valued retrieval over structured records from Volumes I and II of the 2020 Chinese Pharmacopoeia. Given a natural-language query, a system must return the complete set of drug monographs that satisfies every stated constraint.

## Release summary

- Version: v1.0.0
- Corpus: 5,138 records (2,425 from Volume I; 2,713 from Volume II)
- Fusion queries: 1,065
- Single-drug QA examples: 200
- Relevance judgments: 48,335
- Fusion query types: A 43, B 15, C 25, D 982
- Gold-set size: 8 to 595 drugs (mean 45.38; median 18)

This is an evaluation-only release. It does not define a training split.

## Files

```text
ChinesePharmaQA/
├── README.md
├── README_zh-CN.md
├── LICENSE
├── data/
│   ├── corpus.jsonl
│   ├── fusion_queries.jsonl
│   ├── fusion_qrels.tsv
│   └── single_drug_qa.jsonl
└── scripts/
    ├── rebuild_qrels.py
    └── validate.py
```

- `data/corpus.jsonl`: structured pharmacopoeia records and retrieval text.
- `data/fusion_queries.jsonl`: set-valued retrieval queries and their structured constraints.
- `data/fusion_qrels.tsv`: query-to-corpus relevance judgments.
- `data/single_drug_qa.jsonl`: single-record field questions used as a comparison task.
- `scripts/rebuild_qrels.py`: deterministic qrels reconstruction from the released corpus and constraints.
- `scripts/validate.py`: structural and cross-file validation.

## Tasks

| Type | Task |
| --- | --- |
| A | Return all drugs whose inspection-related fields contain a specified reagent. |
| B | Return all drugs associated with a specified instrument or analytical method. |
| C | Return all drugs whose appearance description contains a specified property phrase. |
| D | Return the intersection defined by two or three reagent, instrument/method, property, or volume constraints. |
| S | Answer one field-level question about a specified drug record. |

Fusion-query gold labels use deterministic conjunction over the frozen corpus. Reagent matching is literal occurrence in `鉴别`, `检查`, `含量测定`, or `浸出物`; instrument/method matching additionally uses linked general-rule names; property matching uses `性状`; volume matching is exact.

## Data format

Each corpus row contains `_id`, `title`, `text`, and `metadata`. IDs follow the forms `cpqa-c-xxxxx`, `cpqa-fq-xxxxx`, and `cpqa-sq-xxxxx` and serve as cross-file join keys.

Each fusion query contains `_id`, `text`, `type`, `type_name`, `constraints`, and `difficulty`. Gold answers are stored only in `fusion_qrels.tsv`.

For end-to-end evaluation, systems should use only `text` as the query input. The remaining fields are annotations for analysis and label reproducibility and should not be supplied to the evaluated model.

Each single-drug row contains `_id`, `question`, `answer`, `corpus_id`, `drug`, `field`, and `difficulty`.

`fusion_qrels.tsv` has three tab-separated columns: `query-id`, `corpus-id`, and `score`. Every released relevance score is `1`.

## Validation

From the release root:

```bash
python scripts/validate.py
python scripts/rebuild_qrels.py
```

Both scripts use only the Python standard library.

## Construction and quality control

The corpus was derived from Volumes I and II of the 2020 Chinese Pharmacopoeia through structured extraction, field normalization, and explicit linking to referenced general rules. Gold labels were generated against the frozen release corpus and checked for cross-file consistency.

## License and use

See `LICENSE`. ChinesePharmaQA is intended for non-commercial research and education. It is a benchmark dataset, not medical advice or a substitute for the official pharmacopoeia.

## Paper

ChinesePharmaQA: A Benchmark Dataset for Fusion Knowledge Retrieval over the Chinese Pharmacopoeia / 面向中国药典融合性知识查询的问答数据集构建与基准评测.
