# Dataset guide

ChPSetQA pairs Chinese questions with complete reference sets over a fixed collection of 5,138 drug entries. The collection comes from Volumes I and II of the 2020 Chinese Pharmacopoeia.

## Files

| File | Contents |
|---|---|
| `data/corpus.jsonl` | Drug ID, title, source text, volume, structured sections, and chapter references |
| `data/queries.jsonl` | 1,065 questions with subtype, canonical constraints, and split |
| `data/questions/{train,dev,test}.jsonl` | Question ID and natural-language text for model input |
| `data/splits.json` | Fixed Train, Development, and Test IDs |
| `data/reference_sets.json` | Complete reference drug IDs for each query, including empty sets |
| `data/fusion_qrels.tsv` | Equivalent query–drug pairs in TSV format |
| `data/single_drug_qa.jsonl` | 200 supplementary single-drug field questions |
| `resources/ontology/active_concepts.json` | Canonical values used for exact executability validation |
| `resources/ontology/postings.json` | Drug sets for individual canonical field conditions |

The 639/214/212 split is fixed and stratified by query subtype. Development is used for configuration selection, and Test is used for final evaluation. The methods in the paper do not fine-tune an LLM on Train.

## Question format

```json
{
  "_id": "cpqa-fq-00001",
  "text": "检验过程中需使用高氯酸的药品是哪些？",
  "subtype": "R",
  "constraints": {"reagent": "高氯酸"},
  "split": "dev"
}
```

Model inputs are the `_id` and `text` fields. Canonical constraints and answer sets are supplied separately for evaluation and data reconstruction.

| Subtype | Conditions | Queries |
|---|---|---:|
| R | Reagent | 43 |
| IM | Instrument/method | 15 |
| P | Description | 25 |
| M1 | Reagent + reagent | 240 |
| M2 | Volume + reagent | 91 |
| M3 | Volume + instrument/method | 23 |
| M4 | Description + instrument/method | 84 |
| M5 | Description + reagent | 175 |
| M6 | Instrument/method + reagent | 204 |
| M7 | Volume + Description | 11 |
| M8 | Volume + reagent + reagent | 154 |

Dual-reagent annotations use `reagent1` and `reagent2`; the parser represents them as two constraints with field `reagent`.

## Reference sets

Reference sets are obtained by intersecting the drug sets for all annotated conditions. Reagent matching uses Identification, Tests, Assay, and Extractives. Instrument/method matching additionally uses explicit chapter references. Description conditions use the Description section, and volume conditions use volume metadata.

The implementation normalizes layout whitespace and Unicode width, applies the specified reagent-name boundaries, and recognizes the infrared expressions in `resources/matching_rules.json`. Chapter titles are recorded in `resources/chapter_registry.json`. Matching rules were reviewed against source text, with sample checks of answer-set completeness.

To reconstruct the field postings and all reference sets:

```bash
python src/rebuild_references.py
```

The released annotations contain 56,709 query–drug pairs and four empty reference sets. Empty sets are represented explicitly in `reference_sets.json`; they have no rows in the TSV qrels.

## Evaluation

For each query, predictions are interpreted as a set of corpus IDs. Micro precision and recall pool true positives, false positives, and false negatives across queries. Micro F1 is their harmonic mean. EM is the proportion of queries whose predicted and reference sets are equal. A zero denominator gives a zero precision or recall; two empty sets have EM = 1.

Prediction files map method names to rows:

```json
{"My method": [{"query_id": "cpqa-fq-00001", "answer_ids": ["cpqa-c-00001"]}]}
```

The IDs above illustrate the format, not a reference answer. Full evaluation requires exactly one prediction for every query in the selected split.
