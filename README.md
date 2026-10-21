# ChPSetQA

**ChPSetQA: A Benchmark for Complete-Set Fusion Retrieval over the Chinese Pharmacopoeia**

Dingwei Cai, Yiming Xu, and Tianyong Hao · South China Normal University

[中文说明](README_zh-CN.md) · [Dataset](docs/DATASET.md) · [Reproduction guide](docs/REPRODUCING.md)

ChPSetQA evaluates whether a system can return **all and only the drugs satisfying a question's conditions**. It contains 1,065 Chinese queries over 5,138 drug entries from Volumes I and II of the 2020 Chinese Pharmacopoeia.

For example, “检验过程中需使用高氯酸的药品是哪些？” asks for the complete set of qualifying entries. Multi-condition questions combine reagent, instrument/method, Description, and volume conditions through conjunction. Predictions are evaluated as drug-ID sets using micro precision, recall, F1, and exact match (EM).

Adaptive Evidence PharmaParser first parses the question into canonical constraints. It validates those constraints, retrieves field-specific evidence to repair unresolved values, and executes the resulting conditions through deterministic set intersection.

## Quick start

Clone the repository and reproduce the reported scores without model downloads or API calls:

```bash
git clone https://github.com/yxccai/ChinesePharmaQA.git
cd ChinesePharmaQA
python src/verify.py
python src/evaluation.py --split test
python src/evaluation.py --split dev
```

These commands use the Python standard library. To rebuild every reference answer set from the corpus:

```bash
python src/rebuild_references.py
```

## Dataset

| Item | Count |
|---|---:|
| Drug entries | 5,138 |
| Complete-set queries | 1,065 |
| Train / Development / Test | 639 / 214 / 212 |
| Query–drug reference pairs | 56,709 |
| Supplementary single-drug field questions | 200 |

The split is fixed. `data/questions/` contains question-only inputs, while `data/queries.jsonl` and `data/reference_sets.json` contain annotations for evaluation. The supplementary single-drug questions are separate from the complete-set experiments.

See [the dataset guide](docs/DATASET.md) for record formats and field-matching rules.

## Test results

| Method | Micro P | Micro R | Micro F1 | EM |
|---|---:|---:|---:|---:|
| Random-K | 0.0114 | 0.1196 | 0.0208 | 0.0000 |
| BM25 | 0.2058 | 0.1800 | 0.1921 | 0.0000 |
| BGE | 0.0605 | 0.1058 | 0.0770 | 0.0000 |
| RAG-Filter | 0.2734 | 0.1133 | 0.1602 | 0.0000 |
| Direct LLM | 0.1280 | 0.0083 | 0.0156 | 0.0047 |
| LLM-Parser | 0.8511 | 0.3643 | 0.5103 | 0.4670 |
| RAG+QD | 0.9267 | 0.5531 | 0.6927 | **0.6651** |
| Adaptive Evidence PharmaParser | **0.9340** | **0.5963** | **0.7279** | 0.6509 |

Adaptive Evidence PharmaParser improves recall and F1 over RAG+QD, while RAG+QD has a small EM lead. Random-K scores are averaged over five seeds. Per-query predictions, Development comparisons, answer-size results, and retrieval-depth results are provided in `results/`.

## Run the methods

Use Python 3.10 or later. For fresh inference, install the dependencies in `requirements.txt` with a suitable PyTorch build, then build the retrieval indexes:

```bash
pip install -r requirements.txt
python src/build_indexes.py --device cuda
```

For LLM methods, set `DASHSCOPE_API_KEY` in your environment. All paper methods use `qwen3.7-flash`; fresh API inference incurs provider charges. To run the proposed method:

```bash
python src/run.py --method Ours --split test --device cuda --output outputs/ours_test.json
python src/evaluation.py --split test --predictions outputs/ours_test.json
```

The [reproduction guide](docs/REPRODUCING.md) covers all baselines, reusable first-pass predictions, and Development ablations. Saved predictions reproduce the paper's results exactly; fresh API outputs may vary with the model service.

## Repository layout

```text
data/          Corpus, fixed splits, questions, annotations, and reference sets
resources/     Canonical values, field postings, and matching rules
src/           Parsing, retrieval, repair, execution, and evaluation
configs/       Paper settings
results/       Predictions and results reported in the paper
tests/         Offline method and evaluation tests
docs/          Data formats and reproduction instructions
```

## Citation

```bibtex
@unpublished{cai2026chpsetqa,
  title  = {ChPSetQA: A Benchmark for Complete-Set Fusion Retrieval over the Chinese Pharmacopoeia},
  author = {Cai, Dingwei and Xu, Yiming and Hao, Tianyong},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/yxccai/ChinesePharmaQA}
}
```

## License

Project-authored materials retain the existing [CC BY-NC 4.0 license](LICENSE). See [NOTICE.md](NOTICE.md) for source attribution. The original pharmacopoeia PDFs and model weights are not distributed in this repository.
