# ChPSetQA dataset

This package contains the dataset used in **ChPSetQA: A Benchmark for Complete-Set Fusion Retrieval over the Chinese Pharmacopoeia**, by Dingwei Cai, Yiming Xu, and Tianyong Hao.

It includes 5,138 drug entries, 1,065 complete-set questions with fixed Train/Development/Test splits (639/214/212), 56,709 reference answer pairs, and 200 supplementary single-drug questions.

See [the dataset guide](docs/DATASET.md) for the schema and matching rules. Model inputs are in `data/questions/`; reference annotations are stored separately. The `resources/` directory contains the canonical values, field postings, and matching rules used for deterministic execution.

The code is available at [yxccai/ChinesePharmaQA](https://github.com/yxccai/ChinesePharmaQA). To use the separate data and code archives, extract both into the same parent directory so their `ChPSetQA/` folders merge, then run `python src/verify.py` from that folder. The full archive already contains both.


Project-authored materials retain the [CC BY-NC 4.0 license](LICENSE). Source attribution is provided in [NOTICE.md](NOTICE.md).
