# ChinesePharmaQA 论文复现代码

[English README](README.md)

本代码包用于复现论文《面向中国药典融合性知识查询的问答数据集构建与基准评测》中的基准实验和数据驱动图表。

ChinesePharmaQA 评测的是集合式检索能力：给定“哪些药品使用某种试剂”一类自然语言问题，系统需要返回同时满足全部约束的完整药品专论集合。配套数据集包含 5,138 条结构化记录、1,065 条融合查询、200 条单药问答和 48,335 条查询-语料相关性标注。

本代码包的主实验对象是 1,065 条融合查询。数据集中的 200 条单药问答作为对照任务公开，但当前代码包没有对其运行基准评测。

## 可以复现的内容

- Random、BM25-Original 和 BM25-FullFields 检索基线。
- 基于发布词表的规则解析、全库扫描、结构化上限、BM25+Filter 和受控消融。
- 基于 BGE-large-zh-v1.5 与 FAISS 的稠密检索。
- 对随包发布的脱敏 Qwen-Plus 预测进行离线重新评分。
- 可选的 Qwen-Plus 在线重新生成实验。
- 论文中报告的 200 条程序化问句改写测试。
- 根据冻结论文汇总生成图2和图3。

本包面向基准评测和图表复现，不包含原始药典 PDF、OCR 中间文件、模型权重或向量索引。图1是人工整理的完整流程图，不由 `make_figures.py` 生成。

## 目录结构

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
    ├── evaluate_classic.py
    ├── evaluate_bge.py
    ├── evaluate_qwen.py
    ├── evaluate_paraphrases.py
    ├── make_figures.py
    ├── resources/
    │   └── eval_corpus.json
    └── results/
        ├── paper_summary.json
        ├── qwen_predictions.jsonl
        └── verification_summary.json
```

## 文件逐项说明

| 文件 | 作用 |
| --- | --- |
| `benchmark/common.py` | 公共模块，负责数据加载、中文切词、BM25、结构化倒排索引、问句解析、指标计算、哈希和 JSON I/O；它没有独立命令行入口。 |
| `benchmark/evaluate_classic.py` | 运行 Random、两种 BM25、基于发布词表的 E2E-Rule、FullCorpusScan、OracleStructured、BM25+Filter、解析统计和结构化组件消融。 |
| `benchmark/evaluate_bge.py` | 使用 BGE 编码冻结语料和问题，构建 FAISS 内积索引并评价多个 Top-K；缓存元数据会绑定模型、revision、语料和问题，避免错误复用旧向量。 |
| `benchmark/evaluate_qwen.py` | 默认仅对随包发布的脱敏预测重新评分，不联网；加入 `--generate` 后才调用 DashScope 生成新的 Qwen 结果。 |
| `benchmark/evaluate_paraphrases.py` | 固定随机种子 42，分层抽取 100 个融合问题，每题生成 2 条改写，并评价约束解析和最终集合答案。 |
| `benchmark/make_figures.py` | 读取冻结的 `paper_summary.json`，生成图2和图3的 PDF/PNG；不会自动汇总刚运行得到的实验 JSON。 |
| `benchmark/resources/eval_corpus.json` | 论文基线实际使用的 5,138 条冻结字段视图，用于逐值复现，不是公开数据集的主语料文件。 |
| `benchmark/results/paper_summary.json` | 冻结的论文指标、消融、效率和绘图数据；它是发布产物，不是某个脚本自动合并生成的结果。 |
| `benchmark/results/qwen_predictions.jsonl` | 1,065 条脱敏预测，仅包含 `query_id`、`pred_names` 和 `unmapped_count`，不包含模型原始回答。 |
| `benchmark/results/verification_summary.json` | 机器可读的发布验证记录和关键复现指标。 |
| `VERIFICATION.md` | 面向读者的验证报告，说明每个脚本是否实际运行、使用的环境和复现结果。 |

## 快速开始：离线复现

将数据包和代码包解压为同级目录：

```text
parent/
├── ChinesePharmaQA/
└── ChinesePharmaQA-Code-v1.0.1/
```

在代码包根目录执行以下命令。除绘图外，这一组命令不需要联网、API Key 或第三方 Python 包。

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

根据冻结论文汇总生成两张数据图：

```bash
python benchmark/make_figures.py \
  --summary benchmark/results/paper_summary.json \
  --output-dir reproduced/paper_figures
```

`make_figures.py` 固定读取论文汇总，不读取 `reproduced/classic_results.json`、`bge_results.json` 或 `qwen_results.json`。

## 环境与 BGE 复现

本版本使用 Python 3.13.1 完成验证。经典基线、Qwen 离线评分和改写评测只使用标准库；绘图需要 Matplotlib；BGE 还需要 NumPy、FAISS、Sentence Transformers 和 PyTorch。

先根据本机 CPU/CUDA 环境安装合适的 PyTorch，再安装其余固定版本依赖：

```bash
pip install -r benchmark/requirements.txt
```

论文实验环境为：

- `numpy==2.3.3`
- `sentence-transformers==5.1.2`
- `faiss-cpu==1.12.0`
- `torch==2.6.0+cu118`
- `dashscope==1.25.2`
- `matplotlib==3.10.8`

使用论文模型和 revision 运行 BGE：

```bash
python benchmark/evaluate_bge.py \
  --dataset-root ../ChinesePharmaQA \
  --model BAAI/bge-large-zh-v1.5 \
  --revision 79e7739b6ab944e86d6171e44d24c997fc1e0116 \
  --cache-dir reproduced/bge_cache \
  --output reproduced/bge_results.json
```

首次运行会下载或加载模型并生成向量。缓存带有元数据指纹；模型、revision、冻结语料或问题文本变化时会自动重建。也可以使用 `--rebuild-cache` 强制重新编码。

## 可选：重新调用 Qwen-Plus

前面的 Qwen 命令是离线评分，不会访问网络。在线重新生成属于新的实验，会产生 API 费用，而且模型回答可能与论文时期不同。

在本机设置 `DASHSCOPE_API_KEY`，并使用新的输出路径：

```bash
python benchmark/evaluate_qwen.py \
  --dataset-root ../ChinesePharmaQA \
  --generate \
  --model qwen-plus \
  --predictions reproduced/qwen_new_predictions.jsonl \
  --output reproduced/qwen_new_results.json
```

预测文件、评分结果和可选原始回答必须使用不同路径；在线生成的任一目标文件已存在时，脚本都会拒绝覆盖，只有显式加入 `--overwrite` 才允许替换。除非另行指定 `--raw-output`，否则不会保存原始回答。

## 预期结果

下表是在全部 1,065 条融合查询上计算的集合指标宏平均值。

| 方法 | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| BM25-Original，K=50 | 0.2260 | 0.4591 | 0.2557 | 0.0000 |
| BM25-FullFields，K=50 | 0.2336 | 0.4739 | 0.2646 | 0.0000 |
| BGE，K=200 | 0.0388 | 0.2095 | 0.0523 | 0.0000 |
| Direct Qwen-Plus | 0.0125 | 0.0109 | 0.0084 | 0.0000 |
| E2E-Rule | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 程序化问句改写 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

改写实验从 100 个问题生成 200 条改写，200 条约束均被完整识别。

下列方法逐题读取了发布的标准约束，只能作为受控分析单独解释：

| 受控设置 | Precision | Recall | F1 | EM |
| --- | ---: | ---: | ---: | ---: |
| BM25+Filter，K=200 | 0.9991 | 0.7995 | 0.8591 | 0.3878 |
| BM25+Filter，K=500 | 0.9991 | 0.9262 | 0.9504 | 0.6939 |
| BM25+Filter，K=1000 | 1.0000 | 0.9772 | 0.9856 | 0.8789 |
| OracleStructured | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 评测协议

- 所有预测统一转换为药品名称集合。
- 每题计算 Precision、Recall、F1、Jaccard 和完全匹配率，再对 1,065 题做宏平均。
- BM25、BGE 和 Direct Qwen 在预测阶段输入自然语言问题文本。
- E2E-Rule 属于“发布词表/结构模式辅助”的方法：逐题预测时只接收问题文本，但解析词表由发布的约束清单构建，执行索引由冻结语料构建。
- OracleStructured、BM25+Filter 和组件消融会读取当前问题的标准约束，因此是受控分析，不与自然语言端到端系统直接比较。
- FullCorpusScan 使用与 E2E-Rule 相同的解析结果，只用于评价无索引扫描的效率。

论文运行环境中，E2E-Rule 处理全部问题耗时 47.3 ms，FullCorpusScan 耗时 20,296.1 ms，约为 429.1 倍差异。耗时受硬件影响，不要求在其他机器上逐值一致。

## 冻结评测视图

公开数据集中的 `data/corpus.jsonl` 是正式发布格式。`benchmark/resources/eval_corpus.json` 包含相同的 5,138 个记录名称，但保留论文 BM25/BGE 实验实际使用的字段布局和占位值，仅用于精确恢复当时的文本序列化、排序和指标。

随包冻结产物的 SHA-256：

```text
eval_corpus.json       837471973343F864F605891312A8B50ECB872962271F9CC4A332DAE801087693
qwen_predictions.jsonl 74FD460D794B032850F4C774E1C35FE52996E9EB63BABFFC487B1D526AA45125
paper_summary.json     146EDE86B0B149F9318468CE94ECC9A68B2468FB2DA726372F5535854A91EAC1
```

## 各脚本输出

| 输出 | 内容 |
| --- | --- |
| `classic_results.json` | 总体/分类指标、逐题结果、解析统计、受控结果、消融和耗时。 |
| `bge_results.json` | 各 Top-K 的 BGE 指标和逐题结果。 |
| `qwen_results.json` | 脱敏 Qwen 预测的总体、分类和逐题指标。 |
| `paraphrase_results.json` | 生成的改写、约束识别状态和端到端指标。 |
| `fig02_answer_distribution.pdf/png` | 融合查询答案规模分布。 |
| `fig03_topk_curves.pdf/png` | BM25-FullFields 与 BGE 的 Precision、Recall、F1 曲线。 |

## 常见问题

- **找不到数据集：**将数据目录和代码目录放在同一级，或显式传入 `--dataset-root`。
- **无法下载 BGE：**提前下载固定 revision 的模型，并通过 `--model` 传入本地目录。
- **BGE 重新生成缓存：**模型、revision、语料或问题与缓存元数据不同就会自动重建，这是正常行为。
- **CUDA/PyTorch 安装失败：**先安装与本机显卡驱动匹配的 PyTorch，再安装 `requirements.txt`。
- **Qwen 在线生成拒绝开始：**使用新的 `--predictions` 路径，或明确加入 `--overwrite`。
- **图中的中文字体使用了替代字体：**安装 Microsoft YaHei、SimHei 或 Noto Sans CJK SC 后重新绘图。

## 运行验证

重新打包前，发布包中的每个可执行脚本都使用项目指定解释器实际运行。命令、依赖版本、退出状态、耗时和关键指标见 `VERIFICATION.md` 与 `benchmark/results/verification_summary.json`。

## 许可与引用

许可见 `LICENSE`。当前版本对随包发布的评测代码、数据标注组织和配套产物采用 CC BY-NC 4.0；第三方出版物和模型权重仍遵循其原始权利与许可。

当前论文仍处于初稿阶段，正式作者、年份、发表信息和 DOI/预印本链接尚未确定。公开归档前应在此补全正式 BibTeX，并增加 `CITATION.cff`。
