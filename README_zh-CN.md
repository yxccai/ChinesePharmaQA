# ChPSetQA

**ChPSetQA: A Benchmark for Complete-Set Fusion Retrieval over the Chinese Pharmacopoeia**

Dingwei Cai、Yiming Xu、Tianyong Hao · 华南师范大学

[English](README.md) · [数据说明](docs/DATASET.md) · [复现指南](docs/REPRODUCING.md)

ChPSetQA 面向药典跨条目集合查询，要求系统返回**满足问题全部条件的完整药品集合**。数据集以 2020 年版《中国药典》一、二部的 5,138 条药品条目为语料，包含 1,065 道中文集合查询。

例如，“检验过程中需使用高氯酸的药品是哪些？”要求找出所有符合条件的药品。多约束问题进一步组合试剂、仪器/方法、性状和药典分部条件。评测使用药品 ID 集合，报告微平均 Precision、Recall、F1 和整题完全匹配率 EM。

本文方法 Adaptive Evidence PharmaParser 先将问句解析为规范约束，再通过精确校验识别尚不可执行的约束，按字段检索药典证据进行修复，最后以确定性集合运算生成答案。

## 快速使用

以下命令可直接验证数据并复现论文结果，无需下载模型或调用 API：

```bash
git clone https://github.com/yxccai/ChinesePharmaQA.git
cd ChinesePharmaQA
python src/verify.py
python src/evaluation.py --split test
python src/evaluation.py --split dev
```

这些命令仅依赖 Python 标准库。还可以从语料重新构建全部字段集合与标准答案：

```bash
python src/rebuild_references.py
```

## 数据概况

| 内容 | 数量 |
|---|---:|
| 药品条目 | 5,138 |
| 集合查询 | 1,065 |
| Train / Development / Test | 639 / 214 / 212 |
| 查询—药品标准答案关系 | 56,709 |
| 补充单药字段问答 | 200 |

`data/questions/` 提供仅含问句和 ID 的模型输入；`data/queries.jsonl` 与 `data/reference_sets.json` 保存评测标注。200 道单药字段问答单独提供，不纳入集合查询的主实验。

## Test 结果

| 方法 | Micro P | Micro R | Micro F1 | EM |
|---|---:|---:|---:|---:|
| Random-K | 0.0114 | 0.1196 | 0.0208 | 0.0000 |
| BM25 | 0.2058 | 0.1800 | 0.1921 | 0.0000 |
| BGE | 0.0605 | 0.1058 | 0.0770 | 0.0000 |
| RAG-Filter | 0.2734 | 0.1133 | 0.1602 | 0.0000 |
| Direct LLM | 0.1280 | 0.0083 | 0.0156 | 0.0047 |
| LLM-Parser | 0.8511 | 0.3643 | 0.5103 | 0.4670 |
| RAG+QD | 0.9267 | 0.5531 | 0.6927 | **0.6651** |
| Adaptive Evidence PharmaParser | **0.9340** | **0.5963** | **0.7279** | 0.6509 |

本文方法的 Recall 和 F1 高于 RAG+QD，RAG+QD 的 EM 略高。Random-K 取五个随机种子的平均结果。逐题预测、Development 消融、答案规模分析与 Top-K 分析均位于 `results/`。

## 运行方法

使用 Python 3.10 或以上版本。重新推理前，安装 `requirements.txt` 中的依赖及适合设备的 PyTorch，再构建检索索引：

```bash
pip install -r requirements.txt
python src/build_indexes.py --device cuda
```

LLM 方法通过环境变量 `DASHSCOPE_API_KEY` 读取密钥，统一使用 `qwen3.7-flash`。重新调用模型会产生 API 费用。本文方法的运行示例：

```bash
python src/run.py --method Ours --split test --device cuda --output outputs/ours_test.json
python src/evaluation.py --split test --predictions outputs/ours_test.json
```

更多基线、首次解析结果复用和消融命令见[复现指南](docs/REPRODUCING.md)。保存的预测可精确复现论文分数；重新调用在线模型时，输出可能随模型服务变化。

## 文件组织

`data/` 保存数据，`resources/` 保存规范值和匹配规则，`src/` 提供方法与评测代码，`configs/` 保存实验设置，`results/` 保存论文结果，`tests/` 提供离线测试，`docs/` 说明数据格式和复现步骤。

引用格式见[英文首页](README.md#citation)。项目原创内容沿用 [CC BY-NC 4.0](LICENSE)，来源说明见 [NOTICE.md](NOTICE.md)。仓库不包含药典原始 PDF 和模型权重。
