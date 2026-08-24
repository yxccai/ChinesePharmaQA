# ChinesePharmaQA

[English README](README.md)

ChinesePharmaQA 是面向《中国药典》2020年版结构化记录的集合式检索评测基准。给定自然语言问题，系统需要返回同时满足全部约束的完整药品专论集合。

## 发布统计

- 版本：v1.0.0
- 语料记录：5,138 条（一部 2,425 条，二部 2,713 条）
- 融合查询：1,065 条
- 单药问答：200 条
- 相关性标注：48,335 条
- 融合查询类型：A 43、B 15、C 25、D 982
- 标准答案规模：8 至 595 种药品（均值 45.38，中位数 18）

本版本仅用于评测，不提供训练集划分。

## 目录

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

- `data/corpus.jsonl`：结构化药典记录及检索文本。
- `data/fusion_queries.jsonl`：融合查询、类型与结构化约束。
- `data/fusion_qrels.tsv`：查询与语料记录之间的标准相关性标注。
- `data/single_drug_qa.jsonl`：作为对照任务的单药字段问答。
- `scripts/rebuild_qrels.py`：根据公开语料和约束确定性重建 qrels。
- `scripts/validate.py`：检查文件结构及跨文件一致性。

## 任务

| 类型 | 任务 |
| --- | --- |
| A | 返回检验相关字段中含指定试剂的全部药品。 |
| B | 返回使用指定仪器或分析方法的全部药品。 |
| C | 返回性状描述中含指定性质短语的全部药品。 |
| D | 对试剂、仪器/方法、性状或分部中的两至三个约束取交集。 |
| S | 回答指定药品记录中的一个字段问题。 |

论文正文为便于阅读使用 `R`、`IM`、`P`、`M`、`S` 表示上述类型；数据文件和复现代码中的 `type` 字段保留 `A`、`B`、`C`、`D`、`S` 作为内部类型值。两套记号的对应关系为：`R=A`（试剂聚合）、`IM=B`（仪器/方法聚合）、`P=C`（性状聚合）、`M=D`（多约束组合）、`S=S`（单药字段问答）。
该对应关系仅涉及类型记号，不改变问题文本、结构化约束、标准答案或评测逻辑。

融合查询的标准答案由冻结语料上的确定性合取生成：试剂在 `鉴别`、`检查`、`含量测定`、`浸出物` 中按字面匹配；仪器/方法还会匹配显式关联的通则名称；性状在 `性状` 字段中按字面匹配；分部采用精确匹配。

## 数据格式

语料记录包含 `_id`、`title`、`text` 和 `metadata`。语料、融合查询和单药问答的 ID 分别采用 `cpqa-c-xxxxx`、`cpqa-fq-xxxxx` 和 `cpqa-sq-xxxxx`，用于跨文件关联。

融合查询包含 `_id`、`text`、`type`、`type_name`、`constraints` 和 `difficulty`。标准答案仅保存在 `fusion_qrels.tsv` 中。

端到端评测时，系统只能将 `text` 作为查询输入；其余字段仅用于数据分析和标签复现，不应输入被评测模型。

单药问答包含 `_id`、`question`、`answer`、`corpus_id`、`drug`、`field` 和 `difficulty`。

`fusion_qrels.tsv` 为三列制表符分隔文件，列名依次为 `query-id`、`corpus-id` 和 `score`；本版本中的相关性分数均为 `1`。

## 验证

在发布目录执行：

```bash
python scripts/validate.py
python scripts/rebuild_qrels.py
```

两个脚本均只使用 Python 标准库。

## 构建与质量控制

语料由《中国药典》2020年版一部和二部经结构化提取、字段规范化及通则显式关联后形成。标准答案基于冻结后的发布语料生成，并完成跨文件一致性检查。

## 许可与用途

见 `LICENSE`。ChinesePharmaQA 面向非商业科研与教学使用，仅作为评测数据，不构成医疗建议，也不能替代正式出版的药典。

## 对应论文

《面向中国药典融合性知识查询的问答数据集构建与基准评测》 / ChinesePharmaQA: A Benchmark Dataset for Fusion Knowledge Retrieval over the Chinese Pharmacopoeia.
