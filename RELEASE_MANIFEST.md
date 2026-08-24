# ChinesePharmaQA Release Manifest

本清单对应仓库 `main` 分支在 2026-08-24 整理的当前发布内容，用于统一说明数据集、复现代码、冻结结果和文档之间的关系。

## Release Scope

- 数据集版本：`ChinesePharmaQA v1.0.0`
- 复现代码版本：`ChinesePharmaQA-Code v1.0.1`
- 结构化药品条目：5,138 条（一部 2,425 条，二部 2,713 条）
- 融合查询：1,065 条
- 附加单药字段问答：200 条
- 融合查询 qrels：48,335 条
- 融合查询标准答案规模：8 至 595 个药品名称

论文使用 `R`、`IM`、`P`、`M`、`S`，数据文件和代码中的 `type` 字段使用 `A`、`B`、`C`、`D`、`S`。对应关系为 `R=A`、`IM=B`、`P=C`、`M=D`、`S=S`；该映射只改变类型记号，不改变问题、约束、答案或评测逻辑。

## Current File Inventory

```text
ChinesePharmaQA/
├── README.md
├── README_zh-CN.md
├── NOTICE.md
├── RELEASE_MANIFEST.md
├── ChinesePharmaQA/
│   ├── README.md
│   ├── README_zh-CN.md
│   ├── LICENSE
│   ├── data/
│   │   ├── corpus.jsonl
│   │   ├── fusion_queries.jsonl
│   │   ├── fusion_qrels.tsv
│   │   └── single_drug_qa.jsonl
│   └── scripts/
│       ├── validate.py
│       └── rebuild_qrels.py
└── ChinesePharmaQA-Code-v1.0.1/
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
        ├── resources/eval_corpus.json
        └── results/
            ├── paper_summary.json
            ├── qwen_predictions.jsonl
            ├── semantic_results.json
            └── verification_summary.json
```

## Evaluation Roles

| 角色 | 方法或设置 |
| --- | --- |
| 当前自然语言主比较 | Random-K、BM25-Original、BM25-FullFields、BGE、Direct Qwen-Plus、Semantic-Parser |
| 受控分析 | OracleStructured、BM25+Filter、结构化组件消融 |
| 补充诊断 | E2E-Rule、FullCorpusScan、程序化问句改写 |

Semantic-Parser 先用词汇与句式门控激活约束类别，再以固定版本的 `BAAI/bge-large-zh-v1.5` 对问题和类别化候选描述进行语义排序。普通激活类别取 Top-1，明确的双试剂表达取试剂 Top-2，明确分部直接规范化；不另设相似度阈值。若没有类别被激活，执行器返回空集合。预测约束随后在字段倒排索引上执行集合交集。

## Authoritative SHA-256

以下哈希以 Git 仓库中规范化为 LF 换行的文件字节为准；启用自动 CRLF 转换的本地检出文件可能具有不同的字节级哈希。

### Dataset

```text
corpus.jsonl          E748DD6A95ACE58514C28305F0B023703911A1C5B122C2EB92BD8B621D8E0500
fusion_queries.jsonl  13B2D29867FBA2C207B4FB0A874B3C04C778B4951C69E7A0694E63D25B00203B
fusion_qrels.tsv      322ED53D0BA128C8C0EE33C44E8929DCCF5F8207D16A728FADFBC1D25161B7FB
single_drug_qa.jsonl  F6A14DCACFB464358AD763F499057CB399C223F86431C98992E63BD3A43C283E
validate.py           692BA9468F89E73298F51D9B82C7A2C0D7769C0459F21EF0A0A16EF340C6BF03
rebuild_qrels.py      F2D577D2CD5B2E96DD797F2037D494438F38B77A7D732BFC8DAC97F20A550AB6
```

### Benchmark Code

```text
common.py                58E331EED5A3469F6612F06D305C6A50D83D26B0BF9FF8FDC535C099B4F82925
semantic_parser.py       B104B48D684A1060E2ECEDDB518E28021B3893B8D7AC257E257CDF91B3096604
evaluate_classic.py      B861BFF5834F91CC7903398C9D2299504B5820136BCF87B5ABDE8847B38B2D85
evaluate_bge.py          7EAE22A6A8041A10F4A3E740D3DFA9C218A27AAF633577B6E90520A7EB60C4B3
evaluate_semantic.py     AA8EBF88112B70F4EB56546E68B5C153B6196E2945D8581A8EA9840F4A617202
evaluate_qwen.py         3D8E4C12E94DB9465B0E4BFC3B982E8235F0DF9B5A29356C22DC247D26B26225
evaluate_paraphrases.py  9AAC785DF794888CC19F439CEFECD17A1EBD20CC6EB4ED325A9BED636655AC1A
make_figures.py          FA8D3768ACEF72B0934BF43979EDAB2BD22F98D29B597FEC950C39D090CFAEC9
requirements.txt         42495E820E4D6C08BCE9D46CBDF0B5B471735F4D78383CEC1751F271B32FAC3E
```

### Frozen Evaluation Artifacts

```text
eval_corpus.json        4F17F313FB9C8E66578BD1F8E4237A9C10F39A21BE77E66B36D5FD017C602F27
qwen_predictions.jsonl  74FD460D794B032850F4C774E1C35FE52996E9EB63BABFFC487B1D526AA45125
paper_summary.json      E0B0C1E3AA228C3564D9B0AA618F66B69A8089A0072A5E48BAB45E3EC6ABFE36
semantic_results.json   59147BC78EBA9D1D4CDE86CED2C3A7C6EAE014E39AEA1238BF5371F4957780DE
```

## Validation Record

2026-08-24 使用项目指定虚拟环境完成以下检查，未安装新依赖：

- 数据集结构和跨文件关系检查通过。
- 48,335 条 qrels 可由冻结语料与查询约束逐条确定性重建。
- 所有发布 Python 文件通过语法检查。
- BM25 主结果与冻结结果一致。
- Semantic-Parser 完整复跑结果与发布的 1,065 条逐题记录一致，宏平均 Precision、Recall、F1 和 EM 分别为 0.6732、0.8185、0.6665 和 0.5577。
- Qwen 脱敏预测离线评分、程序化改写补充诊断及图 2、图 3 重建均通过。
- 发布文件未包含 API 密钥、本机绝对路径、模型权重或原始药典 PDF。

详细命令、环境和结果见 `ChinesePharmaQA-Code-v1.0.1/VERIFICATION.md` 与 `ChinesePharmaQA-Code-v1.0.1/benchmark/results/verification_summary.json`。
