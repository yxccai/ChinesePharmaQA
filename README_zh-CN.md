# ChinesePharmaQA

ChinesePharmaQA 是面向《中国药典》融合性知识查询的评测数据集与复现工具包。给定自然语言问题，系统需要返回满足全部约束的完整药品专论集合。

[English README](README.md)

项目仓库：<https://github.com/yxccai/ChinesePharmaQA>

## 仓库内容

```text
ChinesePharmaQA/
├── RELEASE_MANIFEST.md             # 最新文件清单与哈希
├── ChinesePharmaQA/                 # 数据集 v1.0.0
│   ├── data/
│   └── scripts/
└── ChinesePharmaQA-Code-v1.0.1/    # 评测与复现代码
    └── benchmark/
```

数据集包含 5,138 条语料记录、1,065 条融合查询、200 条单药问答和 48,335 条相关性判断。代码包包含 Random、BM25、BGE、Direct Qwen-Plus 和 Semantic-Parser 等评测方法，以及受控分析、补充诊断、图表生成脚本、冻结结果和验证记录。

原始《中国药典》PDF、OCR 中间文件、模型权重、向量缓存和模型原始 API 回答不包含在本仓库中。

## 快速验证

在仓库根目录执行：

```bash
python ChinesePharmaQA/scripts/validate.py
python ChinesePharmaQA/scripts/rebuild_qrels.py
```

复现经典基准：

```bash
python ChinesePharmaQA-Code-v1.0.1/benchmark/evaluate_classic.py \
  --dataset-root ChinesePharmaQA \
  --output reproduced/classic_results.json
```

安装代码包列出的 BGE 相关依赖及适配本机的 PyTorch 后，可复现 Semantic-Parser：

```bash
python ChinesePharmaQA-Code-v1.0.1/benchmark/evaluate_semantic.py \
  --dataset-root ChinesePharmaQA \
  --model BAAI/bge-large-zh-v1.5 \
  --revision 79e7739b6ab944e86d6171e44d24c997fc1e0116 \
  --output reproduced/semantic_results.json
```

当前论文主实验中，Semantic-Parser 的宏平均 Precision、Recall、F1 和 EM 分别为 0.6732、0.8185、0.6665 和 0.5577。E2E-Rule、FullCorpusScan 及程序化改写保留为补充诊断，不属于当前论文的主方法比较。

完整运行说明、依赖版本、预期结果和验证记录见：

- `ChinesePharmaQA/README.md`
- `ChinesePharmaQA-Code-v1.0.1/README.md`
- `ChinesePharmaQA-Code-v1.0.1/VERIFICATION.md`
- `RELEASE_MANIFEST.md`

## 版本

- 数据集：`v1.0.0`
- 复现代码：`v1.0.1`
- 合并发布包：`v1.0.1`

最新文件清单及权威数据、代码与结果文件哈希见 [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md)。

## 许可与来源说明

请阅读各子包中的 `LICENSE` 和仓库根目录的 [NOTICE.md](NOTICE.md)。第三方药典原始出版物的权利仍归其权利人所有。

## 引用

论文作者、正式发表信息、DOI和 `CITATION.cff` 将在论文与公开归档版本确定后补充。
