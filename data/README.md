# Real data workspace

本目录只保存本机生成的下载、解析和向量索引，不随 Git 提交。每次准备脚本都会在 `data/processed/**/manifest.json` 记录输入路径、来源 URL、数量或哈希。

已使用的公开来源：

- [LegalBench](https://huggingface.co/datasets/nguha/legalbench)：多任务法律评测数据；
- [CaseHOLD](https://huggingface.co/datasets/casehold/casehold)：判例 holding 选择数据；
- [LegalBench-RAG](https://github.com/zeroentropy-cc/legalbenchrag)：合同问题、标注片段和合同语料；
- [eCFR Versioner](https://www.ecfr.gov/)，通过官方 API 下载 Title 16/17 XML 作为 authority。

评测数据和法律 authority 分开保存。评测数据不等于法律意见；authority 记录保留来源 URL、版本日期、原文和 SHA-256。`data/raw/`、`data/sources/`、`data/processed/` 的大文件被 `.gitignore` 忽略，重新构建请按 [TESTING.md](../TESTING.md) 执行。
