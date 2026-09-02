# CodexLaw

可重复的法律 Agent A/B 测试底座：Architecture A 使用 Lawgent 条款抽取，Architecture B 使用 CodexLaw 外置 workflow、evidence ledger 和 citation verifier。两边读取同一份真实案例、证据和 K3 配置。

当前已具备：

- 公开 LegalBench、LegalBench-RAG、CaseHOLD 数据下载与统一 JSONL；
- 官方 eCFR Title 16/17 XML 的原文、日期、URL、SHA-256 记录；
- Neo4j authority 导入、唯一约束和来源哈希校验；
- DGX Spark `nvidia/Nemotron-3-Embed-1B-BF16` 的真实 embedding 索引；
- 串行 NVIDIA Kimi K3 A/B runner，以及独立答案、引用和 workflow 判分。

安全边界：API key 只从运行时环境读取，绝不提交；LegalBench 等评测数据不是法律权威；法律结论必须回到带 provenance 的原始法源。数据下载目录和 benchmark 结果均被 `.gitignore` 忽略，许可与来源见 [TESTING.md](TESTING.md) 及 [data/README.md](data/README.md)。

先运行：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -q
python scripts/check_config.py
```

真实数据准备、Neo4j、DGX embedding 和 A/B 命令见 [TESTING.md](TESTING.md)。
