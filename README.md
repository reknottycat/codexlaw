# CodexLaw

可重复的法律 Agent A/B 测试底座。项目对照使用 Lawgent WorkflowExecutor 与原生 Codex CLI，统一通过 MiniMax-M3 回答冻结的证据任务；另保留 K3 内部 A/B 入口。

当前已具备：

- 公开 LegalBench、LegalBench-RAG、CaseHOLD 数据下载与统一 JSONL；
- 官方 eCFR Title 16/17 XML 的原文、日期、URL、SHA-256 记录；
- Neo4j authority 导入、唯一约束和来源哈希校验；
- DGX Spark `nvidia/Nemotron-3-Embed-1B-BF16` 的真实 embedding 索引；
- 串行 NVIDIA Kimi K3 A/B runner，以及独立答案、引用和 workflow 判分。

安全边界：API key 只从运行时环境读取，绝不提交；默认只允许发往 NVIDIA 官方 endpoint，使用自建 endpoint 必须显式设置 `NVIDIA_ALLOW_CUSTOM_ENDPOINT=true`，且 URL 不得含凭据、查询参数或片段。LegalBench 等评测数据不是法律权威；法律结论必须回到带 provenance 的原始法源。数据下载目录和 benchmark 结果均被 `.gitignore` 忽略，许可与来源见 [TESTING.md](TESTING.md) 及 [data/README.md](data/README.md)。

先运行：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -q
python scripts/check_config.py
```

真实数据准备、Neo4j、DGX embedding 和 A/B 命令见 [TESTING.md](TESTING.md)。

评测观察台：运行 `python scripts/build_dashboard.py`，打开生成的 `dist/legal-agent-ab.html`；单文件支持离线分享、批次对照与逐题核验。当前卡点和实测结果见 [项目状态](docs/project-status-2026-09-06.md)。

运行原生对照请明确使用 `--codex-engine cli`。该路径通过仅支持文本的 Responses 转接层调用与 Lawgent 相同的 MiniMax Messages 客户端，保存真实 Codex 事件与模型调用记录。双方禁用外部工具；此实验不覆盖完整自主工具循环。默认 `python` 路径保留为自建封装基线，历史结果不可改称原生 Codex 成绩。

## 文档与记录

- [测试与复现](TESTING.md) · [数据来源与许可](data/README.md) · [变更记录](CHANGELOG.md)
- [项目状态](docs/project-status-2026-09-06.md) · [全量重建计划](docs/rebuild-plan.md) · [Harness 说明稿](docs/harness-video-narration.md)
- [Kimi K3 实时数据验收](benchmark/ACCEPTANCE-2026-09-03-k3-65536.md) · [真实数据验收快照](benchmark/ACCEPTANCE-2026-09-03.md)
- [离线向量 RAG 项目对照](benchmark/OFFLINE-VECTOR-RAG-PROJECT-AB-2026-09-03-minimax-m3.md) · [Lawgent 与 CodexLaw 项目对照](benchmark/PROJECT-AB-2026-09-03-minimax-m3.md)
