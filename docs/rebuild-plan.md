---
prd_contract: v1
---

# CodexLaw 全量重建计划

Complexity: 11 → HIGH mode

## Context

重新构建 Lawgent Native 与 Codex Core + Legal Pack 的可复现 A/B 项目。此前临时工作区的自定义实现已丢失；本仓库是新的唯一真相。

## Solution

- 以 Python workflow controller 作为第一阶段可执行底座，避免嵌套两套 Agent loop。
- 复用 Lawgent 的合同抽取和领域规则；Codex 侧保留独立 workflow/evidence 状态。
- 用 Neo4j 承载法源与合同关系，原文与有效日期进入 Evidence Ledger。
- BM25 + 向量检索 + 图扩展；向量提供者由环境变量选择，默认 NVIDIA Nemotron。
- 所有真实模型测试严格串行，默认每次调用间隔 30 秒。

## Integration Ledger

| New thing | Live caller | Negative control |
|---|---|---|
| workflow controller | benchmark runner | 删除 citation gate 后验证失败 |
| evidence ledger | final-review node | 清空 ledger 后阻止 final |
| Neo4j store | retrieval service | 断开 Neo4j 后明确失败 |
| NVIDIA provider | live benchmark runner | 缺少 key 时明确失败 |
| independent evaluator | benchmark runner | 错误引用必须扣分 |

## Phases

1. **Foundation** — 目录、依赖锁定、配置校验、可重复的本地测试入口。
2. **Shared legal core** — 证据账本、状态机、citation verifier、可测试的检索接口。
3. **Authority graph** — eCFR 下载/解析、SQLite 暂存、Neo4j 导入与 provenance。
4. **Hybrid retrieval** — BM25、NVIDIA embedding、Neo4j graph expansion、rerank contract。
5. **Architecture A** — Lawgent adapter，调用真实合同提取与其领域资产。
6. **Architecture B** — Codex Legal Pack adapter，外置 workflow/evidence state。
7. **Evaluation/observability** — 30+ cases、独立 evaluator、Prometheus trace/metrics。
8. **Live gate** — K3 串行 3–5 条合同问题；仅在运行环境提供 key 与 NVIDIA 出网时执行。
9. **DGX acceptance** — 在持久 Neo4j 与本地模型条件下复跑全量基准。

## Negative Controls

| Gate | Expected red |
|---|---|
| citation verification | 引用不存在时 final review 拒绝输出 |
| evidence ledger | 删除必要 evidence id 时工作流失败 |
| graph ingestion | 换掉官方 XML 后 SHA256/provenance 校验失败 |
| live provider | 未配置 NVIDIA_API_KEY 时退出且不发请求 |
| serial limiter | 调用间隔低于配置时测试拒绝执行 |

## Acceptance Criteria

- 同一数据、模型、token budget 与工具权限下能运行 A/B。
- Citation、jurisdiction、effective date、exception 不能被自由跳过。
- Evidence Ledger 不因上下文压缩丢失。
- 真实法律权威材料与测试数据严格分离并保留来源。
- 真实 K3 结果只有在实际完成配对调用后才可报告。

## Checkpoint Protocol

每阶段先提交 GitHub，再运行该阶段测试并记录命令、输出、caller census 与负向对照；未完成的外部验证标记为 BLOCKED，绝不伪报通过。

Contract conformance: prd_contract: v1
