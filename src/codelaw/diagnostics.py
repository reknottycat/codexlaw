"""Explain the existing scorer's decision without changing its standard."""
from __future__ import annotations

import re
from typing import Any

from .benchmark import _claim_terms, parse_decision


def diagnose(row: dict[str, Any], case: dict[str, Any] | None, *, evidence_limit: int = 0) -> dict[str, Any]:
    reasons: list[dict[str, str]] = []
    def add(code: str, label: str, detail: str):
        reasons.append(dict(code=code, label=label, detail=detail))

    project_row = 'project' in row
    provider_error = row.get('provider_error') if project_row else None
    if provider_error:
        text = str(provider_error)
        code = 'provider_timeout' if any(t in text.lower() for t in ['timeout', 'timed out', 'exceeded']) else 'provider_error'
        add(code, '调用超时' if code == 'provider_timeout' else '模型调用失败', text)
    parsed = parse_decision(row.get('normalised_response') or row.get('model_response', ''))
    if parsed.parse_error and not row.get('answer') and not provider_error:
        add('response_parse_error', '输出格式无法解析', '模型输出没有可解析的 answer JSON，且未提取到明确 Answer: 字段；不是网络失败的同义词。')
    if not row.get('answer_correct') and not parsed.parse_error and not provider_error:
        answer, expected = str(row.get('answer', '')), str(row.get('expected_answer', ''))
        if case and case.get('answer_type') == 'classification' and re.match(r'^'+re.escape(expected)+r'\b', answer.strip(), re.I):
            add('answer_format_mismatch', '答案标签带多余说明', '答案以参考标签开头，但现行分类评分要求整个答案只包含该标签。属于格式扣分，未据此认定法律推理错误。')
        elif case and case.get('answer_type') == 'evidence_span' and answer.strip().lower().strip('.!') in ('yes', 'no'):
            add('answer_type_mismatch', '用是/否代替合同条款', '本题要求引用或抽取具体证据条款，模型只回答 Yes/No，未满足当前输出契约。')
        else:
            add('answer_mismatch', '答案未匹配参考', '分类/选项不一致，或合同回答未达到当前子串/词重叠规则。可对照参考与原文核查；该自动判分本身不提供专家法律结论。')

    ids = row.get('citation_ids') or []
    citations = {str(e['evidence_id']): e for e in case.get('evidence', [])} if case else {}
    selected = citations.get(str(ids[0])) if ids else None
    if not row.get('citation_valid') and not provider_error:
        if not ids:
            add('citation_missing', '没有返回引用 ID', 'citation_ids 为空，无法完成引用验证。旧记录中的 CITATION_VERIFICATION 工作流错误由此产生，不是额外运行故障。')
        elif case and not selected:
            add('citation_unknown', '引用 ID 不在输入证据中', f'首条引用 {ids[0]} 与本题提供的完整 ID 不匹配；可能为缩写、改写或生成了不存在的 ID。')
        elif selected:
            if selected.get('jurisdiction', 'US') != (case.get('jurisdiction') or 'US'):
                add('citation_jurisdiction', '引用法域不符', '首条引用的法域与本题法域不一致。')
            effective = case.get('effective_on') or '9999-12-31'
            if selected.get('effective_on') and selected['effective_on'] > effective:
                add('citation_date', '引用时间不符', '首条引用晚于本题要求的生效日期。')
            missing = [t for t in _claim_terms(case) if t.lower() not in selected['text'].lower()]
            if missing:
                add('citation_claim_terms', '引用片段未通过关键词校验', '现行校验器要求的关键词未出现在首条引用：' + ', '.join(missing))
        else:
            add('citation_unresolved', '引用校验失败，待原输入核查', '尚无可确认的原输入文件，不能细分引用失败原因。')
    if not project_row and row.get('error') and not reasons:
        add('execution_error', '执行错误', str(row['error']))
    if not row.get('common_success', row.get('success')) and not reasons:
        add('unclassified', '未通过，原因未完整记录', str(row.get('error') or '历史日志不足'))

    telemetry = row.get('telemetry') or {}
    stops = telemetry.get('stop_reasons') or []
    if stops and any(s in ('max_tokens', 'length') for s in stops):
        token_state, token_note = 'limit_hit', '至少一次模型响应以 max_tokens/length 结束，已确认触及输出预算；需看是否影响最终答案。'
    elif stops and telemetry.get('all_calls_have_stop_reason'):
        token_state, token_note = 'normal_stop', '所有已记录模型调用正常结束，没有输出 Token 截断记录。'
    else:
        token_state, token_note = 'unknown', '历史未保存 usage/stop_reason，不能确证实际 Token 数或截断；回答长短不能替代结束原因。'
    total = sum(len(str(e.get('text', ''))) for e in case.get('evidence', [])) if case else None
    provided = min(total, evidence_limit) if total is not None and evidence_limit else total
    norm = lambda s: ' '.join(str(s).lower().split())
    target = norm(row.get('expected_answer', ''))
    original_contains = any(target and target in norm(e.get('text','')) for e in case.get('evidence', [])) if case else None
    category = 'passed'
    if reasons:
        codes = {r['code'] for r in reasons}
        if codes & {'provider_error','provider_timeout','execution_error'}: category = '调用/执行失败'
        elif 'response_parse_error' in codes: category = '输出无法解析'
        elif codes & {'answer_format_mismatch','answer_type_mismatch'}: category = '答案格式不符合'
        elif not row.get('answer_correct') and not row.get('citation_valid'): category = '答案和引用均未通过'
        elif not row.get('answer_correct'): category = '仅答案未匹配'
        else: category = '仅引用未通过'
    return dict(category=category, reasons=reasons, token_state=token_state, token_note=token_note,
                input_evidence_chars=total, provided_evidence_chars=provided,
                input_truncated=provided < total if total is not None else None,
                reference_span_in_original_evidence=original_contains,
                selected_citation=ids[0] if ids else None,
                selected_evidence_excerpt=selected['text'][:1600] if selected else None,
                provided_evidence_ids=list(citations))


def telemetry(calls: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [c.get('usage') for c in calls]
    complete = bool(calls) and all(isinstance(u, dict) for u in usages)
    return dict(model_calls=len(calls), input_tokens=sum(u.get('input_tokens', u.get('prompt_tokens', 0)) for u in usages if isinstance(u, dict)) if complete else None,
                output_tokens=sum(u.get('output_tokens', u.get('completion_tokens', 0)) for u in usages if isinstance(u, dict)) if complete else None,
                stop_reasons=[c['stop_reason'] for c in calls if c.get('stop_reason')],
                all_calls_have_stop_reason=bool(calls) and all(c.get('stop_reason') for c in calls),
                actual_models=sorted({str(c['model']) for c in calls if c.get('model')}),
                cost=None, cost_note='Provider did not return billed cost; not estimated from list prices.')
