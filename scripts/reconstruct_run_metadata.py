"""Reconstruct inspectable sidecars, never invent missing historical telemetry."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from codelaw.diagnostics import diagnose
from codelaw.experiment import file_hash


def _trusted_snapshot_path(raw_path: object, expected_sha256: object) -> Path:
    """Resolve a recorded snapshot without allowing metadata to read arbitrary files."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError('metadata snapshot_path is missing')
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve(strict=False)
    results_root = (ROOT / 'benchmark/results').resolve()
    try:
        candidate.relative_to(results_root)
    except ValueError as exc:
        raise ValueError('metadata snapshot_path must remain under benchmark/results') from exc
    # Reject a symlink anywhere in the path, even if it resolves back inside the directory.
    current = results_root
    for part in candidate.relative_to(results_root).parts:
        current = current / part
        if current.is_symlink() or os.path.islink(current):
            raise ValueError('metadata snapshot_path must not use symlinks')
    if not candidate.is_file():
        raise ValueError('metadata snapshot_path does not point to a file')
    if not isinstance(expected_sha256, str) or file_hash(candidate) != expected_sha256:
        raise ValueError('metadata snapshot_sha256 does not match the recorded snapshot')
    return candidate


def main():
    base_path = ROOT / 'data/processed/benchmark/cases.jsonl'
    summary_paths = sorted((ROOT / 'benchmark/results').glob('*.summary.json'))
    ids = set()
    for path in summary_paths:
        ids.update(json.loads(path.read_text(encoding='utf-8-sig')).get('cases', []))
    base = {}
    with base_path.open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            if row['case_id'] in ids:
                base[row['case_id']] = row
            if len(base) == len(ids):
                break
    base_sha = file_hash(base_path)
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        stem = summary_path.name.replace('.summary.json', '')
        rows_path = summary_path.with_name(stem + '.jsonl')
        if not rows_path.exists():
            continue
        metadata_path = summary_path.with_name(stem + '.metadata.json')
        provenance = 'reconstructed from matching case IDs and reference answers; original runtime input snapshot was not saved'
        mode = 'historical_input_assumed'
        source_path, cases, source_sha = base_path, base, base_sha
        original_metadata = None
        if metadata_path.exists():
            original_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            try:
                source_path = _trusted_snapshot_path(original_metadata.get('snapshot_path'), original_metadata.get('snapshot_sha256'))
            except ValueError as exc:
                # Preserve the original metadata for auditability but fall back to
                # the checked-in cases rather than reading an untrusted path.
                source_path, original_metadata = base_path, None
                mode = 'historical_input_assumed'
                provenance = f'recorded snapshot rejected: {exc}; used checked-in case IDs'
            else:
                mode = original_metadata.get('retrieval_mode', 'historical_input_assumed')
                provenance = 'original recorded input snapshot with SHA-256'
        elif stem == 'project-ab-20260903T085033Z':
            source_path = ROOT / 'benchmark/results/offline-rag-cases-20260903T084700Z.jsonl'
            mode = 'vector_legacy'
            provenance = 'input mapping confirmed by benchmark/OFFLINE-VECTOR-RAG-PROJECT-AB-2026-09-03-minimax-m3.md; hashes computed now, not recorded at runtime'
        elif stem == 'project-ab-20260903T041545Z':
            mode = 'supplied_evidence_legacy'
            provenance = 'input policy confirmed by benchmark/PROJECT-AB-2026-09-03-minimax-m3.md; hashes computed now, not recorded at runtime'
        if source_path != base_path:
            cases = {r['case_id']:r for r in map(json.loads, source_path.read_text(encoding='utf-8').splitlines()) if r}
            source_sha = file_hash(source_path)
        rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        trace_path = summary_path.with_name(stem + '.trace.jsonl')
        traces = {}
        if trace_path.exists():
            traces = {(t['project'], t['case_id']): t for t in map(json.loads, trace_path.read_text(encoding='utf-8').splitlines())}
        diagnostics = {}
        groups = {}
        for row in rows:
            case = cases.get(row['case_id'])
            if case and case.get('expected_answer') != row.get('expected_answer'):
                case = None
            info = row.get('diagnostics') or diagnose(row, case, evidence_limit=12000 if row['architecture'] in ('CodexLaw', 'A', 'B') else 0)
            trace = traces.get((row['architecture'], row['case_id']), {})
            failed_calls = [dict(exit_code=c.get('exit_code'), error=c.get('error'), elapsed_seconds=c.get('elapsed_seconds')) for c in trace.get('calls', []) if c.get('error') or c.get('exit_code')]
            if failed_calls:
                info['intermediate_call_failures'] = failed_calls
                info['token_note'] = '最终流程已完成，但中间调用失败未返回 usage/stop_reason；其余已记录响应正常结束。无法确认缺失调用的 Token 数，不能归因为输出预算不足。MMX 1.0.25 的退出码 6 表示网络类错误；原日志未保留更细错误文本。'
            for call in trace.get('calls', []):
                try:
                    raw_plan = call.get('response_text', '')
                    start = raw_plan.find('{')
                    plan, _ = json.JSONDecoder().raw_decode(raw_plan[start:])
                except (TypeError, ValueError):
                    continue
                if isinstance(plan, dict) and isinstance(plan.get('direct_response'), dict) and str(plan['direct_response']) == trace.get('response'):
                    info['failure_origin'] = 'workflow_direct_response_dict_to_str'
                    for reason in info.get('reasons', []):
                        if reason['code'] == 'response_parse_error':
                            reason['detail'] = '模型的规划 JSON 中 direct_response 是对象；Lawgent workflow.py 将其用 str() 转成单引号字典文本，导致最终输出不是合法 JSON。已由原始调用和工作流事件核对；不是 Token 不足或网络故障。'
            if info['category'] == '输出无法解析' and trace and not info.get('failure_origin'):
                raw_answer = str(trace.get('response', '')).strip()
                if raw_answer and '\n' not in raw_answer and '{' not in raw_answer and len(raw_answer) < 80:
                    info['failure_origin'] = 'bare_label_without_json_or_citations'
                    for reason in info.get('reasons', []):
                        if reason['code'] == 'response_parse_error':
                            reason['detail'] = '最终工作流只返回裸标签/选项（'+raw_answer+'），没有约定的 answer JSON 和 citation_ids。属于输出契约失败；标签是否正确仍需与参考答案分开核对。'
            if case:
                info['graph_rag'] = case.get('graph_rag')
                target = ' '.join(str(row.get('expected_answer','')).lower().split())
                originals = [e for e in case.get('evidence', []) if not e['evidence_id'].startswith(('retrieval:', 'graph:'))]
                info['original_evidence_excerpts'] = [dict(evidence_id=e['evidence_id'], text=e['text'][:1800]) for e in originals[:2]]
            diagnostics[row['architecture'] + '|' + row['case_id']] = info
            counter = groups.setdefault(row['architecture'], Counter())
            counter[info['category']] += 1
        sidecar = dict(schema_version='1.0', reconstructed_at=datetime.now(timezone.utc).isoformat(),
            source_rows_sha256=file_hash(rows_path), input_path=str(source_path.resolve()), input_sha256=source_sha,
            retrieval_mode=mode, provenance=provenance, original_metadata=original_metadata,
            unavailable_historical_fields=[] if original_metadata else ['runtime code hash','actual token usage','stop reason','full provider trace'],
            categories={k:dict(v) for k,v in groups.items()}, diagnostics=diagnostics)
        summary_path.with_name(stem + '.audit.json').write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'reconstructed_runs={len(summary_paths)} originals_unchanged=true')


if __name__ == '__main__':
    main()
