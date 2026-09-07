"""Build an offline, shareable evaluation dashboard from immutable result rows."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def load_runs():
    runs = []
    for path in sorted((ROOT / 'benchmark/results').glob('*.summary.json')):
        summary = json.loads(path.read_text(encoding='utf-8-sig'))
        rows_path = path.with_name(path.name.replace('.summary.json', '.jsonl'))
        if not rows_path.exists():
            continue
        raw = rows_path.read_bytes()
        rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
        audit_path = path.with_name(path.name.replace('.summary.json', '.audit.json'))
        audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
        analysis_path = path.with_name(path.name.replace('.summary.json', '.analysis.json'))
        analysis = json.loads(analysis_path.read_text(encoding='utf-8')) if analysis_path.exists() else {}
        if analysis and analysis.get('source_rows_sha256') != hashlib.sha256(raw).hexdigest():
            raise ValueError(f'Stale analysis sidecar for {path.name}')
        if audit and audit.get('source_rows_sha256') != hashlib.sha256(raw).hexdigest():
            raise ValueError(f'Stale audit sidecar for {path.name}; rerun reconstruct_run_metadata.py')
        for row in rows:
            row['diagnostics'] = audit.get('diagnostics', {}).get(row['architecture']+'|'+row['case_id'], row.get('diagnostics', {}))
        project = 'projects' in summary['metrics']
        labels = ['Lawgent', 'CodexLaw'] if project else ['A', 'B']
        groups = {}
        for label in labels:
            group = [r for r in rows if r['architecture'] == label]
            complete = [r for r in group if not r.get('provider_error' if project else 'error')]
            groups[label] = dict(n=len(group), completed=len(complete),
                correct=sum(bool(r['answer_correct']) for r in complete),
                cited=sum(bool(r['citation_valid']) for r in complete),
                success=sum(bool(r.get('common_success' if project else 'success')) for r in group))
        pair_map = {}
        for row in rows:
            pair_map.setdefault(row['case_id'], {})[row['architecture']] = row
        wins = losses = ties = 0
        for pair in pair_map.values():
            if not all(label in pair for label in labels):
                continue
            a, b = [bool(pair[label].get('common_success' if project else 'success')) for label in labels]
            wins += b and not a
            losses += a and not b
            ties += a == b
        discordant = wins + losses
        p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2 ** discordant) if discordant else 1.
        runs.append(dict(id=path.name.replace('.summary.json',''), model=summary['model'],
            project=project, labels=labels, groups=groups, rows=rows, pairs=len(pair_map),
            wins=wins, losses=losses, ties=ties, paired_p=p,
            provenance={k:v for k,v in audit.items() if k not in ('diagnostics', 'categories')},
            categories=audit.get('categories', {}),
            analysis=analysis,
            retrieval_mode=summary.get('retrieval_mode') or audit.get('retrieval_mode', 'unknown'),
            controls=summary.get('fairness_controls'),
            config={k: v for k, v in summary.items() if k not in ('metrics','cases','rows_path','case_sources')},
            sha256=hashlib.sha256(raw).hexdigest(), source=str(rows_path.relative_to(ROOT)).replace('\\','/')))
    return runs


def main():
    runs = load_runs()
    health_path = ROOT / 'benchmark/results/project-health.json'
    health = json.loads(health_path.read_text(encoding='utf-8')) if health_path.exists() else {}
    graph_runs = [r for r in runs if r['retrieval_mode']=='vector_graph' and r['controls']]
    payload = dict(generated=datetime.now(timezone.utc).isoformat(), runs=runs,
                   health=health, default=graph_runs[-1]['id'] if graph_runs else 'project-ab-20260903T085033Z')
    template = (ROOT / 'web/dashboard.html').read_text(encoding='utf-8')
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    output = ROOT / 'dist/legal-agent-ab.html'
    output.parent.mkdir(exist_ok=True)
    output.write_text(template.replace('__BENCHMARK_DATA__', data), encoding='utf-8')
    print(f'dashboard={output}\nruns={len(runs)}\nrows={sum(len(r["rows"]) for r in runs)}')


if __name__ == '__main__':
    main()
