import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import psutil

from codelaw.benchmark import answer_matches, run_case
from codelaw.process import run_process
from scripts import build_dashboard
from scripts import ingest_neo4j
from scripts.ingest_neo4j import _source_scopes


ROOT = Path(__file__).resolve().parents[1]


class ProcessDeadlineRegressionTest(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'Separate POSIX sessions are the regression trigger')
    def test_timeout_stops_nested_session_and_its_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / 'pids.json'
            marker = Path(directory) / 'survived'
            leaf = f'import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).write_text("escaped"); time.sleep(3)'
            inner = ('import os,subprocess,sys,time,json; from pathlib import Path; '
                     f'p=subprocess.Popen([sys.executable,"-c",{leaf!r}],start_new_session=True); '
                     f'Path({str(pid_path)!r}).write_text(json.dumps([os.getpid(),p.pid])); time.sleep(5)')
            outer = ('import sys; from codelaw.process import run_process; '
                     f'run_process([sys.executable,"-c",{inner!r}],timeout=5)')
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
            started = time.monotonic()
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_process([sys.executable, '-c', outer], timeout=1, env=env)
                self.assertLess(time.monotonic() - started, 4)
                self.assertTrue(pid_path.exists(), 'Fixture must start before the deadline')
                pids = json.loads(pid_path.read_text())
                for pid in pids:
                    try:
                        status = psutil.Process(pid).status()
                    except psutil.NoSuchProcess:
                        continue
                    self.assertEqual(status, psutil.STATUS_ZOMBIE, f'Child {pid} survived in {status}')
                self.assertFalse(marker.exists())
            finally:
                if pid_path.exists():
                    for pid in json.loads(pid_path.read_text()):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_process_success_and_check_contract_are_preserved(self):
        result = run_process([sys.executable, '-c', 'print(input())'], input='fixture', text=True, timeout=3)
        self.assertEqual(result.stdout.strip(), 'fixture')
        with self.assertRaises(subprocess.CalledProcessError):
            run_process([sys.executable, '-c', 'raise SystemExit(7)'], timeout=3, check=True)

    @unittest.skipIf(os.name == 'nt', 'Owned POSIX process-group regression')
    def test_nested_helper_is_stopped_after_its_parent_exits_early(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / 'pid'
            leaf = f'import os,time; from pathlib import Path; Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(5)'
            outer = '\n'.join([
                'import os,sys,threading,time', 'from pathlib import Path', 'from codelaw.process import run_process',
                f'threading.Thread(target=lambda:run_process([sys.executable,"-c",{leaf!r}],capture_output=False,timeout=5),daemon=True).start()',
                f'path=Path({str(pid_path)!r})',
                'while not path.exists(): time.sleep(.01)', 'os._exit(0)',
            ])
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_process([sys.executable, '-c', outer], timeout=1, env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')))
                pid = int(pid_path.read_text())
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    try:
                        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                            break
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(.01)
                else:
                    self.fail('Reparented nested helper survived the case group cleanup')
            finally:
                if pid_path.exists():
                    try:
                        os.kill(int(pid_path.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


class GraphReplacementRegressionTest(unittest.TestCase):
    def setUp(self):
        self.old_hash, self.new_hash, self.other_hash = 'a' * 64, 'b' * 64, 'c' * 64
        self.url = 'https://www.ecfr.gov/api/versioner/v1/full/2026-09-30/title-16.xml'
        self.manifest = {'sources': [dict(title=16, source_url=self.url, sha256=self.new_hash, records=1)]}
        self.row = dict(authority_id='ecfr:16:1.1', source_url=self.url, sha256=self.new_hash)

    def test_version_scope_is_stable_and_preserves_other_titles(self):
        scopes = _source_scopes(self.manifest, [self.row])
        self.assertEqual(scopes, [dict(authority_prefix='ecfr:16:', sha256s=[self.new_hash])])
        def obsolete(authority_id, digest):
            return any(authority_id.startswith(s['authority_prefix']) and digest not in s['sha256s'] for s in scopes)
        self.assertTrue(obsolete('ecfr:16:1.2', self.old_hash))
        self.assertFalse(obsolete('ecfr:16:1.1', self.new_hash))
        self.assertFalse(obsolete('ecfr:17:1.2', self.other_hash))
        self.assertFalse(obsolete('ecfr:160:1.2', self.old_hash))

    def test_manifest_binds_hash_to_title_and_url(self):
        for overrides in [dict(authority_id='ecfr:17:1.1'), dict(source_url='https://other.invalid/title.xml'), dict(sha256=self.old_hash)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                _source_scopes(self.manifest, [dict(self.row, **overrides)])

    def test_multiple_allowed_versions_of_one_title_are_retained(self):
        self.manifest['sources'].append(dict(title=16, source_url='old-url', sha256=self.old_hash))
        self.assertEqual(_source_scopes(self.manifest, [self.row])[0]['sha256s'], sorted([self.old_hash, self.new_hash]))

    def test_replacement_rejects_missing_source_or_truncated_records(self):
        self.manifest['sources'].append(dict(title=17, source_url='other-url', sha256=self.other_hash, records=1))
        with self.assertRaisesRegex(ValueError, 'all records'):
            _source_scopes(self.manifest, [self.row], require_complete=True)
        self.manifest['sources'].pop()
        self.manifest['sources'][0]['records'] = 2
        with self.assertRaisesRegex(ValueError, 'all records'):
            _source_scopes(self.manifest, [self.row], require_complete=True)

    def test_duplicate_rows_cannot_satisfy_replacement_record_count(self):
        self.manifest['sources'][0]['records'] = 2
        with self.assertRaisesRegex(ValueError, 'duplicate authority'):
            _source_scopes(self.manifest, [self.row, self.row], require_complete=True)

    def test_importer_passes_stable_title_scopes_to_the_delete_query(self):
        queries = []
        class Driver:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def execute_query(self, query, **kwargs):
                queries.append((query, kwargs))
        module = types.SimpleNamespace(GraphDatabase=types.SimpleNamespace(driver=lambda *args, **kwargs: Driver()))
        with tempfile.TemporaryDirectory() as directory:
            records, manifest = Path(directory) / 'records.jsonl', Path(directory) / 'manifest.json'
            records.write_text(json.dumps(dict(self.row, text='No references.')) + '\n')
            manifest.write_text(json.dumps(self.manifest))
            argv = ['ingest_neo4j.py', str(records), '--uri', 'bolt://localhost:7687', '--manifest', str(manifest), '--replace-source']
            with patch.dict(sys.modules, {'neo4j': module}), patch.dict(os.environ, {'NEO4J_PASSWORD': 'dummy-fixture'}), patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                self.assertEqual(ingest_neo4j.main(), 0)
        delete, parameters = next((q, p) for q, p in queries if 'DETACH DELETE' in q)
        self.assertIn('a.authority_id STARTS WITH source.authority_prefix', delete)
        self.assertEqual(parameters['source_scopes'], [dict(authority_prefix='ecfr:16:', sha256s=[self.new_hash])])
        self.assertNotIn('source_urls', parameters)


class EvidenceSpanRegressionTest(unittest.TestCase):
    def test_trivial_substrings_cannot_pass_the_full_evaluator(self):
        text = 'No assignment shall be permitted without written consent.'
        case = dict(case_id='fixture', source='fixture', task='assignment', prompt='What is the assignment clause?',
                    expected_answer=text, answer_type='evidence_span', jurisdiction='US',
                    evidence=[dict(evidence_id='e1', source_id='s1', text=text, jurisdiction='US')])
        for answer in ['a', 'No', 'No.', 'without', 'yes']:
            with self.subTest(answer=answer):
                row = run_case(case, architecture='B', ask=lambda _, a=answer: json.dumps(dict(answer=a, citation_ids=['e1'])))
                self.assertTrue(row['citation_valid'])
                self.assertFalse(row['answer_correct'])
                self.assertFalse(row['success'])

    def test_exact_short_reference_and_substantive_span_still_match(self):
        self.assertTrue(answer_matches(answer='New York', expected='New York', answer_type='evidence_span'))
        self.assertTrue(answer_matches(answer='written consent is required', expected='Written consent is required by this agreement', answer_type='evidence_span'))
        self.assertFalse(answer_matches(answer='No', expected='No', answer_type='evidence_span'))


class EmptyDashboardRegressionTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed for the offline dashboard script regression')
    def test_single_batch_still_renders_and_enables_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'web').mkdir()
            results = root / 'benchmark/results'
            results.mkdir(parents=True)
            shutil.copyfile(ROOT / 'web/dashboard.html', root / 'web/dashboard.html')
            summary = dict(model='fixture', metrics={'projects': {}}, cases=['fixture:1'])
            (results / 'project-ab-fixture.summary.json').write_text(json.dumps(summary))
            rows = [dict(case_id='fixture:1', source='fixture', architecture=name, provider_error=None,
                         answer_correct=True, citation_valid=True, common_success=True,
                         expected_answer='yes', elapsed_seconds=1) for name in ['Lawgent', 'CodexLaw']]
            (results / 'project-ab-fixture.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            with patch.object(build_dashboard, 'ROOT', root), redirect_stdout(io.StringIO()):
                build_dashboard.main()
            script = r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const html=fs.readFileSync(process.argv[1], 'utf8');
const data=html.match(/<script id="dataset" type="application\/json">([\s\S]*?)<\/script>/)[1];
const source=html.match(/<\/script><script>([\s\S]*?)<\/script>/)[1];
const elements={}, section={id:'metrics'};
const document={getElementById(id){return elements[id]??=(id==='dataset'?{textContent:data}:{value:'',addEventListener(){}})},querySelectorAll(){return [section]},addEventListener(){}};
vm.runInNewContext(source,{document});
assert.equal(JSON.parse(data).default,'project-ab-fixture');
assert.equal(elements.run.value,'project-ab-fixture');
assert.equal(elements.emptyState.hidden,true);
assert.equal(elements.export.disabled,false);
assert.equal(section.hidden,false);
assert.ok(elements.cases.innerHTML.includes('fixture:1'));
'''
            subprocess.run(['node', '-e', script, str(root / 'dist/legal-agent-ab.html')], check=True, timeout=10, capture_output=True, text=True)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed for the offline dashboard script regression')
    def test_fresh_build_initializes_empty_state_and_guards_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'web').mkdir()
            shutil.copyfile(ROOT / 'web/dashboard.html', root / 'web/dashboard.html')
            with patch.object(build_dashboard, 'ROOT', root), redirect_stdout(io.StringIO()):
                build_dashboard.main()
            script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync(process.argv[1], 'utf8');
const data = html.match(/<script id="dataset" type="application\/json">([\s\S]*?)<\/script>/)[1];
const source = html.match(/<\/script><script>([\s\S]*?)<\/script>/)[1];
const elements = {}, sections = [{id:'emptyState'}, {id:'metrics'}, {id:'other'}];
const document = {
  getElementById(id) { return elements[id] ??= (id === 'dataset' ? {textContent:data} : {value:'', addEventListener(){}}); },
  querySelectorAll() { return sections; }, addEventListener() {}
};
const context = vm.createContext({document});
vm.runInContext(source, context);
assert.equal(JSON.parse(data).default, null);
assert.equal(elements.emptyState.hidden, false);
for (const id of ['run','export','search','source','outcome','reason']) assert.equal(elements[id].disabled, true);
assert.equal(sections.find(s=>s.id==='metrics').hidden, true);
elements.export.onclick(); // A synthetic click must also be safe.
vm.runInContext('filter()', context);
assert.ok(html.includes('暂无测试结果'));
assert.ok(html.includes('[hidden]{display:none!important}'));
'''
            subprocess.run(['node', '-e', script, str(root / 'dist/legal-agent-ab.html')], check=True, timeout=10, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
