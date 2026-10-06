import json
import subprocess
import unittest
from unittest.mock import patch

from codelaw.experiment import normalise_response, prepare_case, shared_task
from codelaw.minimax import MiniMaxChatClient, MiniMaxRuntime
from codelaw.diagnostics import diagnose, telemetry
from codelaw.nvidia import NvidiaEmbeddingClient
from codelaw.config import SettingsError
from scripts.run_lawgent_minimax import _task
from scripts.prepare_offline_rag_cases import _top_hits
from codelaw.codex_runtime import messages_text, response_events
from codelaw.graph import AuthorityGraph
from scripts.run_project_ab_benchmark import _project_response


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.case = dict(case_id='fixture:1', prompt='Question?', expected_answer='secret_reference',
                         answer_type='classification', evidence=[dict(evidence_id='e1', text='abcde'), dict(evidence_id='e2', text='fghij')])

    def test_shared_budget_and_task_do_not_leak_reference_or_mutate_original(self):
        prepared = prepare_case(self.case, 7)
        self.assertEqual([e['text'] for e in prepared['evidence']], ['abcde', 'fg'])
        self.assertEqual(self.case['evidence'][1]['text'], 'fghij')
        self.assertEqual(_task(prepared), shared_task(prepared))
        self.assertNotIn('secret_reference', shared_task(prepared))
        self.assertEqual(prepare_case(self.case, 0), self.case)

    def test_cli_json_keeps_usage_stop_reason_and_actual_model(self):
        envelope = dict(content=[dict(type='text', text='answer')], model='MiniMax-M3', usage=dict(input_tokens=10,output_tokens=4),stop_reason='end_turn')
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(envelope), stderr='')
        client=MiniMaxChatClient(MiniMaxRuntime(),run=run)
        self.assertEqual(client.ask('task'), 'answer')
        self.assertNotIn('--quiet', calls[0])
        self.assertEqual(calls[0][calls[0].index('--output')+1], 'json')
        self.assertEqual(telemetry(client.calls)['output_tokens'], 4)
        self.assertEqual(telemetry(client.calls)['stop_reasons'], ['end_turn'])

    def test_diagnostics_distinguish_missing_citation_from_provider_failure(self):
        row=dict(answer='x', expected_answer='x', answer_correct=True, citation_valid=False, project='CodexLaw', common_success=False,
                 citation_ids=[], model_response='{"answer":"x","citation_ids":[]}', error='Workflow violation: CITATION_VERIFICATION',provider_error=None)
        result=diagnose(row,self.case)
        self.assertEqual(result['category'],'仅引用未通过')
        self.assertEqual(result['reasons'][0]['code'],'citation_missing')
        self.assertEqual(result['token_state'],'unknown')
        row['telemetry']=dict(stop_reasons=['max_tokens'],all_calls_have_stop_reason=True)
        self.assertEqual(diagnose(row,self.case)['token_state'],'limit_hit')

    def test_source_quota_is_not_lost_when_one_source_dominates_global_topk(self):
        import numpy as np
        rows=[dict(document_id=f'a:{i}',text='a') for i in range(100)]+[dict(document_id='b:1',text='b')]
        matrix=np.array([[1.,0.]]*100+[[0.,1.]])
        hits=_top_hits(rows,matrix,[[1.,.1]],per_source=1)[0]
        self.assertEqual({r['document_id'].split(':')[0] for r in hits},{'a','b'})

    def test_nemotron_v1_applies_different_prefixes_for_queries_and_documents(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps({'data':[{'index':0,'embedding':[1.,0.]}]}).encode()
        client=NvidiaEmbeddingClient(base_url='http://private/v1',api_key=None,model='nvidia/Nemotron-3-Embed-1B-BF16',require_api_key=False)
        with patch('codelaw.nvidia.urlopen',return_value=Response()) as request:
            client.embed(['text'])
            self.assertEqual(json.loads(request.call_args.args[0].data)['input'],['query: text'])
            client.embed(['text'],input_type='passage')
            self.assertEqual(json.loads(request.call_args.args[0].data)['input'],['passage: text'])

    def test_embedding_key_requires_custom_endpoint_opt_in(self):
        with self.assertRaisesRegex(SettingsError, "NVIDIA_ALLOW_CUSTOM_ENDPOINT"):
            NvidiaEmbeddingClient(base_url='https://collector.invalid/v1', api_key='test-key', model='embed-model')

    def test_native_adapter_preserves_text_and_marks_output_limit(self):
        prompt=messages_text({'input':[{'role':'user','content':[{'type':'input_text','text':'case text'}]}]})
        self.assertIn('case text',prompt)
        with self.assertRaises(ValueError):messages_text({'input':[{'type':'function_call_output'}]})
        events=[json.loads(line[6:]) for line in response_events('answer',{'input_tokens':5,'output_tokens':9},limited=True).decode().splitlines() if line.startswith('data: ')]
        self.assertEqual(events[-1]['type'],'response.incomplete')
        self.assertEqual(events[-1]['response']['usage']['total_tokens'],14)
        self.assertEqual(events[-1]['response']['output'][0]['content'][0]['text'],'answer')

    def test_both_project_workers_receive_identical_budget_deadline_and_input(self):
        calls=[]
        def run(command,**kwargs):
            calls.append((command,kwargs))
            return subprocess.CompletedProcess(command,0,stdout='{"response":"ok","error":null}',stderr='')
        runtime=MiniMaxRuntime(timeout_seconds=37,max_tokens=1536,temperature=.2)
        with patch('scripts.run_project_ab_benchmark.run_process',side_effect=run):
            for project in ['Lawgent','CodexLaw']:_project_response(self.case,project=project,runtime=runtime,codex_engine='cli')
        self.assertEqual(calls[0][1]['input'],calls[1][1]['input'])
        for command,kwargs in calls:
            self.assertEqual(kwargs['timeout'],37)
            self.assertEqual(command[command.index('--max-tokens')+1],'1536')
            self.assertEqual(command[command.index('--timeout')+1],'37')

    def test_graph_rejects_a_source_outside_the_manifest(self):
        graph=object.__new__(AuthorityGraph)
        graph.allowed_hashes={'trusted'}
        graph._validate({'sha256':'trusted'})
        with self.assertRaises(ValueError):graph._validate({'sha256':'different'})


if __name__ == '__main__':unittest.main()
