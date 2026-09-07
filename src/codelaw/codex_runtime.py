"""Run the actual Codex CLI through a bounded text-only MiniMax adapter.

The adapter implements the Responses transport used by Codex, while reusing
the same MiniMax Messages client/budget as Lawgent. External tools are disabled
for this supplied-evidence experiment; this is not a general Responses proxy.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .minimax import MiniMaxChatClient, MiniMaxRuntime
from .process import run_process


def messages_text(payload: dict[str, Any]) -> str:
    items = payload.get('input', [])
    if isinstance(items, str):
        return items
    parts = []
    for item in items:
        if item.get('type', 'message') != 'message':
            # No tool results or hidden reasoning are valid in this experiment.
            raise ValueError('The benchmark adapter accepts text messages only')
        content = item.get('content', '')
        if isinstance(content, list):
            if any(c.get('type') not in ('input_text', 'output_text', 'text') for c in content):
                raise ValueError('Non-text content is disabled in the legal benchmark')
            content = '\n'.join(str(c.get('text', '')) for c in content)
        parts.append(f"[{item.get('role', 'user')}]\n{content}")
    return '\n\n'.join(parts)


def response_events(text: str, usage: dict[str, Any], *, limited: bool) -> bytes:
    response_id, item_id = 'resp_'+secrets.token_hex(12), 'msg_'+secrets.token_hex(12)
    part = dict(type='output_text', text=text, annotations=[])
    item = dict(id=item_id, type='message', role='assistant', status='completed', content=[part])
    response = dict(id=response_id, object='response', status='incomplete' if limited else 'completed',
                    model='MiniMax-M3', output=[item], usage=dict(input_tokens=usage.get('input_tokens', 0),
                    output_tokens=usage.get('output_tokens', 0), total_tokens=usage.get('input_tokens', 0)+usage.get('output_tokens', 0)),
                    incomplete_details={'reason':'max_output_tokens'} if limited else None)
    events = [
        dict(type='response.created', response=dict(response, status='in_progress', output=[])),
        dict(type='response.output_item.added', output_index=0, item=dict(item,status='in_progress',content=[])),
        dict(type='response.content_part.added', item_id=item_id, output_index=0, content_index=0, part=dict(part,text='')),
        dict(type='response.output_text.delta', item_id=item_id, output_index=0, content_index=0, delta=text),
        dict(type='response.output_text.done', item_id=item_id, output_index=0, content_index=0, text=text),
        dict(type='response.content_part.done', item_id=item_id, output_index=0, content_index=0, part=part),
        dict(type='response.output_item.done', output_index=0, item=item),
        dict(type='response.incomplete' if limited else 'response.completed', response=response),
    ]
    return ''.join('event: '+e['type']+'\ndata: '+json.dumps(dict(e, sequence_number=i), ensure_ascii=False)+'\n\n' for i,e in enumerate(events)).encode('utf-8')


def run_codex(prompt: str, runtime: MiniMaxRuntime, workdir: Path) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    catalog = workdir / 'model-catalog.json'
    catalog.write_text(json.dumps(dict(models=[dict(slug=runtime.model, display_name=runtime.model,
        description='MiniMax legal benchmark', default_reasoning_level='none',
        supported_reasoning_levels=[dict(effort='none',description='Text benchmark')],
        shell_type='shell_command', visibility='list', supported_in_api=True, priority=0,
        base_instructions='You are Codex, an agent evaluating a supplied-evidence legal task. Follow the user output contract. External tools are disabled for this benchmark.',
        supports_reasoning_summaries=False, default_reasoning_summary='none', support_verbosity=False,
        truncation_policy=dict(mode='bytes',limit=10000), supports_parallel_tool_calls=False,
        experimental_supported_tools=[], context_window=1000000, max_context_window=1000000,
        input_modalities=['text'])])), encoding='utf-8')
    client = MiniMaxChatClient(runtime)
    requests = []
    local_token = secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path.split('?')[0] not in ('/responses', '/v1/responses'):
                self.send_error(404)
                return
            if self.headers.get('Authorization') != 'Bearer '+local_token:
                self.send_error(403)
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                result, last = client.ask_with_trace(messages_text(payload), system=str(payload.get('instructions') or ''))
                body = response_events(result, last.get('usage') or {}, limited=last.get('stop_reason') in ('max_tokens','length'))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({'error':{'message':str(exc),'type':'benchmark_adapter_error'}}).encode()
                self.send_response(502)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = workdir / 'answer.txt'
    config = dict(model_provider='legal_minimax', model_catalog_json=str(catalog.resolve()),
                  model_reasoning_effort='none', approval_policy='never', project_doc_max_bytes=0, web_search='disabled',
                  **{'features.shell_tool':False, 'skills.include_instructions':False, 'features.recommended_plugins':False,
                     'features.apps':False, 'features.plugins':False,
                     'model_providers.legal_minimax.name':'MiniMax benchmark adapter',
                     'model_providers.legal_minimax.base_url':f'http://127.0.0.1:{server.server_port}/v1',
                     'model_providers.legal_minimax.wire_api':'responses',
                     'model_providers.legal_minimax.experimental_bearer_token':local_token,
                     'model_providers.legal_minimax.request_max_retries':0})
    command = ['codex.cmd', 'exec', '--ignore-user-config', '--strict-config', '--skip-git-repo-check', '--ephemeral',
               '--sandbox','read-only','--cd',str(workdir.resolve()), '--model',runtime.model,'--json','--color','never',
               '--output-last-message',str(output.resolve())]
    for key,value in config.items():
        command.extend(['-c',key+'='+json.dumps(value)])
    command.append('-')
    events, error = [], None
    environment = dict(os.environ)
    isolated_home = workdir / 'codex-home'
    isolated_home.mkdir(exist_ok=True)
    environment['CODEX_HOME'] = str(isolated_home.resolve())
    try:
        result = run_process(command, input=prompt, capture_output=True, text=True, encoding='utf-8', errors='replace',
                             timeout=runtime.timeout_seconds, check=False, env=environment)
        for line in result.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if result.returncode:
            error = f'Codex CLI exited {result.returncode}: '+result.stderr[-1800:].replace(local_token,'[redacted]')
        response = output.read_text(encoding='utf-8').strip() if output.exists() else ''
        if not response and not error:
            error = 'Codex CLI returned no final answer'
    except Exception as exc:
        response, error = '', f'{type(exc).__name__}: Codex CLI did not finish within the case deadline'
    finally:
        server.shutdown()
        server.server_close()
    return dict(response=response, error=error, calls=client.calls, events=events, codex_requests=requests,
                engine='codex-cli', tool_policy='text-only supplied evidence; no shell, web, or external tools',
                adapter='Responses SSE to same MiniMax Messages CLI; shared max_tokens and temperature')
