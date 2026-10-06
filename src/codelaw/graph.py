"""Neo4j authority lookup and auditable one-hop structural expansion."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_GRAPH_SCHEMES = {'bolt', 'bolt+s', 'bolt+ssc', 'neo4j', 'neo4j+s', 'neo4j+ssc'}


def normalise_graph_uri(value: str) -> str:
    """Validate a Neo4j URI and reject embedded credentials or query secrets."""
    raw = str(value).strip().rstrip('/')
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in _GRAPH_SCHEMES or not parsed.hostname:
        raise ValueError('NEO4J_URI must be a bolt/neo4j URI with a hostname')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('NEO4J_URI must not contain userinfo, query, or fragment')
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError('NEO4J_URI contains an invalid port') from exc
    host = parsed.hostname.lower()
    if ':' in host and not host.startswith('['):
        host = f'[{host}]'
    netloc = host if port is None else f'{host}:{port}'
    path = parsed.path.rstrip('/')
    return f'{parsed.scheme.lower()}://{netloc}{path}'


def connection(root: Path) -> tuple[str, str, str]:
    saved = root / '.runtime/graph-connection.json'
    config = json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else {}
    uri = os.environ.get('NEO4J_URI') or config.get('uri')
    user = os.environ.get('NEO4J_USER') or config.get('user', 'neo4j')
    password = os.environ.get('NEO4J_PASSWORD')
    if not password and config.get('password_file'):
        password = Path(config['password_file']).read_text(encoding='utf-8').strip()
    if not uri or not password:
        raise RuntimeError('Set NEO4J_URI/NEO4J_PASSWORD or the ignored .runtime/graph-connection.json before graph RAG')
    return normalise_graph_uri(uri), user, password


class AuthorityGraph:
    def __init__(self, uri: str, user: str, password: str, *, allowed_hashes: set[str]):
        from neo4j import GraphDatabase
        uri = normalise_graph_uri(uri)
        self.driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10, max_transaction_retry_time=5)
        self.allowed_hashes = allowed_hashes
        self.uri = uri

    def close(self):
        self.driver.close()

    def expand(self, authority_ids: list[str], *, limit: int = 4) -> list[dict[str, Any]]:
        if not authority_ids:
            return []
        self.driver.verify_connectivity()
        seeds, _, _ = self.driver.execute_query(
            'MATCH (a:Authority) WHERE a.authority_id IN $ids RETURN properties(a) AS authority ORDER BY a.authority_id',
            ids=authority_ids, database_='neo4j', routing_='r')
        found = {r['authority']['authority_id'] for r in seeds}
        if found != set(authority_ids):
            raise ValueError('Graph is missing retrieved authority IDs: ' + ', '.join(sorted(set(authority_ids)-found)))
        result = []
        for record in seeds:
            authority = record['authority']
            self._validate(authority)
            result.append(dict(authority=authority, relation='SEED_LOOKUP', via=authority['authority_id']))
        records, _, _ = self.driver.execute_query(
            'MATCH (seed:Authority) WHERE seed.authority_id IN $ids '
            'CALL (seed) { '
            'MATCH (seed)-[:REFERENCES]->(related:Authority) '
            'RETURN related, "REFERENCES" AS relation, 0 AS priority '
            'UNION '
            'MATCH (seed)-[:IN_PART]->(:Part)<-[:IN_PART]-(related:Authority) '
            'RETURN related, "SAME_PART" AS relation, 1 AS priority '
            '} '
            'WITH seed, related, relation, priority WHERE NOT related.authority_id IN $ids '
            'RETURN DISTINCT properties(related) AS authority, relation, seed.authority_id AS via, priority '
            'ORDER BY priority, authority.authority_id, via LIMIT $limit',
            ids=authority_ids, limit=limit, database_='neo4j', routing_='r')
        used = set(found)
        for record in records:
            authority = record['authority']
            self._validate(authority)
            if authority['authority_id'] in used:
                continue
            used.add(authority['authority_id'])
            result.append(dict(authority=authority, relation=record['relation'], via=record['via']))
        return result

    def _validate(self, authority: dict[str, Any]):
        if authority.get('sha256') not in self.allowed_hashes:
            raise ValueError('Graph authority hash is outside the local source manifest')

    def counts(self) -> dict[str, int]:
        records, _, _ = self.driver.execute_query(
            'CALL () { MATCH (a:Authority) RETURN count(a) AS authorities } '
            'CALL () { MATCH (p:Part) RETURN count(p) AS parts } '
            'CALL () { MATCH ()-[r:IN_PART]->() RETURN count(r) AS in_part } '
            'CALL () { MATCH ()-[r:REFERENCES]->() RETURN count(r) AS references } '
            'CALL () { MATCH (s:SourceVersion) RETURN count(s) AS source_versions } '
            'RETURN authorities, parts, in_part, references, source_versions', database_='neo4j', routing_='r')
        return dict(records[0])
