"""Production Phase P4 gate — auth, quotas, tenant isolation, secrets hygiene.

Proves: unauthenticated and over-quota requests are refused with typed errors;
one tenant can never read/mutate/observe another tenant's memory (adversarial);
privileged actions are audited; and no secret material lives in config files.
Runs fully in-proc.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aegismem.api.app import create_app
from aegismem.errors import NotFoundError
from aegismem.execution.agent import AgentRuntime
from aegismem.execution.llm import MockProvider
from aegismem.guardrails import SecurityBoundary
from aegismem.memory.models import MemoryCreate, MemoryStatus, MemoryType, TrustLevel
from aegismem.memory.store import SQLiteMemoryStore
from aegismem.memory.tenant import TenantOwnership, TenantScopedMemoryStore
from aegismem.observability.trace_store import JSONLTraceStore
from aegismem.observability.tracer import Tracer
from aegismem.retrieval.bm25 import BM25Index
from aegismem.retrieval.embeddings import HashingEmbedder
from aegismem.retrieval.reranker import OverlapReranker
from aegismem.retrieval.router import JITRouter
from aegismem.retrieval.vector import VectorIndex
from aegismem.security.audit import AuditLog
from aegismem.security.auth import ApiKeyStore, FixedWindowQuota, Principal

_HAS_JWT = importlib.util.find_spec("jwt") is not None


def _runtime(tmp_path) -> AgentRuntime:
    router = JITRouter(BM25Index(), VectorIndex(HashingEmbedder()), OverlapReranker(), top_k=3)
    rt = AgentRuntime(
        llm=MockProvider(),
        router=router,
        tracer=Tracer(JSONLTraceStore(tmp_path / "t.jsonl")),
        boundary=SecurityBoundary(),
    )
    rt.remember("mem_sop", "runbook for checkout")
    return rt


# -- authentication ----------------------------------------------------------


def test_unauthenticated_request_is_401(tmp_path) -> None:
    keys = ApiKeyStore.from_mapping({"secret-a": Principal("alice", "tenant-a")})
    c = TestClient(create_app(_runtime(tmp_path), api_keys=keys))
    r = c.post("/runs", json={"session_id": "s", "input": "checkout?"})
    assert r.status_code == 401
    assert r.json()["category"] == "auth"


def test_invalid_key_is_401(tmp_path) -> None:
    keys = ApiKeyStore.from_mapping({"secret-a": Principal("alice", "tenant-a")})
    c = TestClient(create_app(_runtime(tmp_path), api_keys=keys))
    r = c.post("/runs", json={"session_id": "s", "input": "x"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_valid_key_authorizes(tmp_path) -> None:
    keys = ApiKeyStore.from_mapping({"secret-a": Principal("alice", "tenant-a")})
    c = TestClient(create_app(_runtime(tmp_path), api_keys=keys))
    r = c.post(
        "/runs", json={"session_id": "s", "input": "checkout?"}, headers={"X-API-Key": "secret-a"}
    )
    assert r.status_code == 200


def test_no_key_store_runs_open(tmp_path) -> None:
    # Backward-compatible: without an ApiKeyStore the service is open (dev mode).
    c = TestClient(create_app(_runtime(tmp_path)))
    assert c.post("/runs", json={"session_id": "s", "input": "x"}).status_code == 200


# -- quota -------------------------------------------------------------------


def test_over_quota_is_429(tmp_path) -> None:
    keys = ApiKeyStore.from_mapping({"k": Principal("bob", "tenant-b")})
    quota = FixedWindowQuota(max_requests=2, window_seconds=60)
    c = TestClient(create_app(_runtime(tmp_path), api_keys=keys, quota=quota))
    h = {"X-API-Key": "k"}
    body = {"session_id": "s", "input": "x"}
    assert c.post("/runs", json=body, headers=h).status_code == 200
    assert c.post("/runs", json=body, headers=h).status_code == 200
    r = c.post("/runs", json=body, headers=h)
    assert r.status_code == 429
    assert r.json()["retryable"] is True


# -- audit -------------------------------------------------------------------


def test_privileged_actions_are_audited(tmp_path) -> None:
    keys = ApiKeyStore.from_mapping({"k": Principal("carol", "tenant-c")})
    audit = AuditLog()
    c = TestClient(create_app(_runtime(tmp_path), api_keys=keys, audit=audit))
    c.post("/runs", json={"session_id": "s", "input": "x"}, headers={"X-API-Key": "k"})
    c.post("/runs", json={"session_id": "s", "input": "x"}, headers={"X-API-Key": "bad"})
    actions = [(e.action, e.outcome) for e in audit.entries]
    assert ("run.create", "ok") in actions
    assert ("auth.fail", "denied") in actions


# -- durable audit sink ------------------------------------------------------


def test_jsonl_audit_sink_persists_entries(tmp_path) -> None:
    from aegismem.security.audit import AuditLog, JSONLAuditSink

    path = tmp_path / "audit.jsonl"
    log = AuditLog(sink=JSONLAuditSink(path))
    log.record("run.create", principal_id="alice", tenant="tenant-a", detail="req_1")
    log.record("auth.fail", outcome="denied")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    import json

    first = json.loads(lines[0])
    assert first["action"] == "run.create" and first["principal_id"] == "alice"


# -- OIDC / JWT bearer auth (gated on PyJWT) ---------------------------------


@pytest.mark.skipif(not _HAS_JWT, reason="PyJWT not installed (auth group)")
def test_oidc_bearer_authorizes_and_maps_tenant(tmp_path) -> None:
    import jwt

    from aegismem.security.auth import OIDCVerifier

    secret = "test-signing-secret"
    verifier = OIDCVerifier(
        issuer="https://idp.example",
        audience="aegismem",
        signing_key=secret,
        algorithms=("HS256",),
        tenant_claim="tenant",
    )
    token = jwt.encode(
        {
            "sub": "user-9",
            "tenant": "tenant-z",
            "scope": "runs:write",
            "iss": "https://idp.example",
            "aud": "aegismem",
        },
        secret,
        algorithm="HS256",
    )
    c = TestClient(create_app(_runtime(tmp_path), oidc=verifier))
    # No token -> 401.
    assert c.post("/runs", json={"session_id": "s", "input": "x"}).status_code == 401
    # Valid bearer -> 200.
    r = c.post(
        "/runs",
        json={"session_id": "s", "input": "checkout?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    # And the verifier maps claims to a Principal with the tenant.
    principal = verifier.verify(f"Bearer {token}")
    assert principal.principal_id == "user-9" and principal.tenant == "tenant-z"


@pytest.mark.skipif(not _HAS_JWT, reason="PyJWT not installed (auth group)")
def test_oidc_rejects_tampered_token(tmp_path) -> None:
    from aegismem.errors import AuthError
    from aegismem.security.auth import OIDCVerifier

    verifier = OIDCVerifier(issuer="i", audience="a", signing_key="k", algorithms=("HS256",))
    with pytest.raises(AuthError):
        verifier.verify("Bearer not.a.jwt")


# -- tenant isolation (adversarial) ------------------------------------------


def test_tenant_cannot_read_another_tenants_memory() -> None:
    inner = SQLiteMemoryStore(":memory:")
    own = TenantOwnership()
    a = TenantScopedMemoryStore(inner, "tenant-a", own)
    b = TenantScopedMemoryStore(inner, "tenant-b", own)

    secret = a.create(
        MemoryCreate(
            type=MemoryType.SEMANTIC, content="A's private infra fact", trust=TrustLevel.USER
        )
    )
    a.transition(secret.id, MemoryStatus.ACTIVE, reason="seed", source="t", trigger="t")

    # B cannot get it (existence never leaks), cannot list it, cannot mutate it.
    with pytest.raises(NotFoundError):
        b.get(secret.id)
    assert all(m.id != secret.id for m in b.list())
    with pytest.raises(NotFoundError):
        b.transition(secret.id, MemoryStatus.SUPERSEDED, reason="x", source="t", trigger="t")
    with pytest.raises(NotFoundError):
        b.provenance(secret.id)

    # A still sees its own.
    assert a.get(secret.id).content == "A's private infra fact"
    assert any(m.id == secret.id for m in a.list())
    inner.close()


def test_cross_tenant_edge_is_refused() -> None:
    inner = SQLiteMemoryStore(":memory:")
    own = TenantOwnership()
    a = TenantScopedMemoryStore(inner, "tenant-a", own)
    b = TenantScopedMemoryStore(inner, "tenant-b", own)
    ma = a.create(MemoryCreate(type=MemoryType.SEMANTIC, content="a", trust=TrustLevel.USER))
    mb = b.create(MemoryCreate(type=MemoryType.SEMANTIC, content="b", trust=TrustLevel.USER))
    with pytest.raises(NotFoundError):  # B cannot link to A's memory
        b.add_edge(mb.id, ma.id, "supersedes")
    inner.close()


# -- secrets hygiene ---------------------------------------------------------


def test_config_files_contain_no_secrets() -> None:
    cfg = Path("src/aegismem/config")
    # API-key / token shaped material must never be committed in config.
    patterns = [
        re.compile(p)
        for p in (
            r"sk-[A-Za-z0-9]{16,}",
            r"AKIA[0-9A-Z]{16}",
            r"AIza[0-9A-Za-z_\-]{30,}",
            r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}",
        )
    ]
    for path in cfg.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        for pat in patterns:
            assert not pat.search(text), f"possible secret in {path.name}: {pat.pattern}"
