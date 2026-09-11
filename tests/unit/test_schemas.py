"""Phase 1 gate: schema validation for memory records and the runtime contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aegismem.api.models import AgentRequest, AgentResponse, RunMode, RunStatus
from aegismem.errors import ErrorCategory, ErrorEnvelope
from aegismem.memory.models import MemoryRecord, MemoryType


def test_memory_record_defaults() -> None:
    rec = MemoryRecord(type=MemoryType.SEMANTIC, content="hello")
    assert rec.id.startswith("mem_")
    assert rec.version == 1
    assert rec.confidence == 1.0
    assert rec.derived_from == []


def test_memory_record_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(type=MemoryType.SEMANTIC, content="")


def test_memory_record_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(type=MemoryType.SEMANTIC, content="x", confidence=1.5)


def test_agent_request_defaults_and_ids() -> None:
    req = AgentRequest(session_id="s1", input="what changed?")
    assert req.request_id.startswith("req_")
    assert req.mode is RunMode.SYNC


def test_agent_request_rejects_empty_input() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(session_id="s1", input="")


def test_agent_response_carries_error_envelope() -> None:
    env = ErrorEnvelope(
        code="not_found",
        category=ErrorCategory.NOT_FOUND,
        message="missing",
        stage="memory.get",
    )
    resp = AgentResponse(request_id="req_1", run_id="run_1", status=RunStatus.FAILED, error=env)
    assert resp.error is not None
    assert resp.error.category is ErrorCategory.NOT_FOUND
    # Round-trips through JSON without leaking a stack trace.
    assert "not_found" in resp.model_dump_json()
