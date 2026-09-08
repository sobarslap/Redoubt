"""AegisMem service-security subsystem (Production Phase P4).

API-key authentication, per-principal quotas, tenant identity, and an audit log —
the perimeter around the runtime service. Distinct from ``guardrails`` (which
defends the agent's *reasoning* against adversarial content); this defends the
*service* against unauthenticated, over-quota, or cross-tenant access.
"""

from aegismem.security.audit import AuditEntry, AuditLog
from aegismem.security.auth import (
    ApiKeyStore,
    AuthContext,
    FixedWindowQuota,
    Principal,
)

__all__ = [
    "ApiKeyStore",
    "AuditEntry",
    "AuditLog",
    "AuthContext",
    "FixedWindowQuota",
    "Principal",
]
