"""API-key authentication, tenant identity, and per-principal quotas.

Keys are never stored in plaintext at rest: the `ApiKeyStore` holds SHA-256
hashes and compares in constant time, so a leaked store does not leak usable
keys. Each key resolves to a `Principal` carrying its `tenant` — the identity the
tenant-scoped memory store enforces isolation on. Provider/API secrets come from
the environment (see `execution.factory`), never from config files.

The quota is a simple fixed-window counter per principal; a distributed
deployment swaps it for Redis, but the interface (`allow`) is the same.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any

from aegismem.errors import AuthError, QuotaExceededError


@dataclass(frozen=True)
class Principal:
    """An authenticated caller and the tenant it acts within."""

    principal_id: str
    tenant: str
    scopes: frozenset[str] = frozenset()


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class ApiKeyStore:
    """Maps API keys (by hash) to principals. Register with the raw key once;
    only the hash is retained."""

    _by_hash: dict[str, Principal] = field(default_factory=dict)

    def register(self, raw_key: str, principal: Principal) -> None:
        self._by_hash[_hash_key(raw_key)] = principal

    @classmethod
    def from_mapping(cls, mapping: dict[str, Principal]) -> ApiKeyStore:
        store = cls()
        for raw, p in mapping.items():
            store.register(raw, p)
        return store

    def authenticate(self, raw_key: str | None) -> Principal:
        if not raw_key:
            raise AuthError("missing API key")
        candidate = _hash_key(raw_key)
        for known, principal in self._by_hash.items():
            if hmac.compare_digest(candidate, known):  # constant-time
                return principal
        raise AuthError("invalid API key")


@dataclass
class FixedWindowQuota:
    """Per-principal request quota over a fixed time window.

    ``allow`` raises :class:`QuotaExceededError` once a principal exceeds
    ``max_requests`` within ``window_seconds``."""

    max_requests: int = 60
    window_seconds: float = 60.0
    max_tracked: int = 100_000  # cap the principal map; sweep expired past this
    _counts: dict[str, tuple[float, int]] = field(default_factory=dict)

    def allow(self, principal_id: str, *, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        # Lazily evict principals whose window has fully expired so the map can't
        # grow without bound as callers rotate (a distributed deployment uses Redis).
        if len(self._counts) > self.max_tracked:
            self._counts = {
                p: (s, c) for p, (s, c) in self._counts.items() if now - s < self.window_seconds
            }
        start, count = self._counts.get(principal_id, (now, 0))
        if now - start >= self.window_seconds:
            start, count = now, 0  # window rolled over
        count += 1
        self._counts[principal_id] = (start, count)
        if count > self.max_requests:
            raise QuotaExceededError(
                f"quota exceeded for {principal_id!r} "
                f"({self.max_requests}/{self.window_seconds:.0f}s)"
            )


@dataclass
class OIDCVerifier:
    """Verify an OIDC/JWT bearer token and map its claims to a `Principal`.

    Import-guarded (`PyJWT`, `auth` group). In production, pass a `jwks_url` and the
    tokens are verified against the IdP's rotating public keys; for tests/offline a
    static `signing_key` (symmetric) can be supplied instead. The `tenant_claim`
    names the JWT claim that carries the tenant (defaults to ``"tenant"``).
    """

    issuer: str
    audience: str
    jwks_url: str | None = None
    signing_key: str | None = None
    algorithms: tuple[str, ...] = ("RS256",)
    tenant_claim: str = "tenant"
    scopes_claim: str = "scope"
    _jwk_client: Any = field(default=None, repr=False)

    def _key_for(self, token: str) -> Any:
        if self.signing_key is not None:
            return self.signing_key
        import jwt

        if self.jwks_url is None:
            raise AuthError("OIDC verifier needs a signing_key or a jwks_url")
        if self._jwk_client is None:
            self._jwk_client = jwt.PyJWKClient(self.jwks_url)
        return self._jwk_client.get_signing_key_from_jwt(token).key

    def verify(self, bearer_token: str | None) -> Principal:
        if not bearer_token:
            raise AuthError("missing bearer token")
        token = bearer_token.removeprefix("Bearer ").strip()
        try:
            import jwt

            claims = jwt.decode(
                token,
                self._key_for(token),
                algorithms=list(self.algorithms),
                audience=self.audience,
                issuer=self.issuer,
            )
        except AuthError:
            raise
        except Exception as exc:
            raise AuthError(f"invalid bearer token: {exc}") from exc
        sub = claims.get("sub")
        if not sub:
            raise AuthError("token missing subject")
        scopes = claims.get(self.scopes_claim, "")
        scope_set = frozenset(scopes.split()) if isinstance(scopes, str) else frozenset(scopes)
        return Principal(
            principal_id=str(sub),
            tenant=str(claims.get(self.tenant_claim, "default")),
            scopes=scope_set,
        )


@dataclass(frozen=True)
class AuthContext:
    """The resolved principal for a request, plus the raw key hash for auditing."""

    principal: Principal
    key_fingerprint: str  # first 8 hex of the key hash — identifies without leaking
