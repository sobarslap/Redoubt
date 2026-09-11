# SECURITY — Trust Model & Guardrails (Phase 6)

AegisMem's security posture is a **structural trust boundary**, not "a prompt-injection
classifier." External content is *data, never instructions*, and that property is enforced by
architecture first and by a detector second. This document is the reproducible spec behind the
`src/aegismem/guardrails/` subsystem and the red-team gate in `tests/adversarial/`.

## Trust hierarchy

Every piece of context carries a trust label (`aegismem.memory.models.TrustLevel`). Privilege
decreases down the list; everything **below `USER` is untrusted data**:

```
SYSTEM      (highest — the immutable rules)
DEVELOPER   (trusted configuration)
USER        (user-controlled input)
──────────────────────────────────────  ← trust boundary
MEMORY        }
TOOL_RESULT   }  untrusted data — may be attacker-controlled
RAG_CONTENT   }
```

`guardrails.trust.is_untrusted` and `trust_rank` are the single source of truth for this ordering.

## The five layers (plan §6)

1. **Privilege separation** — trust labels attached at ingestion; `assemble_context` places the
   immutable-rules block first, trusted material next, and untrusted segments last, each fenced.
2. **Input sanitization** — resource limits (`max_input_chars`), the injection scan, and a PII scan
   run on inbound content at the boundary.
3. **Prompt hardening / scope** — the `IMMUTABLE_RULES` block plus **randomized-delimiter fences**
   around untrusted data. The nonce is unpredictable per assembly, so injected text cannot forge a
   closing fence to structurally "escape" the fenced region. This is a *mitigation*, not an enforced
   boundary: a sufficiently gullible model can still be persuaded to act on fenced instructions
   (see `GullibleAdversaryModel` in the red-team suite), so the fence reduces attack surface but does
   not by itself guarantee the model ignores injected content.
4. **Output validation / authorization** — the outbound leakage filter scrubs known secrets and PII;
   tool authorization stays with the Execute_Tool gateway (Phase 5), never the model. This is the
   control that actually *enforces* safety: even a model fully taken in by an injection cannot run an
   unauthorized tool or exfiltrate a secret, because authorization and output filtering sit outside it.
5. **Monitoring** — every guardrail decision is a typed, loggable verdict (Phase 7 traces it).

Enforcement lives in the scanner, the Execute_Tool authorization gateway, and the output filter —
controls outside the model. The fences and immutable rules are defense-in-depth that shrink the
attack surface; they are not relied on as the sole barrier against indirect injection.

## Injection detector

`guardrails.injection.PromptInjectionScanner` is a deterministic, weighted regex taxonomy over named
attack families (`AttackFamily`): instruction override, fake-system, role hijack, exfiltration,
tool hijack, memory poisoning, secret probe, many-shot, obfuscation. It returns a typed
`InjectionVerdict{blocked, score, families, matches}` and is **fail-closed** — an input it cannot
scan is reported as blocked, never allowed.

## Ingest choke point — how the invariants hold

- **`memory_corruption = 0`**: `guardrails.sanitize.MemoryWriteSanitizer` runs on every write.
  Externally-originated writes cannot be stored at a trust more privileged than the ceiling
  (default `USER`), even if the payload claims `SYSTEM`. Injection-bearing untrusted writes are
  **quarantined to `CANDIDATE`** so they can never be retrieved as an ACTIVE fact — the poisoned
  log is still visible to the agent as data, but never becomes truth.
- **`unauthorized_tool_exec = 0`**: authorization lives in the Execute_Tool gateway, not the model.
  A poisoned tool result instructing "restart prod" is flagged as carrying instructions; the gateway
  refuses the call regardless, and the handler never runs.

## Measured results (reproducible)

```
uv run python -m benchmarks.security.run
```

| Metric | Value | Target |
|---|---|---|
| Attack-success-rate (17 named attacks) | **0.000** | 0 |
| False-positive rate (benign SRE corpus) | **0.000** | ≤ 0.10 |
| Memory-corruption | **0** | 0 |
| Unauthorized-tool-exec | **0** | 0 |

Per-family coverage and the CSV are written to `benchmarks/security/results/`. The gate lives in
`tests/adversarial/test_red_team.py`; a rise in attack-success-rate fails CI.

## End-to-end red-team (Production P6)

Phase 6 scored the injection *scanner* in isolation. Production P6 scores the
**whole runtime**: attacks are seeded into untrusted content and a fully-compliant
"gullible" model — one that emits whatever tool call the injected text asks for
and parrots any secret it sees — drives the run
(`tests/adversarial/test_redteam_e2e.py`). The measured end-to-end
attack-success-rate is **0** regardless, because the defense is *structural*: the
Execute_Tool gateway refuses the unauthorized call and the output filter scrubs
the secret no matter how thoroughly the model was fooled. A live-model variant is
network+key gated for a real deployment. An optional ML classifier
(`guardrails.classifier`, `deberta-v3-prompt-injection`, `guard-ml` group) layers
on as defense-in-depth via `CombinedScanner`. CI adds a dependency CVE scan
(`pip-audit`) and an SBOM (CycloneDX).

## Configuration

All thresholds are in `src/aegismem/config/policies.yaml` under `security:` — block threshold,
resource limits, the untrusted-write trust ceiling, and `fail_closed`. Nothing is hardcoded in the
enforcement path.
