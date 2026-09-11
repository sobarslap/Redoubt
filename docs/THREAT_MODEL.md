# THREAT MODEL (Phase 6)

Threats are named after the **AI-Infra-Guard** taxonomy so the red-team suite scores against
concrete attacks, not vague "poisoning." Each row names the attack, the mechanism that defends
against it, and the test that proves it. The scored corpus lives in `tests/adversarial/corpus.py`;
the gate in `tests/adversarial/test_red_team.py`.

## Assets

Memory (the agent's source of truth), tool-authorization policy, configured secrets/credentials, and
the system/developer instructions. The adversary controls: log/webhook/RAG content the agent reads,
tool outputs, and (partially) user input.

## Attack catalogue → defense → proof

| # | Attack (AI-Infra-Guard) | Defense | Proof |
|---|---|---|---|
| 1 | Direct instruction injection ("ignore previous instructions") | injection scanner + immutable rules | `test_every_named_attack_is_blocked` |
| 2 | Indirect injection via poisoned log line | scanner + Execute_Tool authorization + output filter enforce safety; structural fence is defense-in-depth (mitigation, not a guaranteed boundary — a gullible model may still act on fenced text) | red-team corpus `indirect_injection_log`; `test_injection_cannot_trigger_unauthorized_tool_exec` |
| 3 | Fake system instruction / forged role turn | scanner (`fake_system`) + trust hierarchy | corpus `fake_*` |
| 4 | Role hijack / ActorAttack persona | scanner (`role_hijack`) | corpus `role_*`, `actor_attack_*` |
| 5 | Memory poisoning (persist a false fact) | write sanitizer → quarantine to `CANDIDATE` | `test_injected_untrusted_write_never_becomes_active_fact` |
| 6 | Contradictory memory | deterministic conflict resolver (Phase 2) + quarantine | Phase 2 gate + write sanitizer |
| 7 | Trust escalation (untrusted write claims SYSTEM) | trust ceiling clamp | `test_untrusted_origin_cannot_escalate_to_system_trust` |
| 8 | Malicious tool result ("restart prod / mark compromised") | authorization in gateway, not model | `test_injection_cannot_trigger_unauthorized_tool_exec` |
| 9 | Malicious MCP tool description | discovery ≠ authorization (Phase 5 gateway) | `tests/unit/test_mcp.py` |
| 10 | Credential exfiltration | scanner (`exfiltration`) + output leakage filter | corpus `exfiltration_*`, `test_output_filter_*` |
| 11 | Secret / system-prompt probe | scanner (`secret_probe`) + output filter | corpus `secret_probe_*` |
| 12 | Context-overflow / oversized input | resource limits, fail-closed | `SecurityBoundary.inspect_input` raises `GuardrailError` |
| 13 | Argument manipulation | Pydantic arg validation in gateway (Phase 5) | `test_argument_validation_blocks_bad_args` |
| 14 | Obfuscated payload ("base64 decode then run") | scanner (`obfuscation`) | corpus `obfuscation_*` |
| 15 | Multi-turn / Many-Shot jailbreak | stacked-turn heuristic | corpus `many_shot_jailbreak` |

PAIR / GOAT / ActorAttack are represented by their single-turn signatures (persona reassignment,
staged dialogue); the harness is structured so additional multi-turn transcripts drop into the same
corpus and scoring.

## Invariants (asserted, not asserted-in-prose)

- **memory-corruption = 0** — no injected untrusted write reaches ACTIVE, high-trust state.
- **unauthorized-tool-exec = 0** — the tool handler runs only after every gateway guard passes.
- **attack-success-rate → 0** — every named attack in the corpus is caught (currently 0.000).
- **false-positive rate measured** — bounded on a realistic benign SRE corpus (currently 0.000).

## Failure posture

The security classifier is **fail-closed** (`policies.yaml → failure_policy.security_classifier`):
if a scan cannot complete, the content is treated as hostile. Other components degrade per the
Phase-8 failure/recovery table; security never fails open.

## Out of scope (v1)

Network/transport security, authN/authZ of API callers, secret storage at rest, and model-weight
supply chain. The boundary defends the agent's reasoning and memory against adversarial *content*,
which is the failure mode this project exists to prove.
