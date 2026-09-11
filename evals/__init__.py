"""AegisMem evaluation & regression harness (Phase 8).

A categorized golden dataset run through deterministic, offline evaluators over
the *real* subsystems. The harness aggregates per-category pass rates and the
cross-cutting invariants (memory-corruption, unauthorized-tool-exec,
attack-success-rate) into a report the CI gate blocks a regression on.
"""
