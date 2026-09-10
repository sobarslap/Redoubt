"""Production Phase P8 gate — deploy artifacts exist, are well-formed, and the
smoke checks pass against a served app.

Doesn't build the image in CI (that's the deploy workflow's job), but locks in
that the Dockerfile, compose files, deploy workflow, and runbook are present and
coherent, that the ASGI factory serves the smoke endpoints, and that a version is
exposed for image tagging.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from aegismem import __version__
from aegismem.api.app import create_app


def test_deploy_artifacts_present() -> None:
    for rel in (
        "Dockerfile",
        ".dockerignore",
        "ops/docker-compose.yml",
        "ops/docker-compose.langfuse.yml",
        "ops/alerts.yml",
        "ops/smoke.py",
        ".github/workflows/deploy.yml",
        "docs/RUNBOOK.md",
    ):
        assert Path(rel).exists(), f"missing deploy artifact: {rel}"


def test_dockerfile_runs_as_nonroot_and_serves() -> None:
    text = Path("Dockerfile").read_text(encoding="utf-8")
    assert "USER aegis" in text  # not root
    assert "aegismem" in text and "serve" in text  # entrypoint runs the service
    assert "HEALTHCHECK" in text


def test_compose_files_are_valid_yaml_with_expected_services() -> None:
    stack = yaml.safe_load(Path("ops/docker-compose.yml").read_text(encoding="utf-8"))
    assert set(stack["services"]) >= {"db", "service"}
    assert stack["services"]["db"]["image"].startswith("pgvector/pgvector")
    lf = yaml.safe_load(Path("ops/docker-compose.langfuse.yml").read_text(encoding="utf-8"))
    assert "langfuse" in lf["services"]


def test_alerts_yaml_is_valid_and_references_metrics() -> None:
    rules = yaml.safe_load(Path("ops/alerts.yml").read_text(encoding="utf-8"))
    exprs = " ".join(r["expr"] for g in rules["groups"] for r in g["rules"])
    assert "aegismem_runs_total" in exprs
    assert "aegismem_run_latency_ms" in exprs


def test_version_is_exposed_for_image_tagging() -> None:
    assert __version__
    assert TestClient(create_app()).get("/healthz").json()["status"] == "ok"


def test_smoke_checks_pass_against_served_app(tmp_path) -> None:
    # Exercise the smoke logic in-process (the deploy pipeline runs it over HTTP).
    c = TestClient(create_app())
    assert c.get("/healthz").status_code == 200
    assert c.get("/readyz").status_code == 200
    run = c.post("/runs", json={"session_id": "smoke", "input": "ping"})
    assert run.status_code == 200 and "output" in run.json()
    metrics = c.get("/metrics")
    assert metrics.status_code == 200 and "aegismem_runs_total" in metrics.text
