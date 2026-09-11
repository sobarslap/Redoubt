"""Provenance graph types.

Edges answer *why does the agent believe this, where did it come from, what
replaced it, and what depends on it?* Three edge kinds are stored, each directed
``src -> dst``:

* ``supersedes``    — ``src`` replaced ``dst``
* ``derived_from``  — ``src`` was produced from ``dst`` (a run/tool output)
* ``supported_by``  — ``src`` is corroborated by ``dst``

Persistence lives on the store (a ``provenance_edges`` table); this module owns
the vocabulary and the read-model returned to callers and the API.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EdgeKind(StrEnum):
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    SUPPORTED_BY = "supported_by"


class ProvenanceView(BaseModel):
    """The provenance neighbourhood of a single memory."""

    memory_id: str
    supersedes: list[str] = Field(default_factory=list)  # this replaced these
    superseded_by: list[str] = Field(default_factory=list)  # these replaced this
    derived_from: list[str] = Field(default_factory=list)  # this came from these
    supports: list[str] = Field(default_factory=list)  # this corroborates these
    supported_by: list[str] = Field(default_factory=list)  # these corroborate this
