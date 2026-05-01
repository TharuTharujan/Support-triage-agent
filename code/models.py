from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    path: Path
    company: str
    product_area: str
    title: str
    text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class Chunk:
    document: Document
    text: str
    tokens: frozenset[str]
    index: int


@dataclass(frozen=True)
class RetrievalHit:
    score: float
    chunk: Chunk


@dataclass(frozen=True)
class TriageResult:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


@dataclass(frozen=True)
class RunSummary:
    llm_provider_name: str = "none"
    llm_generation_used: bool = False
