from __future__ import annotations

import re

from llm_client import OptionalLLMClient
from models import RetrievalHit


def invalid_response() -> str:
    return (
        "I can only help with HackerRank, Claude, or Visa support topics covered by "
        "the provided support documentation. This request is outside that scope, so "
        "I cannot provide a product support action for it."
    )


def escalation_response() -> str:
    return (
        "I cannot safely resolve this directly from the available support documentation. "
        "I am escalating this to the appropriate support team so they can verify account, "
        "billing, security, service, or authority-specific details."
    )


def grounded_response(
    company: str,
    product_area: str,
    hits: list[RetrievalHit],
    llm_client: OptionalLLMClient | None = None,
) -> str:
    lead = {
        "hackerrank": "Based on the HackerRank support documentation",
        "claude": "Based on the Claude support documentation",
        "visa": "Based on the Visa support documentation",
    }.get(company, "Based on the provided support documentation")
    titles = ", ".join(_unique_titles(hits[:2]))
    snippet = _clean_snippet(hits[0].chunk.text if hits else "")
    if llm_client is not None and llm_client.enabled:
        llm_text = llm_client.generate(
            "Write a concise support response from this local context only.\n\n"
            f"Company: {company}\n"
            f"Product area: {product_area}\n"
            f"Retrieved article titles: {titles}\n"
            f"Local context:\n{snippet}\n\n"
            "If the context is insufficient, say the case should be routed to support."
        )
        if llm_text:
            return llm_text
    return (
        f"{lead}, the relevant area is {product_area}. The most relevant article is "
        f"'{titles}'. {snippet} If those steps do not match the user's account, "
        "permissions, or exact product state, route the case to the product support team."
    )


def justification_for_hits(
    hits: list[RetrievalHit],
    llm_client: OptionalLLMClient | None = None,
) -> str:
    titles = "; ".join(_unique_titles(hits[:2]))
    snippet = _clean_snippet(hits[0].chunk.text if hits else "", max_chars=260)
    if llm_client is not None and llm_client.enabled:
        llm_text = llm_client.generate(
            "Write one concise justification for this support triage decision using only "
            "the retrieved local support context.\n\n"
            f"Retrieved article titles: {titles}\n"
            f"Local context:\n{snippet}\n\n"
            "Explain why the response is grounded or why escalation is appropriate. "
            "Do not add policies or facts not present in the context."
        )
        if llm_text:
            return llm_text
    return f"Answered from matching local support articles: {titles}."


def _unique_titles(hits: list[RetrievalHit]) -> list[str]:
    titles: list[str] = []
    for hit in hits:
        title = hit.chunk.document.title
        if title not in titles:
            titles.append(title)
    return titles


def _clean_snippet(text: str, max_chars: int = 360) -> str:
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"\s+", " ", line.strip())
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            continue
        if stripped.startswith("[") or stripped.startswith("!"):
            continue
        lines.append(stripped)
        if sum(len(item) for item in lines) > max_chars:
            break
    snippet = " ".join(lines)
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return snippet or "Use the matching support article for the exact product guidance."
