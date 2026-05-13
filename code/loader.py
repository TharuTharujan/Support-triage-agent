from __future__ import annotations

import re
from pathlib import Path

from chunker import tokenize
from models import Document


ALLOWED_CORPUS_ROOTS = {"claude", "hackerrank", "visa"}


def load_corpus(data_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative = path.relative_to(data_dir)
        parts = relative.parts
        company = parts[0].lower() if parts else "unknown"
        if company not in ALLOWED_CORPUS_ROOTS:
            continue
        title = _title_from_markdown(path, text)
        documents.append(
            Document(
                path=path,
                company=company,
                product_area=_area_from_path(company, parts),
                title=title,
                text=text,
                tokens=frozenset(tokenize(f"{title} {path.stem} {text}")),
            )
        )
    return documents


def normalize_area(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _area_from_path(company: str, parts: tuple[str, ...]) -> str:
    if company == "visa":
        if "travel-support" in parts or "travelers-cheques.md" in parts:
            return "travel_support"
        if "small-business" in parts and len(parts) > 2:
            return normalize_area(Path(parts[-1]).stem)
        if "small-business" in parts:
            return "small_business"
        if "merchant" in parts:
            return "merchant_support"
        return "general_support"

    if company == "claude":
        preferred = {
            "account-management": "account_management",
            "conversation-management": "conversation_management",
            "features-and-capabilities": "features_and_capabilities",
            "privacy-and-security": "privacy",
            "trust-and-safety": "privacy",
            "troubleshooting": "troubleshooting",
            "usage-and-limits": "usage_and_limits",
            "pricing-and-billing": "billing",
            "claude-api-and-console": "api_console",
            "claude-code": "claude_code",
            "amazon-bedrock": "amazon_bedrock",
            "claude-for-education": "education",
            "claude-desktop": "desktop",
        }
        for part in parts:
            if part in preferred:
                return preferred[part]
        return "claude"

    if company == "hackerrank":
        for part in parts[1:]:
            if part != "index.md":
                return normalize_area(part)
        return "hackerrank"

    return normalize_area(parts[1]) if len(parts) > 1 else "general"


def _title_from_markdown(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("-", " ").replace("%2c", ",")
