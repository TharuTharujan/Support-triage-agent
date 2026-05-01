from __future__ import annotations

import re

from models import Chunk, Document


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "when",
    "where",
    "why",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9_'-]*", text.lower()):
        if token in STOPWORDS or len(token) <= 1:
            continue
        tokens.append(token)
        tokens.extend(_variants(token))
    return tokens


def _variants(token: str) -> list[str]:
    variants: list[str] = []
    if len(token) > 4 and token.endswith("s"):
        variants.append(token[:-1])
    if len(token) > 5 and token.endswith("ing"):
        variants.append(token[:-3])
        if token[:-3].endswith("t"):
            variants.append(token[:-3] + "e")
    return variants


def chunk_document(document: Document, max_words: int = 180) -> list[Chunk]:
    paragraphs = [
        re.sub(r"\s+", " ", line.strip())
        for line in document.text.splitlines()
        if line.strip() and not line.strip().startswith(("!", "|"))
    ]
    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        words = paragraph.split()
        if current and current_words + len(words) > max_words:
            chunks.append(_make_chunk(document, current, len(chunks)))
            current = []
            current_words = 0
        current.append(paragraph)
        current_words += len(words)
    if current:
        chunks.append(_make_chunk(document, current, len(chunks)))
    return chunks or [_make_chunk(document, [document.title], 0)]


def _make_chunk(document: Document, paragraphs: list[str], index: int) -> Chunk:
    text = " ".join(paragraphs)
    token_text = f"{document.path.stem} {text}"
    return Chunk(
        document=document,
        text=text,
        tokens=frozenset(tokenize(token_text)),
        index=index,
    )
