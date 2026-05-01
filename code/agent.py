from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from generator import (
    escalation_response,
    grounded_response,
    invalid_response,
    justification_for_hits,
)
from llm_client import OptionalLLMClient
from loader import load_corpus
from models import Document, RetrievalHit, RunSummary, TriageResult
from retriever import HybridRetriever
from rules import escalation_reason, infer_company, is_invalid, product_area, request_type
from validator import OUTPUT_FIELDS, result_to_row, validate_output_rows


class SupportTriageAgent:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.documents: list[Document] = load_corpus(data_dir)
        self.retriever = HybridRetriever(self.documents)
        self.llm_client = OptionalLLMClient()

    def triage(self, issue: str, subject: str = "", company: str = "") -> TriageResult:
        issue = issue or ""
        subject = subject or ""
        company = company or ""
        combined = f"{subject}\n{issue}".strip()
        inferred_company = infer_company(company, combined)
        kind = request_type(combined, inferred_company)

        if is_invalid(combined, inferred_company):
            return TriageResult(
                status="replied",
                product_area="unsupported",
                response=invalid_response(),
                justification="The message is unrelated to the supported local product corpus.",
                request_type="invalid",
            )

        matches = self.retrieve(combined, inferred_company, limit=4)
        area = product_area(combined, inferred_company, matches)
        reason = escalation_reason(combined, inferred_company)
        unsupported = not matches or matches[0].score < 0.05

        if reason or unsupported:
            escalation_basis = reason or "the local corpus did not contain enough matching support guidance"
            return TriageResult(
                status="escalated",
                product_area=area,
                response=escalation_response(),
                justification=f"Escalated because {escalation_basis}.",
                request_type=kind,
            )

        return TriageResult(
            status="replied",
            product_area=area,
            response=grounded_response(inferred_company, area, matches, self.llm_client),
            justification=justification_for_hits(matches, self.llm_client),
            request_type=kind,
        )

    def retrieve(
        self, query: str, company: str = "", limit: int = 5
    ) -> list[RetrievalHit]:
        return self.retriever.retrieve(query, company, limit)


def process_csv(input_path: Path, output_path: Path, data_dir: Path) -> RunSummary:
    agent = SupportTriageAgent(data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for row in _read_input_rows(input_path):
        issue = row.get("Issue") or row.get("issue") or ""
        subject = row.get("Subject") or row.get("subject") or ""
        company = row.get("Company") or row.get("company") or ""
        rows.append(result_to_row(agent.triage(issue, subject, company), issue, subject, company))
    validate_output_rows(rows)
    if _write_with_pandas(rows, output_path):
        return _run_summary(agent)
    _write_with_csv(rows, output_path)
    return _run_summary(agent)


def _run_summary(agent: SupportTriageAgent) -> RunSummary:
    return RunSummary(
        llm_provider_name=agent.llm_client.provider_name,
        llm_generation_used=agent.llm_client.generation_used,
    )


def _read_input_rows(input_path: Path) -> list[dict[str, str]]:
    try:
        import pandas as pd
    except Exception:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    return frame.to_dict(orient="records")


def _write_with_pandas(rows: list[dict[str, str]], output_path: Path) -> bool:
    try:
        import pandas as pd
    except Exception:
        return False
    frame = pd.DataFrame(rows, columns=OUTPUT_FIELDS)
    validate_output_rows(frame.to_dict(orient="records"))
    frame.to_csv(output_path, index=False)
    return True


def _write_with_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def iter_results(output_path: Path) -> Iterable[dict[str, str]]:
    with output_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            yield {field: row.get(field, "") for field in OUTPUT_FIELDS}
