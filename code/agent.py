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
from rules import (
    escalation_reason,
    infer_company,
    invalid_product_area,
    is_claude_bedrock_support_route,
    is_claude_public_vulnerability_report,
    is_generic_unknown_outage,
    is_hackerrank_hiring_user_removal,
    is_visa_charge_dispute_faq,
    is_invalid,
    product_area,
    request_type,
)
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
                product_area=invalid_product_area(combined, inferred_company),
                response=invalid_response(),
                justification="The message is unrelated to the supported local product corpus.",
                request_type="invalid",
            )

        retrieval_query = _enriched_retrieval_query(combined, inferred_company)
        matches = self.retrieve(retrieval_query, inferred_company, limit=4)
        area = "" if is_generic_unknown_outage(combined, inferred_company) else product_area(combined, inferred_company, matches)
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

        direct_response = _direct_grounded_response(combined, inferred_company, area)
        if direct_response:
            return TriageResult(
                status="replied",
                product_area=area,
                response=direct_response,
                justification=justification_for_hits(matches, self.llm_client),
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


def _enriched_retrieval_query(query: str, company: str) -> str:
    additions: list[str] = []
    if is_hackerrank_hiring_user_removal(query, company):
        additions.append(
            "Teams Management User Management remove users teams company admin "
            "Remove from Teams user roles employees"
        )
    if is_visa_charge_dispute_faq(query, company):
        additions.append(
            "Visa Consumer Support How do I dispute a charge contact your issuer or bank"
        )
    if is_claude_public_vulnerability_report(query, company):
        additions.append(
            "Public Vulnerability Reporting Responsible Disclosure Policy "
            "Model Safety Bug Bounty Program"
        )
    if is_claude_bedrock_support_route(query, company):
        additions.append(
            "Amazon Bedrock Contact AWS Support AWS account manager AWS re:Post support inquiries"
        )
    if not additions:
        return query
    return f"{query}\n{' '.join(additions)}"


def _direct_grounded_response(query: str, company: str, area: str) -> str:
    if is_visa_charge_dispute_faq(query, company):
        return (
            "Based on Visa Consumer Support, to dispute a charge you should contact "
            "your issuer or bank using the phone number on the front or back of your "
            "Visa card. The issuer or bank may require detailed information about "
            "the transaction before resolving the dispute."
        )
    if is_claude_public_vulnerability_report(query, company):
        return (
            "Based on Claude's Public Vulnerability Reporting and Model Safety Bug "
            "Bounty documentation, report the vulnerability through Anthropic's "
            "public responsible disclosure or bug bounty reporting channels. Do not "
            "share or request hidden prompts or private handling details in the ticket."
        )
    if is_claude_bedrock_support_route(query, company):
        return (
            "Based on Claude's Amazon Bedrock support guidance, contact AWS Support "
            "for Claude in Amazon Bedrock support inquiries, or reach out to your "
            "AWS account manager. For community-based support, use AWS re:Post."
        )
    return ""


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
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False).fillna("")
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
