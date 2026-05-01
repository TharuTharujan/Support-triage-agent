from __future__ import annotations

from models import TriageResult


OUTPUT_FIELDS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]
RESULT_FIELDS = ["response", "product_area", "status", "request_type", "justification"]
STATUSES = {"replied", "escalated"}
REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}


def validate_result(result: TriageResult) -> None:
    if result.status not in STATUSES:
        raise ValueError(f"Invalid status: {result.status}")
    if result.request_type not in REQUEST_TYPES:
        raise ValueError(f"Invalid request_type: {result.request_type}")
    for field_name in RESULT_FIELDS:
        value = getattr(result, field_name)
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        if not value.strip():
            raise ValueError(f"{field_name} must not be blank")


def result_to_row(
    result: TriageResult,
    issue: str,
    subject: str,
    company: str,
) -> dict[str, str]:
    validate_result(result)
    return {
        "issue": issue,
        "subject": subject,
        "company": company,
        "response": result.response,
        "product_area": result.product_area,
        "status": result.status,
        "request_type": result.request_type,
        "justification": result.justification,
    }


def validate_output_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if list(row.keys()) != OUTPUT_FIELDS:
            raise ValueError(f"Output columns must be exactly: {OUTPUT_FIELDS}")
        if row["status"] not in STATUSES:
            raise ValueError(f"Invalid status: {row['status']}")
        if row["request_type"] not in REQUEST_TYPES:
            raise ValueError(f"Invalid request_type: {row['request_type']}")
        for field_name in OUTPUT_FIELDS:
            if field_name in {"subject", "company"}:
                continue
            if not str(row.get(field_name, "")).strip():
                raise ValueError(f"{field_name} must not be blank")
