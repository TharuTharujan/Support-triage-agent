from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]
INPUT_COLUMNS = ["Issue", "Subject", "Company"]
STATUSES = {"replied", "escalated"}
REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
GENERATED_REQUIRED_FIELDS = ["response", "status", "request_type", "justification"]
GENERIC_ESCALATION_RESPONSE = (
    "I cannot safely resolve this directly from the available support documentation. "
    "I am escalating this to the appropriate support team so they can verify account, "
    "billing, security, service, or authority-specific details."
)
SUPPORTED_PRODUCTS = ("hackerrank", "claude", "visa", "anthropic")
HIGH_RISK_PATTERNS = [
    ("fraud", ["fraud"]),
    ("unauthorized transaction", ["unauthorized transaction", "unauthorised transaction"]),
    ("identity theft", ["identity theft"]),
    ("refund/payment/billing/charge", ["refund", "payment", "billing", "charge", "invoice"]),
    (
        "account access restoration without authority",
        ["restore my access", "removed my seat", "not the workspace owner", "admin removed"],
    ),
    (
        "hiring outcome/score change/rescheduling",
        ["increase my score", "score change", "tell the company", "rejected me", "rescheduling"],
    ),
    (
        "prompt injection/internal rules/system prompt",
        [
            "internal rules",
            "system prompt",
            "developer message",
            "print your instructions",
            "logic exact",
            "exact logic",
            "retrieved documents",
            "documents recuperes",
            "documents récupérés",
        ],
    ),
    (
        "legal/privacy sensitive request",
        ["legal", "subpoena", "court order", "law enforcement", "personal data", "private information", "gdpr"],
    ),
]


@dataclass(frozen=True)
class Finding:
    row_number: int | None
    reason: str


@dataclass(frozen=True)
class Evaluation:
    input_exists: bool
    output_exists: bool
    schema_ok: bool
    row_count_ok: bool
    input_rows: int
    output_rows: int
    invalid_values: list[Finding]
    blank_fields: list[Finding]
    preservation_errors: list[Finding]
    warnings: list[Finding]
    status_distribution: Counter[str]
    request_type_distribution: Counter[str]
    product_area_distribution: Counter[str]

    @property
    def failures(self) -> list[Finding]:
        failures: list[Finding] = []
        if not self.input_exists:
            failures.append(Finding(None, "support_tickets/support_tickets.csv is missing"))
        if not self.output_exists:
            failures.append(Finding(None, "support_tickets/output.csv is missing"))
        if not self.schema_ok:
            failures.append(Finding(None, "output.csv columns do not exactly match the required 8-column schema"))
        if not self.row_count_ok:
            failures.append(
                Finding(
                    None,
                    f"row count mismatch: input={self.input_rows}, output={self.output_rows}",
                )
            )
        failures.extend(self.invalid_values)
        failures.extend(self.blank_fields)
        failures.extend(self.preservation_errors)
        return failures


def evaluate(input_path: Path, output_path: Path) -> Evaluation:
    input_exists = input_path.exists()
    output_exists = output_path.exists()
    input_rows: list[dict[str, str]] = []
    output_rows: list[dict[str, str]] = []
    output_columns: list[str] = []
    invalid_values: list[Finding] = []
    blank_fields: list[Finding] = []
    preservation_errors: list[Finding] = []
    warnings: list[Finding] = []

    if input_exists:
        input_rows = _read_rows(input_path)
    if output_exists:
        output_rows, output_columns = _read_output_rows(output_path)

    schema_ok = output_columns == REQUIRED_COLUMNS if output_exists else False
    row_count_ok = len(input_rows) == len(output_rows) if input_exists and output_exists else False

    if output_exists:
        invalid_values.extend(_invalid_value_findings(output_rows))
        blank_fields.extend(_blank_field_findings(output_rows))
        warnings.extend(_product_area_warnings(output_rows))
        warnings.extend(_suspicious_vague_row_warnings(output_rows))
        warnings.extend(_high_risk_not_escalated_warnings(output_rows))
        warnings.extend(_unsupported_company_warnings(output_rows))
        warnings.extend(_overly_generic_response_warnings(output_rows))

    if input_exists and output_exists:
        preservation_errors.extend(_preservation_findings(input_rows, output_rows))

    return Evaluation(
        input_exists=input_exists,
        output_exists=output_exists,
        schema_ok=schema_ok,
        row_count_ok=row_count_ok,
        input_rows=len(input_rows),
        output_rows=len(output_rows),
        invalid_values=invalid_values,
        blank_fields=blank_fields,
        preservation_errors=preservation_errors,
        warnings=warnings,
        status_distribution=Counter(_normalized(row.get("status", "")) for row in output_rows),
        request_type_distribution=Counter(_normalized(row.get("request_type", "")) for row in output_rows),
        product_area_distribution=Counter(_display_blank(row.get("product_area", "")) for row in output_rows),
    )


def print_report(report: Evaluation) -> None:
    failures = report.failures
    print("Output CSV Evaluation")
    print("=" * 21)
    print(f"Summary: {'PASS' if not failures else 'FAIL'}")
    print(f"Final recommendation: {'READY' if not failures and not report.warnings else 'NEEDS REVIEW'}")
    print()

    print("Files")
    print("-" * 5)
    print(f"support_tickets.csv exists: {_yes_no(report.input_exists)}")
    print(f"output.csv exists: {_yes_no(report.output_exists)}")
    print()

    print("Row Counts")
    print("-" * 10)
    print(f"input rows: {report.input_rows}")
    print(f"output rows: {report.output_rows}")
    print(f"row count match: {_yes_no(report.row_count_ok)}")
    print()

    print("Schema")
    print("-" * 6)
    print(f"exact 8-column schema: {_yes_no(report.schema_ok)}")
    print(f"required header: {', '.join(REQUIRED_COLUMNS)}")
    print()

    _print_findings("Invalid Values", report.invalid_values)
    _print_findings("Blank Generated Fields", report.blank_fields)
    _print_findings("Preservation Errors", report.preservation_errors)
    _print_counter("Status Distribution", report.status_distribution)
    _print_counter("Request Type Distribution", report.request_type_distribution)
    _print_counter("Product Area Distribution", report.product_area_distribution)
    _print_findings("Warnings", report.warnings)
    _print_findings("Failures", failures)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_output_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _invalid_value_findings(rows: list[dict[str, str]]) -> list[Finding]:
    findings: list[Finding] = []
    for index, row in enumerate(rows, start=1):
        status = _normalized(row.get("status", ""))
        request_type = _normalized(row.get("request_type", ""))
        if status not in STATUSES:
            findings.append(Finding(index, f"invalid status {status!r}"))
        if request_type not in REQUEST_TYPES:
            findings.append(Finding(index, f"invalid request_type {request_type!r}"))
    return findings


def _blank_field_findings(rows: list[dict[str, str]]) -> list[Finding]:
    findings: list[Finding] = []
    for index, row in enumerate(rows, start=1):
        for field in GENERATED_REQUIRED_FIELDS:
            if not _normalized(row.get(field, "")):
                findings.append(Finding(index, f"blank required generated field: {field}"))
    return findings


def _product_area_warnings(rows: list[dict[str, str]]) -> list[Finding]:
    warnings: list[Finding] = []
    for index, row in enumerate(rows, start=1):
        if _normalized(row.get("product_area", "")):
            continue
        if _product_area_blank_allowed(row):
            continue
        warnings.append(Finding(index, "blank product_area outside allowed invalid/generic-outage cases"))
    return warnings


def _preservation_findings(
    input_rows: list[dict[str, str]],
    output_rows: list[dict[str, str]],
) -> list[Finding]:
    findings: list[Finding] = []
    for index, (source, generated) in enumerate(zip(input_rows, output_rows), start=1):
        comparisons = [
            ("issue", source.get("Issue", ""), generated.get("issue", "")),
            ("subject", source.get("Subject", ""), generated.get("subject", "")),
            ("company", source.get("Company", ""), generated.get("company", "")),
        ]
        for field, expected, actual in comparisons:
            if expected != actual:
                findings.append(Finding(index, f"{field} not preserved from input"))
    return findings


def _suspicious_vague_row_warnings(rows: list[dict[str, str]]) -> list[Finding]:
    warnings: list[Finding] = []
    for index, row in enumerate(rows, start=1):
        issue = row.get("issue", "")
        short_issue = len(_tokens(issue)) <= 4
        none_company = _is_none_company(row.get("company", ""))
        generic_response = _looks_overly_generic(row.get("response", ""))
        if short_issue:
            warnings.append(Finding(index, "issue is very short"))
        if none_company and generic_response:
            warnings.append(Finding(index, "None/blank company with overly generic response"))
    return warnings


def _high_risk_not_escalated_warnings(rows: list[dict[str, str]]) -> list[Finding]:
    warnings: list[Finding] = []
    for index, row in enumerate(rows, start=1):
        text = _normalized(" ".join([row.get("subject", ""), row.get("issue", "")]))
        status = _normalized(row.get("status", ""))
        if status == "escalated":
            continue
        for label, patterns in HIGH_RISK_PATTERNS:
            if any(pattern in text for pattern in patterns):
                warnings.append(Finding(index, f"high-risk pattern not escalated: {label}"))
                break
    return warnings


def _unsupported_company_warnings(rows: list[dict[str, str]]) -> list[Finding]:
    warnings: list[Finding] = []
    for index, row in enumerate(rows, start=1):
        company = _normalized(row.get("company", ""))
        if not _is_none_company(company):
            continue
        text = _normalized(" ".join([row.get("subject", ""), row.get("issue", "")]))
        can_infer_product = any(product in text for product in SUPPORTED_PRODUCTS)
        status = _normalized(row.get("status", ""))
        request_type = _normalized(row.get("request_type", ""))
        justification = _normalized(row.get("justification", ""))
        acceptable = (
            can_infer_product
            or (status == "replied" and request_type == "invalid")
            or (status == "escalated" and bool(justification))
        )
        if not acceptable:
            warnings.append(
                Finding(
                    index,
                    "None/blank company without inferred product should usually be invalid or clearly escalated",
                )
            )
    return warnings


def _overly_generic_response_warnings(rows: list[dict[str, str]]) -> list[Finding]:
    warnings: list[Finding] = []
    response_counts = Counter(_normalized(row.get("response", "")) for row in rows)
    generic_escalation = _normalized(GENERIC_ESCALATION_RESPONSE)
    for index, row in enumerate(rows, start=1):
        response = _normalized(row.get("response", ""))
        status = _normalized(row.get("status", ""))
        if response == generic_escalation and response_counts[response] > 1:
            warnings.append(Finding(index, "repeated generic escalation response"))
        if status == "replied" and not _mentions_product_or_context(row):
            warnings.append(Finding(index, "replied response does not mention product or document context"))
    return warnings


def _mentions_product_or_context(row: dict[str, str]) -> bool:
    response = _normalized(row.get("response", ""))
    company = _normalized(row.get("company", ""))
    product_area = _normalized(row.get("product_area", ""))
    markers = [
        "based on",
        "support documentation",
        "support guidance",
        "support article",
        "visa",
        "hackerrank",
        "claude",
        "aws support",
        "issuer or bank",
        "responsible disclosure",
    ]
    if company and company != "none":
        markers.append(company)
    if product_area:
        markers.append(product_area.replace("_", " "))
    return any(marker in response for marker in markers)


def _product_area_blank_allowed(row: dict[str, str]) -> bool:
    request_type = _normalized(row.get("request_type", ""))
    status = _normalized(row.get("status", ""))
    company = _normalized(row.get("company", ""))
    issue = _normalized(row.get("issue", ""))
    generic_outage = _is_none_company(company) and any(
        phrase in issue
        for phrase in ["site is down", "pages are down", "none of the pages", "not working"]
    )
    return request_type == "invalid" or (status == "escalated" and generic_outage)


def _looks_overly_generic(value: str) -> bool:
    text = _normalized(value)
    generic_markers = [
        _normalized(GENERIC_ESCALATION_RESPONSE),
        "cannot safely resolve",
        "outside that scope",
        "provided support documentation",
    ]
    return any(marker in text for marker in generic_markers)


def _tokens(value: str) -> list[str]:
    return [token for token in _normalized(value).replace("/", " ").split() if token]


def _is_none_company(value: str) -> bool:
    normalized = _normalized(value)
    return normalized in {"", "none", "null", "n/a"}


def _normalized(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _display_blank(value: str) -> str:
    normalized = _normalized(value)
    return normalized if normalized else "<blank>"


def _yes_no(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    print("-" * len(title))
    if not counter:
        print("None")
    else:
        for value, count in sorted(counter.items()):
            print(f"{value}: {count}")
    print()


def _print_findings(title: str, findings: list[Finding]) -> None:
    print(title)
    print("-" * len(title))
    if not findings:
        print("None")
    else:
        for finding in findings:
            prefix = f"Row {finding.row_number}: " if finding.row_number is not None else ""
            print(f"{prefix}{finding.reason}")
    print()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "support_tickets" / "support_tickets.csv"
    output_path = repo_root / "support_tickets" / "output.csv"
    print_report(evaluate(input_path, output_path))


if __name__ == "__main__":
    main()
