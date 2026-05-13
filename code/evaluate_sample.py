from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from agent import process_csv


COMPARE_FIELDS = [
    ("status", "Status"),
    ("product_area", "Product Area"),
    ("request_type", "Request Type"),
]


@dataclass(frozen=True)
class FieldScore:
    field: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total * 100.0) if self.total else 0.0


@dataclass(frozen=True)
class FieldMismatch:
    row_number: int
    field: str
    expected: str
    generated: str
    issue: str
    subject: str
    company: str


@dataclass(frozen=True)
class EvaluationReport:
    row_count: int
    scores: list[FieldScore]
    mismatches: list[FieldMismatch]


def evaluate_sample(
    sample_path: Path,
    temp_output_path: Path,
    data_dir: Path,
) -> EvaluationReport:
    try:
        process_csv(sample_path, temp_output_path, data_dir)
        expected_rows = _read_rows(sample_path)
        generated_rows = _read_rows(temp_output_path)
        return compare_rows(expected_rows, generated_rows)
    finally:
        temp_output_path.unlink(missing_ok=True)


def compare_rows(
    expected_rows: list[dict[str, str]],
    generated_rows: list[dict[str, str]],
) -> EvaluationReport:
    if len(expected_rows) != len(generated_rows):
        raise ValueError(
            f"Row count mismatch: expected {len(expected_rows)}, generated {len(generated_rows)}"
        )

    scores: list[FieldScore] = []
    mismatches: list[FieldMismatch] = []
    for generated_field, expected_field in COMPARE_FIELDS:
        correct = 0
        for index, (expected, generated) in enumerate(
            zip(expected_rows, generated_rows), start=1
        ):
            expected_value = _normalize(expected.get(expected_field, ""))
            generated_value = _normalize(generated.get(generated_field, ""))
            if expected_value == generated_value:
                correct += 1
                continue
            mismatches.append(
                FieldMismatch(
                    row_number=index,
                    field=generated_field,
                    expected=expected_value,
                    generated=generated_value,
                    issue=expected.get("Issue") or expected.get("issue") or "",
                    subject=expected.get("Subject") or expected.get("subject") or "",
                    company=expected.get("Company") or expected.get("company") or "",
                )
            )
        scores.append(FieldScore(generated_field, correct, len(expected_rows)))
    return EvaluationReport(len(expected_rows), scores, mismatches)


def print_report(report: EvaluationReport) -> None:
    print("Sample Support Triage Evaluation")
    print("=" * 32)
    print(f"Rows compared: {report.row_count}")
    for score in report.scores:
        print(
            f"{score.field}: {score.accuracy:.2f}% "
            f"({score.correct}/{score.total})"
        )

    print()
    print("Mismatches")
    print("-" * 10)
    if not report.mismatches:
        print("None")
        return
    for mismatch in report.mismatches:
        print(
            f"Row {mismatch.row_number} | {mismatch.field}: "
            f"expected={mismatch.expected!r}, predicted={mismatch.generated!r}"
        )
        print(f"  issue: {_truncate(mismatch.issue, 140)}")
        print(f"  subject: {_truncate(mismatch.subject, 100)}")
        print(f"  company: {mismatch.company}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _truncate(value: str, max_chars: int) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample_path = repo_root / "support_tickets" / "sample_support_tickets.csv"
    temp_output_path = Path(__file__).resolve().parent / "_sample_predictions_tmp.csv"
    data_dir = repo_root / "data"
    report = evaluate_sample(sample_path, temp_output_path, data_dir)
    print_report(report)


if __name__ == "__main__":
    main()
