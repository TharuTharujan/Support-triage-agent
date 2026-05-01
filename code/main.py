from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from agent import process_csv
from models import RunSummary
from validator import OUTPUT_FIELDS


def resolve_default_input(repo_root: Path) -> Path:
    support_dir = repo_root / "support_tickets"
    candidates = [
        support_dir / "support_tickets.csv",
        support_dir / "support_issues.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the HackerRank Orchestrate support triage agent."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=resolve_default_input(repo_root),
        help=(
            "CSV file containing Issue, Subject, and Company columns. Defaults to "
            "support_tickets.csv, or support_issues.csv when that platform filename is present."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "support_tickets" / "output.csv",
        help="CSV file to write predictions to.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help="Directory containing the markdown support corpus.",
    )
    return parser


def print_run_report(
    input_path: Path,
    output_path: Path,
    summary: RunSummary | None = None,
    preview_rows: int = 3,
) -> None:
    summary = summary or RunSummary()
    rows = _read_output_rows(output_path)
    status_counts = Counter(row.get("status", "") for row in rows)
    request_type_counts = Counter(row.get("request_type", "") for row in rows)

    print()
    print("Multi-Domain Support Triage Challenge")
    print("=" * 43)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Rows processed: {len(rows)}")
    print(f"LLM provider: {summary.llm_provider_name}")
    print(
        "Generation mode: "
        f"{'llm-assisted' if summary.llm_generation_used else 'local fallback'}"
    )
    print(f"Output schema: {', '.join(OUTPUT_FIELDS)}")
    print(f"Status counts: {_format_counts(status_counts)}")
    print(f"Request type counts: {_format_counts(request_type_counts)}")
    print()
    print("Preview")
    print("-" * 7)
    for index, row in enumerate(rows[:preview_rows], start=1):
        response = _truncate(row.get("response", ""), 120)
        justification = _truncate(row.get("justification", ""), 100)
        print(
            f"{index}. status={row.get('status', '')} | "
            f"product_area={row.get('product_area', '')} | "
            f"request_type={row.get('request_type', '')}"
        )
        print(f"   response: {response}")
        print(f"   justification: {justification}")
    if len(rows) > preview_rows:
        print(f"... {len(rows) - preview_rows} more rows written to CSV")


def _read_output_rows(output_path: Path) -> list[dict[str, str]]:
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_counts(counts: Counter[str]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def _truncate(value: str, max_chars: int) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def main() -> None:
    args = build_parser().parse_args()
    summary = process_csv(args.input, args.output, args.data_dir)
    print_run_report(args.input, args.output, summary)


if __name__ == "__main__":
    main()
