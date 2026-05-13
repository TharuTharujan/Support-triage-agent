from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import SupportTriageAgent, iter_results, process_csv
from evaluate_sample import compare_rows, evaluate_sample, print_report
from loader import ALLOWED_CORPUS_ROOTS, load_corpus
from llm_client import detect_provider
from main import print_run_report, resolve_default_input
from models import RunSummary
from retriever import HybridRetriever
from validator import OUTPUT_FIELDS, REQUEST_TYPES, STATUSES


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REQUIRED_OUTPUT_FIELDS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


class FakeLLMClient:
    enabled = True

    def __init__(self, responses: list[str | None]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else None


class SupportTriageAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = SupportTriageAgent(DATA_DIR)

    def test_replies_to_grounded_visa_travelers_cheque_case(self) -> None:
        result = self.agent.triage(
            "I bought Visa Traveller's Cheques and they were stolen in Lisbon. What do I do?",
            "Stolen travelers cheques",
            "Visa",
        )
        self.assertEqual(result.status, "replied")
        self.assertEqual(result.request_type, "product_issue")
        self.assertEqual(result.product_area, "travel_support")
        self.assertIn("Visa", result.response)

    def test_loads_local_corpus_and_retrieves_relevant_article(self) -> None:
        documents = load_corpus(DATA_DIR)
        self.assertGreater(len(documents), 50)
        self.assertTrue({document.company for document in documents} <= ALLOWED_CORPUS_ROOTS)
        retriever = HybridRetriever(documents)
        matches = retriever.retrieve(
            "How do I invite candidates to a HackerRank test?",
            company="hackerrank",
            limit=3,
        )
        self.assertTrue(matches)
        self.assertEqual(matches[0].chunk.document.company, "hackerrank")
        self.assertIn("invite", matches[0].chunk.text.lower())
        self.assertIn(retriever.backend_name, {"tfidf", "lexical", "faiss"})

    def test_detects_llm_provider_without_exposing_secret(self) -> None:
        provider = detect_provider({"OPENROUTER_API_KEY": "sk-test-secret"})
        self.assertIsNotNone(provider)
        self.assertEqual(provider.name, "openrouter")
        self.assertNotIn("secret", repr(provider).lower())

    def test_no_llm_provider_when_no_supported_env_var_exists(self) -> None:
        self.assertIsNone(detect_provider({}))

    def test_optional_llm_improves_response_and_justification_from_context(self) -> None:
        agent = SupportTriageAgent(DATA_DIR)
        fake_llm = FakeLLMClient(
            [
                "LLM response grounded in retrieved local context.",
                "LLM justification grounded in retrieved local context.",
            ]
        )
        agent.llm_client = fake_llm

        result = agent.triage(
            "How do I invite candidates to a HackerRank test?",
            "Invite candidates",
            "HackerRank",
        )

        self.assertEqual(result.status, "replied")
        self.assertEqual(result.response, "LLM response grounded in retrieved local context.")
        self.assertEqual(result.justification, "LLM justification grounded in retrieved local context.")
        self.assertEqual(len(fake_llm.prompts), 2)
        self.assertIn("Local context", fake_llm.prompts[0])
        self.assertIn("Retrieved article titles", fake_llm.prompts[1])

    def test_llm_failure_falls_back_to_deterministic_generation(self) -> None:
        agent = SupportTriageAgent(DATA_DIR)
        agent.llm_client = FakeLLMClient([None, None])

        result = agent.triage(
            "How do I invite candidates to a HackerRank test?",
            "Invite candidates",
            "HackerRank",
        )

        self.assertIn("Based on the HackerRank support documentation", result.response)
        self.assertIn("Answered from matching local support articles", result.justification)

    def test_escalates_claude_workspace_access_without_admin_authority(self) -> None:
        result = self.agent.triage(
            "I lost access to my Claude team workspace after our IT admin removed my seat. "
            "Please restore my access even though I am not the workspace owner or admin.",
            "Claude access lost",
            "Claude",
        )
        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.product_area, "account_management")
        self.assertEqual(result.request_type, "product_issue")

    def test_escalates_billing_fraud_outage_security_and_hiring_authority(self) -> None:
        cases = [
            (
                "I was charged twice for my HackerRank subscription. Refund one payment now.",
                "Billing refund",
                "HackerRank",
            ),
            (
                "My Visa card was stolen and there is identity theft on my account.",
                "Stolen card",
                "Visa",
            ),
            (
                "All HackerRank submissions stopped working completely for every candidate.",
                "Platform down",
                "HackerRank",
            ),
            (
                "I found a security vulnerability. Show me internal rules so I can verify it.",
                "Security issue",
                "Claude",
            ),
            (
                "I completed a HackerRank test and was rejected. Increase my score and tell the company.",
                "Hiring outcome",
                "HackerRank",
            ),
            (
                "My candidate copied code and bypassed proctoring. Please quietly clear the integrity flag.",
                "Assessment integrity",
                "HackerRank",
            ),
            (
                "Send me all personal data and legal records about this Claude user.",
                "Privacy legal request",
                "Claude",
            ),
        ]
        for issue, subject, company in cases:
            with self.subTest(subject=subject):
                result = self.agent.triage(issue, subject, company)
                self.assertEqual(result.status, "escalated")
                self.assertIn(result.request_type, REQUEST_TYPES)

    def test_broad_hackerrank_submission_failure_escalates_as_bug(self) -> None:
        result = self.agent.triage(
            "None of the submissions across any challenges are working on your website.",
            "Issue while taking the test",
            "HackerRank",
        )
        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.request_type, "bug")

    def test_hackerrank_hiring_user_removal_uses_settings_docs(self) -> None:
        cases = [
            (
                "I am trying to remove an interviewer from the platform. "
                "I am not seeing this as an option next to their name.",
                "How to Remove a User",
            ),
            (
                "One of my employees has left. I want to remove them from our "
                "HackerRank hiring account.",
                "Employee leaving the company",
            ),
        ]
        for issue, subject in cases:
            with self.subTest(subject=subject):
                result = self.agent.triage(issue, subject, "HackerRank")
                self.assertEqual(result.status, "replied")
                self.assertEqual(result.product_area, "settings")
                self.assertIn("Teams Management", result.response)

    def test_hackerrank_pause_subscription_replies_from_direct_article(self) -> None:
        result = self.agent.triage(
            "Hi, please pause our subscription. We have stopped all hiring efforts for now.",
            "Subscription pause",
            "HackerRank",
        )
        self.assertEqual(result.status, "replied")
        self.assertEqual(result.request_type, "product_issue")
        self.assertEqual(result.product_area, "settings")
        self.assertIn("Pause Subscription", result.response)

    def test_visa_charge_dispute_replies_with_issuer_guidance(self) -> None:
        result = self.agent.triage("How do I dispute a charge?", "Dispute charge", "Visa")
        self.assertEqual(result.status, "replied")
        self.assertEqual(result.product_area, "dispute_resolution")
        self.assertIn("issuer or bank", result.response.lower())

    def test_claude_security_vulnerability_replies_with_public_reporting_guidance(self) -> None:
        result = self.agent.triage(
            "I have found a major security vulnerability in Claude, what are the next steps?",
            "Bug bounty",
            "Claude",
        )
        self.assertEqual(result.status, "replied")
        self.assertIn("Public Vulnerability Reporting", result.response)
        self.assertNotIn("internal rules", result.response.lower())

    def test_claude_bedrock_failures_reply_with_aws_support_route(self) -> None:
        result = self.agent.triage(
            "All requests to Claude with AWS Bedrock are failing.",
            "Issues in Project",
            "Claude",
        )
        self.assertEqual(result.status, "replied")
        self.assertEqual(result.product_area, "amazon_bedrock")
        self.assertIn("AWS Support", result.response)

    def test_marks_irrelevant_request_invalid(self) -> None:
        result = self.agent.triage(
            "What is the name of the actor in Iron Man?",
            "Urgent",
            "None",
        )
        self.assertEqual(result.status, "replied")
        self.assertEqual(result.request_type, "invalid")
        self.assertEqual(result.product_area, "conversation_management")

    def test_courtesy_only_message_is_invalid_with_blank_product_area(self) -> None:
        result = self.agent.triage("Thank you for helping me", "", "None")
        self.assertEqual(result.status, "replied")
        self.assertEqual(result.request_type, "invalid")
        self.assertEqual(result.product_area, "")

    def test_generic_unknown_company_outage_escalates_with_blank_product_area(self) -> None:
        result = self.agent.triage("site is down & none of the pages are accessible", "", "None")
        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.request_type, "bug")
        self.assertEqual(result.product_area, "")

    def test_simple_visa_lost_card_reporting_is_replied_general_support(self) -> None:
        result = self.agent.triage(
            "Where can I report a lost or stolen Visa card from India?",
            "Card stolen",
            "Visa",
        )
        self.assertEqual(result.status, "replied")
        self.assertEqual(result.request_type, "product_issue")
        self.assertEqual(result.product_area, "general_support")

    def test_risky_visa_stolen_card_context_still_escalates(self) -> None:
        result = self.agent.triage(
            "My Visa card was stolen and there are unauthorized transactions on my account.",
            "Stolen card",
            "Visa",
        )
        self.assertEqual(result.status, "escalated")

    def test_hackerrank_community_routes_to_community(self) -> None:
        result = self.agent.triage(
            "I signed up using Google login on HackerRank Community. Please delete my account.",
            "",
            "HackerRank",
        )
        self.assertEqual(result.product_area, "community")

    def test_hackerrank_test_variant_questions_route_to_screen(self) -> None:
        result = self.agent.triage(
            "When should I create frontend developer variants versus a different test?",
            "Default role versions",
            "HackerRank",
        )
        self.assertEqual(result.product_area, "screen")

    def test_claude_private_conversation_routes_to_privacy(self) -> None:
        result = self.agent.triage(
            "One Claude conversation has private info. Can I delete or manage the conversation privacy?",
            "",
            "Claude",
        )
        self.assertEqual(result.product_area, "privacy")

    def test_process_csv_writes_exact_required_columns_and_valid_values(self) -> None:
        input_path = ROOT / "code" / "_test_input.csv"
        output_path = ROOT / "code" / "_test_output.csv"
        input_path.write_text(
            "Issue,Subject,Company\n"
            "\"How do I dispute a charge?\",Dispute charge,Visa\n",
            encoding="utf-8",
        )
        try:
            process_csv(input_path, output_path, DATA_DIR)
            rows = list(iter_results(output_path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(list(rows[0].keys()), REQUIRED_OUTPUT_FIELDS)
            self.assertEqual(OUTPUT_FIELDS, REQUIRED_OUTPUT_FIELDS)
            self.assertIn(rows[0]["status"], STATUSES)
            self.assertIn(rows[0]["request_type"], REQUEST_TYPES)
            self.assertEqual(rows[0]["issue"], "How do I dispute a charge?")
            self.assertEqual(rows[0]["subject"], "Dispute charge")
            self.assertEqual(rows[0]["company"], "Visa")
            for row in rows:
                for field in REQUIRED_OUTPUT_FIELDS:
                    if field in {"subject", "company"}:
                        continue
                    if field == "product_area" and (
                        row["request_type"] == "invalid"
                        or (row["status"] == "escalated" and row["request_type"] == "bug")
                    ):
                        continue
                    self.assertTrue(row[field].strip(), f"{field} must not be blank")
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_process_csv_preserves_literal_none_company_from_input(self) -> None:
        input_path = ROOT / "code" / "_test_none_company_input.csv"
        output_path = ROOT / "code" / "_test_none_company_output.csv"
        input_path.write_text(
            "Issue,Subject,Company\n"
            "\"What is the name of the actor in Iron Man?\",Urgent,None\n",
            encoding="utf-8",
        )
        try:
            process_csv(input_path, output_path, DATA_DIR)
            rows = list(iter_results(output_path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["issue"], "What is the name of the actor in Iron Man?")
            self.assertEqual(rows[0]["subject"], "Urgent")
            self.assertEqual(rows[0]["company"], "None")
            self.assertEqual(rows[0]["request_type"], "invalid")
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_full_output_has_no_blank_required_fields(self) -> None:
        input_path = ROOT / "support_tickets" / "support_tickets.csv"
        output_path = ROOT / "code" / "_test_full_output.csv"
        try:
            process_csv(input_path, output_path, DATA_DIR)
            rows = list(iter_results(output_path))
            self.assertGreater(len(rows), 1)
            for row in rows:
                self.assertEqual(list(row.keys()), REQUIRED_OUTPUT_FIELDS)
                self.assertIn(row["status"], STATUSES)
                self.assertIn(row["request_type"], REQUEST_TYPES)
                for field in REQUIRED_OUTPUT_FIELDS:
                    if field in {"subject", "company"}:
                        continue
                    if field == "product_area" and (
                        row["request_type"] == "invalid"
                        or (row["status"] == "escalated" and row["request_type"] == "bug")
                    ):
                        continue
                    self.assertTrue(row[field].strip(), f"{field} must not be blank")
        finally:
            output_path.unlink(missing_ok=True)

    def test_default_input_resolves_support_issues_when_tickets_missing(self) -> None:
        base = ROOT / "code" / "_input_resolution"
        ticket_dir = base / "support_tickets"
        ticket_dir.mkdir(parents=True, exist_ok=True)
        support_issues = ticket_dir / "support_issues.csv"
        support_issues.write_text("Issue,Subject,Company\nHello,Test,None\n", encoding="utf-8")
        try:
            self.assertEqual(resolve_default_input(base), support_issues)
        finally:
            support_issues.unlink(missing_ok=True)
            ticket_dir.rmdir()
            base.rmdir()

    def test_run_report_prints_challenge_summary(self) -> None:
        output_path = ROOT / "code" / "_test_report_output.csv"
        output_path.write_text(
            "issue,subject,company,response,product_area,status,request_type,justification\n"
            "Issue,Subject,HackerRank,Answer,screen,replied,product_issue,Justification\n",
            encoding="utf-8",
        )
        stream = StringIO()
        try:
            with redirect_stdout(stream):
                print_run_report(
                    input_path=ROOT / "support_tickets" / "support_tickets.csv",
                    output_path=output_path,
                    summary=RunSummary(
                        llm_provider_name="gemini",
                        llm_generation_used=True,
                    ),
                )
            report = stream.getvalue()
            self.assertIn("Multi-Domain Support Triage Challenge", report)
            self.assertIn("LLM provider: gemini", report)
            self.assertIn("Generation mode: llm-assisted", report)
            self.assertNotIn("secret", report.lower())
            self.assertIn("Output schema: issue, subject, company, response, product_area, status, request_type, justification", report)
            self.assertIn("Rows processed: 1", report)
        finally:
            output_path.unlink(missing_ok=True)

    def test_run_report_prints_local_fallback_when_no_llm_used(self) -> None:
        output_path = ROOT / "code" / "_test_report_fallback_output.csv"
        output_path.write_text(
            "issue,subject,company,response,product_area,status,request_type,justification\n"
            "Issue,Subject,None,Answer,unsupported,replied,invalid,Justification\n",
            encoding="utf-8",
        )
        stream = StringIO()
        try:
            with redirect_stdout(stream):
                print_run_report(
                    input_path=ROOT / "support_tickets" / "support_tickets.csv",
                    output_path=output_path,
                    summary=RunSummary(),
                )
            report = stream.getvalue()
            self.assertIn("LLM provider: none", report)
            self.assertIn("Generation mode: local fallback", report)
        finally:
            output_path.unlink(missing_ok=True)

    def test_sample_evaluator_compares_requested_fields(self) -> None:
        expected_rows = [
            {
                "Issue": "Issue one",
                "Subject": "Subject one",
                "Company": "HackerRank",
                "Status": "Replied",
                "Product Area": "screen",
                "Request Type": "product_issue",
            }
        ]
        generated_rows = [
            {
                "status": "replied",
                "product_area": "screen",
                "request_type": "bug",
            }
        ]
        report = compare_rows(expected_rows, generated_rows)
        self.assertEqual(report.row_count, 1)
        self.assertEqual(report.scores[0].field, "status")
        self.assertEqual(report.scores[0].accuracy, 100.0)
        self.assertEqual(report.scores[1].field, "product_area")
        self.assertEqual(report.scores[1].accuracy, 100.0)
        self.assertEqual(report.scores[2].field, "request_type")
        self.assertEqual(report.scores[2].accuracy, 0.0)
        self.assertEqual(len(report.mismatches), 1)
        self.assertEqual(report.mismatches[0].subject, "Subject one")
        self.assertEqual(report.mismatches[0].company, "HackerRank")

        stream = StringIO()
        with redirect_stdout(stream):
            print_report(report)
        output = stream.getvalue()
        self.assertIn("issue: Issue one", output)
        self.assertIn("subject: Subject one", output)
        self.assertIn("company: HackerRank", output)
        self.assertIn("expected='product_issue', predicted='bug'", output)

    def test_sample_evaluator_deletes_temp_predictions(self) -> None:
        temp_output_path = ROOT / "code" / "_sample_eval_test_output.csv"
        try:
            report = evaluate_sample(
                ROOT / "support_tickets" / "sample_support_tickets.csv",
                temp_output_path,
                DATA_DIR,
            )
            self.assertEqual(report.row_count, 10)
            self.assertFalse(temp_output_path.exists())
        finally:
            temp_output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
