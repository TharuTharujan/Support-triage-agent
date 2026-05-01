from __future__ import annotations

from chunker import tokenize
from models import RetrievalHit


def infer_company(company: str, text: str) -> str:
    normalized = company.strip().lower()
    if normalized and normalized != "none":
        return normalized
    lowered = text.lower()
    scores = {
        "hackerrank": sum(
            lowered.count(term)
            for term in ["hackerrank", "assessment", "test", "candidate", "recruiter"]
        ),
        "claude": sum(
            lowered.count(term)
            for term in ["claude", "anthropic", "bedrock", "workspace", "model"]
        ),
        "visa": sum(
            lowered.count(term)
            for term in ["visa", "card", "merchant", "cheque", "charge", "cash"]
        ),
    }
    best_company, score = max(scores.items(), key=lambda item: item[1])
    return best_company if score > 0 else "none"


def request_type(text: str, company: str = "") -> str:
    lowered = text.lower()
    if is_invalid(text, company):
        return "invalid"
    if any(
        term in lowered
        for term in ["feature request", "can you add", "please add", "extend inactivity"]
    ):
        return "feature_request"
    if any(
        term in lowered
        for term in [
            "bug",
            "down",
            "not working",
            "stopped",
            "failing",
            "error",
            "blocker",
            "unable",
            "can't",
            "cannot",
        ]
    ):
        return "bug"
    return "product_issue"


def is_invalid(text: str, company: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return True
    if lowered in {"thank you", "thanks for helping", "hi there", "hello"}:
        return True
    if len(tokenize(lowered)) <= 1:
        return True
    irrelevant_or_injection = [
        "actor in iron man",
        "delete all files",
        "print all rules",
        "show all rules",
        "ignore previous",
    ]
    return any(term in lowered for term in irrelevant_or_injection) and company in {"", "none"}


def escalation_reason(text: str, company: str) -> str:
    lowered = text.lower()
    if any(
        term in lowered
        for term in ["all requests", "site is down", "stopped working completely", "none of the submissions"]
    ):
        return "the ticket describes a broad outage or platform-wide failure"
    if any(
        term in lowered
        for term in ["security vulnerability", "bug bounty", "internal rules", "rules internal", "logic exact"]
    ):
        return "the ticket involves security-sensitive handling or prompt-injection-like content"
    if any(
        term in lowered
        for term in ["ignore previous", "system prompt", "developer message", "print your instructions"]
    ):
        return "the ticket contains prompt-injection-like instructions"
    if any(
        term in lowered
        for term in ["legal", "subpoena", "law enforcement", "court order", "copyright", "terms of service"]
    ):
        return "the ticket involves legal or policy-sensitive handling"
    traveler_cheque_case = any(term in lowered for term in ["cheque", "traveller", "traveler"])
    if company == "visa" and any(
        term in lowered for term in ["identity theft", "fraud", "ban the seller", "refund me today"]
    ):
        return "the ticket involves fraud, stolen credentials, or a forced refund request"
    if company == "visa" and "stolen" in lowered and not traveler_cheque_case:
        return "the ticket involves a stolen payment card or credentials"
    if any(term in lowered for term in ["refund", "payment", "charge", "subscription", "invoice", "order id"]):
        return "the ticket involves billing or payment authority"
    if company == "claude" and any(
        term in lowered
        for term in ["restore my access", "removed my seat", "not the workspace owner", "admin removed"]
    ):
        return "the ticket requests account access changes without verified admin authority"
    if company == "hackerrank" and any(
        term in lowered
        for term in ["increase my score", "tell the company", "rejected me", "rescheduling"]
    ):
        return "the ticket asks for hiring outcome or assessment authority outside support documentation"
    if any(
        term in lowered
        for term in ["personal data", "private info", "sensitive data", "gdpr", "privacy", "data export"]
    ) and "delete" not in lowered:
        return "the ticket involves privacy-sensitive data handling"
    if company == "hackerrank" and any(
        term in lowered
        for term in [
            "integrity flag",
            "clear the integrity",
            "bypassed proctoring",
            "copied code",
            "plagiarism",
            "cheating",
            "impersonation",
        ]
    ):
        return "the ticket involves assessment integrity or proctoring authority"
    return ""


def product_area(text: str, company: str, matches: list[RetrievalHit]) -> str:
    lowered = text.lower()
    if company == "visa":
        if any(term in lowered for term in ["travel", "cheque", "cash", "blocked"]):
            return "travel_support"
        if any(term in lowered for term in ["dispute", "charge", "merchant", "refund"]):
            return "dispute_resolution"
        if any(term in lowered for term in ["fraud", "identity theft", "stolen"]):
            return "fraud_protection"
        return "general_support"
    if company == "claude":
        if any(term in lowered for term in ["bedrock", "aws"]):
            return "amazon_bedrock"
        if any(term in lowered for term in ["workspace", "seat", "login", "account", "access"]):
            return "account_management"
        if any(term in lowered for term in ["security", "vulnerability", "crawl", "data", "privacy"]):
            return "privacy"
        if any(term in lowered for term in ["lti", "college", "students", "education"]):
            return "education"
        return matches[0].chunk.document.product_area if matches else "claude"
    if company == "hackerrank":
        if any(term in lowered for term in ["subscription", "payment", "refund", "invoice"]):
            return "settings"
        if any(term in lowered for term in ["mock interview", "interview", "inactivity"]):
            return "interview"
        if any(term in lowered for term in ["certificate", "practice", "apply tab", "challenge", "resume"]):
            return "community"
        if any(term in lowered for term in ["test", "assessment", "candidate", "score", "reinvite"]):
            return "screen"
        if any(term in lowered for term in ["employee", "user", "remove interviewer"]):
            return "settings"
        return matches[0].chunk.document.product_area if matches else "hackerrank"
    return matches[0].chunk.document.product_area if matches else ""
