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
    if is_hackerrank_pause_subscription(text, company):
        return "product_issue"
    if any(
        term in lowered
        for term in ["feature request", "can you add", "please add", "extend inactivity"]
    ):
        return "feature_request"
    if is_hackerrank_broad_submission_failure(text, company):
        return "bug"
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
    if is_courtesy_only(text):
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


def is_courtesy_only(text: str) -> bool:
    lowered = text.lower().strip(" .,!?\n\t")
    return lowered in {
        "thank you",
        "thanks",
        "thank you for helping me",
        "thanks for helping",
        "hi there",
        "hello",
    }


def is_generic_unknown_outage(text: str, company: str) -> bool:
    lowered = text.lower()
    if company != "none":
        return False
    mentions_product = any(term in lowered for term in ["hackerrank", "claude", "visa", "anthropic"])
    outage_terms = ["site is down", "pages are accessible", "pages are down", "none of the pages"]
    return not mentions_product and any(term in lowered for term in outage_terms)


def is_hackerrank_broad_submission_failure(text: str, company: str) -> bool:
    lowered = text.lower()
    if company != "hackerrank":
        return False
    has_submission_failure = "submission" in lowered and any(
        term in lowered for term in ["not working", "stopped working", "failing", "unable"]
    )
    broad_scope = any(
        term in lowered
        for term in ["across any challenges", "across all challenges", "all submissions", "none of the submissions"]
    )
    if "none of the submissions" in lowered and "working" in lowered:
        has_submission_failure = True
    return has_submission_failure and broad_scope


def is_hackerrank_hiring_user_removal(text: str, company: str) -> bool:
    lowered = text.lower()
    if company != "hackerrank":
        return False
    wants_removal = any(term in lowered for term in ["remove", "removing", "delete", "deactivate"])
    target_user = any(term in lowered for term in ["user", "interviewer", "employee", "team member"])
    account_context = any(
        term in lowered
        for term in ["hiring account", "platform", "company account", "hackerrank account", "for work"]
    )
    return wants_removal and target_user and account_context


def is_hackerrank_pause_subscription(text: str, company: str) -> bool:
    lowered = text.lower()
    return company == "hackerrank" and "pause" in lowered and "subscription" in lowered


def is_visa_charge_dispute_faq(text: str, company: str) -> bool:
    lowered = text.lower()
    if company != "visa":
        return False
    has_dispute = "dispute" in lowered and any(term in lowered for term in ["charge", "transaction"])
    high_risk_demand = any(
        term in lowered
        for term in ["refund me today", "ban the seller", "force a refund", "unauthorized", "fraud"]
    )
    return has_dispute and not high_risk_demand


def is_claude_public_vulnerability_report(text: str, company: str) -> bool:
    lowered = text.lower()
    if company != "claude":
        return False
    has_vulnerability = any(term in lowered for term in ["security vulnerability", "vulnerability", "bug bounty"])
    asks_for_internal_handling = any(
        term in lowered
        for term in [
            "internal rules",
            "rules internal",
            "logic exact",
            "exact logic",
            "system prompt",
            "developer message",
            "print your instructions",
        ]
    )
    return has_vulnerability and not asks_for_internal_handling


def is_claude_bedrock_support_route(text: str, company: str) -> bool:
    lowered = text.lower()
    if company != "claude":
        return False
    return "bedrock" in lowered and any(
        term in lowered for term in ["aws", "failing", "failure", "not working", "support", "requests"]
    )


def is_simple_visa_lost_card_report(text: str, company: str) -> bool:
    lowered = text.lower()
    if company != "visa":
        return False
    has_card = "card" in lowered
    has_lost_or_stolen = "lost" in lowered or "stolen" in lowered
    asks_reporting = any(term in lowered for term in ["where can i report", "how can i report", "where do i report", "how do i report", "report a"])
    risky_terms = [
        "identity theft",
        "fraud",
        "unauthorized",
        "unauthorised",
        "credential",
        "compromise",
        "refund",
        "transaction",
        "account",
    ]
    return has_card and has_lost_or_stolen and asks_reporting and not any(term in lowered for term in risky_terms)


def invalid_product_area(text: str, company: str) -> str:
    if is_courtesy_only(text):
        return ""
    if company == "none":
        return "conversation_management"
    return product_area(text, company, [])


def escalation_reason(text: str, company: str) -> str:
    lowered = text.lower()
    if is_generic_unknown_outage(text, company):
        return "the ticket describes a generic outage without identifying a supported product"
    if is_hackerrank_broad_submission_failure(text, company):
        return "the ticket describes a broad submissions failure across challenges"
    prompt_injection_terms = [
        "internal rules",
        "rules internal",
        "logic exact",
        "exact logic",
        "system prompt",
        "developer message",
        "print your instructions",
        "règles internes",
        "regles internes",
        "logique exacte",
        "documents récupérés",
        "documents recuperes",
    ]
    if any(term in lowered for term in prompt_injection_terms):
        return "the ticket asks for internal rules, retrieved documents, or decision logic that must not be exposed"
    if is_claude_bedrock_support_route(text, company):
        return ""
    if any(
        term in lowered
        for term in ["all requests", "site is down", "stopped working completely", "none of the submissions"]
    ):
        return "the ticket describes a broad outage or platform-wide failure"
    if is_claude_public_vulnerability_report(text, company):
        return ""
    if any(term in lowered for term in ["security vulnerability", "bug bounty"]):
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
    if company == "visa" and any(term in lowered for term in ["ban the seller", "refund me today"]):
        return "the ticket asks Visa to force a refund or merchant enforcement action beyond the support guidance"
    if company == "visa" and is_visa_charge_dispute_faq(text, company):
        return ""
    if company == "visa" and any(
        term in lowered
        for term in [
            "identity theft",
            "fraud",
            "ban the seller",
            "refund me today",
            "unauthorized transaction",
            "unauthorised transaction",
            "stolen credential",
            "account compromise",
            "account-specific compromise",
        ]
    ):
        return "the ticket involves fraud, stolen credentials, or a forced refund request"
    if company == "visa" and "stolen" in lowered and not traveler_cheque_case and not is_simple_visa_lost_card_report(text, company):
        return "the ticket involves a stolen payment card or credentials"
    if is_hackerrank_pause_subscription(text, company):
        return ""
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
        if is_simple_visa_lost_card_report(text, company):
            return "general_support"
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
        if any(
            term in lowered
            for term in [
                "security",
                "vulnerability",
                "crawl",
                "data",
                "privacy",
                "private info",
                "private information",
                "sensitive data",
                "temporary chat",
                "incognito",
                "who can view my conversations",
            ]
        ):
            return "privacy"
        if any(term in lowered for term in ["lti", "college", "students", "education"]):
            return "education"
        return matches[0].chunk.document.product_area if matches else "claude"
    if company == "hackerrank":
        if is_hackerrank_hiring_user_removal(text, company):
            return "settings"
        if any(term in lowered for term in ["test variant", "test variants", "default versions of roles", "create a variant", "different test"]):
            return "screen"
        if any(term in lowered for term in ["hackerrank community", "google login", "community account", "delete my account"]):
            return "community"
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
        if matches:
            return _normalize_product_area(matches[0].chunk.document.product_area)
        return "hackerrank"
    return _normalize_product_area(matches[0].chunk.document.product_area) if matches else ""


def _normalize_product_area(area: str) -> str:
    return "community" if area == "hackerrank_community" else area
