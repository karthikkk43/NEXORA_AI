
"""Conservative demo guardrails for Nexora AI.

These checks supplement, but do not replace,
authentication, IAM permissions, or authorization.
"""

import re
from fastapi import HTTPException


# Prompt injection and secret extraction patterns
INJECTION = [
    r"\bignore\s+(?:all\s+|any\s+|the\s+|previous\s+|prior\s+)?(?:instructions|rules|system\s+prompt)\b",
    r"\b(?:reveal|print|show|dump|disclose)\s+(?:your\s+|the\s+|hidden\s+|internal\s+)*(?:system|developer)\s+(?:prompt|instructions)\b",
    r"\b(?:reveal|print|dump|expose|send)\s+(?:your\s+|all\s+|the\s+)*(?:api\s*keys?|passwords?|credentials?|secrets?|access\s+tokens?)\b",
    r"\b(?:bypass|disable|turn\s+off)\s+(?:all\s+|the\s+)?(?:guardrails|safety\s+checks|security\s+controls)\b",
    r"\b(?:pretend|act)\s+(?:you\s+are|as\s+if\s+you\s+are)\s+(?:the\s+)?(?:system|developer|administrator)\b",
]

INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in INJECTION
]


# Sensitive information detection
SENSITIVE_OUTPUT = [
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(
        r"\b(?:aws_secret_access_key|pinecone_api_key|database_url)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:password|api[_ -]?key|secret[_ -]?key)\s*[:=]\s*[\"']?[^\s\"']{12,}",
        re.IGNORECASE,
    ),
]


class GuardrailBlocked(ValueError):
    """Raised when a request violates a guardrail."""
    pass


def classify_input(question: str) -> str | None:
    """Classify input without calling an LLM."""

    if not isinstance(question, str) or not question.strip():
        return "empty_input"

    if any(
        pattern.search(question)
        for pattern in INJECTION_PATTERNS
    ):
        return "prompt_injection_or_secret_request"

    return None


def check_input(question: str) -> str:
    """Validate incoming user questions."""

    reason = classify_input(question)

    if reason:
        raise HTTPException(
            status_code=400,
            detail=f"Input guardrail blocked this request ({reason}).",
        )

    return question


def contains_sensitive_output(text: str) -> bool:
    """Detect common secret patterns in generated responses."""

    return any(
        pattern.search(text or "")
        for pattern in SENSITIVE_OUTPUT
    )


def check_output(answer: str) -> str:
    """Prevent detected secrets from reaching the user."""

    if contains_sensitive_output(answer):
        return (
            "The response was blocked by the "
            "sensitive-data output guardrail."
        )

    return answer


def check_evidence(text: str) -> str:
    """Return retrieved evidence unchanged.

    Retrieved content must be treated as untrusted
    by the generation workflow.
    """

    return text
