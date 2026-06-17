"""
Input validation and sanitization for URCM.
Protects against prompt injection and malicious input.
"""

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Maximum input length
MAX_INPUT_LENGTH = 4096

# Blocked patterns for prompt injection
BLOCKED_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|below|prior)",
    r"system\s*(prompt|message|instruction)",
    r"forget\s+(everything|all|previous)",
    r"you\s+(are\s+)?(now|will\s+now)\s+",
    r"output\s+(only|just|exactly)",
    r"respond\s+in\s+",
    r"do\s+not\s+(follow|obey|listen)",
    r"override\s+(your|core|all|safety)",
    r"bypass\s+(safety|filter|restriction)",
    r"<!--|-->|```|\$\{|\$\(|`.*`",
]

# Compiled patterns
_COMPILED_BLOCKED = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


def validate_input(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user input before processing.
    Returns (is_valid, error_message).
    """
    if not text or not text.strip():
        return False, "Input cannot be empty."

    if len(text) > MAX_INPUT_LENGTH:
        return False, f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters."

    for i, pattern in enumerate(_COMPILED_BLOCKED):
        if pattern.search(text):
            logger.warning(f"Blocked input matching pattern {i}: {text[:100]}")
            return False, "Input contains disallowed patterns."

    return True, None


def sanitize_for_llm(text: str) -> str:
    """
    Sanitize text before passing to LLM.
    Strips control characters and limits length.
    """
    sanitized = text.strip()[:MAX_INPUT_LENGTH]
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', sanitized)
    return sanitized
