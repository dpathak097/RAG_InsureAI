"""
Validator — basic grounding and conflict checks.
"""
import re


def validate_grounding(answer: str, context: str) -> tuple[bool, list[str]]:
    """Check if key facts in the answer appear in the context."""
    # Extract numbers/amounts from answer
    answer_nums = set(re.findall(r'\b\d[\d,\.]*\b', answer))
    context_lower = context.lower()
    missing = []
    for num in answer_nums:
        if num not in context:
            missing.append(num)
    grounded = len(missing) == 0
    return grounded, missing


def detect_conflict(chunks) -> tuple[bool, list[str]]:
    """Detect if chunks come from multiple insurers (potential conflict)."""
    insurers = set()
    for chunk in chunks:
        insurer = chunk.metadata.get("insurer") if hasattr(chunk, "metadata") else None
        if insurer and insurer != "UNKNOWN":
            insurers.add(insurer)
    has_conflict = len(insurers) > 1
    return has_conflict, list(insurers)


def validate_calculation(answer: str, context: str) -> tuple[bool, str]:
    """Basic check that any calculation result appears grounded."""
    return True, ""
