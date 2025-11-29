"""
Shared helpers for EPI Suite parsing.
"""
import re
from typing import Optional


def extract_multiline_field(text: str,
                            label: str,
                            *,
                            joiner: Optional[str] = None,
                            last: bool = True) -> Optional[str]:
    """
    Extract the value for a label that may span multiple wrapped lines.

    Args:
        text: Source text to search.
        label: Field label (without trailing colon).
        joiner: Optional string inserted between wrapped lines. Defaults to
            empty string (no separator).
        last: When True, return the last occurrence; otherwise the first.

    Returns:
        The concatenated field value or None if not found.
    """
    pattern = re.compile(
        rf'{re.escape(label)}\s*:\s*([^\n]+(?:\n\s+[^\n]+)*)'
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    match = matches[-1] if last else matches[0]
    raw = match.group(1)
    lines = raw.splitlines()
    if not lines:
        return None

    parts = [lines[0].strip()]
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if ':' in stripped:
            break
        parts.append(stripped)
    if joiner is None:
        joiner = ''
    return joiner.join(parts)
