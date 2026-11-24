"""Input processing: validate, normalize, preserve original text for evidence checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re
import unicodedata


@dataclass
class ProcessedInput:
    """Normalized input with preserved original text and sentence spans."""

    original_text: str
    normalized_text: str
    is_valid: bool
    error_message: Optional[str] = None
    sentence_spans: List[Tuple[int, int, str]] = field(default_factory=list)


class InputProcessor:
    """Accept raw text, validate, normalize lightly, preserve original for verification."""

    MAX_LENGTH = 10000
    MIN_LENGTH = 1

    def process(self, raw_text: Optional[str]) -> ProcessedInput:
        if raw_text is None:
            return ProcessedInput(
                original_text="",
                normalized_text="",
                is_valid=False,
                error_message="Input is None.",
            )

        if not isinstance(raw_text, str):
            return ProcessedInput(
                original_text=str(raw_text),
                normalized_text="",
                is_valid=False,
                error_message="Input must be a string.",
            )

        original = raw_text
        stripped = original.strip()

        if len(stripped) < self.MIN_LENGTH:
            return ProcessedInput(
                original_text=original,
                normalized_text="",
                is_valid=False,
                error_message="Input is empty.",
            )

        if len(stripped) > self.MAX_LENGTH:
            return ProcessedInput(
                original_text=original,
                normalized_text="",
                is_valid=False,
                error_message=f"Input exceeds maximum length of {self.MAX_LENGTH} characters.",
            )

        # Light normalization: NFKC unicode, collapse excessive whitespace.
        # Do not destroy punctuation or casing that may be evidence.
        normalized = unicodedata.normalize("NFKC", stripped)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        spans = self._split_sentences(normalized)
        return ProcessedInput(
            original_text=original,
            normalized_text=normalized,
            is_valid=True,
            error_message=None,
            sentence_spans=spans,
        )

    def _split_sentences(self, text: str) -> List[Tuple[int, int, str]]:
        """Split into sentence-like spans with character offsets into `text`."""
        spans: List[Tuple[int, int, str]] = []
        # Split on sentence-ending punctuation while keeping offsets.
        pattern = re.compile(r"[^.!?\n]+[.!?]*(?:\n+|$)")
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            chunk = text[start:end].strip()
            if not chunk:
                continue
            # Recompute offsets for stripped content within the match.
            local = text[start:end]
            inner_start = local.find(chunk)
            abs_start = start + inner_start
            abs_end = abs_start + len(chunk)
            spans.append((abs_start, abs_end, chunk))
        if not spans and text.strip():
            t = text.strip()
            i = text.find(t)
            spans.append((i, i + len(t), t))
        return spans
