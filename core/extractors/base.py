from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ExtractedContent:
    """Standardized output from any text extractor."""
    text: str
    chunks: List[Dict[str, Any]]
    source_label: str


class BaseTextExtractor(ABC):
    """Abstract base for all input source extractors."""

    @abstractmethod
    def extract(self, source: str, max_tokens_per_chunk: int = 5000) -> ExtractedContent:
        """Extract text from the given source identifier (URL, file path, raw text, etc.)."""
        ...
