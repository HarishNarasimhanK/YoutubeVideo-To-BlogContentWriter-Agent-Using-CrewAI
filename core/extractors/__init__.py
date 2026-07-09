from __future__ import annotations

from core.extractors.base import BaseTextExtractor, ExtractedContent
from core.extractors.youtube import YouTubeExtractor
from core.extractors.raw_text import RawTextExtractor

EXTRACTOR_REGISTRY = {
    "youtube": YouTubeExtractor,
    "raw_text": RawTextExtractor,
}


def get_extractor(source_type: str) -> BaseTextExtractor:
    """Factory function to get the appropriate extractor for the given source type."""
    cls = EXTRACTOR_REGISTRY.get(source_type.lower().strip())
    if cls is None:
        raise ValueError(
            f"Unknown source type '{source_type}'. "
            f"Supported: {', '.join(EXTRACTOR_REGISTRY.keys())}"
        )
    return cls()


__all__ = [
    "BaseTextExtractor",
    "ExtractedContent",
    "get_extractor",
]
