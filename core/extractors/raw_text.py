from __future__ import annotations

from typing import Any, Dict, List
from core.extractors.base import BaseTextExtractor, ExtractedContent


class RawTextExtractor(BaseTextExtractor):
    """Chunk raw text input by token count with sequential dummy timestamps."""

    MAX_TOKENS_PER_CHUNK = 5000

    def extract(self, source: str) -> ExtractedContent:
        if not source.strip():
            raise ValueError("Input text is empty.")

        chunks = self._chunk_raw_text(source)

        return ExtractedContent(
            text=source,
            chunks=chunks,
            source_label="Raw Text",
        )

    def _chunk_raw_text(self, text: str) -> List[Dict[str, Any]]:
        words = text.split()
        chunks: List[Dict[str, Any]] = []
        idx = 0
        chunk_id = 0

        while idx < len(words):
            end = min(idx + self.MAX_TOKENS_PER_CHUNK, len(words))
            chunk_text = " ".join(words[idx:end])
            chunks.append({
                "text": chunk_text,
                "start_time": float(chunk_id),
                "end_time": float(chunk_id + 1),
            })
            idx = end
            chunk_id += 1

        return chunks
