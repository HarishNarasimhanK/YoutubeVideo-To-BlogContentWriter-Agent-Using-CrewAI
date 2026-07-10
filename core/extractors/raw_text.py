from typing import Any, Dict, List
import tiktoken
from core.extractors.base import BaseTextExtractor, ExtractedContent


class RawTextExtractor(BaseTextExtractor):
    """Chunk raw text input by exact token count with sequential dummy timestamps."""

    def extract(self, source: str, max_tokens_per_chunk: int = 5000) -> ExtractedContent:
        if not source.strip():
            raise ValueError("Input text is empty.")

        chunks = self._chunk_raw_text(source, max_tokens_per_chunk)

        return ExtractedContent(
            text=source,
            chunks=chunks,
            source_label="Raw Text",
        )

    def _chunk_raw_text(self, text: str, max_tokens: int) -> List[Dict[str, Any]]:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_ids = encoding.encode(text)

        chunks: List[Dict[str, Any]] = []
        chunk_id = 0

        for i in range(0, len(token_ids), max_tokens):
            chunk_token_ids = token_ids[i : i + max_tokens]
            chunk_text = encoding.decode(chunk_token_ids)
            chunks.append({
                "text": chunk_text,
                "start_time": float(chunk_id),
                "end_time": float(chunk_id + 1),
            })
            chunk_id += 1

        return chunks
