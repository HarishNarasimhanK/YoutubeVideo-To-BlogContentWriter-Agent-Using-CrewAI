import os
from typing import Any, Dict, List
import pypdf
import tiktoken
from core.extractors.base import BaseTextExtractor, ExtractedContent


class PDFExtractor(BaseTextExtractor):
    """Extract text from local PDF file paths and chunk by exact tokens."""

    def extract(self, source: str, max_tokens_per_chunk: int = 5000) -> ExtractedContent:
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found at path: {source}")

        pages_text = []
        try:
            reader = pypdf.PdfReader(source)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
        except Exception as exc:
            raise ValueError(f"Failed to read PDF file '{source}': {exc}") from exc

        full_text = "\n\n".join(pages_text)
        if not full_text.strip():
            raise ValueError(f"No text extracted from PDF file: {source}")

        chunks = self._chunk_pdf_text(full_text, max_tokens_per_chunk)

        return ExtractedContent(
            text=full_text,
            chunks=chunks,
            source_label="PDF Document",
        )

    def _chunk_pdf_text(self, full_text: str, max_tokens: int) -> List[Dict[str, Any]]:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_ids = encoding.encode(full_text)

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
