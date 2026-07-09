from __future__ import annotations

import os
from typing import Any, Dict, List
import pypdf
from core.extractors.base import BaseTextExtractor, ExtractedContent


class PDFExtractor(BaseTextExtractor):
    """Extract text from local PDF file paths."""

    MAX_TOKENS_PER_CHUNK = 5000

    def extract(self, source: str) -> ExtractedContent:
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

        chunks = self._chunk_pdf_text(pages_text)

        return ExtractedContent(
            text=full_text,
            chunks=chunks,
            source_label="PDF Document",
        )

    def _chunk_pdf_text(self, pages_text: List[str]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        current_texts: List[str] = []
        current_tokens = 0
        chunk_id = 0

        for page_num, text in enumerate(pages_text):
            words = text.split()
            page_tokens = int(len(words) * 1.3) or 1

            if current_tokens + page_tokens > self.MAX_TOKENS_PER_CHUNK and current_texts:
                chunks.append({
                    "text": "\n\n".join(current_texts),
                    "start_time": float(chunk_id),
                    "end_time": float(chunk_id + 1),
                })
                current_texts = [text]
                current_tokens = page_tokens
                chunk_id += 1
            else:
                current_texts.append(text)
                current_tokens += page_tokens

        if current_texts:
            chunks.append({
                "text": "\n\n".join(current_texts),
                "start_time": float(chunk_id),
                "end_time": float(chunk_id + 1),
            })

        return chunks
