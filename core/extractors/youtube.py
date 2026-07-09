from __future__ import annotations

import re
from typing import Any, Dict, List

from youtube_transcript_api import YouTubeTranscriptApi
from core.extractors.base import BaseTextExtractor, ExtractedContent


class YouTubeExtractor(BaseTextExtractor):
    """Extract transcript text from a YouTube video URL."""

    MAX_TOKENS_PER_CHUNK = 5000

    def extract(self, source: str) -> ExtractedContent:
        video_id = self._get_video_id(source)

        try:
            client = YouTubeTranscriptApi()
            items = client.fetch(video_id)
            full_text = " ".join(item.text for item in items)
        except Exception as exc:
            raise ValueError(f"Failed to fetch transcript for video '{video_id}': {exc}") from exc

        if not full_text.strip():
            raise ValueError("Extracted transcript is empty. The video may have no captions.")

        chunks = self._chunk_by_tokens(items)

        return ExtractedContent(
            text=full_text,
            chunks=chunks,
            source_label="YouTube Video",
        )

    def _get_video_id(self, url: str) -> str:
        pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/)|youtu\.be/)([\w-]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        return url

    def _chunk_by_tokens(self, items: list) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        current_texts: List[str] = []
        current_tokens = 0
        current_start = None

        for item in items:
            start = item.start
            text = item.text
            words = text.split()
            item_tokens = int(len(words) * 1.3) or 1

            if current_start is None:
                current_start = start

            if current_tokens + item_tokens > self.MAX_TOKENS_PER_CHUNK and current_texts:
                chunks.append({
                    "text": " ".join(current_texts),
                    "start_time": current_start,
                    "end_time": start,
                })
                current_texts = [text]
                current_tokens = item_tokens
                current_start = start
            else:
                current_texts.append(text)
                current_tokens += item_tokens

        if current_texts:
            last_duration = getattr(items[-1], "duration", 0.0) or 0.0
            chunks.append({
                "text": " ".join(current_texts),
                "start_time": current_start,
                "end_time": items[-1].start + last_duration,
            })

        return chunks
