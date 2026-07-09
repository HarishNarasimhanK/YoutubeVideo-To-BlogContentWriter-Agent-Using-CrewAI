from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

class TranscriptRAGStore:
    """In-memory indexing and hybrid search (BM25 + Cosine Semantic Similarity) for transcript segments."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.chunks = chunks
        self.provider = provider.lower().strip() if provider else None
        self.api_key = api_key

        self.df: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_len = 0.0

        self.embeddings: List[List[float]] = []
        self.embed_model: Any = None

        self._build_lexical_index()
        self._initialize_embeddings()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build_lexical_index(self) -> None:
        total_len = 0
        for chunk in self.chunks:
            tokens = self._tokenize(chunk["text"])
            total_len += len(tokens)
            self.doc_lengths.append(len(tokens))
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1
        self.avg_doc_len = (total_len / len(self.chunks)) if self.chunks else 0.0

    def _initialize_embeddings(self) -> None:
        if not self.provider or not self.chunks:
            return

        try:
            if self.provider == "gemini":
                key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if key:
                    self.embed_model = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004",
                        google_api_key=key
                    )
            elif self.provider == "openai":
                key = self.api_key or os.getenv("OPENAI_API_KEY")
                if key:
                    self.embed_model = OpenAIEmbeddings(
                        model="text-embedding-3-small",
                        openai_api_key=key
                    )

            if self.embed_model:
                texts = [c["text"] for c in self.chunks]
                self.embeddings = self.embed_model.embed_documents(texts)
                print(f"[TranscriptRAGStore] Loaded {len(self.embeddings)} semantic embeddings successfully.")
        except Exception as exc:
            print(f"[TranscriptRAGStore] ⚠️ Semantic embeddings init failed: {exc}. Falling back to lexical BM25 search only.")
            self.embeddings = []
            self.embed_model = None

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def retrieve_lexical(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.chunks[:top_k]

        scores = [0.0] * len(self.chunks)
        k1 = 1.5
        b = 0.75
        N = len(self.chunks)

        for token in query_tokens:
            if token not in self.df:
                continue
            idf = math.log((N - self.df[token] + 0.5) / (self.df[token] + 0.5) + 1.0)
            for i, chunk in enumerate(self.chunks):
                tokens = self._tokenize(chunk["text"])
                tf = tokens.count(token)
                dl = self.doc_lengths[i]
                score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / self.avg_doc_len)))
                scores[i] += score

        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        results = [self.chunks[idx] for idx in ranked_indices if scores[idx] > 0.0]
        return results[:top_k] if results else self.chunks[:top_k]

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.embed_model or not self.embeddings:
            return self.retrieve_lexical(query, top_k)

        try:
            query_embedding = self.embed_model.embed_query(query)
            similarities = [
                self._cosine_similarity(query_embedding, doc_emb)
                for doc_emb in self.embeddings
            ]
            ranked_indices = sorted(
                range(len(similarities)),
                key=lambda idx: similarities[idx],
                reverse=True
            )
            return [self.chunks[idx] for idx in ranked_indices[:top_k]]
        except Exception as exc:
            print(f"[TranscriptRAGStore] ⚠️ Semantic query failed: {exc}. Falling back to lexical.")
            return self.retrieve_lexical(query, top_k)
