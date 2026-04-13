"""
Unified RAG that searches across documents, videos, and webpages.
Uses the existing RAGPipeline for documents + new stores for videos/webpages.
"""
import asyncio
import logging
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from rag import RAGPipeline, _build_structured_context, _sources_from_chunks
from router import get_insurance_llm
from video_store import VideoVectorStore
from webpage_store import WebpageVectorStore

logger = logging.getLogger(__name__)

class MultiSourceRAG:
    def __init__(self):
        self.doc_pipeline = RAGPipeline()   # untouched
        self.video_store = VideoVectorStore()
        self.webpage_store = WebpageVectorStore()
        self.max_context_chars = 3000

    def _merge_chunks(self, chunks: List[Document]) -> List[Document]:
        """Deduplicate by content hash and keep highest similarity."""
        seen = {}
        for chunk in chunks:
            h = hash(chunk.page_content[:200])
            if h not in seen or chunk.metadata.get("similarity", 0) > seen[h].metadata.get("similarity", 0):
                seen[h] = chunk
        return list(seen.values())

    async def ask(self, question: str) -> Tuple[str, List[str]]:
        """
        Search all three sources, combine context, and generate answer with citations.
        Returns (answer, list_of_sources).
        """
        # 1. Get chunks from all three stores
        doc_chunks = await asyncio.to_thread(self.doc_pipeline._vector_store.search, question, top_k=5)
        video_chunks = await asyncio.to_thread(self.video_store.search, question, top_k=4)
        webpage_chunks = await asyncio.to_thread(self.webpage_store.search, question, top_k=4)

        all_chunks = self._merge_chunks(doc_chunks + video_chunks + webpage_chunks)
        # keep top 12 after merging
        all_chunks.sort(key=lambda x: x.metadata.get("similarity", 0), reverse=True)
        all_chunks = all_chunks[:12]

        if not all_chunks:
            return "No relevant information found in any source (documents, videos, or webpages).", []

        # 2. Build context with source type annotation
        context_parts = []
        sources = []
        for chunk in all_chunks:
            source_type = chunk.metadata.get("source_type", "document")
            if source_type == "document":
                src = chunk.metadata.get("source", "Unknown")
                page = chunk.metadata.get("page", "?")
                label = f"Document: {src} (Page {page})"
                sources.append(f"{src} (page {page})")
            elif source_type == "video":
                url = chunk.metadata.get("source_url", "Unknown URL")
                label = f"Video: {url}"
                sources.append(url)
            else:  # webpage
                url = chunk.metadata.get("source_url", "Unknown URL")
                label = f"Webpage: {url}"
                sources.append(url)
            context_parts.append(f"[{label}]\n{chunk.page_content}")
        full_context = "\n\n".join(context_parts)
        if len(full_context) > self.max_context_chars:
            full_context = full_context[:self.max_context_chars] + "... (truncated)"

        # 3. Prompt with citation requirement
        prompt = f"""You are an Insurance Policy Analyst. Answer based ONLY on the CONTEXT below.
CONTEXT may come from documents, video transcripts, or webpages.

RULES (strict):
- For every fact, number, condition, or limit, cite the exact source as shown in brackets: [Document: name, Page X] or [Video: URL] or [Webpage: URL].
- If information is not found, say "Not mentioned in any source."
- Do not invent anything.
- Use markdown: headings, bullet points, bold for key numbers.
- If calculation is needed, show step‑by‑step using only numbers from context.

CONTEXT:
{full_context}

QUESTION: {question}

ANSWER (with citations):"""

        llm = get_insurance_llm(temperature=0)
        response = await asyncio.to_thread(llm.invoke, prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # Post‑process warning if no citations
        if "[Document:" not in answer and "[Video:" not in answer and "[Webpage:" not in answer and "Not mentioned" not in answer:
            answer += "\n\n⚠️ **Warning:** Answer lacks explicit citations. Verify against original sources."

        # Deduplicate sources
        unique_sources = list(dict.fromkeys(sources))
        return answer, unique_sources

    # Separate methods for managing video/webpage stores
    def video_exists(self, url: str) -> bool:
        return self.video_store.url_exists(url)

    def add_video_chunks(self, url: str, chunks: List[Document]):
        self.video_store.add_video_chunks(url, chunks)

    def delete_video(self, url: str):
        self.video_store.delete_by_url(url)

    def list_videos(self) -> List[str]:
        return self.video_store.list_urls()

    def webpage_exists(self, url: str) -> bool:
        return self.webpage_store.url_exists(url)

    def add_webpage_chunks(self, url: str, chunks: List[Document]):
        self.webpage_store.add_webpage_chunks(url, chunks)

    def delete_webpage(self, url: str):
        self.webpage_store.delete_by_url(url)

    def list_webpages(self) -> List[str]:
        return self.webpage_store.list_urls()