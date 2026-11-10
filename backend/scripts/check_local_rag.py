"""Opt-in synthetic PDF -> real embeddings -> local Qdrant -> real agent check.

No existing documents, databases, buckets, collections or secrets are modified.
This is not a substitute for a PostgreSQL/Redis/S3/Celery deployment test.
"""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf
from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.services.agents.orchestrator import orchestrator
from app.services.ingestion.ingestion_service import process_document
from app.services.retrieval.embedding_service import embedding_service
from app.services.retrieval.vector_store import VectorStore
from app.services.retrieval.hybrid_retriever import HybridRetriever


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize bounded generation/embedding calls for a synthetic PDF")
    parser.add_argument("--stream", action="store_true", help="Exercise actual provider streaming through the agent")
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live to authorize Google API calls")
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Quantiva travel policy. The annual travel allowance is 4800 USD per employee.")
    raw = pdf.tobytes()
    pdf.close()
    doc_id, user_id = str(uuid.uuid4()), str(uuid.uuid4())
    chunks = process_document(doc_id, raw, "application/pdf")
    embeddings = await asyncio.to_thread(embedding_service.embed_batch, [c["text"] for c in chunks])
    store = VectorStore.__new__(VectorStore)
    store.client = AsyncQdrantClient(location=":memory:")
    store.collection_name = "synthetic_live_rag"
    store.vector_size = settings.EMBEDDING_DIMENSIONS
    try:
        await store.create_collection()
        point_ids = await store.upsert_chunks(chunks, embeddings, doc_id, user_id)
        db_chunks = [SimpleNamespace(id=uuid.uuid4(), doc_id=doc_id, qdrant_id=point, chunk_index=i, text=c["text"], page_number=c["metadata"].get("page")) for i, (c, point) in enumerate(zip(chunks, point_ids))]
        deltas = []

        async def receive(delta):
            deltas.append(delta)

        result = await orchestrator.run(query="What is the annual travel allowance?", session_id=str(uuid.uuid4()), doc_ids=[doc_id], db_chunks=db_chunks, user_id=user_id, history=[], document_names={doc_id: "synthetic-travel-policy.pdf"}, use_cache=False, retriever=HybridRetriever(store), on_token=receive if args.stream else None)
        assert "4800" in result.answer.answer.replace(",", ""), "Answer did not contain the source fact"
        assert result.answer.citations and all(c.doc_id == doc_id for c in result.answer.citations), "Missing or invalid source citations"
        assert result.answer.tokens_used > 0, "Provider usage was not recorded"
        if args.stream:
            assert deltas and "4800" in "".join(deltas).replace(",", ""), "Expected answer was not streamed"
        print(json.dumps({"status": "passed", "streamed_events": len(deltas), "chunks": len(chunks), "citations": len(result.answer.citations), "tool_calls": result.tool_calls_made, "total_generation_tokens": result.answer.tokens_used, "grounding_heuristic": result.answer.confidence, "model": result.answer.model_used}, indent=2))
    finally:
        await store.client.close()


if __name__ == "__main__":
    asyncio.run(main())
