import asyncio
import logging
from sqlalchemy import update, select
from app.workers.celery_app import celery_app
from app.db.database import AsyncSessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.core.storage import get_file
from app.services.ingestion.ingestion_service import process_document as process_file
from app.services.retrieval.embedding_service import embedding_service
from app.services.retrieval.vector_store import vector_store

logger = logging.getLogger(__name__)

async def _process_document_async(doc_id: str):
    """Async helper to perform the database operations and orchestration."""
    async with AsyncSessionLocal() as session:
        # 1. Fetch document metadata
        stmt = select(Document).where(Document.id == doc_id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()
        
        if not doc:
            logger.error(f"Document {doc_id} not found in database.")
            return
            
        try:
            # 2. Fetch raw file bytes from S3/MinIO
            logger.info(f"Fetching {doc.filename} from storage...")
            file_bytes = get_file(doc.storage_path)
            
            # 3. Parse and chunk the document
            logger.info(f"Parsing {doc.filename} ({doc.file_type})...")
            chunks = process_file(str(doc.id), file_bytes, doc.file_type)
            
            if not chunks:
                logger.warning(f"No chunks extracted from document {doc_id}")
                stmt = update(Document).where(Document.id == doc_id).values(status="ready", chunk_count=0)
                await session.execute(stmt)
                await session.commit()
                return

            # 4. Embed chunks via Google text-embedding-004
            logger.info(f"Embedding {len(chunks)} chunks...")
            texts = [chunk["text"] for chunk in chunks]
            
            # Batch embeddings in groups of 100
            embeddings = []
            for i in range(0, len(texts), 100):
                batch = texts[i:i+100]
                batch_embs = embedding_service.embed_batch(batch)
                embeddings.extend(batch_embs)
                
            # 5. Ensure Qdrant collection exists, then upsert
            logger.info("Upserting vectors to Qdrant...")
            await vector_store.create_collection()
            qdrant_ids = await vector_store.upsert_chunks(chunks, embeddings, str(doc.id), str(doc.user_id))
            
            # 6. Save chunk metadata to Postgres
            logger.info("Saving chunk metadata to Postgres...")
            db_chunks = []
            for i, (chunk, q_id) in enumerate(zip(chunks, qdrant_ids)):
                db_chunks.append(
                    Chunk(
                        doc_id=doc.id,
                        chunk_index=i,
                        text_preview=chunk["text"][:200],  # Preview text
                        page_number=chunk["metadata"].get("page"),
                        chunk_type=chunk["metadata"].get("chunk_type", "text"),
                        qdrant_id=q_id
                    )
                )
            session.add_all(db_chunks)

            # 7. Update Document status
            stmt = update(Document).where(Document.id == doc_id).values(
                status="ready", 
                chunk_count=len(chunks)
            )
            await session.execute(stmt)
            await session.commit()
            logger.info(f"Successfully processed document {doc_id} with {len(chunks)} chunks.")
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {str(e)}")
            stmt = update(Document).where(Document.id == doc_id).values(status="failed")
            await session.execute(stmt)
            await session.commit()

@celery_app.task(name="process_document")
def process_document(doc_id: str):
    """
    Celery task to process the uploaded document.
    """
    asyncio.run(_process_document_async(doc_id))

