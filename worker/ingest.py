
import io
import logging
from pathlib import Path

from docx import Document
from pypdf import PdfReader

# Initialize observability before importing traced modules
from app.observability import initialize, trace

initialize()

from app.aws import client, embed
from app.vector import upsert, index, namespace
from app.config import settings
from app.db import connection, audit


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger("atlas.worker")


# --------------------------------------------------
# 1. Document Parsing
# --------------------------------------------------

def parse(blob: bytes, suffix: str):
    """Extract text from PDF, DOCX, or TXT files."""

    suffix = suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(blob))

        return [
            (i + 1, page.extract_text() or "")
            for i, page in enumerate(reader.pages)
        ]

    if suffix == ".docx":
        doc = Document(io.BytesIO(blob))

        texts = [p.text for p in doc.paragraphs]

        # Include text from Word tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    texts.append(cell.text)

        return [(1, "\n".join(texts))]

    if suffix == ".txt":
        return [(1, blob.decode("utf-8"))]

    raise ValueError(
        f"Unsupported document format: {suffix}"
    )


# --------------------------------------------------
# 2. Text Chunking
# --------------------------------------------------

def chunks(text, size=1200, overlap=180):
    """Split text into overlapping chunks."""

    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid chunk size or overlap")

    text = " ".join(text.split())

    pos = 0

    while pos < len(text):
        yield text[pos:pos + size]

        if pos + size >= len(text):
            break

        pos += size - overlap


# --------------------------------------------------
# 3. Document Ingestion
# --------------------------------------------------

@trace("process_s3_key")
def process_s3_key(key):
    """
    Download document from S3, extract text,
    create Titan embeddings, store vectors
    in Pinecone, and update PostgreSQL.
    """

    s = settings()

    doc = None
    uploaded_ids = []

    # Atomically claim a document for processing
    with connection() as conn:
        doc = conn.execute(
            """
            UPDATE documents
            SET status = 'processing'
            WHERE s3_key = %s
              AND status IN ('pending', 'failed')
            RETURNING *
            """,
            (key,)
        ).fetchone()

    if doc is None:
        log.info(
            "Document not found or not eligible for ingestion: %s",
            key
        )
        return

    try:
        # ------------------------------------------
        # Step 1: Download document from Amazon S3
        # ------------------------------------------

        obj = client("s3").get_object(
            Bucket=s.s3_bucket_name,
            Key=key
        )

        content_length = obj.get("ContentLength", 0)

        if content_length > s.max_upload_bytes:
            raise ValueError("Upload too large")

        blob = obj["Body"].read(
            s.max_upload_bytes + 1
        )

        if len(blob) > s.max_upload_bytes:
            raise ValueError("Upload too large")

        # ------------------------------------------
        # Step 2: Validate document
        # ------------------------------------------

        suffix = Path(key).suffix.lower()

        if suffix not in (".pdf", ".docx", ".txt"):
            raise ValueError(
                f"Unsupported document format: {suffix}"
            )

        if suffix == ".pdf":
            if not blob.startswith(b"%PDF"):
                raise ValueError("Invalid PDF file")

        # ------------------------------------------
        # Step 3: Extract document text
        # ------------------------------------------

        pages = parse(blob, suffix)

        # ------------------------------------------
        # Step 4: Split text into chunks
        # ------------------------------------------

        parts = [
            (page, part)
            for page, text in pages
            for part in chunks(text)
        ]

        if not parts:
            raise ValueError(
                "No extractable text found. "
                "Scanned PDFs may require Amazon Textract OCR."
            )

        log.info(
            "Document %s split into %d chunks",
            doc["id"],
            len(parts)
        )

        # ------------------------------------------
        # Step 5: Generate Titan embeddings
        # ------------------------------------------

        vectors = []

        for i, (page, part) in enumerate(parts):

            embedding = embed(part)

            if len(embedding) != s.embedding_dimension:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {s.embedding_dimension}, "
                    f"received {len(embedding)}"
                )

            vector_id = (
                f"{doc['id']}:"
                f"{doc['current_version']}:"
                f"{i}"
            )

            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "document_id": str(doc["id"]),
                    "version": doc["current_version"],
                    "title": doc["title"],
                    "page": page,
                    "text": part,
                    "owner_sub": doc["owner_sub"],
                    "allowed_groups": doc["allowed_groups"],
                },
            })

        # ------------------------------------------
        # Step 6: Upsert embeddings to Pinecone
        # ------------------------------------------

        batch_size = 50

        for i in range(0, len(vectors), batch_size):

            batch = vectors[i:i + batch_size]

            # Track IDs before attempting the batch
            # so failed batches can be cleaned up.
            uploaded_ids.extend(
                vector["id"] for vector in batch
            )

            upsert(
                doc["tenant_id"],
                batch
            )

        # ------------------------------------------
        # Step 7: Update PostgreSQL and audit
        # ------------------------------------------

        with connection() as conn:

            conn.execute(
                """
                UPDATE documents
                SET status = 'ready'
                WHERE id = %s
                  AND status = 'processing'
                """,
                (doc["id"],)
            )

            audit(
                conn,
                doc["owner_sub"],
                doc["tenant_id"],
                "ingest_ready",
                {
                    "document_id": str(doc["id"]),
                    "chunks": len(parts),
                    "version": doc["current_version"],
                }
            )

        log.info(
            "Successfully ingested document %s with %d chunks",
            doc["id"],
            len(parts)
        )

    except Exception:

        log.exception(
            "Document ingestion failed for key %s",
            key
        )

        # Best-effort cleanup of vectors from this attempt
        if uploaded_ids:
            try:
                pinecone_index = index()

                for i in range(0, len(uploaded_ids), 1000):
                    pinecone_index.delete(
                        ids=uploaded_ids[i:i + 1000],
                        namespace=namespace(doc["tenant_id"])
                    )

            except Exception:
                log.exception(
                    "Pinecone cleanup failed for document %s",
                    doc["id"]
                )

        # Update PostgreSQL failure status
        try:
            with connection() as conn:

                conn.execute(
                    """
                    UPDATE documents
                    SET status = 'failed'
                    WHERE id = %s
                      AND status = 'processing'
                    """,
                    (doc["id"],)
                )

                audit(
                    conn,
                    doc["owner_sub"],
                    doc["tenant_id"],
                    "ingest_failed",
                    {
                        "document_id": str(doc["id"]),
                        "version": doc["current_version"],
                    }
                )

        except Exception:
            log.exception(
                "Failed to record ingestion failure in PostgreSQL"
            )

        raise
