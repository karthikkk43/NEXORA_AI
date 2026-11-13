
import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Initialize observability before traced modules
from app.observability import initialize

initialize()

# Application imports
from app.demo_context import principal, Principal
from app.aws import client
from app.config import settings
from app.db import connection, audit
from app.graph import chat
from app.guardrails import classify_input
from app.memory import history, save_turn, list_messages
from worker.ingest import process_s3_key


# --------------------------------------------------
# 1. Application Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger("nexora.api")

BASE = Path(__file__).resolve().parent

app = FastAPI(
    title="Nexora AI - Enterprise Agentic RAG",
    description="Enterprise AI Knowledge Assistant",
    version="1.0.0",
    docs_url="/api/docs"
)

app.mount(
    "/static",
    StaticFiles(directory=BASE / "static"),
    name="static"
)

templates = Jinja2Templates(
    directory=BASE / "templates"
)


# --------------------------------------------------
# 2. Request Models
# --------------------------------------------------

class Upload(BaseModel):
    filename: str = Field(
        min_length=1,
        max_length=180
    )


class CompleteUpload(BaseModel):
    document_id: uuid.UUID


class Ask(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=2000
    )
    conversation_id: uuid.UUID | None = None


class Feedback(BaseModel):
    document_id: str | None = None
    rating: int = Field(ge=-1, le=1)
    note: str = Field(default="", max_length=500)


class ApprovalDecision(BaseModel):
    approve: bool


# --------------------------------------------------
# 3. Frontend
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "region": settings().aws_region,
            "mode": "single-workspace demo"
        }
    )


# --------------------------------------------------
# 4. Health Checks
# --------------------------------------------------

@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():

    try:
        with connection() as conn:
            conn.execute("SELECT 1").fetchone()

        return {"status": "ready"}

    except Exception as exc:
        log.exception("Database readiness check failed")

        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        ) from exc


# --------------------------------------------------
# 5. Generate S3 Upload URL
# --------------------------------------------------

@app.post("/api/documents/uploads")
def create_upload(
    body: Upload,
    p: Principal = Depends(principal)
):

    suffix = Path(body.filename).suffix.lower()

    allowed_types = {".pdf", ".docx", ".txt"}

    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Allowed types: PDF, DOCX, TXT"
        )

    doc_id = uuid.uuid4()

    key = (
        f"tenants/{p.tenant}/incoming/"
        f"{doc_id}/v1{suffix}"
    )

    # Generate presigned S3 upload URL
    try:
        url = client("s3").generate_presigned_url(
            "put_object",
            Params={
                "Bucket": settings().s3_bucket_name,
                "Key": key,
                "ContentType": "application/octet-stream"
            },
            ExpiresIn=300
        )

    except Exception as exc:
        log.exception("Failed to generate S3 upload URL")

        raise HTTPException(
            status_code=502,
            detail="Could not generate upload URL"
        ) from exc

    # Save document metadata to PostgreSQL
    with connection() as conn:

        conn.execute(
            """
            INSERT INTO documents (
                id,
                tenant_id,
                title,
                owner_sub,
                allowed_groups,
                s3_key,
                status,
                current_version
            )
            VALUES (%s,%s,%s,%s,%s,%s,'pending',1)
            """,
            (
                doc_id,
                p.tenant,
                body.filename,
                p.subject,
                ["workspace"],
                key
            )
        )

        audit(
            conn,
            p.subject,
            p.tenant,
            "upload_requested",
            {
                "document_id": str(doc_id)
            }
        )

    return {
        "document_id": str(doc_id),
        "upload_url": url,
        "headers": {
            "Content-Type": "application/octet-stream"
        }
    }


# --------------------------------------------------
# 6. Complete Document Upload
# --------------------------------------------------

@app.post("/api/documents/complete")
def complete_upload(
    body: CompleteUpload,
    p: Principal = Depends(principal)
):

    with connection() as conn:

        row = conn.execute(
            """
            SELECT s3_key, status
            FROM documents
            WHERE id = %s
              AND tenant_id = %s
              AND owner_sub = %s
            """,
            (
                body.document_id,
                p.tenant,
                p.subject
            )
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    if row["status"] == "ready":
        return {"status": "ready"}

    if row["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail="Document already processing"
        )

    if row["status"] == "retired":
        raise HTTPException(
            status_code=409,
            detail="Retired documents cannot be ingested"
        )

    try:
        process_s3_key(row["s3_key"])

    except Exception as exc:
        log.exception("Document ingestion failed")

        raise HTTPException(
            status_code=502,
            detail="Ingestion failed. Check server logs."
        ) from exc

    # Verify actual document status
    with connection() as conn:

        updated = conn.execute(
            """
            SELECT status
            FROM documents
            WHERE id = %s
              AND tenant_id = %s
            """,
            (
                body.document_id,
                p.tenant
            )
        ).fetchone()

    if not updated or updated["status"] != "ready":
        raise HTTPException(
            status_code=409,
            detail="Document ingestion did not complete"
        )

    return {"status": "ready"}


# --------------------------------------------------
# 7. List Documents
# --------------------------------------------------

@app.get("/api/documents")
def documents(
    p: Principal = Depends(principal)
):

    with connection() as conn:

        rows = conn.execute(
            """
            SELECT
                id,
                title,
                status,
                current_version,
                created_at,
                allowed_groups
            FROM documents
            WHERE tenant_id = %s
              AND (
                  allowed_groups && %s::text[]
                  OR owner_sub = %s
              )
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (
                p.tenant,
                list(p.groups),
                p.subject
            )
        ).fetchall()

    return {"documents": rows}


# --------------------------------------------------
# 8. Agentic RAG Chat
# --------------------------------------------------

@app.post("/api/chat")
def ask(
    body: Ask,
    p: Principal = Depends(principal)
):

    # Input guardrails
    reason = classify_input(body.question)

    if reason:

        with connection() as conn:

            audit(
                conn,
                p.subject,
                p.tenant,
                "guardrail_input_blocked",
                {"reason": reason}
            )

        raise HTTPException(
            status_code=400,
            detail=(
                "Input guardrail blocked "
                f"this request ({reason})."
            )
        )

    # Human-in-the-loop approval
    restricted_actions = (
        "delete",
        "export",
        "send to",
        "update policy"
    )

    if (
        settings().hitl_enabled
        and any(
            action in body.question.lower()
            for action in restricted_actions
        )
    ):

        with connection() as conn:

            row = conn.execute(
                """
                INSERT INTO approvals (
                    tenant_id,
                    requester,
                    question,
                    status
                )
                VALUES (%s,%s,%s,'pending')
                RETURNING id
                """,
                (
                    p.tenant,
                    p.subject,
                    body.question
                )
            ).fetchone()

            audit(
                conn,
                p.subject,
                p.tenant,
                "approval_requested",
                {"id": str(row["id"])}
            )

        return {
            "answer": (
                "This request requires administrator "
                "approval before any action."
            ),
            "sources": [],
            "status": "pending_approval",
            "approval_id": str(row["id"])
        }

    # Conversation memory
    conversation_id = (
        body.conversation_id or uuid.uuid4()
    )

    prior = history(
        p.tenant,
        conversation_id
    )

    # Execute LangGraph
    result = chat(
        p,
        body.question,
        prior
    )

    # Handle blocked LLM output
    if result.get("status") == "blocked":

        with connection() as conn:

            audit(
                conn,
                p.subject,
                p.tenant,
                "guardrail_output_blocked",
                {
                    "conversation_id": str(conversation_id)
                }
            )

        result["status"] = "guardrail_blocked"
        result["sources"] = []

    # Save conversation history
    save_turn(
        p.tenant,
        conversation_id,
        body.question,
        result["answer"]
    )

    return {
        **result,
        "conversation_id": str(conversation_id)
    }


# --------------------------------------------------
# 9. Conversation History
# --------------------------------------------------

@app.get(
    "/api/conversations/{conversation_id}/messages"
)
def conversation_messages(
    conversation_id: uuid.UUID,
    p: Principal = Depends(principal)
):

    messages = list_messages(
        p.tenant,
        conversation_id
    )

    return {
        "conversation_id": str(conversation_id),
        "messages": messages
    }


# --------------------------------------------------
# 10. User Feedback
# --------------------------------------------------

@app.post("/api/feedback")
def feedback(
    body: Feedback,
    p: Principal = Depends(principal)
):

    with connection() as conn:

        conn.execute(
            """
            INSERT INTO feedback (
                tenant_id,
                actor,
                rating,
                note
            )
            VALUES (%s,%s,%s,%s)
            """,
            (
                p.tenant,
                p.subject,
                body.rating,
                body.note
            )
        )

    return {"saved": True}


# --------------------------------------------------
# 11. Admin Metrics
# --------------------------------------------------

@app.get("/api/admin/metrics")
def metrics(
    p: Principal = Depends(principal)
):

    if not p.admin:
        raise HTTPException(
            status_code=403,
            detail="Admin only"
        )

    with connection() as conn:

        docs = conn.execute(
            """
            SELECT status, COUNT(*) AS total
            FROM documents
            WHERE tenant_id = %s
            GROUP BY status
            """,
            (p.tenant,)
        ).fetchall()

        fb = conn.execute(
            """
            SELECT rating, COUNT(*) AS total
            FROM feedback
            WHERE tenant_id = %s
            GROUP BY rating
            """,
            (p.tenant,)
        ).fetchall()

    return {
        "documents": docs,
        "feedback": fb
    }


# --------------------------------------------------
# 12. List HITL Approvals
# --------------------------------------------------

@app.get("/api/approvals")
def list_approvals(
    p: Principal = Depends(principal)
):

    if not p.admin:
        raise HTTPException(
            status_code=403,
            detail="Admin only"
        )

    with connection() as conn:

        rows = conn.execute(
            """
            SELECT
                id,
                requester,
                question,
                status,
                reviewer,
                created_at,
                reviewed_at
            FROM approvals
            WHERE tenant_id = %s
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (p.tenant,)
        ).fetchall()

    return {"approvals": rows}


# --------------------------------------------------
# 13. HITL Approval Decision
# --------------------------------------------------

@app.post(
    "/api/approvals/{approval_id}/decision"
)
def decide(
    approval_id: uuid.UUID,
    body: ApprovalDecision,
    p: Principal = Depends(principal)
):

    if not p.admin:
        raise HTTPException(
            status_code=403,
            detail="Admin only"
        )

    status = (
        "approved"
        if body.approve
        else "rejected"
    )

    with connection() as conn:

        row = conn.execute(
            """
            UPDATE approvals
            SET
                status = %s,
                reviewer = %s,
                reviewed_at = NOW()
            WHERE id = %s
              AND tenant_id = %s
              AND status = 'pending'
            RETURNING id
            """,
            (
                status,
                p.subject,
                approval_id,
                p.tenant
            )
        ).fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="Pending approval not found"
            )

        audit(
            conn,
            p.subject,
            p.tenant,
            "approval_decided",
            {
                "id": str(approval_id),
                "approved": body.approve
            }
        )

    # This endpoint records the decision only.
    # It does not execute destructive actions.

    return {
        "status": status,
        "executed": False
    }


# --------------------------------------------------
# 14. Guardrail Metrics
# --------------------------------------------------

@app.get("/api/admin/guardrails")
def guardrail_metrics(
    p: Principal = Depends(principal)
):

    if not p.admin:
        raise HTTPException(
            status_code=403,
            detail="Admin only"
        )

    with connection() as conn:

        rows = conn.execute(
            """
            SELECT action, COUNT(*) AS total
            FROM audit_events
            WHERE tenant_id = %s
              AND action IN (
                  'guardrail_input_blocked',
                  'guardrail_output_blocked'
              )
            GROUP BY action
            ORDER BY action
            """,
            (p.tenant,)
        ).fetchall()

    return {
        "events": rows,
        "mode": "deterministic_demo",
        "managed_bedrock_guardrails": False
    }
