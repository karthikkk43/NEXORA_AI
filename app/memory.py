
import uuid

from app.db import connection
from app.config import settings


def history(tenant: str, conversation_id: uuid.UUID):
    """Retrieve recent conversation history."""

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM chat_messages
                WHERE tenant_id = %s
                  AND conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            (
                tenant,
                conversation_id,
                settings().memory_turns * 2,
            ),
        ).fetchall()

    return rows


def save_turn(
    tenant: str,
    conversation_id: uuid.UUID,
    question: str,
    response: str,
):
    """Save the user's question and AI response."""

    with connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (
                tenant_id,
                conversation_id,
                role,
                content
            )
            VALUES
                (%s, %s, 'user', %s),
                (%s, %s, 'assistant', %s)
            """,
            (
                tenant,
                conversation_id,
                question,
                tenant,
                conversation_id,
                response,
            ),
        )


def list_messages(
    tenant: str,
    conversation_id: uuid.UUID,
):
    """Retrieve up to 100 conversation messages."""

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE tenant_id = %s
              AND conversation_id = %s
            ORDER BY id ASC
            LIMIT 100
            """,
            (
                tenant,
                conversation_id,
            ),
        ).fetchall()

    return rows
