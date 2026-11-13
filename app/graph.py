
"""Nexora AI Agentic RAG workflow.

KB -> Grade -> Tavily -> Grade -> Rewrite -> Retry.

Preserves:
- Input/output guardrails
- Existing FastAPI chat interface
- Pinecone access control
- Conversation history
- Observability decorators
"""

import hashlib
import logging
import re

from typing import Any, TypedDict

from fastapi import HTTPException
from langgraph.graph import StateGraph, START, END

from app.observability import trace

from app.aws import (
    embed,
    answer,
    answer_web,
    grade_evidence,
    classify_web_safety,
    rewrite_query,
)

from app.vector import search
from app.config import settings

from app.guardrails import (
    check_input,
    check_output,
)

from app.web_search import (
    search_web,
    WebSearchError,
)


log = logging.getLogger(__name__)


# ============================================
# 1. RETRY CONFIGURATION
# ============================================

MAX_REWRITES = 1

MAX_KB_SEARCHES = 1 + MAX_REWRITES

MAX_WEB_SEARCHES = 1 + MAX_REWRITES


# ============================================
# 2. LANGGRAPH STATE
# ============================================

class State(TypedDict, total=False):

    question: str
    current_query: str

    principal: Any
    history: list

    passages: list
    sources: list

    kb_grade: dict

    web_results: list
    web_grade: dict

    scope: str

    rewrite_count: int
    kb_count: int
    web_count: int

    web_error: str
    kb_error: str

    answer: str
    status: str


# ============================================
# 3. INPUT GUARDRAILS
# ============================================

@trace("guard")
def guard(state: State):

    question = check_input(
        state["question"]
    )

    return {
        "question": question,
        "current_query": question,

        "rewrite_count": 0,
        "kb_count": 0,
        "web_count": 0,

        "scope": "unknown",

        "passages": [],
        "sources": [],
        "web_results": [],

        "web_grade": {
            "relevant": False,
            "sufficient": False,
        },

        "kb_grade": {
            "relevant": False,
            "sufficient": False,
        },

        "web_error": "",
        "kb_error": "",
    }


# ============================================
# 4. GREETING DETECTION
# ============================================

def _is_greeting(question):

    q = re.sub(
        r"[^a-z\s]",
        "",
        question.lower()
    ).strip()

    return q in {
        "hi",
        "hello",
        "hey",
        "hi nexora",
        "hello nexora",
        "good morning",
        "good evening",
        "good afternoon",
        "thanks",
        "thank you",
    }


# ============================================
# 5. SEARCH QUERY PREPARATION
# ============================================

def _search_query(state):

    q = state["current_query"]

    followup_pattern = (
        r"^(and\b|what about\b|how about\b|"
        r"tell me more\b|what is (?:its|their)\b|"
        r"who is (?:he|she|it)\b)"
    )

    if not re.match(
        followup_pattern,
        q,
        re.IGNORECASE,
    ):
        return q

    previous_questions = [
        str(message.get("content", ""))
        for message in state.get("history", [])
        if message.get("role") == "user"
    ]

    if previous_questions:
        return (
            previous_questions[-1][:350]
            + "\nFollow-up: "
            + q
        )

    return q


# ============================================
# 6. REMOVE DUPLICATE KB RESULTS
# ============================================

def _deduplicate(hits):

    unique = []
    seen = set()

    for hit in hits:

        metadata = (
            hit.get("metadata") or {}
        )

        content = str(
            metadata.get("text", "")
        ).strip()

        key = (
            str(
                metadata.get(
                    "document_id", ""
                )
            ),
            hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(hit)

    return unique


# ============================================
# 7. RETRIEVE COMPANY KB
# ============================================

@trace("retrieve_kb")
def retrieve_kb(state: State):

    principal = state["principal"]

    count = (
        state.get("kb_count", 0) + 1
    )

    try:
        query_vector = embed(
            _search_query(state)
        )

        hits = search(
            principal.tenant,
            query_vector,
            principal.groups,
            principal.subject,
            limit=5,
        )

        hits = _deduplicate([
            hit
            for hit in hits
            if (
                hit.get("score") is not None
                and hit["score"]
                >= settings().min_relevance
            )
        ])

        log.info(
            "KB retrieval attempt=%d candidates=%d",
            count,
            len(hits),
        )

        return {
            "kb_count": count,
            "passages": hits,
            "kb_error": "",
            "sources": [],
        }

    except Exception:
        log.exception(
            "Knowledge-base retrieval failed"
        )

        return {
            "kb_count": count,
            "passages": [],
            "kb_error": "retrieval_unavailable",
            "sources": [],
        }


# ============================================
# 8. GRADE KB EVIDENCE
# ============================================

@trace("grade_kb")
def grade_kb(state: State):

    grade = grade_evidence(
        state["question"],
        state.get("passages", []),
        source="kb",
    )

    return {
        "kb_grade": grade
    }


# ============================================
# 9. ROUTE AFTER KB GRADING
# ============================================

def after_kb(state: State):

    if state.get(
        "kb_grade", {}
    ).get("sufficient"):

        return "generate_kb"

    return "check_scope"


# ============================================
# 10. PUBLIC / PRIVATE CLASSIFICATION
# ============================================

@trace("check_scope")
def check_scope(state: State):

    # Private scope must not be relaxed
    # after a query rewrite.
    if (
        state.get("rewrite_count", 0) > 0
        and state.get("scope") == "private"
    ):
        return {"scope": "private"}

    # Relevant documents explicitly marked
    # internal should remain private.
    if (
        state.get(
            "kb_grade", {}
        ).get("relevant")
        and any(
            re.search(
                r"\b(internal|confidential|private)\b",
                str(
                    (
                        hit.get("metadata") or {}
                    ).get("title", "")
                ),
                re.IGNORECASE,
            )
            for hit in state.get(
                "passages", []
            )
        )
    ):
        return {"scope": "private"}

    scope = classify_web_safety(
        state["question"],
        state.get("history", []),
    )

    log.info(
        "Web-safety scope=%s",
        scope,
    )

    return {
        "scope": scope
    }


# ============================================
# 11. ROUTE AFTER SCOPE CHECK
# ============================================

def after_scope(state: State):

    # Private company question:
    # Never send to Tavily.
    if state.get("scope") != "public":

        if (
            state.get(
                "rewrite_count", 0
            ) < MAX_REWRITES
        ):
            return "rewrite"

        return "no_evidence"

    # Permanent Tavily failures:
    # Do not retry.
    permanent_errors = {
        "authentication_failed",
        "missing_api_key",
        "rate_limited",
        "request_rejected",
    }

    if state.get("web_error") in permanent_errors:
        return "web_unavailable"

    if (
        state.get("web_count", 0)
        < MAX_WEB_SEARCHES
    ):
        return "search_web"

    return "web_unavailable"


# ============================================
# 12. TAVILY WEB SEARCH
# ============================================

@trace("search_web")
def web_search_node(state: State):

    count = (
        state.get("web_count", 0) + 1
    )

    try:
        results = search_web(
            state["current_query"]
        )

        log.info(
            "Tavily attempt=%d results=%d",
            count,
            len(results),
        )

        return {
            "web_count": count,
            "web_results": results,
            "web_error": "",

            "web_grade": {
                "relevant": False,
                "sufficient": False,
            },
        }

    except WebSearchError as exc:

        log.warning(
            "Tavily failed kind=%s attempt=%d",
            exc.kind,
            count,
        )

        return {
            "web_count": count,
            "web_results": [],
            "web_error": exc.kind,

            "web_grade": {
                "relevant": False,
                "sufficient": False,
            },
        }

    except Exception:
        log.exception(
            "Unexpected Tavily failure"
        )

        return {
            "web_count": count,
            "web_results": [],
            "web_error": "unexpected_error",

            "web_grade": {
                "relevant": False,
                "sufficient": False,
            },
        }


# ============================================
# 13. GRADE WEB EVIDENCE
# ============================================

@trace("grade_web")
def grade_web(state: State):

    grade = grade_evidence(
        state["question"],
        state.get("web_results", []),
        source="web",
    )

    return {
        "web_grade": grade
    }


# ============================================
# 14. ROUTE AFTER WEB GRADING
# ============================================

def after_web(state: State):

    if state.get(
        "web_grade", {}
    ).get("sufficient"):

        return "generate_web"

    permanent_errors = {
        "authentication_failed",
        "missing_api_key",
        "rate_limited",
        "request_rejected",
    }

    if state.get("web_error") in permanent_errors:
        return "web_unavailable"

    # KB weak -> Tavily weak -> rewrite
    if (
        state.get("rewrite_count", 0)
        < MAX_REWRITES
    ):
        return "rewrite"

    if state.get("web_error"):
        return "web_unavailable"

    return "no_evidence"


# ============================================
# 15. QUERY REWRITING
# ============================================

@trace("rewrite_query")
def rewrite(state: State):

    rewritten = rewrite_query(
        state["question"],
        state.get("history", []),
    )

    # Revalidate public scope before
    # another external search.
    scope = state.get(
        "scope", "private"
    )

    if (
        scope == "public"
        and classify_web_safety(
            rewritten,
            state.get("history", []),
        ) != "public"
    ):
        scope = "private"

    return {
        "current_query": rewritten,
        "rewrite_count": (
            state.get("rewrite_count", 0) + 1
        ),
        "scope": scope,
    }


# ============================================
# 16. FORMAT KB EVIDENCE
# ============================================

def kb_evidence(hits):

    return "\n\n".join(
        (
            f"[{i}] "
            f"{((hit.get('metadata') or {}).get('title') or 'Document')}, "
            f"page {((hit.get('metadata') or {}).get('page') or 1)}: "
            f"{str((hit.get('metadata') or {}).get('text', ''))[:2500]}"
        )
        for i, hit in enumerate(hits, 1)
    )


# ============================================
# 17. GENERATE COMPANY ANSWER
# ============================================

@trace("generate_kb")
def generate_kb(state: State):

    hits = state.get(
        "passages", []
    )

    sources = []

    for i, hit in enumerate(hits, 1):

        metadata = (
            hit.get("metadata") or {}
        )

        sources.append({
            "index": i,
            "document_id": metadata.get(
                "document_id", ""
            ),
            "title": metadata.get(
                "title", "Document"
            ),
            "page": metadata.get(
                "page", 1
            ),
            "excerpt": str(
                metadata.get("text", "")
            )[:320],
        })

    response = answer(
        state["question"],
        kb_evidence(hits),
        state.get("history", []),
    )

    return {
        "answer": response,
        "sources": sources,
        "status": "answered",
    }


# ============================================
# 18. GENERATE WEB ANSWER
# ============================================

@trace("generate_web")
def generate_web(state: State):

    results = state.get(
        "web_results", []
    )

    sources = [
        {
            "index": i,
            "title": item["title"],
            "url": item["url"],
            "excerpt": item["content"][:320],
            "page": "web",
        }
        for i, item in enumerate(results, 1)
    ]

    response = answer_web(
        state["question"],
        results,
        state.get("history", []),
    )

    return {
        "answer": response,
        "sources": sources,
        "status": "web_answer",
    }


# ============================================
# 19. GREETING RESPONSE
# ============================================

@trace("greeting")
def greeting(state: State):

    return {
        "answer": (
            "Hello! I'm Nexora AI. "
            "Ask me about your knowledge base "
            "or a public topic."
        ),
        "sources": [],
        "status": "greeting",
    }


# ============================================
# 20. INSUFFICIENT EVIDENCE
# ============================================

@trace("no_evidence")
def no_evidence(state: State):

    if state.get("scope") == "public":
        message = (
            "I couldn't find reliable supporting "
            "evidence in the knowledge base or "
            "public search results."
        )

    else:
        message = (
            "I couldn't verify that from the "
            "company documents you are "
            "authorized to access."
        )

    return {
        "answer": message,
        "sources": [],
        "status": "no_evidence",
    }


# ============================================
# 21. WEB SEARCH FAILURE
# ============================================

@trace("web_unavailable")
def web_unavailable(state: State):

    return {
        "answer": (
            "I couldn't verify an answer because "
            "live web search is unavailable or "
            "did not return reliable evidence. "
            "I won't guess about potentially "
            "current information."
        ),
        "sources": [],
        "status": "web_unavailable",
    }


# ============================================
# 22. OUTPUT GUARDRAILS
# ============================================

@trace("validate")
def validate(state: State):

    original = state.get(
        "answer", ""
    )

    checked = check_output(
        original
    )

    blocked = (
        checked != original
    )

    return {
        "answer": checked,

        "sources": (
            []
            if blocked
            else state.get("sources", [])
        ),

        "status": (
            "blocked"
            if blocked
            else state.get(
                "status", "answered"
            )
        ),
    }


# ============================================
# 23. CREATE LANGGRAPH
# ============================================

def make_graph():

    workflow = StateGraph(State)

    nodes = (
        ("guard", guard),
        ("greeting", greeting),
        ("retrieve_kb", retrieve_kb),
        ("grade_kb", grade_kb),
        ("check_scope", check_scope),
        ("search_web", web_search_node),
        ("grade_web", grade_web),
        ("rewrite", rewrite),
        ("generate_kb", generate_kb),
        ("generate_web", generate_web),
        ("no_evidence", no_evidence),
        ("web_unavailable", web_unavailable),
        ("validate", validate),
    )

    for name, node in nodes:
        workflow.add_node(
            name,
            node
        )

    workflow.add_edge(
        START,
        "guard"
    )

    workflow.add_conditional_edges(
        "guard",
        lambda state: (
            "greeting"
            if _is_greeting(state["question"])
            else "retrieve_kb"
        ),
        {
            "greeting": "greeting",
            "retrieve_kb": "retrieve_kb",
        },
    )

    workflow.add_edge(
        "retrieve_kb",
        "grade_kb"
    )

    workflow.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_kb": "generate_kb",
            "check_scope": "check_scope",
        },
    )

    workflow.add_conditional_edges(
        "check_scope",
        after_scope,
        {
            "rewrite": "rewrite",
            "no_evidence": "no_evidence",
            "search_web": "search_web",
            "web_unavailable": "web_unavailable",
        },
    )

    workflow.add_edge(
        "search_web",
        "grade_web"
    )

    workflow.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_web": "generate_web",
            "rewrite": "rewrite",
            "no_evidence": "no_evidence",
            "web_unavailable": "web_unavailable",
        },
    )

    # Retry KB after rewriting the query.
    workflow.add_edge(
        "rewrite",
        "retrieve_kb"
    )

    # Every final response passes through
    # the existing output guardrails.
    final_nodes = (
        "greeting",
        "generate_kb",
        "generate_web",
        "no_evidence",
        "web_unavailable",
    )

    for name in final_nodes:
        workflow.add_edge(
            name,
            "validate"
        )

    workflow.add_edge(
        "validate",
        END
    )

    return workflow.compile()


graph = make_graph()


# ============================================
# 24. CHAT ENTRY POINT
# ============================================

@trace("chat")
def chat(
    principal,
    question,
    history=None,
):

    if principal is None or any(
        not hasattr(
            principal,
            field,
        )
        for field in (
            "tenant",
            "groups",
            "subject",
        )
    ):
        raise HTTPException(
            status_code=401,
            detail=(
                "Authenticated principal required."
            ),
        )

    result = graph.invoke({
        "principal": principal,
        "question": question,
        "history": history or [],
    })

    return {
        "answer": result["answer"],
        "sources": result.get(
            "sources", []
        ),
        "status": result.get(
            "status", "answered"
        ),
    }
