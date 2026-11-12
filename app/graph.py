
from typing import Any, TypedDict

from fastapi import HTTPException
from langgraph.graph import StateGraph, START, END

from app.observability import trace
from app.aws import embed, answer
from app.vector import search
from app.config import settings
from app.guardrails import check_input, check_output


# -----------------------------------------
# Graph State
# -----------------------------------------

class State(TypedDict, total=False):
    question: str
    principal: Any
    passages: list
    answer: str
    sources: list
    status: str
    history: list


# -----------------------------------------
# Input Guardrails
# -----------------------------------------

@trace("guard")
def guard(state: State):
    return {
        "question": check_input(state["question"])
    }


# -----------------------------------------
# Retrieve Relevant Documents
# -----------------------------------------

@trace("retrieve")
def retrieve(state: State):
    principal = state["principal"]

    tenant = principal.tenant
    groups = principal.groups
    subject = principal.subject

    history = state.get("history", [])

    prior_questions = [
        message["content"]
        for message in history
        if message.get("role") == "user"
    ]

    question = state["question"]

    if prior_questions:
        query = (
            prior_questions[-1]
            + "\nFollow-up: "
            + question
        )
    else:
        query = question

    query_embedding = embed(query)

    hits = search(
        tenant,
        query_embedding,
        groups,
        subject,
    )

    hits = [
        hit for hit in hits
        if hit["score"] >= settings().min_relevance
    ]

    sources = [
        {
            "index": i + 1,
            "document_id": hit["metadata"].get(
                "document_id", ""
            ),
            "title": hit["metadata"].get(
                "title", ""
            ),
            "page": hit["metadata"].get(
                "page", 1
            ),
            "excerpt": hit["metadata"].get(
                "text", ""
            )[:320],
        }
        for i, hit in enumerate(hits)
    ]

    return {
        "passages": hits,
        "sources": sources,
    }


# -----------------------------------------
# Conditional Routing
# -----------------------------------------

def route(state: State):
    if state.get("passages"):
        return "generate"

    return "no_evidence"


# -----------------------------------------
# No Evidence
# -----------------------------------------

@trace("no_evidence")
def no_evidence(state: State):
    return {
        "answer": (
            "I could not find supporting evidence "
            "in documents you are authorized to access."
        ),
        "status": "no_evidence",
    }


# -----------------------------------------
# Generate LLM Answer
# -----------------------------------------

@trace("generate")
def generate(state: State):
    evidence = "\n\n".join(
        (
            f"[{i + 1}] "
            f"{hit['metadata'].get('title', 'Document')}, "
            f"page {hit['metadata'].get('page', 1)}: "
            f"{hit['metadata'].get('text', '')}"
        )
        for i, hit in enumerate(state["passages"])
    )

    response = answer(
        state["question"],
        evidence,
        state.get("history", []),
    )

    return {
        "answer": response,
        "status": "answered",
    }


# -----------------------------------------
# Output Guardrails
# -----------------------------------------

@trace("validate")
def validate(state: State):
    original_answer = state["answer"]
    checked_answer = check_output(original_answer)

    return {
        "answer": checked_answer,
        "status": (
            "blocked"
            if checked_answer != original_answer
            else state.get("status", "answered")
        ),
    }


# -----------------------------------------
# Build LangGraph Workflow
# -----------------------------------------

def make_graph():
    workflow = StateGraph(State)

    workflow.add_node("guard", guard)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("no_evidence", no_evidence)
    workflow.add_node("generate", generate)
    workflow.add_node("validate", validate)

    workflow.add_edge(START, "guard")
    workflow.add_edge("guard", "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        route,
        {
            "generate": "generate",
            "no_evidence": "no_evidence",
        },
    )

    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", END)
    workflow.add_edge("no_evidence", END)

    return workflow.compile()


graph = make_graph()


# -----------------------------------------
# Chat Entry Point
# -----------------------------------------

@trace("chat")
def chat(principal, question, history=None):

    if principal is None or any(
        not hasattr(principal, field)
        for field in ("tenant", "groups", "subject")
    ):
        raise HTTPException(
            status_code=401,
            detail="Authenticated principal required.",
        )

    result = graph.invoke(
        {
            "principal": principal,
            "question": question,
            "history": history or [],
        }
    )

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "status": result.get("status", "answered"),
    }
