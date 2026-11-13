# NEXORA_AI
Nexora AI adds an enterprise governance harness around an existing Agentic RAG assistant, focusing on security, guardrails, human-in-the-loop (HITL) approvals, observability, and auditability.
Key Features
- Agentic RAG: LangGraph orchestrates Pinecone retrieval, Bedrock evidence grading, Tavily web-search fallback, query rewriting, and bounded retries.
- Input/output guardrails: Detect common prompt-injection attempts and sensitive output patterns.
- HITL approvals: Queue restricted requests for administrator review; decisions are logged but do not execute actions.
- Security controls: Tenant/group/owner-filtered Pinecone retrieval (demo authentication only).
- Monitoring and audit: Rius tracing integration, PostgreSQL audit events, feedback, and conversation history.
- Document ingestion: Amazon S3 uploads, PDF/DOCX/TXT parsing, Amazon Titan embeddings, and Pinecone indexing.
Tech Stack
Python · FastAPI · LangGraph · Amazon Bedrock (Nova 2 Lite, Titan Embeddings V2) · Pinecone · Tavily · Amazon S3 · PostgreSQL · Rius
