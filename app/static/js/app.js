
"use strict";

// ============================================
// NEXORA AI — FRONTEND JAVASCRIPT
// ============================================

const $ = (id) => document.getElementById(id);

const STORAGE_KEY = "nexora_conversation_id";

// ============================================
// 1. UUID GENERATOR — HTTP + HTTPS
// ============================================

function generateUUID() {
    if (
        typeof crypto !== "undefined" &&
        typeof crypto.randomUUID === "function"
    ) {
        return crypto.randomUUID();
    }

    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);

    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    const hex = Array.from(bytes, b =>
        b.toString(16).padStart(2, "0")
    ).join("");

    return [
        hex.slice(0, 8),
        hex.slice(8, 12),
        hex.slice(12, 16),
        hex.slice(16, 20),
        hex.slice(20)
    ].join("-");
}

let conversationId =
    sessionStorage.getItem(STORAGE_KEY) ||
    generateUUID();

sessionStorage.setItem(STORAGE_KEY, conversationId);

let chatBusy = false;
let uploadBusy = false;
let toastTimer;

// ============================================
// 2. NOTIFICATIONS
// ============================================

function toast(message) {
    const element = $("toast");

    if (!element) {
        console.log(message);
        return;
    }

    element.textContent = message;
    element.style.display = "block";

    clearTimeout(toastTimer);

    toastTimer = setTimeout(() => {
        element.style.display = "none";
    }, 4500);
}

// ============================================
// 3. FASTAPI REQUEST HELPER
// ============================================

async function api(path, options = {}) {
    const response = await fetch(path, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {})
        }
    });

    const contentType =
        response.headers.get("content-type") || "";

    const data = contentType.includes("application/json")
        ? await response.json()
        : { detail: await response.text() };

    if (!response.ok) {
        const detail = data.detail || "Request failed";

        throw new Error(
            typeof detail === "string"
                ? detail
                : JSON.stringify(detail)
        );
    }

    return data;
}

// ============================================
// 4. SIDEBAR NAVIGATION
// ============================================

function view(name) {
    const tabs = ["chat", "docs", "approval", "admin"];

    tabs.forEach(tab => {
        $(tab + "View").classList.toggle(
            "hidden",
            tab !== name
        );

        $(tab + "Tab").classList.toggle(
            "active",
            tab === name
        );
    });

    const headings = {
        chat: "Ask your organization's knowledge",
        docs: "Document management",
        approval: "Approval requests",
        admin: "Operational overview"
    };

    $("heading").textContent = headings[name];

    if (name === "docs") {
        loadDocs();
    }

    if (name === "approval") {
        loadApprovals();
    }

    if (name === "admin") {
        loadMetrics();
    }
}

$("chatTab").onclick = () => view("chat");
$("docsTab").onclick = () => view("docs");
$("approvalTab").onclick = () => view("approval");
$("adminTab").onclick = () => view("admin");

$("refreshApprovals").onclick = loadApprovals;

// ============================================
// 5. CHAT WELCOME SCREEN
// ============================================

function showHero(show) {
    const hero = document.querySelector(".hero");

    if (hero) {
        hero.classList.toggle("hidden", !show);
    }
}

// ============================================
// 6. NEW CONVERSATION
// ============================================

$("newChat").onclick = () => {
    conversationId = generateUUID();

    sessionStorage.setItem(
        STORAGE_KEY,
        conversationId
    );

    $("messages").replaceChildren();
    $("question").value = "";

    showHero(true);

    toast("New conversation started");
};

// ============================================
// 7. DISPLAY CHAT MESSAGES
// ============================================

function addMessage(content, kind, sources = []) {
    const element = document.createElement("div");

    element.className = "message " + kind;
    element.textContent = String(content ?? "");

    if (Array.isArray(sources) && sources.length) {
        const wrapper = document.createElement("div");
        wrapper.className = "sources";

        sources.forEach((source, index) => {
            const details = document.createElement("details");
            const summary = document.createElement("summary");
            const excerpt = document.createElement("p");

            const citationNumber =
                source.index ?? index + 1;

            summary.textContent =
                `[${citationNumber}] ` +
                `${source.title || "Document"}` +
                (source.page
                    ? ` — page ${source.page}`
                    : "");

            excerpt.textContent = source.excerpt || "";

            details.append(summary, excerpt);
            wrapper.append(details);
        });

        element.append(wrapper);

        const feedbackButton =
            document.createElement("button");

        feedbackButton.textContent = "👍 Helpful";
        feedbackButton.type = "button";

        feedbackButton.onclick = async () => {
            try {
                await api("/api/feedback", {
                    method: "POST",
                    body: JSON.stringify({
                        rating: 1,
                        note: "Helpful answer"
                    })
                });

                feedbackButton.disabled = true;
                feedbackButton.textContent =
                    "✓ Feedback saved";

                toast("Feedback saved");
            } catch (error) {
                toast(error.message);
            }
        };

        element.append(feedbackButton);
    }

    $("messages").append(element);

    $("messages").scrollTop =
        $("messages").scrollHeight;

    return element;
}

// ============================================
// 8. SEND MESSAGE TO AGENTIC RAG
// ============================================

$("chatForm").onsubmit = async (event) => {
    event.preventDefault();

    if (chatBusy) return;

    const questionInput = $("question");
    const question = questionInput.value.trim();

    if (question.length < 3) {
        toast("Please enter at least 3 characters");
        return;
    }

    const requestConversationId = conversationId;

    const sendButton =
        $("chatForm").querySelector(
            'button[type="submit"]'
        );

    chatBusy = true;
    sendButton.disabled = true;

    showHero(false);

    addMessage(question, "user");
    questionInput.value = "";

    const loading = addMessage(
        "Nexora AI is searching your knowledge base...",
        "assistant"
    );

    try {
        const result = await api("/api/chat", {
            method: "POST",
            body: JSON.stringify({
                question,
                conversation_id: requestConversationId
            })
        });

        loading.remove();

        // Ignore responses from an old conversation.
        if (requestConversationId !== conversationId) {
            return;
        }

        const answerElement = addMessage(
            result.answer || "No answer returned.",
            "assistant",
            result.sources || []
        );

        if (result.conversation_id) {
            conversationId = result.conversation_id;

            sessionStorage.setItem(
                STORAGE_KEY,
                conversationId
            );
        }

        if (result.status === "pending_approval") {
            const button =
                document.createElement("button");

            button.type = "button";
            button.textContent =
                "Review approval request →";

            button.onclick = () => view("approval");

            answerElement.append(button);
        }

    } catch (error) {
        loading.remove();

        if (requestConversationId === conversationId) {
            addMessage(
                "Unable to process request: " +
                error.message,
                "assistant"
            );
        }

        toast(error.message);

    } finally {
        chatBusy = false;
        sendButton.disabled = false;
    }
};

// ============================================
// 9. SUGGESTED QUESTIONS
// ============================================

document.querySelectorAll("[data-q]").forEach(
    button => {
        button.onclick = () => {
            $("question").value = button.dataset.q;
            $("question").focus();
        };
    }
);

// ============================================
// 10. LOAD DOCUMENTS
// ============================================

async function loadDocs() {
    const root = $("docs");

    root.textContent = "Loading documents...";

    try {
        const result = await api("/api/documents");

        root.replaceChildren();

        const documents = result.documents || [];

        if (!documents.length) {
            root.textContent =
                "No documents uploaded yet.";
            return;
        }

        documents.forEach(doc => {
            const row =
                document.createElement("div");

            row.className = "document-row";

            row.textContent =
                `${doc.title} — ${doc.status} ` +
                `(v${doc.current_version})`;

            root.append(row);
        });

    } catch (error) {
        root.textContent =
            "Unable to load documents: " +
            error.message;

        toast(error.message);
    }
}

// ============================================
// 11. UPLOAD TO AMAZON S3
// ============================================

$("uploadForm").onsubmit = async (event) => {
    event.preventDefault();

    if (uploadBusy) return;

    const file = $("file").files[0];

    if (!file) {
        toast("Please select a file");
        return;
    }

    const extension =
        "." + file.name.split(".").pop().toLowerCase();

    const allowed = [".pdf", ".docx", ".txt"];

    if (!allowed.includes(extension)) {
        toast("Allowed file types: PDF, DOCX, TXT");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        toast("Maximum upload size is 10 MB");
        return;
    }

    const uploadButton =
        $("uploadForm").querySelector(
            'button[type="submit"]'
        );

    uploadBusy = true;
    uploadButton.disabled = true;
    uploadButton.textContent = "Preparing upload...";

    try {
        // STEP 1: Generate an S3 presigned URL.

        const result = await api(
            "/api/documents/uploads",
            {
                method: "POST",
                body: JSON.stringify({
                    filename: file.name
                })
            }
        );

        // STEP 2: Upload file directly to S3.

        uploadButton.textContent =
            "Uploading to S3...";

        const uploadResponse = await fetch(
            result.upload_url,
            {
                method: "PUT",
                headers: result.headers || {
                    "Content-Type":
                        "application/octet-stream"
                },
                body: file
            }
        );

        if (!uploadResponse.ok) {
            const errorText =
                await uploadResponse.text();

            // Do not log the presigned URL.
            // It contains temporary authorization.

            console.error("S3 upload error:", {
                status: uploadResponse.status,
                response: errorText
            });

            const xmlErrorCode =
                errorText.match(
                    /<Code>([^<]+)<\/Code>/
                )?.[1];

            const xmlMessage =
                errorText.match(
                    /<Message>([^<]+)<\/Message>/
                )?.[1];

            throw new Error(
                `S3 upload failed (${uploadResponse.status})` +
                (xmlErrorCode
                    ? `: ${xmlErrorCode}`
                    : "") +
                (xmlMessage
                    ? ` — ${xmlMessage}`
                    : "")
            );
        }

        toast("File uploaded to S3");

        // STEP 3: Process and index the document.

        uploadButton.textContent =
            "Processing document...";

        const completion = await api(
            "/api/documents/complete",
            {
                method: "POST",
                body: JSON.stringify({
                    document_id: result.document_id
                })
            }
        );

        if (completion.status !== "ready") {
            throw new Error(
                "Document processing did not complete"
            );
        }

        $("uploadForm").reset();

        await loadDocs();

        toast(
            "Document successfully indexed in Pinecone!"
        );

    } catch (error) {
        console.error(
            "Nexora document upload failed:",
            error.message
        );

        toast(error.message);

    } finally {
        uploadBusy = false;
        uploadButton.disabled = false;

        uploadButton.textContent =
            "Upload Document ↗";
    }
};

// ============================================
// 12. LOAD WORKSPACE METRICS
// ============================================

async function loadMetrics() {
    $("metrics").textContent =
        "Loading application metrics...";

    $("guardrailMetrics").textContent =
        "Loading guardrail events...";

    try {
        const [metrics, guardrails] =
            await Promise.all([
                api("/api/admin/metrics"),
                api("/api/admin/guardrails")
            ]);

        $("metrics").textContent =
            JSON.stringify(metrics, null, 2);

        $("guardrailMetrics").textContent =
            JSON.stringify(guardrails, null, 2);

    } catch (error) {
        $("metrics").textContent =
            "Unable to load metrics: " +
            error.message;

        $("guardrailMetrics").textContent =
            "Unable to load guardrail metrics.";

        toast(error.message);
    }
}

// ============================================
// 13. LOAD HUMAN APPROVALS
// ============================================

async function loadApprovals() {
    const root = $("approvalList");

    root.textContent = "Loading approvals...";

    try {
        const response = await api("/api/approvals");

        const approvals =
            Array.isArray(response.approvals)
                ? response.approvals
                : [];

        const pending = approvals.filter(
            approval => approval.status === "pending"
        ).length;

        $("approvalSummary").textContent =
            `${pending} pending · ` +
            `${approvals.length} total requests`;

        root.replaceChildren();

        if (!approvals.length) {
            root.textContent =
                "No approval requests yet.";
            return;
        }

        approvals.forEach(approval => {
            const card =
                document.createElement("article");

            card.className = "approval-card";

            const header =
                document.createElement("div");

            header.className = "approval-header";

            const title =
                document.createElement("strong");

            title.textContent = approval.question;

            const badge =
                document.createElement("span");

            badge.className =
                "approval-status " +
                approval.status;

            badge.textContent = approval.status;

            header.append(title, badge);
            card.append(header);

            const metadata =
                document.createElement("p");

            metadata.textContent =
                `Requested by ${approval.requester}` +
                " · " +
                new Date(
                    approval.created_at
                ).toLocaleString();

            card.append(metadata);

            if (approval.status === "pending") {
                const actions =
                    document.createElement("div");

                actions.className = "approval-actions";

                [
                    ["Approve", true],
                    ["Reject", false]
                ].forEach(([label, approve]) => {
                    const button =
                        document.createElement("button");

                    button.type = "button";
                    button.textContent = label;

                    button.className = approve
                        ? "approve-button"
                        : "reject-button";

                    button.onclick = async () => {
                        const action = approve
                            ? "approve"
                            : "reject";

                        const confirmed = confirm(
                            `Do you want to ${action} ` +
                            "this request? " +
                            "This only records a decision."
                        );

                        if (!confirmed) return;

                        button.disabled = true;

                        try {
                            const result = await api(
                                `/api/approvals/${
                                    encodeURIComponent(
                                        approval.id
                                    )
                                }/decision`,
                                {
                                    method: "POST",
                                    body: JSON.stringify({
                                        approve
                                    })
                                }
                            );

                            toast(
                                `Request ${result.status}. ` +
                                "No action executed."
                            );

                            await loadApprovals();

                        } catch (error) {
                            button.disabled = false;
                            toast(error.message);
                        }
                    };

                    actions.append(button);
                });

                card.append(actions);

            } else {
                const statusInfo =
                    document.createElement("p");

                statusInfo.textContent =
                    `Reviewed by ${
                        approval.reviewer || "unknown"
                    } · No action executed`;

                card.append(statusInfo);
            }

            root.append(card);
        });

    } catch (error) {
        root.textContent =
            "Unable to load approvals: " +
            error.message;

        toast(error.message);
    }
}

// ============================================
// 14. RESTORE CONVERSATION HISTORY
// ============================================

async function restoreConversation() {
    try {
        const result = await api(
            `/api/conversations/${conversationId}/messages`
        );

        const messages = result.messages || [];

        if (!messages.length) {
            showHero(true);
            return;
        }

        $("messages").replaceChildren();

        showHero(false);

        messages.forEach(message => {
            addMessage(
                message.content,
                message.role === "user"
                    ? "user"
                    : "assistant"
            );
        });

    } catch (error) {
        console.warn(
            "Conversation history unavailable:",
            error.message
        );
    }
}

// ============================================
// 15. INITIALIZE FRONTEND
// ============================================

document.addEventListener("DOMContentLoaded", () => {
    view("chat");
    restoreConversation();

    console.log("Nexora AI frontend initialized");
});
