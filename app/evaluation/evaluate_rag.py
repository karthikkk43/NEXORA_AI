
"""
Automated Nexora AI RAGAS Evaluation

Runs actual LangGraph workflow.
Collects real Pinecone retrieval contexts.
Evaluates generated answers using RAGAS.

Evaluation scope: Private KB questions.
"""

import json
import time
from pathlib import Path

import pandas as pd

from langchain_aws import ChatBedrockConverse

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
)

from app.graph import graph
from app.demo_context import Principal
from app.config import settings


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

DATASET_PATH = BASE / "test_dataset.json"
RESULTS_PATH = BASE / "ragas_results.csv"
RUNS_PATH = BASE / "pipeline_results.json"


# ==========================================
# 2. LOAD TEST DATASET
# ==========================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
) as file:
    test_cases = json.load(file)

if not test_cases:
    raise ValueError("Evaluation dataset is empty")


# ==========================================
# 3. EVALUATION PRINCIPAL
# ==========================================

# Use the existing Nexora demo workspace.
# Replace with a dedicated least-privilege
# evaluation identity before production use.

evaluation_principal = Principal(
    subject="demo-operator",
    tenant="demo-workspace",
    groups=("workspace",),
    admin=False,
)


# ==========================================
# 4. RUN REAL NEXORA WORKFLOW
# ==========================================

records = []

print("\nStarting Nexora Automated RAG Evaluation")
print("=" * 55)

for index, case in enumerate(test_cases, start=1):

    question = case["user_input"]
    reference = case["reference"]

    print(f"\nTest {index}/{len(test_cases)}")
    print("Question:", question)

    start = time.perf_counter()

    try:
        result = graph.invoke({
            "principal": evaluation_principal,
            "question": question,
            "history": [],
        })

        latency = round(
            time.perf_counter() - start,
            3,
        )

        answer = result.get("answer", "")
        intent = result.get("intent", "")
        status = result.get("status", "")

        # Collect actual retrieved Pinecone chunks.
        contexts = []

        for passage in result.get("passages", []):

            metadata = passage.get("metadata") or {}

            content = str(
                metadata.get("text", "")
            ).strip()

            if content:
                contexts.append(content)

        record = {
            "user_input": question,
            "response": answer,
            "reference": reference,
            "retrieved_contexts": contexts,
            "intent": intent,
            "status": status,
            "latency_seconds": latency,
            "kb_grade": result.get("kb_grade", {}),
            "rewrite_count": result.get(
                "rewrite_count", 0
            ),
        }

        records.append(record)

        print("Route:", intent)
        print("Status:", status)
        print("Retrieved chunks:", len(contexts))
        print("Latency:", latency, "seconds")

    except Exception as exc:

        print("Workflow failed:", type(exc).__name__)

        records.append({
            "user_input": question,
            "reference": reference,
            "response": "",
            "retrieved_contexts": [],
            "intent": "error",
            "status": "error",
            "latency_seconds": round(
                time.perf_counter() - start, 3
            ),
            "error": type(exc).__name__,
        })


# ==========================================
# 5. SAVE PIPELINE RESULTS
# ==========================================

with open(
    RUNS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        records,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("\nPipeline results saved:", RUNS_PATH)


# ==========================================
# 6. SELECT VALID KB RUNS
# ==========================================

# RAGAS context metrics only apply when
# the workflow really retrieved KB documents.

valid_records = [
    record
    for record in records
    if (
        record["intent"] == "kb"
        and record["status"] == "answered"
        and bool(record["retrieved_contexts"])
        and bool(record["response"])
    )
]

failed_count = len(records) - len(valid_records)

print("\nValid KB runs:", len(valid_records))
print("Skipped/failed runs:", failed_count)

if not valid_records:
    raise RuntimeError(
        "No valid KB runs available for RAGAS. "
        "Check pipeline_results.json."
    )


# ==========================================
# 7. BUILD RAGAS DATASET
# ==========================================

ragas_rows = []

for record in valid_records:

    ragas_rows.append({
        "user_input": record["user_input"],
        "response": record["response"],
        "reference": record["reference"],
        "retrieved_contexts": record[
            "retrieved_contexts"
        ],
    })

dataset = EvaluationDataset.from_list(
    ragas_rows
)


# ==========================================
# 8. AMAZON BEDROCK EVALUATOR
# ==========================================

s = settings()

bedrock_llm = ChatBedrockConverse(
    model=s.bedrock_chat_model,
    region_name=s.aws_region,
    temperature=0,
    max_tokens=1500,
)

evaluator_llm = LangchainLLMWrapper(
    bedrock_llm
)


# ==========================================
# 9. RAGAS METRICS
# ==========================================

metrics = [
    Faithfulness(),
    LLMContextRecall(),
]

print("\nRunning RAGAS evaluation...")

results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=evaluator_llm,
    raise_exceptions=False,
)


# ==========================================
# 10. SAVE EVALUATION SCORES
# ==========================================

df = results.to_pandas()

df["intent"] = [
    r["intent"] for r in valid_records
]

df["latency_seconds"] = [
    r["latency_seconds"]
    for r in valid_records
]

df["rewrite_count"] = [
    r["rewrite_count"]
    for r in valid_records
]

df.to_csv(
    RESULTS_PATH,
    index=False,
)

print("\n" + "=" * 55)
print("NEXORA AI - EVALUATION RESULTS")
print("=" * 55)

score_columns = [
    "faithfulness",
    "context_recall",
]

for column in score_columns:

    if column in df.columns:

        score = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        print(
            f"{column}: "
            f"{score.mean():.4f}"
        )

        print(
            f"Scored cases: "
            f"{score.count()}/{len(df)}"
        )

print(
    "\nAverage pipeline latency:",
    round(df["latency_seconds"].mean(), 3),
    "seconds",
)

print("\nResults saved:", RESULTS_PATH)
print("\nEvaluation completed.")

# ==========================================
# LATENCY PERFORMANCE METRICS
# ==========================================

print("\nNEXORA AI - LATENCY METRICS")

print(
    "Average:",
    round(df["latency_seconds"].mean(), 3),
    "seconds"
)

print(
    "P50:",
    round(df["latency_seconds"].quantile(0.50), 3),
    "seconds"
)

print(
    "P95:",
    round(df["latency_seconds"].quantile(0.95), 3),
    "seconds"
)

print(
    "P99:",
    round(df["latency_seconds"].quantile(0.99), 3),
    "seconds"
)
