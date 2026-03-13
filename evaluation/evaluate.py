"""
evaluation.py — Phase 5a: RAGAS Evaluation

RAGAS measures 4 key qualities of a RAG system:
  1. Faithfulness      — Is the answer grounded in the retrieved context?
                         (catches hallucinations)
  2. Answer Relevancy  — Does the answer actually address the question?
  3. Context Precision — Are the retrieved chunks relevant to the question?
  4. Context Recall    — Were all the important chunks retrieved?

Run:
  python evaluation/evaluate.py

Output:
  evaluation/results.json   ← scores for each question
  evaluation/summary.csv    ← mean scores (attach to GitHub README)

Interview talking point:
  - "I measured my RAG system using RAGAS and achieved 0.87 faithfulness"
  - Shows you know how to validate AI outputs beyond just eyeballing them
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from loguru import logger

from src.config import validate_config
from src.retrieval import vector_search, rerank, build_context, generate_answer


# ── Test questions (replace with questions about YOUR document) ───
# Format: question + ground_truth (the correct answer you know)
TEST_QUESTIONS = [
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "Replace this with the actual correct answer from your document.",
    },
    {
        "question": "What are the key findings or conclusions?",
        "ground_truth": "Replace this with the actual correct answer from your document.",
    },
    {
        "question": "Who is the author or organization behind this document?",
        "ground_truth": "Replace this with the actual correct answer from your document.",
    },
    {
        "question": "What methodology or approach is described?",
        "ground_truth": "Replace this with the actual correct answer from your document.",
    },
    {
        "question": "What are the limitations mentioned?",
        "ground_truth": "Replace this with the actual correct answer from your document.",
    },
]


def run_rag_pipeline(question: str) -> dict:
    """Run full RAG pipeline and return answer + context for RAGAS."""
    # Retrieve
    candidates = vector_search(question)
    top_docs = rerank(question, candidates)
    context, sources = build_context(top_docs)

    # Generate
    answer = generate_answer(question, context)

    # Return in RAGAS format
    return {
        "question": question,
        "answer": answer,
        "contexts": [doc.page_content for doc in top_docs],  # list of context strings
    }


def evaluate_rag(questions: list = None) -> dict:
    """
    Run RAGAS evaluation on the test questions.
    Returns a dict with per-question scores and overall averages.
    """
    validate_config()

    if questions is None:
        questions = TEST_QUESTIONS

    logger.info(f"Running RAGAS evaluation on {len(questions)} questions...")

    # Run RAG pipeline for each question
    results = []
    for i, q in enumerate(questions):
        logger.info(f"[{i+1}/{len(questions)}] {q['question'][:60]}...")
        try:
            result = run_rag_pipeline(q["question"])
            result["ground_truth"] = q["ground_truth"]
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on question {i+1}: {e}")

    if not results:
        raise RuntimeError("All questions failed. Check that documents are ingested.")

    # Convert to HuggingFace Dataset (RAGAS requirement)
    dataset = Dataset.from_list(results)

    # Run RAGAS metrics
    logger.info("Computing RAGAS metrics...")
    scores = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,        # 0-1: higher = less hallucination
            answer_relevancy,    # 0-1: higher = more on-topic answers
            context_precision,   # 0-1: higher = retrieved chunks are relevant
            context_recall,      # 0-1: higher = nothing important was missed
        ],
    )

    # Build output
    scores_df = scores.to_pandas()
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "mean_scores": {
            "faithfulness":       round(float(scores_df["faithfulness"].mean()),       3),
            "answer_relevancy":   round(float(scores_df["answer_relevancy"].mean()),   3),
            "context_precision":  round(float(scores_df["context_precision"].mean()),  3),
            "context_recall":     round(float(scores_df["context_recall"].mean()),     3),
        },
        "per_question": scores_df.to_dict(orient="records"),
    }

    # Save results
    output_dir = Path(__file__).parent
    results_path = output_dir / "results.json"
    csv_path = output_dir / "summary.csv"

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    scores_df.to_csv(csv_path, index=False)

    logger.info("\n" + "="*50)
    logger.info("RAGAS EVALUATION RESULTS")
    logger.info("="*50)
    for metric, score in summary["mean_scores"].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        logger.info(f"  {metric:<22} {bar} {score:.3f}")
    logger.info("="*50)
    logger.info(f"Results saved to: {results_path}")

    return summary


if __name__ == "__main__":
    summary = evaluate_rag()
    print("\n📊 Mean RAGAS Scores:")
    for metric, score in summary["mean_scores"].items():
        print(f"  {metric:<22}: {score:.3f}")
