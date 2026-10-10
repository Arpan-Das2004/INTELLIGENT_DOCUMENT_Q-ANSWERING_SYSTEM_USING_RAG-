"""
Step 7: Evaluate answer quality.

Two metrics, kept intentionally simple so you can explain and defend
them easily to your mentor:

1. Retrieval relevance - did we retrieve chunks that are actually
   related to the question? (keyword-overlap heuristic, fast/free)

2. Faithfulness - is the generated answer actually supported by the
   retrieved context? (LLM-as-judge: a second call asks the model to
   check the answer against the context)

Run this AFTER you have a set of test questions with generated answers.
"""

import csv
import os
import re
import sys
import time
from llm import ask_llm, has_api_key, missing_key_message, SECONDS_BETWEEN_CALLS, DailyQuotaError

FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is fully \
supported by the given context.

Context:
{context}

Question: {question}

Generated Answer: {answer}

Ignore any mention of source file names or "Based on <file>" phrases in the answer; judge only the facts.
Is the generated answer fully supported by the context above? Respond with ONLY one word:
"YES" if every claim in the answer is backed by the context (or if the answer correctly \
says the information isn't available),
"NO" if the answer includes information not found in the context (hallucination),
"PARTIAL" if some but not all of the answer is supported.
"""


def keyword_relevance_score(question: str, retrieved_chunks: list[dict]) -> float:
    """Simple, fast, free heuristic: what fraction of the question's
    meaningful words appear somewhere in the retrieved chunks?
    Useful as a quick sanity check before running the more expensive
    LLM-as-judge faithfulness check."""
    stopwords = {"the", "a", "an", "is", "are", "do", "does", "how", "what",
                 "when", "where", "who", "why", "many", "much", "of", "for",
                 "to", "in", "on", "and", "or", "i", "can"}
    q_words = set(re.findall(r"[a-zA-Z]+", question.lower())) - stopwords
    if not q_words:
        return 0.0

    combined_text = " ".join(c["text"] for c in retrieved_chunks).lower()
    found = sum(1 for w in q_words if w in combined_text)
    return round(found / len(q_words), 2)


def faithfulness_judge(question: str, answer: str, retrieved_chunks: list[dict]) -> str:
    """Uses the LLM itself as a judge to check if the answer is grounded
    in the retrieved context. Returns 'YES', 'NO', or 'PARTIAL'."""
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks)
    prompt = FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)

    verdict = ask_llm(prompt, max_tokens=20).strip().upper()
    # Be forgiving if the model adds punctuation or a short phrase
    for word in ("PARTIAL", "YES", "NO"):
        if verdict.startswith(word):
            return word
    return "UNKNOWN"


def print_summary(rows: list[dict]):
    """Short scoreboard to quote in your presentation."""
    if not rows:
        return
    total = len(rows)
    counts = {v: sum(1 for r in rows if r["faithfulness"] == v) for v in ("YES", "PARTIAL", "NO", "UNKNOWN")}
    avg_rel = sum(float(r["retrieval_relevance"]) for r in rows) / total
    print("\n===== SUMMARY =====")
    print(f"Questions evaluated : {total}")
    print(f"Faithful (YES)      : {counts['YES']}  ({counts['YES'] / total:.0%})")
    print(f"Partial             : {counts['PARTIAL']}")
    print(f"Not faithful (NO)   : {counts['NO']}")
    print(f"Judge unclear       : {counts['UNKNOWN']}")
    print(f"Avg retrieval score : {avg_rel:.2f}")


def run_evaluation(test_questions: list[str], output_csv: str = "evaluation_results.csv"):
    """Runs generate + evaluate over a list of test questions and writes
    results to a CSV so you can show before/after comparisons to your mentor.

    Progress is saved after EVERY question, and questions already in the CSV
    are skipped - so if the free daily quota runs out, just run it again
    later (or tomorrow) and it continues where it stopped."""
    from generate import generate_answer

    fieldnames = ["question", "answer", "sources", "retrieval_relevance", "faithfulness"]

    def save(rows):
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    rows = []
    if os.path.exists(output_csv):
        with open(output_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {r["question"] for r in rows}
    if done:
        print(f"Resuming: {len(done)} question(s) already in {output_csv}, skipping them.\n")

    try:
        for q in test_questions:
            if q in done:
                continue
            result = generate_answer(q)
            time.sleep(SECONDS_BETWEEN_CALLS)  # stay under free-tier requests/minute
            relevance = keyword_relevance_score(q, result["retrieved_chunks"])
            faithfulness = faithfulness_judge(q, result["answer"], result["retrieved_chunks"])
            rows.append({
                "question": q,
                "answer": result["answer"],
                "sources": ", ".join(c["source"] for c in result["retrieved_chunks"]),
                "retrieval_relevance": relevance,
                "faithfulness": faithfulness,
            })
            save(rows)
            print(f"[{faithfulness}] (relevance={relevance}) {q}")
            time.sleep(SECONDS_BETWEEN_CALLS)
    except DailyQuotaError as err:
        print(f"\nSTOPPED: {err}")
        print("Your progress so far is saved - run this script again later to continue.")

    print(f"\n{len(rows)} result(s) saved in {output_csv}")
    print_summary(rows)
    return rows


if __name__ == "__main__":
    if not has_api_key():
        print(missing_key_message())
    else:
        # Optional: choose the CSV name, e.g.  python evaluate.py results_tfidf.csv
        out_file = sys.argv[1] if len(sys.argv) > 1 else "evaluation_results.csv"

        test_questions = [
            # --- Real questions (answers exist in the documents) ---
            "How many days of annual leave do employees get?",
            "How long is maternity leave?",
            "What is the notice period for resignation?",
            "How many days of sick leave are allowed per year?",
            "What are the password requirements?",
            "Is two-factor authentication mandatory?",
            "How soon must a lost or stolen device be reported?",
            "How much is the home office setup stipend for remote employees?",
            "How many approvals are needed for a pull request on payment systems?",
            "What is the first response time for a priority 1 support ticket?",
            "Who must approve expenses above $1,000?",
            "What is the primary brand color?",
            "How many weeks of extended parental leave can be requested?",
            "What is the limit for accepting gifts from vendors?",
            # --- Trap questions (answers are NOT in the documents) ---
            "What is the company's policy on pet insurance?",
            "What is the company's policy on remote work in Antarctica?",
            "What is the CEO's salary?",
        ]
        run_evaluation(test_questions, output_csv=out_file)