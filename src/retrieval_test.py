"""
Objective retrieval test - NO API key, NO LLM needed.

Asks reworded questions (that avoid the documents' own wording) and checks
whether the correct document appears in the top results. Run it once with
the TF-IDF embedder and once with the sentence-transformer embedder
(re-run ingest.py between them) to get a real before/after comparison.

    python retrieval_test.py
"""

from retrieve import retrieve

# (question, set of acceptable source files)
TESTS = [
    ("How much time off do I get each year?", {"company_leave_policy.txt"}),
    ("Can I work from my house?", {"remote_work_policy.txt"}),
    ("How long can I stay off sick?", {"company_leave_policy.txt"}),
    ("How often do I have to change my login password?", {"it_security_guidelines.txt"}),
    ("What should I do if someone tries to trick me with a fake email?", {"it_security_guidelines.txt"}),
    ("Are we allowed to accept presents from suppliers?", {"code_of_conduct.txt"}),
    ("How do I get a standing desk?", {"health_and_safety_policy.txt"}),
    ("What fonts should be used in our marketing?", {"brand_guidelines.txt"}),
    ("How long do we keep customer information?", {"data_privacy_policy.txt"}),
    ("Do new hires get a mentor?", {"onboarding_guide.txt"}),
    ("Which open source licenses are safe to use?", {"software_licensing_policy.txt"}),
    ("How quickly must a teammate look at my code change?", {"code_review_guidelines.txt"}),
    ("What is the deadline to tell the company I am leaving?",
     {"company_leave_policy.txt", "exit_offboarding_process.txt"}),
    ("How fast must support reply when the whole system is down?",
     {"customer_support_sla.txt", "incident_response_runbook.txt"}),
    ("Who has to sign off before I spend company money on a trip?",
     {"expense_reimbursement_policy.txt", "business_travel_policy.txt"}),
    ("What happens to my laptop when I quit?",
     {"equipment_and_asset_policy.txt", "exit_offboarding_process.txt"}),
]


def main():
    top1 = top3 = 0
    print(f"{'#':>2}  {'Top1':<5}{'Top3':<5} Question  ->  top result")
    for i, (question, expected) in enumerate(TESTS, 1):
        results = retrieve(question, top_k=3)
        sources = [r["source"] for r in results]
        hit1 = sources[0] in expected
        hit3 = any(s in expected for s in sources)
        top1 += hit1
        top3 += hit3
        print(f"{i:>2}  {'YES' if hit1 else 'no':<5}{'YES' if hit3 else 'no':<5} {question}  ->  {sources[0]}")
    n = len(TESTS)
    print("\n===== RETRIEVAL SCORE =====")
    print(f"Correct document ranked #1     : {top1}/{n}  ({top1 / n:.0%})")
    print(f"Correct document in top 3      : {top3}/{n}  ({top3 / n:.0%})")


if __name__ == "__main__":
    main()
