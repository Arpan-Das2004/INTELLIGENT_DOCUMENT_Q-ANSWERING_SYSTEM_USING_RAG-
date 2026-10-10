"""
Step 6: Combine the question + retrieved chunks into a prompt, and call
the LLM to generate a grounded answer.

Uses llm.py, which calls the FREE Groq API.
Set your key in PowerShell before running:
    $env:GROQ_API_KEY="your-key-here"
"""

from retrieve import retrieve
from llm import ask_llm, has_api_key, missing_key_message

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
provided context. Follow these rules strictly:

1. Answer using only information found in the context below.
2. If the answer is not present in the context, say clearly: \
"I don't have enough information in the provided documents to answer this."
3. Do not use outside knowledge, even if you know the answer.
4. Keep answers concise and directly address the question.
5. When possible, mention which source document the answer came from.
"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        context_blocks.append(
            f"[Source: {chunk['source']}, chunk {chunk['chunk_index']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""Context:
{context}

Question: {question}

Answer the question using only the context above."""


def generate_answer(question: str, top_k: int = 3) -> dict:
    retrieved_chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, retrieved_chunks)

    answer_text = ask_llm(prompt, system=SYSTEM_PROMPT, max_tokens=500)

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_chunks": retrieved_chunks,
    }


if __name__ == "__main__":
    if not has_api_key():
        print(missing_key_message())
    else:
        'test_question = "How many days of annual leave do employees get?"'
        test_question = "What is the company's policy on pet insurance?"
        result = generate_answer(test_question)
        print(f"Q: {result['question']}\n")
        print(f"A: {result['answer']}\n")
        print("Sources used:", [c["source"] for c in result["retrieved_chunks"]])
