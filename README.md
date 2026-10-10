# Mini RAG Q&A System — Starter Project

A working Retrieval-Augmented Generation pipeline: load documents → chunk →
embed → store → retrieve → generate a grounded answer → evaluate quality.

This has already been tested end-to-end (ingestion + retrieval) on the
21 sample documents included in `data/sample_docs/` — a mock internal
knowledge base spanning HR, IT, Engineering, Legal, and Marketing
policies. Retrieval was verified to correctly find the right source
document across all 21 files for 8 different test questions.

---

## Project Structure

```
rag_project/
├── data/sample_docs/       # Put your .txt or .pdf documents here
├── chroma_db/               # Auto-created vector database (after running ingest.py)
├── src/
│   ├── chunking.py          # Step 1-2: load + chunk documents
│   ├── embeddings.py        # Step 3: embedding backends (TF-IDF / local model / OpenAI)
│   ├── ingest.py             # Steps 1-4: full ingestion pipeline
│   ├── retrieve.py           # Step 5: retrieval
│   ├── generate.py           # Step 6: LLM answer generation (needs API key)
│   ├── evaluate.py           # Step 7: evaluation framework (needs API key)
│   └── app.py                 # Streamlit demo UI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

To use Step 6 (answer generation) and Step 7 (evaluation), set your
Anthropic API key:

```bash
export GROQ_API_KEY="your-key-here"
```

Steps 1-5 (ingestion + retrieval) work with **no API key required** —
they use a free, local TF-IDF based embedder by default.

## How to Run

### 1. Add your documents
Drop `.txt` or `.pdf` files into `data/sample_docs/` — 21 sample policy
documents (HR, IT, Engineering, Legal, Marketing) are already included
so you can test immediately. Swap these for real documents relevant to
your internship/company before your final demo.

### 2. Ingest (build the vector database)
```bash
cd src
python3 ingest.py
```
This loads documents, chunks them, generates embeddings, and stores
everything in a local Chroma database (`chroma_db/`).

### 3. Test retrieval only (no API key needed)
```bash
python3 retrieve.py
```
Edit the `test_question` variable in the file to try your own questions.

### 4. Generate a grounded answer (requires API key)
```bash
python3 generate.py
```

### 5. Run the full evaluation (requires API key)
```bash
python3 evaluate.py
```
This runs a batch of test questions, scores retrieval relevance and
answer faithfulness, and saves results to `evaluation_results.csv`.

### 6. Launch the demo UI
```bash
streamlit run app.py
```

---

## How to Improve This for Your Final Submission

1. **Swap the embedder**: the default TF-IDF embedder works with zero
   setup but isn't truly semantic. Once you have full internet access,
   switch `EMBEDDING_BACKEND` in `ingest.py` to `"sentence_transformer"`
   (free, local, much better) or `"openai"` (best quality, paid).
   No other code changes needed — `embeddings.py` is built so every
   backend is a drop-in replacement.

2. **Tune chunk size/overlap**: try different values in `chunking.py`
   and see how retrieval quality changes.

3. **Expand your evaluation set**: add 15-20 real test questions,
   including a few "trap" questions whose answers are NOT in your
   documents, to prove the system says "I don't know" instead of
   hallucinating.

4. **Add source citations in the UI**: the retrieved chunks already
   carry the source filename — surface it next to the generated answer.

5. **Log before/after results**: once you tune chunking/prompts, re-run
   `evaluate.py` and keep both CSVs — this before/after comparison is
   the strongest part to show your mentor.
