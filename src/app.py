"""
Simple Streamlit demo UI.

Run with:
    streamlit run src/app.py

Requires GROQ_API_KEY (free) to be set for answer
generation. Retrieval alone works without any API key.
"""

import streamlit as st
from retrieve import retrieve
from llm import has_api_key

st.set_page_config(page_title="RAG Q&A Demo", page_icon="📄")
st.title("📄 Document Q&A (RAG Demo)")
st.caption("Ask a question — the answer is grounded in the ingested documents, not general knowledge.")

question = st.text_input("Your question:", placeholder="e.g. How many days of annual leave do employees get?")
top_k = st.slider("Number of chunks to retrieve", 1, 5, 3)

if st.button("Ask") and question:
    with st.spinner("Retrieving relevant chunks..."):
        chunks = retrieve(question, top_k=top_k)

    has_key = has_api_key()

    if has_key:
        from generate import generate_answer
        with st.spinner("Generating grounded answer..."):
            result = generate_answer(question, top_k=top_k)
        st.subheader("Answer")
        st.write(result["answer"])
    else:
        st.warning(
            "No LLM API key set — showing retrieved context only. "
            "Set the key to see the generated answer step too."
        )

    st.subheader("Retrieved Source Chunks")
    for i, c in enumerate(chunks):
        with st.expander(f"{i+1}. {c['source']} (chunk {c['chunk_index']}, distance={c['distance']:.3f})"):
            st.write(c["text"])
