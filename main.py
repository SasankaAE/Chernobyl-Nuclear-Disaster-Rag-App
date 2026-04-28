import streamlit as st
import requests
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# PAGE CONFIG
st.set_page_config(page_title="Chernobyl Chat RAG",page_icon="☢️", layout="wide")
st.title("☢️ Chernobyl Chat RAG App")


# SESSION STATE (CHAT MEMORY)
if "messages" not in st.session_state:
    st.session_state.messages = []


# FETCH WIKIPEDIA
@st.cache_data
def fetch_wikipedia():
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "titles": "Chernobyl disaster",
        "prop": "extracts",
        "explaintext": True
    }

    headers = {
        "User-Agent": "Chernobyl-RAG-App/1.0"
    }

    res = requests.get(url, params=params, headers=headers)
    data = res.json()

    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


# CHUNKING
def chunk_text(text, size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks


# RAG SETUP
@st.cache_resource
def setup_rag():
    text = fetch_wikipedia()
    chunks = chunk_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return model, index, chunks

model, index, chunks = setup_rag()


# RETRIEVAL
def retrieve(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [chunks[i] for i in I[0]]


# OPENROUTER LLM
def ask_llm(query, context):
    api_key = st.secrets["OPENROUTER_API_KEY"]

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the given context.
If not in context, say "I don't know".

Context:
{context}

User: {query}
"""

    body = {
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    res = requests.post(url, headers=headers, json=body)

    if res.status_code != 200:
        return "Error: " + res.text

    return res.json()["choices"][0]["message"]["content"]


# DISPLAY CHAT HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# CHAT INPUT
user_input = st.chat_input("Ask about Chernobyl...")

if user_input:

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # RAG retrieval
    context_chunks = retrieve(user_input)
    context = "\n\n".join(context_chunks)

    # LLM response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_llm(user_input, context)
            st.write(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})