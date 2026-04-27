# Chernobyl Chat RAG

Streamlit app that answers questions about the Chernobyl disaster using a retrieval-augmented generation workflow. The app pulls the Chernobyl disaster article from Wikipedia, chunks the text, builds FAISS embeddings with `sentence-transformers`, and sends the retrieved context to an OpenRouter chat model.

## Features

- Chat-style Streamlit interface
- Wikipedia-based knowledge source
- Semantic retrieval with FAISS
- Chat history stored in session state

## Requirements

- Python 3.12 or newer
- An `OPENROUTER_API_KEY`

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies used by the app:

```bash
pip install streamlit requests faiss-cpu numpy sentence-transformers python-dotenv
```

3. Create a `.env` file in the project root and add your OpenRouter key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

## Run the app

```bash
streamlit run main.py
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

## How It Works

1. The app fetches the Wikipedia extract for the Chernobyl disaster.
2. The text is split into overlapping chunks.
3. Each chunk is embedded with `all-MiniLM-L6-v2`.
4. FAISS retrieves the most relevant chunks for each user question.
5. The retrieved context is sent to OpenRouter, which generates the answer.

## Notes

- If the OpenRouter request fails, the response is shown in the chat window as an error message.
- Answers are constrained to the retrieved context, so the model may return "I don't know" when the article does not contain enough information.
