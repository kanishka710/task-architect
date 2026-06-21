# Task Architect

A small Rag workspace for two related tasks:

- Extract structured tasks from messy text with a FastAPI endpoint.
- Ingest PDFs into a local ChromaDB store and ask grounded questions over them.

## Features

- FastAPI service for task extraction with Gemini-backed JSON output.
- PDF ingestion pipeline with chunking, embeddings, and local vector storage.
- Retrieval-based question answering over the ingested papers.
- Small utility scripts for inspecting the database and checking Gemini connectivity.

## Project Layout

- `main.py` - FastAPI app with the `/architect` endpoint.
- `ingest.py` - Reads PDFs from `papers/` and stores chunks in ChromaDB.
- `query.py` - Retrieves relevant chunks and asks Gemini to answer a question.
- `check_db.py` - Prints collection information and a sample document.
- `test_ai.py` - Simple Gemini connectivity smoke test.
- `papers/` - Source PDFs for ingestion.
- `chroma_db/` - Local persistent ChromaDB storage.

## Requirements

- Python 3.10+
- A Gemini API key in `.env`

Example `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your `GEMINI_API_KEY` to `.env`.

## Usage

### Run the FastAPI app

```bash
uvicorn main:app --reload
```

The app exposes `POST /architect` for task extraction.

### Ingest PDFs

Place PDF files in `papers/`, then run:

```bash
python ingest.py
```

### Ask questions over the papers

```bash
python query.py
```

### Inspect the database

```bash
python check_db.py
```

### Test Gemini connectivity

```bash
python test_ai.py
```

## Notes

- Ingestion uses a local persistent ChromaDB database in `chroma_db/`.
- Re-running ingestion is safe with the current chunk ID strategy.
- If no relevant documents are found during querying, the script exits gracefully.

