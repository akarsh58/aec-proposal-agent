# Structura — AEC Proposal Intelligence Agent

A working local MVP for PDF RFP analysis, approved-experience retrieval, citation-grounded response drafting, and compliance evaluation.

## Features

- Upload a searchable PDF (up to 25 MB) and extract requirement statements with page references.
- Classify requirements into technical, experience, submission, commercial, and compliance categories.
- Retrieve the most relevant approved project records using a deterministic local embedding index.
- Draft with OpenRouter's free-model router and enforce evidence IDs such as `[EXP-101]`.
- Continue in demo mode without an API key, using a conservative grounded fallback draft.
- Review missing drafts, evidence gaps, and citation risks in a compliance matrix.
- Export the complete response as Markdown.

## Run locally

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenRouter API key to .env
set -a; source .env; set +a
uvicorn app:app --reload
```

Open <http://localhost:8000>. On Windows PowerShell, set the variables with `$env:OPENROUTER_API_KEY="..."` before starting Uvicorn.

## Demo flow

1. Click **load the demo pursuit**, or upload a searchable PDF.
2. Select a requirement and open **Response studio**.
3. Click **Generate** to retrieve relevant experience and draft a cited response.
4. Open **Compliance** to inspect missing content and evidence risks.
5. Click **Export response** to download Markdown.

## Data and safety

- `data/experience.json` is the approved evidence library. Replace the demo records with validated firm projects.
- The LLM receives extracted requirement text and retrieved evidence only; it is explicitly instructed not to invent claims.
- Free OpenRouter models have variable availability and rate limits, so this MVP includes deterministic fallback behavior.
- Uploaded files stay in the local `uploads/` folder. Add authentication, encryption, retention controls, malware scanning, and a production database before handling confidential client documents.

## Architecture

Browser UI → FastAPI → PyMuPDF extraction → requirement classifier → local hashed-vector retrieval → OpenRouter `/chat/completions` → compliance checks.

The local vector implementation is dependency-light and transparent for a demo. For larger libraries, replace `retrieve()` with pgvector, Qdrant, or another production vector store, and add reranking and tenant isolation.
