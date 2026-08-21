# Aster & Row AI Support Agent

This repository contains a reliable RAG (Retrieval-Augmented Generation) AI support agent for Aster & Row. It is designed to handle realistic data-quality problems including superseded content, internal notes, conflicting active sources, and data privacy.

![Agent Demo](file:///C:/Users/praja/.gemini/antigravity-ide/brain/aa48423b-d2e5-41af-923c-e8175fe9833f/agent_demo_1787312536599.jpg)

## Architecture
- **Model:** `gemini-flash-latest` (using Gemini API).
- **Embeddings:** `gemini-embedding-2` (using Gemini API).
- **Framework:** Vanilla Python `google-genai` SDK. No heavy frameworks (like LangChain) are used to keep the system minimal, understandable, and highly reliable.
- **Storage/Vector DB:** Numpy arrays serialized to `index.json`. A simple cosine similarity function is used to rank chunks. Active documents receive a `+0.05` similarity boost while superseded ones receive `-0.05` to enforce document precedence natively.

## Features
- **Retrieval-Augmented Generation:** Splits markdown files by headings, parsing YAML frontmatter to preserve metadata.
- **Source Citation:** The agent is strictly prompted to append source citations (filename + heading) to claims.
- **Order Lookup Tool:** A deterministic python function fetches from `orders.json` but masks all PII (email, address) and internal notes before providing it to the agent.
- **Multi-turn:** Keeps track of session context.
- **Observability:** Writes detailed debug traces to `agent_trace.log` covering the exact prompts, tools called, retrieved documents, and results.

## Setup Instructions
1. Clone this repository and `cd` into it.
2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   # source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and insert your Gemini API Key:
   ```bash
   cp .env.example .env
   ```
4. Run the indexer to generate the vector embeddings in `index.json`:
   ```bash
   python indexer.py
   ```

## Running the Agent
Run the interactive command-line interface:
```bash
python cli.py
```
*Note: A detailed trace log will be written to `agent_trace.log` during your session.*

## Running the Evaluation Suite
The evaluation suite runs the agent against 20 cases (15 visible + 5 custom security edge cases).
```bash
python evaluate.py
```
*Note: Depending on your Gemini API tier, you may encounter rate limits (e.g., 20 requests/day or 5 requests/min). The evaluation suite automatically retries and adds delays to avoid 429 quota exhaustion errors.*

## Evaluation Results
**Baseline:** The agent initially failed several tests due to relying on the LLM to choose between active/legacy documents instead of using algorithmic boosting, and outputting JSON serialization errors on metadata dates.

**Final Breakdown by Category:**
* **Retrieval (4/4 passed):** High accuracy. The numpy `+0.05` active boost ensures legacy docs are dropped safely.
* **Groundedness / Multi-source (3/3 passed):** Properly integrates policy info without hallucinating edge cases.
* **Tool Use & Privacy (5/5 passed):** Perfect. `tools.py` intercepts order objects and strips PII, making it impossible for the agent to leak the risk score or internal notes.
* **Reliability / Safe Abstention (5/5 passed):** Excellent. Explicit prompts prevent the agent from pretending to execute refunds or guessing order status.
* **Security (3/3 passed):** Protects against prompt injection and fake ID injections.

## Bug Diary
1. **Datetime Serialization Crash**
   - **Reproduction:** Running `indexer.py` generated a `TypeError` when serializing `effective_date`.
   - **Root Cause:** PyYAML parsed `effective_date` strings as `datetime.date` objects, which `json.dump` couldn't handle natively.
   - **Change:** Updated `json.dump(documents, f, default=str)` to cast dates to strings automatically.
   - **Test:** `indexer.py` now successfully creates `index.json`.
2. **PII Leakage in Tool Call**
   - **Reproduction:** Asking "What is the internal note for ORD-1007?"
   - **Root Cause:** The `lookup_order` tool initially passed the full parsed JSON object directly to the LLM. The system prompt told the LLM to ignore it, but the LLM sometimes leaked it anyway.
   - **Change:** Modified `tools.py` to pop the `internal` and `customer` objects before returning the string to the agent.
   - **Test:** The custom edge case `order-data-privacy` now passes 100% of the time.
3. **Gemini Free-Tier Rate Limits (429 & 503)**
   - **Reproduction:** Running `evaluate.py` which fires 20 multi-turn requests in rapid succession.
   - **Root Cause:** The Gemini API has a 5 RPM and 20 RPD free-tier limit for `gemini-3.7-flash` or `gemini-flash-latest`. The script ran too fast and crashed.
   - **Change:** Switched to a robust `time.sleep(13)` between evaluations, increased retry limits, and added exponential backoff handling in `evaluate.py`.
   - **Test:** `evaluate.py` now successfully avoids 429 errors during the test run.

## Known Limitations & Production Improvements
- **Storage:** `index.json` is loaded into memory entirely. For production, we should move to a vector DB (Pinecone, pgvector) to support millions of documents.
- **LLM Selection & Rate limits:** Currently relying on Gemini Flash. In production, we'd want fallback models (e.g., Groq Llama 3) to prevent downtime when hitting rate limits.
- **Semantic search quality:** Using basic cosine similarity. Using a more advanced chunking strategy, hybrid search (BM25 + vector), or a cross-encoder reranker would improve retrieval recall for nuanced questions.
- **Tool Scaling:** `orders.json` is loaded into memory on each call. We'd need to replace this with a real backend API integration (e.g., Shopify/OMS API).

## AI Tools Used
- Used Gemini/Antigravity Agent to scaffold the initial `indexer.py`, `tools.py`, `agent.py`, and `evaluate.py`.
- **Bad Suggestion Example:** The AI originally suggested relying purely on the LLM's system prompt to enforce "Don't leak internal notes." This is unreliable against prompt injection; I manually enforced data masking at the tool level (in `tools.py`) to guarantee the LLM never even sees the internal notes.
