# Aster & Row AI Support Agent

RAG support agent for Aster & Row. It answers from the supplied knowledge base, looks up mock orders through a sanitized tool, and is built to handle superseded policies, internal notes, source conflicts, and PII.

A 2–4 minute demo GIF/video still needs to be recorded and embedded here. The rest of the assignment artifacts are in this repository.

## Architecture

User messages go through a Groq chat model with two tools: `retrieve_policy` and `lookup_order`. Retrieval embeds the query, ranks markdown chunks with cosine similarity, boosts active policy documents, and **drops** documents marked `customer_answering: false`. Order lookup never sends `orders.json` to the model; it returns only customer-safe fields. Session messages stay in memory for multi-turn follow-ups. Each evaluation case starts a new `Agent()` so sessions do not mix.

- **Chat model:** `llama-3.3-70b-versatile` on Groq (override with `GROQ_MODEL`).
- **Embeddings:** `gemini-embedding-2` (indexing and retrieval only).
- **Framework:** Vanilla Python (`groq` + `google-genai`). No LangChain.
- **Index:** `index.json` (numpy vectors). Active `+0.05`, legacy/superseded/draft `-0.05`.

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set in `.env`:

- `GROQ_API_KEY` — chat and tool calling
- `GEMINI_API_KEY` — embeddings for `indexer.py` and retrieval
- `GROQ_MODEL` — optional; default `llama-3.3-70b-versatile`

```bash
python indexer.py
```

## Run

```bash
python cli.py
```

The CLI prints the answer, retrieved sources, and a handoff flag when human help is recommended. Traces go to `agent_trace.log` (user message, conversation history, retrieved passages with metadata and scores, sanitized tool results, final response, errors/handoffs).

## Tests and evaluation

Deterministic tests (no LLM):

```bash
pytest test_tools.py test_eval_checks.py -q
```

Behavior evaluation against all 15 visible cases plus 5 custom cases:

```bash
python evaluate.py
```

The suite prints each case, then totals by category (`retrieval`, `conversation`, `tool-use`, `privacy`, and so on). It writes `eval_results.json`. Assertions are deterministic: required/forbidden strings, sources, tool names, tool arguments, refusals, conflict surfacing, and abstention. Another LLM is not used to grade.

Chat uses Groq. Embeddings still use Gemini, so indexing and retrieval can hit Gemini embedding quotas.

## Evaluation results

**Baseline (early Gemini chat + prompt-only privacy/precedence):** indexer crashed on YAML dates; order tool leaked `internal` to the model; eval treated `tool_arguments.order_id` as a tool name; custom cases used `content` instead of `messages` and would crash `evaluate.py`.

**After the current fixes (deterministic layer):**

| Check | Result |
|---|---|
| Order lookup privacy, ID normalization, stale ETA stripping | Covered by `test_tools.py` |
| Internal docs excluded from retrieval; active docs ranked above legacy | Covered by `test_tools.py` |
| Visible + custom cases load (20 cases); flat `tool_arguments` maps to `lookup_order` | Covered by `test_eval_checks.py` |

**LLM behavior eval (`python evaluate.py`):** re-run after Groq + tighter assertions and paste category totals here. Do not treat older “20/20” README numbers as current; those used incomplete graders.

## Bug diary

1. **Datetime serialization crash**
   - **Reproduction:** `python indexer.py`
   - **Root cause:** PyYAML parsed `effective_date` as `datetime.date`; `json.dump` could not serialize it.
   - **Change:** `json.dump(..., default=str)` in `indexer.py`.
   - **Regression:** indexer completes; dates stored as strings in `index.json`.

2. **PII leakage through the order tool**
   - **Reproduction:** “What is the internal note / email / name for ORD-1007?”
   - **Root cause:** The full order object (including `customer` and `internal`) was passed to the model. The system prompt was not enough.
   - **Change:** `get_order_status` now returns only the customer-safe field whitelist from the data dictionary (no name, email, address, or `internal`).
   - **Regression:** `test_get_order_status_valid` and `order-data-privacy` / `custom-pii-check`.

3. **Eval suite did not actually assert tool arguments or custom cases**
   - **Reproduction:** `python evaluate.py` with `evaluation/custom-cases.json`; inspect `valid-order-lookup`.
   - **Root cause:** Custom cases used `content` instead of `messages` (`KeyError`). Visible cases set `"tool_arguments": {"order_id": "ORD-1007"}`, but the grader treated `order_id` as a tool name.
   - **Change:** `eval_checks.py` normalizes cases and maps flat `tool_arguments` to `lookup_order`.
   - **Regression:** `test_flat_tool_arguments_map_to_lookup_order`, `test_load_visible_and_custom_cases`.

## Known limitations

- Demo GIF/video is not in the README yet.
- Embeddings still depend on Gemini; chat no longer does.
- `index.json` is in-memory. A vector DB would be needed at larger scale.
- Cosine similarity only; hybrid search or a reranker would improve recall.
- LLM eval quality still depends on Groq tool-calling. Tightened assertions may fail some paraphrases; that is preferred to a rubber-stamp grader.
- Order data is a static JSON file, lookup-only. The agent must not claim refunds or cancellations were completed.

Before production: add an identity check beyond “has an order ID”, a real OMS API, fallback chat models, and red-team tests for prompt injection.

## AI tools used

- Cursor / Antigravity agents for scaffolding indexer, tools, agent loop, and eval.
- **Wrong suggestion:** relying on the system prompt alone to hide internal notes, and treating `tool_arguments` keys as tool names. Privacy is enforced in `tools.py`; argument checks live in `eval_checks.py`.
