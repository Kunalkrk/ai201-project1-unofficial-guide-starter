# Generation & Interface — Implementation Notes

Documentation for [`generate.py`](../src/generate.py): the **Generation** stage and
**Gradio interface** of the RAG pipeline (see [`planning.md`](planning.md)).

```
chunks.json → [embed.py] → ChromaDB
                               ↓
                         retrieve(query)          top-5 chunks + metadata
                               ↓
                         build_prompt()           numbered context blocks + grounding rules
                               ↓
                         Groq API                 llama-3.3-70b-versatile, temp=0.0
                               ↓
                         format_sources()         deterministic — reads metadata, NOT LLM output
                               ↓
                         Answer + Sources         displayed in Gradio UI
```

Input: live ChromaDB index (`data/chroma_db/`) built by `src/embed.py`.
Output: grounded natural language answer + deterministic source attribution.

---

## How to run

```bash
# Launch Gradio UI (opens in browser automatically):
python src/generate.py

# One-shot CLI answer (no UI):
python src/generate.py --query "What is dbt used for?"
python src/generate.py --query "What skills should beginners learn first?" --top-k 5
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--query` | — | Run a single query from the CLI; skips the UI |
| `--top-k` | `5` | Number of chunks to retrieve per query |
| `--no-ui` | off | Load the index without launching the UI |

Requires `GROQ_API_KEY` in `.env`. The key is never committed (`.gitignore` covers `.env`).

---

## Functions

| Function | Responsibility |
|----------|----------------|
| `build_prompt()` | Format retrieved chunks as numbered context blocks and attach the grounding system prompt. |
| `generate_answer()` | Full pipeline: retrieve → filter → prompt → Groq → return `(answer, sources, raw_chunks)`. |
| `format_sources()` | Build the Sources section from retrieval metadata — no LLM involvement. |
| `build_ui()` | Construct the Gradio `Blocks` layout wired to `generate_answer()`. |

### `generate_answer()` return values

```python
answer, sources, chunks = generate_answer(query, collection, embed_model, groq_client)

# answer  — str: grounded answer text from Groq, or the refusal phrase
# sources — str: formatted Sources section built from metadata
# chunks  — list[dict]: raw retrieve() output for inspection/testing
```

---

## How it works

### Full pipeline flow

1. `retrieve(query, collection, embed_model, top_k=5)` — embeds the query and fetches
   the 5 most similar chunks from ChromaDB (cosine similarity).

2. Chunks below `MIN_SCORE_THRESHOLD = 0.50` are dropped. If no chunks remain, the
   refusal phrase is returned immediately without making a Groq API call.

3. `build_prompt(query, filtered_chunks)` constructs two messages:
   - A **system message** containing 7 hard grounding rules (see below).
   - A **user message** containing the query followed by each chunk as a numbered,
     labelled context block.

4. `groq_client.chat.completions.create(model=..., temperature=0.0)` sends the
   messages to Groq. `temperature=0.0` makes the response deterministic — grounded
   answers must not vary between calls.

5. `format_sources(filtered_chunks)` builds the Sources section directly from the
   `retrieve()` result dict. The LLM never writes citation text.

6. The final output is `format_response(answer, sources)`:

```
Answer:
<grounded answer text>

Sources:
  - dbt.pdf  |  page 1-1  |  score 0.8246  (chunk 0)
  - dbt.pdf  |  page 1-1  |  score 0.8184  (chunk 2)
  ...
```

### Grounding contract

The system prompt enforces seven rules as system-level constraints:

1. Answer ONLY using the numbered context blocks provided.
2. Do NOT use external knowledge, training data, or information not in the context.
3. Do NOT invent, infer, or extrapolate facts, names, tools, dates, or claims.
4. Do NOT generate citations — sources are handled separately.
5. If the context does not answer the question, respond with exactly:
   `I don't have enough information in the provided sources.`
6. Keep answers concise and directly responsive to the question.
7. You may quote or paraphrase the context, but not blend it with outside knowledge.

Rule 4 is what makes source attribution deterministic: the LLM is explicitly
prohibited from generating citation text. `format_sources()` reads `source`,
`page_start`, `page_end`, `score`, and `chunk_id` from the `retrieve()` result and
constructs the Sources section entirely in Python.

### Numbered context blocks

Each chunk is inserted into the prompt as:

```
[Context 1 | ApacheSpark.pdf | page 1-1]
Apache Spark is a unified analytics engine for large-scale data processing ...

[Context 2 | Dataquest.pdf | page 21-21]
Learning Spark and PySpark is core to modern data engineering ...
```

Numbering lets the model reference specific blocks precisely. The raw chunk text is
passed verbatim — nothing is summarised or pre-processed before Groq sees it.

---

## Refusal behaviour

Out-of-domain or unanswerable queries return the exact refusal phrase:

```
Answer:
I don't have enough information in the provided sources.

Sources:
  - Uber_Michelangelo.pdf  |  page 9-9  |  score 0.5685  (chunk 29)
  ...
```

Tested with "What is the capital of France?" — Groq correctly returned the refusal
phrase even though 5 chunks passed the score filter, because none of the context
addressed the question.

---

## Gradio interface

`build_ui()` uses `gr.Blocks` for layout control. The interface has:

- **Question box** — text input, 2 lines, with placeholder example
- **Ask button** — primary variant, triggers `generate_answer()`
- **Answer box** — displays grounded answer text (non-editable)
- **Sources box** — displays deterministic source attribution (non-editable)
- **Example questions** — the 5 evaluation-plan queries from `planning.md`,
  clickable to pre-fill the question box

Both the Ask button click and Enter keypress (`query_box.submit`) trigger the same
`respond()` handler.

---

## Sample outputs

**On-domain query:**

```
Query: "What is dbt used for?"

Answer:
dbt is used to transform raw warehouse data into trusted data products by creating
modular, maintainable data models that power analytics, operations, and AI, replacing
the need for complex and fragile transformation code. It helps teams build high-quality,
trustworthy data pipelines faster by applying software engineering best practices
to analytics workflows.

Sources:
  - dbt.pdf  |  page 1-1  |  score 0.8246  (chunk 0)
  - dbt.pdf  |  page 1-1  |  score 0.8184  (chunk 2)
  - dbt.pdf  |  page 5-5  |  score 0.7819  (chunk 13)
  - dbt.pdf  |  page 1-1  |  score 0.7738  (chunk 1)
  - dbt.pdf  |  page 4-5  |  score 0.7732  (chunk 12)
```

**Out-of-domain query:**

```
Query: "What is the capital of France?"

Answer:
I don't have enough information in the provided sources.

Sources:
  - Uber_Michelangelo.pdf  |  page 9-9  |  score 0.5685  (chunk 29)
  - Uber_DatabaseFederation.pdf  |  page 2-2  |  score 0.5537  (chunk 3)
  ...
```

---

## Three grounding checks

| Check | Status | How it is enforced |
|-------|--------|--------------------|
| System prompt strictly enforces grounding | Pass | 7 numbered rules; rules 1–3 prohibit external knowledge; rule 5 mandates exact refusal phrase |
| Source attribution is deterministic | Pass | `format_sources()` reads metadata from `retrieve()` output; rule 4 prohibits LLM from generating citations |
| Retrieval output passed directly into prompt | Pass | `build_prompt()` inserts raw chunk text verbatim as numbered context blocks; nothing is summarised before Groq sees it |

---

## Configuration

| Constant | Value | Meaning |
|----------|-------|---------|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model per planning.md |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MIN_SCORE_THRESHOLD` | `0.50` | Chunks below this score are dropped before prompting |
| `temperature` | `0.0` | Deterministic generation — grounded answers must not vary |

---

## Known limitations

- **Score threshold is approximate.** `MIN_SCORE_THRESHOLD = 0.50` prevents the weakest
  chunks from reaching the LLM, but in a small single-domain corpus even a 0.55-scoring
  chunk may be weakly related. Raising the threshold (e.g. to 0.60) would reduce noise
  at the cost of occasionally filtering useful context.
- **TOC/navigation chunks.** `ApacheSpark.pdf` contains index pages that score highly on
  Spark queries but contain only link slugs. These reach the LLM as context. The fix
  (filtering chunks where >40% of tokens are URL/path patterns) is described in
  `embedding_retrieval.md`.
- **Groq rate limits.** The free tier has token-per-minute limits. If many concurrent
  users submit queries, the API will throttle. For production, add retry logic with
  exponential backoff around the `groq_client.chat.completions.create()` call.

---

## Pipeline map

| Stage | File | Status |
|-------|------|--------|
| 1. Document Ingestion | `src/ingest.py` | Complete — see `docs/ingestion.md` |
| 2. Chunking | `src/ingest.py` | Complete — see `docs/ingestion.md` |
| 3. Embedding | `src/embed.py` | Complete — see `docs/embedding_retrieval.md` |
| 4. Vector Store | `src/embed.py` | Complete — see `docs/embedding_retrieval.md` |
| 5. Retrieval | `src/embed.py` | Complete — see `docs/embedding_retrieval.md` |
| 6. Generation + UI | `src/generate.py` | Complete — this document |
