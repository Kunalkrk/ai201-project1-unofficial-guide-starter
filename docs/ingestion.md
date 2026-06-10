# Ingestion & Chunking — Implementation Notes

Documentation for [`ingest.py`](../src/ingest.py): the **Document Ingestion** and **Chunking**
stages of the RAG pipeline (see [`planning.md`](planning.md)).

```
PDFs → load_pdfs() → extract_text() → clean_text() → chunk_documents() → chunks.json
                     └──────────── ingest.py ────────────┘
```

Output (`chunks.json`) is a list of chunks ready for the embedding stage
(`sentence-transformers/all-MiniLM-L6-v2`) and the ChromaDB vector store.

---

## How to run

```bash
python src/ingest.py                      # uses documents/, writes data/chunks.json
python src/ingest.py --chunk-size 200 --overlap 50 --out data/chunks.json
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--doc-dir` | `documents/` | Folder scanned for `*.pdf` |
| `--chunk-size` | `200` | Target tokens per chunk (full window, overlap included) |
| `--overlap` | `50` | Tokens shared between consecutive chunks |
| `--out` | `data/chunks.json` | Output path for the embedding stage |

Dependency added to `requirements.txt`: `pdfplumber==0.11.4`
(`transformers`, used for token-accurate counting, ships with `sentence-transformers`).

---

## Functions

| Function | Responsibility |
|----------|----------------|
| `load_pdfs()` | Discover `*.pdf` files in `documents/`, sorted for deterministic output. |
| `extract_text()` | Extract text page-by-page with `pdfplumber`, preserving reading order. |
| `clean_text()` | Normalize whitespace, repair hyphenation, keep paragraph structure. |
| `chunk_documents()` | Paragraph-aware splitting into ~`chunk_size`-token chunks with `overlap`. |
| `TokenCounter` | Counts/slices tokens with the **embedding model's own tokenizer**. |
| `build_chunks()` | Orchestrates load → extract → clean → chunk across every PDF. |

### Output schema

Each chunk is a `Chunk(id, text, metadata)`:

```json
{
  "id": "ApacheSpark.pdf::chunk0000",
  "text": "Apache Spark is a unified analytics engine ...",
  "metadata": {
    "source": "ApacheSpark.pdf",
    "page": 1,
    "page_start": 1,
    "page_end": 1,
    "token_count": 180,
    "chunk_index": 0
  }
}
```

All metadata values are scalars (str/int) — directly accepted by ChromaDB.
`page_start`/`page_end` give an honest page **range** because a chunk can span pages.

---

## How it works

### PDF parsing
`extract_text()` calls `pdfplumber.open()` and iterates `pdf.pages` **sequentially**,
so multi-page reading order is preserved. Each page's `extract_text()` output is tagged
with its 1-based page number, then passed through `clean_text()`. Pages with no text
layer log a warning — image-only/scanned PDFs return zero characters and need OCR
(out of scope here).

### Text cleaning
`clean_text()`:
- repairs words hyphenated across a line break (`engineer-\ning` → `engineering`),
- collapses single newlines (PDF soft-wraps) into spaces **while keeping blank lines
  as paragraph separators**,
- drops common web-export chrome (cookie/nav/"skip to content" lines).

### Chunk boundaries (paragraph-aware)
Forward boundaries fall **between paragraphs** whenever possible. A paragraph larger
than the content budget is split on sentence boundaries; a single oversized sentence
is hard-split by tokens as a last resort. Blocks are pre-sized to
`chunk_size − overlap`, so `overlap_prefix + one block` never exceeds `chunk_size`
(and therefore never the model's 256-token truncation limit).

### Overlap (sliding window, token-level)
Each new chunk begins with the **last ~`overlap` tokens of the previous chunk**. The
start position is located via the fast tokenizer's offset map and sliced from the
*original* text (snapped to a word boundary), so the overlap is a clean, readable
substring — no WordPiece artifacts (`##s`) or respaced punctuation from decoding token
IDs.

> An earlier paragraph-boundary-only carry produced ~0 overlap whenever a chunk was a
> single large paragraph (common in these docs). The token-level window is what makes
> overlap actually work.

### Token accuracy
"Tokens" are counted with `all-MiniLM-L6-v2`'s own WordPiece tokenizer via `TokenCounter`,
not word counts — so "200 tokens" means what the embedder actually sees.

---

## Chunk size: why 200/50 (not the spec's 500/100)

`all-MiniLM-L6-v2` has `max_seq_length = 256`. Chunks longer than 256 tokens are
**silently truncated** by sentence-transformers before embedding — the tail never
contributes to the vector, and no error is raised. The original 500-token plan would
have lost roughly half of every chunk at embedding time.

Settling on **200 tokens / 50 overlap** keeps every chunk safely under 256 (no
truncation) while preserving ~25% overlap for cross-boundary context. `chunk_documents()`
still accepts any size and prints a loud warning when `chunk_size > 256`.

---

## Verified results (current `documents/`)

| Metric | Result |
|--------|--------|
| PDFs ingested | 11 |
| Total chunks | **402** |
| Tokens/chunk (min / mean / max) | 66 / 163 / 200 — **none over 256** |
| Consecutive overlap, tokens (min / mean / max) | 3 / 49 / 50 |
| Adjacent pairs with no overlap | **0 of 391** |

Per-document chunk counts:

| Document | Pages | Chunks |
|----------|------:|-------:|
| Alasdairb_Roadmap.pdf | 8 | 21 |
| ApacheSpark.pdf | 3 | 21 |
| Datadriven.pdf | 7 | 22 |
| Dataquest.pdf | 34 | 63 |
| Datatalks.pdf | 10 | 44 |
| dbt.pdf | 7 | 20 |
| Netflix_DataPipeline.pdf | 11 | 18 |
| Reddit_Path.pdf | 17 | 66 |
| Reddit_Roadmap.pdf | 2 | 7 |
| Uber_DatabaseFederation.pdf | 25 | 60 |
| Uber_Michelangelo.pdf | 25 | 60 |

---

## Known limitations
- **Image-only PDFs** (no text layer) yield no chunks; they would require OCR.
- Some PDFs contain link/TOC regions that `pdfplumber` extracts lowercased and
  respaced — this is source extraction noise, not introduced by chunking.
- `clean_text()`'s chrome filter is heuristic and targets common cases only.

---

## Next pipeline stages (see `planning.md`)
- **Milestone 4 — Embedding + vector store:** embed `chunks.json` with
  `all-MiniLM-L6-v2`, persist embeddings + metadata in ChromaDB, retrieve top-k = 5
  via cosine similarity.
- **Milestone 5 — Generation:** combine retrieved chunks into a grounded prompt and
  answer via Groq (`llama-3.3-70b-versatile`).
