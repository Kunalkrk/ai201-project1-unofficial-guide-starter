# Embedding, Retrieval & Testing — Implementation Notes

Documentation for [`embed.py`](../src/embed.py) and the test suite
([`test_chunks.py`](../tests/test_chunks.py), [`test_retrieval.py`](../tests/test_retrieval.py)):
the **Embedding**, **Vector Store**, and **Retrieval** stages of the RAG pipeline
(see [`planning.md`](planning.md)).

```
chunks.json → embed_chunks() → store_in_chroma() → ChromaDB
                                                       ↓
                              retrieve(query) ← query embedding
                                                       ↓
                                              top-5 chunks + scores
```

Input: `data/chunks.json` produced by `src/ingest.py`.
Output: `data/chroma_db/` (persisted vector index) and ranked chunk results at query time.

---

## How to run

```bash
# Build the index (reads data/chunks.json, writes data/chroma_db/):
python src/embed.py --index

# Query the existing index:
python src/embed.py --query "What is Apache Spark used for?"

# Index and query in one command:
python src/embed.py --index --query "What skills should beginners learn first?"

# Run chunk quality inspection (5 representative chunks, graded):
python tests/test_chunks.py

# Run full retrieval quality test (5 eval + 4 adversarial queries):
python tests/test_retrieval.py
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--index` | off | Build/rebuild the ChromaDB index from `data/chunks.json` |
| `--query` | — | Query string; returns top-k results with scores |
| `--top-k` | `5` | Number of results to return |
| `--chunks-file` | `data/chunks.json` | Input chunks path |
| `--chroma-dir` | `data/chroma_db` | ChromaDB persistence directory |
| `--collection` | `rag_chunks` | ChromaDB collection name |

---

## Functions

| Function | File | Responsibility |
|----------|------|----------------|
| `load_chunks()` | `embed.py` | Read `chunks.json`, return list of chunk dicts. |
| `embed_chunks()` | `embed.py` | Batch-encode chunk texts into 384-dim vectors. |
| `store_in_chroma()` | `embed.py` | Upsert vectors + metadata into ChromaDB. |
| `retrieve()` | `embed.py` | Embed a query and return top-k similar chunks. |
| `load_index()` | `embed.py` | Load an existing index + model for the generation stage. |
| `build_index()` | `embed.py` | Orchestrate load → embed → store in one call. |

### `retrieve()` return schema

Each result in the returned list:

```python
{
    "rank":       1,             # 1-based position
    "score":      0.8766,        # cosine similarity [0, 1]; higher = more relevant
    "text":       "...",         # chunk text passed to the LLM as context
    "source":     "ApacheSpark.pdf",
    "page":       1,
    "page_start": 1,
    "page_end":   1,
    "chunk_id":   "ApacheSpark.pdf::chunk0000"
}
```

---

## How it works

### Embedding
`embed_chunks()` calls `SentenceTransformer("all-MiniLM-L6-v2").encode()` with
`normalize_embeddings=True`, producing **384-dimensional unit vectors**. Normalization
means cosine similarity equals the dot product — ChromaDB can use the faster dot-product
path internally. Chunks are processed in batches of 64 to keep memory use predictable.

The same model and normalization setting are used for query embedding in `retrieve()`,
ensuring query and chunk vectors live in the same space.

### How ChromaDB stores vectors
ChromaDB uses a **two-part store**:

- `data/chroma_db/chroma.sqlite3` — metadata, collection configuration, and the raw
  chunk text. Every call to `store_in_chroma()` writes source, page, token_count, and
  chunk_index here as scalar metadata fields.
- `data/chroma_db/<uuid>/*.bin` — an **HNSW (Hierarchical Navigable Small World)**
  index. HNSW is a graph where each vector is a node connected to its approximate
  nearest neighbours. Similarity search walks this graph rather than scanning all 402
  vectors linearly, making retrieval fast even as the corpus grows.

The collection is created with `metadata={"hnsw:space": "cosine"}`. Without this,
ChromaDB defaults to L2 (Euclidean) distance — a different metric that would produce
different rankings. This must be set at collection creation time; it cannot be changed
after the fact.

`store_in_chroma()` uses `collection.upsert()` rather than `collection.add()`.
`add()` raises a `DuplicateIDError` if any chunk ID already exists; `upsert()` inserts
new IDs and overwrites existing ones, so re-running `python src/embed.py --index` after
adding new PDFs is always safe.

### How retrieval works
`retrieve()` encodes the query string into a 384-dim vector, then calls
`collection.query(query_embeddings=[...], n_results=5, include=["documents",
"metadatas", "distances"])`. ChromaDB walks the HNSW graph from the query vector,
identifies the 5 nearest neighbours, and joins back to SQLite for the text and metadata.

Results are returned sorted by **ascending distance** (best match first).

### Distance vs. similarity
ChromaDB's cosine mode returns **distance**, not similarity:

```
distance = 1 - cosine_similarity      range: [0, 2]
```

`retrieve()` converts this to an interpretable similarity score:

```
similarity = 1 - (distance / 2)       range: [0, 1]
```

| Distance | Similarity | Meaning |
|----------|------------|---------|
| 0.00 | 1.00 | Identical vectors |
| 0.25 | 0.875 | Very close (typical strong hit) |
| 1.00 | 0.50 | Orthogonal — no relation |
| 2.00 | 0.00 | Opposite vectors |

Higher score = lower distance = more similar = better retrieval result.

---

## On-disk layout

```
data/
├── chunks.json              402 chunks produced by ingest.py
└── chroma_db/
    ├── chroma.sqlite3       Collection config, chunk text, metadata
    └── <uuid>/
        ├── data_level0.bin  HNSW graph edges
        ├── header.bin       HNSW index header
        ├── length.bin       Vector count
        └── link_lists.bin   HNSW neighbour lists
```

---

## Using the index in the generation stage (Milestone 5)

```python
from src.embed import load_index, retrieve

# Load once at startup — avoids re-embedding on every request.
collection, model = load_index()

# Retrieve context for a user query.
results = retrieve("What is dbt used for?", collection, model)

# results[0]["text"]   -> chunk text to include in the LLM prompt
# results[0]["source"] -> source attribution for the response
# results[0]["score"]  -> relevance score; filter below a threshold if needed
```

---

## Test suite

### `tests/test_chunks.py` — chunk quality inspection

Samples 5 representative chunks (one per source, mid-document index) and grades each
on **standalone retrievability**: can the chunk answer a question without adjacent
context?

| Grade | Meaning |
|-------|---------|
| GOOD | Complete, self-contained idea; strong standalone retrieval value |
| WARN | Readable but starts/ends mid-thought, mixes topics, or contains nav noise |
| BAD | Fragment with no standalone meaning, or multiple unrelated topics merged |

**Results (current corpus):** 2 GOOD, 3 WARN, 0 BAD

| Chunk | Grade | Issue |
|-------|-------|-------|
| `Reddit_Path.pdf::chunk0033` | WARN | Three different commenters fused in one chunk; overlap prefix is a dangling fragment |
| `ApacheSpark.pdf::chunk0010` | WARN | Pure TOC/navigation chunk — link slugs, no substantive explanation |
| `Uber_Michelangelo.pdf::chunk0030` | GOOD | Complete description of Michelangelo model training lifecycle |
| `Dataquest.pdf::chunk0031` | GOOD | Self-contained skill explanation (Python for large datasets) with reasoning |
| `dbt.pdf::chunk0010` | WARN | Mixes install instructions with a pricing note |

### `tests/test_retrieval.py` — retrieval quality

Runs two groups of queries against the live ChromaDB index and grades each on source
relevance and keyword signal.

**Part 1 — Evaluation plan queries** (exact terminology from `planning.md`):

| ID | Query | Score | Source | Grade |
|----|-------|-------|--------|-------|
| Q1 | Is there one data engineering roadmap? | 0.875 | Alasdairb_Roadmap.pdf | GOOD |
| Q2 | What skills should beginners learn first? | 0.734 | Dataquest.pdf | GOOD |
| Q3 | What is Apache Spark used for? | 0.877 | ApacheSpark.pdf | GOOD |
| Q4 | What is dbt used for? | 0.825 | dbt.pdf | GOOD |
| Q5 | What project is recommended for aspiring data engineers? | 0.846 | Reddit_Path.pdf | GOOD |

**Part 2 — Adversarial queries** (colloquial phrasing, no exact document keywords):

| ID | Query | Score | Source | Grade |
|----|-------|-------|--------|-------|
| A1 | is spark worth learning in 2026 | 0.802 | Dataquest.pdf | GOOD |
| A2 | how do i get my first job in data | 0.820 | Alasdairb_Roadmap.pdf | GOOD |
| A3 | whats the sql transformation tool everyone uses | 0.788 | dbt.pdf | GOOD |
| A4 | I want to transition from software engineering to data engineering | 0.879 | Alasdairb_Roadmap.pdf | GOOD |

**Total: 9 / 9 GOOD**

---

## Why evaluation scores are high

All scores fall in the 0.73–0.88 range. This does **not** mean retrieval is perfect —
three compounding factors inflate the numbers:

1. **Evaluation queries use exact document vocabulary.** Q3 says "Apache Spark" — the
   exact document name and opening words of the top chunk. Q4 says "dbt" — the top
   source is literally titled "What is dbt?" These are confirmation checks, not
   stress tests.

2. **The corpus is small and single-domain.** With 402 chunks all about data
   engineering, there is nothing truly unrelated to compete for the top-k slots. The
   semantic floor is elevated; even a loose match scores above 0.60.

3. **The score conversion amplifies appearance.** A ChromaDB distance of 0.25 maps to
   similarity 0.875 via `1 - (0.25 / 2)`. That distance is ordinary for a focused
   corpus — it does not mean "nearly identical."

The adversarial queries (A1–A4) are the more honest signal. A3 — *"whats the sql
transformation tool everyone uses"* — retrieved `dbt.pdf` at rank 1 with score 0.788
without the word "dbt" appearing anywhere in the query. That demonstrates genuine
semantic bridging, not keyword matching.

---

## Known limitations

- **TOC/navigation chunks** (e.g. `ApacheSpark.pdf::chunk0010`) score highly on topic
  queries but contain only link slugs — no substantive content for generation. A
  post-indexing filter discarding chunks where >40% of tokens are URL/path patterns
  would remove these without affecting other results.
- **Reddit multi-voice chunks** merge 2–3 different commenters into one unit, making
  attribution and coherence unreliable. Pre-processing Reddit PDFs to split on commenter
  header lines before ingestion would fix this at the source.
- **PDF header noise** (timestamps, page numbers, URLs printed at the top of each page)
  survives into chunk text. Extending `clean_text()` to strip these patterns would raise
  average chunk quality across all sources.
- **Score inflation in small corpora** means retrieval quality looks better than it is.
  Adding out-of-domain documents to the corpus (or testing with off-topic queries) would
  reveal the true discrimination boundary.

---

## Next pipeline stage

**Milestone 5 — Generation:** pass retrieved chunks as context to Groq
(`llama-3.3-70b-versatile`) via a grounded system prompt, and return a sourced natural
language answer. See `planning.md` for the architecture diagram and AI tool plan.
