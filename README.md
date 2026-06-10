# The Unofficial Data Engineering Guide — Project 1

---

## Domain

The Data Engineering Career Guide covers skills, tools, career paths, and real-world responsibilities for aspiring and early-career data engineers. It aggregates community experiences (Reddit threads), learning roadmaps (Dataquest, Datadriven, Alasdairb), technical documentation (Apache Spark, dbt), and engineering case studies (Netflix, Uber).

This knowledge is difficult to find in one place because practical data engineering advice is scattered across Reddit discussions, personal blogs, vendor documentation, and engineering blogs. Beginners often struggle to distinguish foundational skills from specific tools, and no single official source covers the full picture — from what to learn first to how production data systems actually work.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Alasdairb | Career advice blog post | https://alasdairb.com/posts/there-is-no-data-engineering-roadmap |
| 2 | Reddit r/dataengineering | Community thread — career paths | https://www.reddit.com/r/dataengineering/comments/1ibkmlj/what_path_did_you_take_to_become_a_data_engineer/ |
| 3 | Reddit r/dataengineer | Community thread — roadmap 2026 | https://www.reddit.com/r/dataengineer/comments/1qe7od5/the_roadmap_to_becoming_a_data_engineer_in_2026/ |
| 4 | Dataquest | Beginner learning roadmap | https://www.dataquest.io/blog/the-data-engineer-roadmap-for-beginners/ |
| 5 | Datadriven.io | Technical skills & career roadmap | https://datadriven.io/data-engineer-roadmap |
| 6 | Uber Engineering | Database Federation architecture | https://www.uber.com/us/en/blog/database-federation/ |
| 7 | DataTalks.Club | Learning resources & portfolio projects | https://datatalks.club/blog/data-engineering-zoomcamp.html |
| 8 | dbt Labs | Data transformation documentation | https://docs.getdbt.com/docs/introduction |
| 9 | Apache Spark | Distributed computing documentation | https://spark.apache.org/docs/latest/index.html |
| 10 | Netflix Tech Blog | Data pipeline engineering case study | https://netflixtechblog.com/evolution-of-the-netflix-data-pipeline-da246ca36905 |
| 11 | Uber Engineering | Michelangelo ML platform | https://www.uber.com/us/en/blog/michelangelo-machine-learning-platform/ |

---

## Chunking Strategy

**Chunk size:** 200 tokens

**Overlap:** 50 tokens

**Why these choices fit your documents:** Most documents are medium-to-long articles, technical documentation, and Reddit discussions where important explanations span multiple paragraphs. The 200-token chunk size is large enough to preserve a complete idea — such as a section describing a career path or a data pipeline concept — while staying under the embedding model's 256-token `max_seq_length` limit (exceeding it causes silent truncation). The 50-token overlap is implemented as a sliding-window carry: the overlap prefix is sliced from the preceding chunk using the tokenizer's character-offset mapping, which avoids WordPiece artifacts (`##s`) that appear when decoding token IDs. This ensures that important information near chunk boundaries is never lost.

The chunker uses a paragraph-aware greedy packing approach: paragraphs (blank-line separated) are grouped into blocks sized to the content budget (`chunk_size − overlap`), and the overlap prefix from the previous chunk is prepended at the start of each new chunk.

**Final chunk count:** 402 chunks across 11 documents  
Token distribution: min 66 / mean 163 / max 200 | Mean overlap: 49 tokens | Zero-overlap pairs: 0

---

## Embedding Model

**Model used:** `sentence-transformers/all-MiniLM-L6-v2`

This model was chosen because it runs locally (no API cost or rate limits), produces 384-dimensional vectors well-suited to a small corpus, and has a 256-token max sequence length that matches the 200-token chunk size. It performs well on semantic similarity tasks in English, which is the language of all 11 source documents.

**Production tradeoff reflection:** For a production deployment, the main tradeoffs are retrieval quality, context length, multilingual support, and latency. A larger model such as `text-embedding-3-large` (OpenAI) or `e5-large-v2` would better capture relationships between concepts like data pipelines, Spark, ETL, and analytics engineering, but would require API calls (latency + cost) or more GPU memory for local hosting. If the corpus were expanded to multilingual content, a multilingual model like `paraphrase-multilingual-MiniLM-L12-v2` would be necessary. For this single-domain English corpus at 402 chunks, `all-MiniLM-L6-v2` gives fast local inference with no quality degradation visible in retrieval tests.

---

## Grounded Generation

**System prompt grounding instruction:**

The system prompt contains seven numbered rules enforced as system-level constraints:

1. Answer ONLY using the numbered context blocks provided below the user question.
2. Do NOT use any external knowledge, training data, or information not present in the context.
3. Do NOT invent, infer, or extrapolate facts, names, tools, dates, or claims beyond what the context explicitly states.
4. Do NOT generate citations, source names, or document references — sources are handled separately.
5. If the provided context does not contain enough information to answer the question, respond with exactly: `I don't have enough information in the provided sources.`
6. Keep your answer concise and directly responsive to the question.
7. You may quote or paraphrase the context, but do not blend it with outside knowledge.

Each retrieved chunk is inserted into the user message as a numbered, labelled block:
```
[Context 1 | dbt.pdf | page 1-1]
dbt is an open source tool that enables data analysts and engineers ...
```

Chunks with a similarity score below 0.50 are dropped before the prompt is built. If no chunks pass this filter, the refusal phrase is returned immediately without making an API call.

**How source attribution is surfaced in the response:**

Source attribution is fully deterministic — the LLM never generates citation text (rule 4 above). After the Groq API call returns, `format_sources()` builds the Sources section directly from the `retrieve()` result metadata: filename, page range, similarity score, and chunk index. This means the sources listed are always exactly the chunks that were passed into the prompt, not whatever the model decided to cite.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Is there one data engineering roadmap? | No — successful engineers follow different paths and focus on core fundamentals | No — the system cited Context 1 ("There is no Data Engineering roadmap"), noted contradictions across roadmap sources, then concluded there is no single roadmap | Relevant | Accurate |
| 2 | What skills should beginners learn first? | SQL, Python, databases, and data modeling | SQL first — "SQL is the foundation of data engineering and is not going to become obsolete" | Relevant | Partially accurate (SQL confirmed, Python omitted — LLM gave a focused one-sentence answer rather than listing all skills) |
| 3 | What is Apache Spark used for? | Large-scale distributed data processing and ETL | Large-scale data analytics; unified engine with Spark SQL, MLlib, GraphX, and Structured Streaming | Relevant | Accurate |
| 4 | What is dbt used for? | Data transformation, testing, and documentation in data warehouses | Transforms raw warehouse data into trusted data products via modular, maintainable models; replaces complex transformation code; applies software engineering practices to analytics | Relevant | Accurate |
| 5 | What project is recommended for aspiring data engineers? | End-to-end data pipeline using ingestion, transformation, and orchestration tools | A project covering design/architecture, engineering pipelines, modeling, cleaning, and analysis/visualization on a dataset you care about | Relevant | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** "What path did people take to become a data engineer?"

**What the system returned:** A chunk containing three different Reddit commenters' responses fused into a single block — an Art major career pivot, an Azure course recommendation, and a deleted reply. The overlap prefix ("that rare breed") was a dangling sentence fragment from the previous chunk with no standalone meaning. No single commenter's answer was complete, so the LLM had to synthesize across fragmented, unattributed voices.

**Root cause (tied to a specific pipeline stage):** Chunking stage. The Reddit PDF is a printed thread where each commenter's response is a short paragraph separated by a username and timestamp. Because the chunker packs paragraphs greedily up to the token budget, multiple commenters' paragraphs (each ~40–60 tokens) get merged into one 179-token chunk. The chunker has no awareness that a username line signals a speaker change — it treats it as a normal paragraph boundary.

**What you would change to fix it:** Pre-process Reddit PDFs before ingestion: split the raw text on commenter header lines (username + timestamp pattern) so each comment becomes its own document. Feed those mini-documents through the chunker individually. This keeps each chunk to one voice and eliminates the attribution confusion at generation time.

---

**Question that failed:** "What APIs does Apache Spark provide?"

**What the system returned:** A chunk that was purely a table-of-contents listing — a sequence of hyperlink slugs ("rdd-programming-guide.html", "sql-programming-guide.html") with one-line labels and no explanatory content. The chunk would rank highly in retrieval because it names every Spark API, but it cannot answer the question — it only points to where answers live.

**Root cause (tied to a specific pipeline stage):** Ingestion + chunking stages. The Apache Spark PDF is a web-exported documentation page that includes a full "Where to Go from Here" navigation section. `pdfplumber` extracts this as regular text, and the chunker has no signal to distinguish navigation prose from substantive content. The result is a chunk that matches many queries but satisfies none of them.

**What you would change to fix it:** Add a post-chunking filter that discards chunks where more than 40% of tokens match a URL/filepath pattern (`re.search(r'\.\w{2,5}\b', token)`). Alternatively, detect and strip navigation sections in `clean_text()` by recognizing header lines like "Where to Go from Here" followed by dense link lists.

---

## Spec Reflection

**One way the spec helped you during implementation:**

The chunking strategy section of `planning.md` specified 200-token chunks with 50-token overlap before any code was written. This made it possible to hand the spec directly to Claude as a prompt and get a working implementation immediately — the AI tool had a concrete target rather than generic requirements. More importantly, having the chunk size written down revealed the conflict with the embedding model's 256-token `max_seq_length` limit early: the original spec had 500-token chunks, and the model constraint was caught before any chunking code ran, not after. The spec forced the decision to be explicit and recordable.

**One way your implementation diverged from the spec, and why:**

The spec described a "recursive text splitter" for chunking (from the architecture diagram). The actual implementation uses a sliding-window token-level carry with paragraph-aware greedy packing — a different algorithm. The reason for the divergence was that a simple recursive character split produced 0 shared tokens between consecutive chunks in testing, because each chunk ended exactly at a paragraph boundary with no carry. The overlap only worked correctly when implemented at the token level using the tokenizer's character-offset mapping to slice the overlap prefix from the tail of the previous chunk. The spec was updated in `planning.md` after this was confirmed working, but the architecture diagram still says "recursive text splitter."

---

## AI Usage

**Instance 1**

- *What I gave the AI:* The full `planning.md` chunking strategy section (200-token chunks, 50-token overlap, paragraph-aware splitting, pdfplumber ingestion) and a list of required function signatures: `load_pdfs()`, `extract_text()`, `clean_text()`, `chunk_documents()`.
- *What it produced:* A complete `src/ingest.py` with a `TokenCounter` class wrapping HuggingFace tokenizer, greedy paragraph packing, and a CLI that outputs `data/chunks.json`.
- *What I changed or overrode:* The initial chunk size was 500 tokens. I overrode it to 200 before running because the embedding model (`all-MiniLM-L6-v2`) has a 256-token max sequence length — 500-token chunks would be silently truncated. I also requested a rewrite of the overlap logic after the first version produced 0 shared tokens between consecutive chunks; the fix used tokenizer offset mapping to slice the overlap prefix at a word boundary from the original text.

**Instance 2**

- *What I gave the AI:* The embedding and retrieval section of `planning.md` (all-MiniLM-L6-v2 + ChromaDB + cosine similarity + top-k=5), plus the output schema from `ingest.py` and required function signatures: `embed_chunks()`, `store_in_chroma()`, `retrieve()`.
- *What it produced:* A complete `src/embed.py` with batched ChromaDB upserts, cosine distance-to-similarity score conversion (`1 - distance/2`), and a `load_index()` function for the generation stage.
- *What I changed or overrode:* I directed the use of `upsert()` instead of `add()` to make indexing idempotent (re-running never raises `DuplicateIDError`). I also directed the `hnsw:space: cosine` metadata to be set at collection creation time, since ChromaDB does not allow changing the distance metric after the collection exists.
