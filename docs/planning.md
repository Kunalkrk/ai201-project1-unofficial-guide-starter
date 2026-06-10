# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

The Data Engineering Career Guide domain focuses on helping aspiring and early-career data engineers understand the skills, tools, career paths, projects, and real-world responsibilities involved in the field. It combines community experiences, learning roadmaps, technical documentation, and engineering case studies from major technology companies.

This knowledge is difficult to find in one place because practical data engineering advice is scattered across Reddit discussions, personal blogs, vendor documentation, and engineering blogs. Beginners often struggle to distinguish foundational skills from specific tools, making it valuable to aggregate perspectives from both industry practitioners and official technical resources.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Alasdairb | Career Development & Learning Strategy | https://alasdairb.com/posts/there-is-no-data-engineering-roadmap |
| 2 | Reddit | Career Paths & Industry Experiences | https://www.reddit.com/r/dataengineering/comments/1ibkmlj/what_path_did_you_take_to_become_a_data_engineer/ |
| 3 | Reddit | Skills Roadmaps & Interview Preparation | https://www.reddit.com/r/dataengineer/comments/1qe7od5/the_roadmap_to_becoming_a_data_engineer_in_2026/ |
| 4 | Dataquest | Beginner Learning Roadmap | https://www.dataquest.io/blog/the-data-engineer-roadmap-for-beginners/ |
| 5 | Datadriven | Technical Skills & Career Roadmaps | https://datadriven.io/data-engineer-roadmap |
| 6 | Uber | Data Architecture & Large-Scale Systems | https://www.uber.com/us/en/blog/database-federation/ |
| 7 | Datatalks | Learning Resources & Portfolio Projects | https://datatalks.club/blog/data-engineering-zoomcamp.html |
| 8 | DBT | Data Transformation & Analytics Engineering | https://docs.getdbt.com/docs/introduction |
| 9 | Apache | Big Data Processing & Distributed Computing | https://spark.apache.org/docs/latest/index.html |
| 10 | Netflix | Data Pipelines & Data Infrastructure | https://netflixtechblog.com/evolution-of-the-netflix-data-pipeline-da246ca36905 |
| 11 | Uber | Data Platforms & Machine Learning Infrastructure | https://www.uber.com/us/en/blog/michelangelo-machine-learning-platform/ |

---

## Chunking Strategy

Most documents are medium-to-long articles, technical documentation pages, blog posts, and Reddit discussions, not short reviews. Because important explanations often span multiple paragraphs (e.g., descriptions of Spark, dbt, career advice, or data architecture), I would use chunks of 200 tokens with a 50-token overlap.

**Chunk size: 200**

**Overlap: 50**

**Reasoning: The 200-token chunk size is large enough to preserve context around a complete idea, such as a section describing data pipelines or career progression, while still being small enough for precise retrieval. The 50-token overlap helps ensure that important information near chunk boundaries is not lost**

---

## Retrieval Approach

**Embedding model: all-MiniLM-L6-v2**

**Top-k: top 5 chunks (top-k = 5) for each query**

**Production tradeoff reflection:The main tradeoffs would be retrieval quality, context length, multilingual support, and latency. A larger model may better capture relationships between concepts such as data pipelines, Spark, ETL, and analytics engineering, but it would require more computational resources and increase response times.**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Is there one data engineering roadmap? | No. Successful data engineers follow different paths and focus on core fundamentals. |
| 2 | What skills should beginners learn first? | SQL, Python, databases, and data modeling. |
| 3 | What is Apache Spark used for? | Large-scale distributed data processing and ETL. |
| 4 | What is dbt used for? | Data transformation, testing, and documentation in data warehouses. |
| 5 | What project is recommended for aspiring data engineers? | An end-to-end data pipeline using ingestion, transformation, and orchestration tools. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Conflicting or inconsistent advice: Community sources such as Reddit may provide differing opinions on the best learning path, tools, or career advice. The system may retrieve contradictory information and generate an unclear answer.

2. Chunk boundary issues: Important explanations may be split across two chunks. If only one chunk is retrieved, the answer could be incomplete or miss key context.

3. Off-topic retrieval: A query about data engineering careers may retrieve chunks focused on machine learning platforms, data science, or software engineering because of overlapping terminology.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

┌──────────────────────────────────────┐
│ 1. Document Ingestion               │
│                                      │
│ Tools: pdfplumber     │
│ Sources: PDFs, blogs, Reddit, docs   │
└──────────────────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────┐
│ 2. Chunking                         │
│                                      │
│ Tool: Recursive text splitter       │
│ Chunk size: ~200 tokens             │
│ Overlap: ~50 tokens                │
└──────────────────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────┐
│ 3. Embedding Generation             │
│                                      │
│ Model: sentence-transformers        │
│        all-MiniLM-L6-v2             │
│ Runs locally                        │
└──────────────────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────┐
│ 4. Vector Store                     │
│                                      │
│ DB: ChromaDB                        │
│ Stores embeddings + metadata        │
│ Local persistence                   │
└──────────────────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────┐
│ 5. Retrieval                        │
│                                      │
│ Similarity search (cosine)          │
│ Top-k = 5 chunks                    │
│ Returns most relevant context       │
└──────────────────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────┐
│ 6. Generation                       │
│                                      │
│ LLM: Groq API                      │
│ Model: llama-3.3-70b-versatile     │
│ Uses retrieved context + prompt     │
└──────────────────────────────────────┘

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**
Prompt Claude with planning.md ingestion + chunking specs (PDF ingestion using pdfplumber, 200-token chunks, 50-token overlap) and ask it to implement a load_documents() and chunk_documents() pipeline that outputs clean text chunks with metadata.

**Milestone 4 — Embedding and retrieval:**
Provide Claude embedding + vector store design (sentence-transformers all-MiniLM-L6-v2 + ChromaDB + top-k=5) and ask it to generate code for embedding chunks, persisting them in Chroma, and implementing a retrieve(query) function using cosine similarity search.

**Milestone 5 — Generation and interface:**
Provide Claude my RAG architecture diagram + prompt requirements and ask it to build the Groq (llama-3.3-70b-versatile) query function that combines retrieved chunks into a structured prompt and returns a final response via a simple CLI or lightweight API endpoint.