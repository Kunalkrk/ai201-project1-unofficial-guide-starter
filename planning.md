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

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**

**Overlap:**

**Reasoning:**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.

2.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

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

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
