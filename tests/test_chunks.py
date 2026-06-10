"""
Chunk quality inspection for the RAG pipeline.

Prints 5 representative chunks (one per source document, mid-index) and grades
each on standalone retrievability: can a user's question be answered from the
chunk alone, without reading what comes before or after?

Run:
    python tests/test_chunks.py
"""

import json
import textwrap
from pathlib import Path

CHUNKS_FILE = Path(__file__).parent.parent / "data" / "chunks.json"
WRAP_WIDTH = 90

# ── Grading rubric ─────────────────────────────────────────────────────────────
#
# GOOD  — chunk contains a complete, self-contained idea; metadata (source/page)
#         is correct; a retrieval hit on this chunk would let the LLM answer a
#         specific question without needing adjacent chunks.
#
# WARN  — chunk is readable but starts/ends mid-thought (overlap artifact), or
#         contains navigation/URL noise that dilutes the content.
#
# BAD   — fragment with no standalone meaning, or multiple unrelated topics
#         merged into one chunk (too diluted to match any specific query).
#
# ── Evaluation criteria per chunk ─────────────────────────────────────────────
#
# 1. Is the main topic clear from the text alone?
# 2. Could a reader answer at least one specific question from this chunk?
# 3. Is the chunk focused (not jumping across 3+ unrelated subtopics)?
# 4. Are the token count and page metadata plausible?
# ──────────────────────────────────────────────────────────────────────────────

# One representative source from each major domain in the corpus.
TARGET_SOURCES = [
    "Reddit_Path.pdf",         # community career advice
    "ApacheSpark.pdf",         # technical documentation
    "Uber_Michelangelo.pdf",   # industry engineering blog
    "Dataquest.pdf",           # structured learning guide
    "dbt.pdf",                 # tool-specific documentation
]

GRADES = {
    "Reddit_Path.pdf::chunk0033": {
        "grade": "WARN",
        "reason": (
            "Multiple unrelated user voices in one chunk (three different Reddit "
            "commenters). A query about 'Azure data engineering' would retrieve "
            "this, but the answer is buried between an 'Art major' career path "
            "and a deleted comment. Standalone meaning is weak — no commenter's "
            "thought is complete. Overlap prefix ('that rare breed') is a dangling "
            "sentence fragment that carries no meaning without the previous chunk."
        ),
        "question_answerable": "Partially — 'Is SQL/Python needed for data engineering?' gets a yes, but it's attribution-less.",
        "fix": "Acceptable for this project size. A higher-quality fix would be to strip Reddit header/footer noise and treat each top-level comment as its own document before chunking.",
    },
    "ApacheSpark.pdf::chunk0010": {
        "grade": "WARN",
        "reason": (
            "A table-of-contents / navigation chunk. Contains hyperlink slugs "
            "('rdd-programming-guide.html', 'sql-programming-guide.html') and no "
            "substantive explanation. A query like 'What is Spark Streaming?' would "
            "retrieve this, but the chunk only names the guide — it gives no actual "
            "answer. It is technically complete (a list of Spark APIs) but low-value "
            "for generation."
        ),
        "question_answerable": "Only 'What guides exist for Spark?' — too shallow for real questions.",
        "fix": "Filter out chunks whose text is >40% URL/path tokens during ingestion, or deduplicate TOC pages before chunking.",
    },
    "Uber_Michelangelo.pdf::chunk0030": {
        "grade": "GOOD",
        "reason": (
            "Describes two distinct phases of Uber's ML platform workflow: (1) how "
            "training jobs pull from the Feature Store, and (2) the model evaluation "
            "process during exploratory training. Both ideas belong to the same topic "
            "(model training lifecycle) and together form a complete thought. The "
            "page-header noise ('6/10/26, 1:04 AM ... 9/25') is minor. A query about "
            "'how Uber trains ML models' would get a useful, grounded answer from "
            "this chunk alone."
        ),
        "question_answerable": "Yes — 'What is the model training workflow in Michelangelo?'",
        "fix": "Strip the PDF timestamp/URL header line ('https://www.uber.com/... 9/25') in clean_text() for a cleaner chunk.",
    },
    "Dataquest.pdf::chunk0031": {
        "grade": "GOOD",
        "reason": (
            "Focused on a single skill stage: handling large datasets in Python before "
            "reaching for distributed tools. Explains *why* this skill matters "
            "(real-world data exceeds memory), references a concrete tool (NumPy/pandas), "
            "and gives a timeline. The chunk stands alone — it does not require reading "
            "the surrounding curriculum to be useful. Token count (179) is well-sized: "
            "complete idea, not padded."
        ),
        "question_answerable": "Yes — 'What Python skills do data engineers need for large datasets?'",
        "fix": "None needed. This is the target quality for all chunks.",
    },
    "dbt.pdf::chunk0010": {
        "grade": "WARN",
        "reason": (
            "Mixes two separate topics: (1) how to access dbt via VS Code/CLI and "
            "(2) a pricing/plans note. A query about 'how to install dbt' gets a "
            "partially useful answer, but the pricing sentence ('learn about plans on "
            "www.getdbt.com') adds noise that could mislead the LLM into discussing "
            "cost when the user only asked about setup. Not a blocking issue at this "
            "scale, but a clear example of a chunk that serves two masters."
        ),
        "question_answerable": "Partially — 'How do I use dbt locally?' is answerable, but the pricing sentence dilutes it.",
        "fix": "Increase the paragraph break sensitivity in the dbt PDF, or add a post-processing step to split chunks on topic-shift indicators ('pricing', 'plans').",
    },
}


def load_chunks(path: Path) -> dict[str, list[dict]]:
    """Return chunks grouped by source document."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_source: dict[str, list[dict]] = {}
    for chunk in raw:
        src = chunk["metadata"]["source"]
        by_source.setdefault(src, []).append(chunk)
    return by_source


def pick_sample(by_source: dict[str, list[dict]], source: str) -> dict:
    """Return the mid-index chunk for a given source."""
    chunks = by_source.get(source, [])
    if not chunks:
        raise KeyError(f"Source not found in chunks.json: {source}")
    return chunks[len(chunks) // 2]


def grade_label(grade: str) -> str:
    labels = {"GOOD": "[GOOD]", "WARN": "[WARN]", "BAD": "[BAD ]"}
    return labels.get(grade, grade)


def print_divider(char: str = "-", width: int = WRAP_WIDTH) -> None:
    print(char * width)


def print_chunk(n: int, chunk: dict, evaluation: dict) -> None:
    meta = chunk["metadata"]
    chunk_id = chunk["id"]
    grade = evaluation["grade"]

    print_divider("=")
    print(f"  CHUNK {n} of 5  |  {chunk_id}")
    print(f"  Source: {meta['source']}  |  Page: {meta['page_start']}-{meta['page_end']}"
          f"  |  Tokens: {meta['token_count']}")
    print_divider()

    print("\n  TEXT\n")
    for line in textwrap.wrap(chunk["text"], width=WRAP_WIDTH - 4):
        print(f"    {line}")

    print(f"\n  GRADE: {grade_label(grade)}\n")
    print_divider()

    print("  REASONING")
    for line in textwrap.wrap(evaluation["reason"], width=WRAP_WIDTH - 4):
        print(f"    {line}")

    print(f"\n  QUESTION ANSWERABLE FROM THIS CHUNK ALONE?")
    for line in textwrap.wrap(evaluation["question_answerable"], width=WRAP_WIDTH - 4):
        print(f"    {line}")

    print(f"\n  SUGGESTED FIX (if any)")
    for line in textwrap.wrap(evaluation["fix"], width=WRAP_WIDTH - 4):
        print(f"    {line}")
    print()


def print_summary(samples: list[dict], evals: dict) -> None:
    print_divider("=")
    print("  SUMMARY\n")
    counts = {"GOOD": 0, "WARN": 0, "BAD": 0}
    rows = []
    for chunk in samples:
        cid = chunk["id"]
        ev = evals.get(cid, {})
        grade = ev.get("grade", "?")
        counts[grade] = counts.get(grade, 0) + 1
        rows.append((cid, chunk["metadata"]["token_count"], grade))

    for cid, tokens, grade in rows:
        short = cid.split("::")[-1]
        src = cid.split("::")[0]
        print(f"    {grade_label(grade):<10}  {src:<32}  {short}  ({tokens} tok)")

    print(f"\n  {counts['GOOD']} GOOD  |  {counts['WARN']} WARN  |  {counts['BAD']} BAD  "
          f"(out of {len(samples)} sampled)")

    print("""
  PIPELINE-LEVEL OBSERVATIONS

    1. Token sizes (66–200) are correctly bounded below all-MiniLM-L6-v2's
       256-token limit. No silent truncation at embedding time.

    2. Overlap (mean ~49 tokens) ensures no hard cuts at paragraph edges —
       context that spans a boundary appears in at least two chunks.

    3. Reddit chunks mix multiple commenters' voices in one chunk (3+ speakers).
       For community-sourced PDFs, comment-level splitting would improve quality.

    4. Navigation/TOC pages (Apache Spark) produce low-value chunks. A filter
       on chunks with >40% URL/path tokens would reduce retrieval noise.

    5. PDF header lines (timestamps, page numbers, URLs) survive into chunk text.
       Extending clean_text() to strip these patterns would raise average quality.
""")
    print_divider("=")


def main() -> None:
    print()
    print("  CHUNK QUALITY INSPECTION")
    print("  5 representative chunks — one per source, mid-document index")
    print("  Criterion: standalone retrievability without adjacent context")
    print()

    by_source = load_chunks(CHUNKS_FILE)
    samples = [pick_sample(by_source, src) for src in TARGET_SOURCES]

    for n, chunk in enumerate(samples, start=1):
        evaluation = GRADES.get(chunk["id"])
        if evaluation is None:
            # Fallback if chunk IDs shift after re-running ingest.py
            evaluation = {
                "grade": "?",
                "reason": f"No pre-written evaluation for {chunk['id']}. Re-run ingest.py and update GRADES.",
                "question_answerable": "Unknown.",
                "fix": "Update GRADES dict in test_chunks.py with this chunk id.",
            }
        print_chunk(n, chunk, evaluation)

    print_summary(samples, GRADES)


if __name__ == "__main__":
    main()
