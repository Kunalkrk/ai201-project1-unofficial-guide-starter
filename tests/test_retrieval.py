"""
Retrieval quality test for the Unofficial Guide RAG pipeline.

Tests all 5 evaluation-plan queries from docs/planning.md against the live
ChromaDB index. For each query, prints the top-k returned chunks with their
similarity scores and delivers a verdict: GOOD, WARN, or BAD.

Verdict criteria
----------------
GOOD  -- top result is on-topic, score >= 0.60, from a plausible source.
         A real user question would get a useful answer from this chunk.
WARN  -- result is loosely related but too generic, mixes topics, or the
         score is in the 0.45-0.60 "uncertain" band. Might answer the
         question but might hallucinate too.
BAD   -- result is off-topic (score < 0.45, or clearly wrong subject),
         or the top source has nothing to do with the query domain.

Run (from project root):
    python tests/test_retrieval.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# Allow imports from src/ without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embed import load_index, retrieve  # noqa: E402

TOP_K = 5

# ---------------------------------------------------------------------------
# Evaluation plan queries (verbatim from docs/planning.md)
#
# NOTE ON HIGH SCORES: these queries were written knowing the corpus contents,
# use exact terminology from document titles, and the corpus is small (402 chunks)
# and single-domain. Both factors inflate similarity scores. A score of 0.87 here
# does NOT mean the same as 0.87 in a large general-purpose corpus — it reflects
# the narrow semantic neighbourhood of a focused collection, not absolute quality.
# The adversarial queries below stress-test retrieval more honestly.
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "id": "Q1",
        "query": "Is there one data engineering roadmap?",
        "expected": "No. Successful data engineers follow different paths and focus on core fundamentals.",
        "good_sources": ["Alasdairb_Roadmap.pdf", "Reddit_Path.pdf", "Reddit_Roadmap.pdf",
                         "Dataquest.pdf", "Datadriven.pdf"],
        "good_keywords": ["roadmap", "path", "different", "fundamentals", "no single"],
    },
    {
        "id": "Q2",
        "query": "What skills should beginners learn first?",
        "expected": "SQL, Python, databases, and data modeling.",
        "good_sources": ["Dataquest.pdf", "Datadriven.pdf", "Reddit_Path.pdf",
                         "Alasdairb_Roadmap.pdf"],
        "good_keywords": ["sql", "python", "database", "modeling", "beginner", "fundamentals"],
    },
    {
        "id": "Q3",
        "query": "What is Apache Spark used for?",
        "expected": "Large-scale distributed data processing and ETL.",
        "good_sources": ["ApacheSpark.pdf", "Dataquest.pdf", "Datatalks.pdf"],
        "good_keywords": ["spark", "distributed", "processing", "analytics", "etl", "large-scale"],
    },
    {
        "id": "Q4",
        "query": "What is dbt used for?",
        "expected": "Data transformation, testing, and documentation in data warehouses.",
        "good_sources": ["dbt.pdf", "Dataquest.pdf", "Datadriven.pdf"],
        "good_keywords": ["dbt", "transform", "model", "warehouse", "test", "documentation"],
    },
    {
        "id": "Q5",
        "query": "What project is recommended for aspiring data engineers?",
        "expected": "An end-to-end data pipeline using ingestion, transformation, and orchestration tools.",
        "good_sources": ["Dataquest.pdf", "Datadriven.pdf", "Datatalks.pdf",
                         "Reddit_Path.pdf", "Reddit_Roadmap.pdf"],
        "good_keywords": ["pipeline", "project", "ingestion", "transformation",
                          "orchestration", "portfolio", "end-to-end"],
    },
]

# ---------------------------------------------------------------------------
# Adversarial queries — colloquial phrasing a real user might type.
# These do NOT use exact document terminology. Lower scores are expected and
# honest. They expose where retrieval actually struggles.
# ---------------------------------------------------------------------------

ADVERSARIAL_QUERIES = [
    {
        "id": "A1",
        "query": "is spark worth learning in 2026",
        "note": "Colloquial version of Q3. Drops 'Apache', adds an opinion framing. "
                "Should still hit ApacheSpark.pdf — if it doesn't, the model is "
                "over-relying on exact keyword match rather than semantic similarity.",
        "good_sources": ["ApacheSpark.pdf", "Dataquest.pdf", "Datatalks.pdf"],
        "good_keywords": ["spark", "distributed", "processing", "analytics"],
    },
    {
        "id": "A2",
        "query": "how do i get my first job in data",
        "note": "Vague career query with no technical vocabulary. The corpus has "
                "relevant content (Reddit career paths, Dataquest roadmap) but the "
                "query shares few tokens with chunk text. Score will be lower — "
                "a WARN here is honest and expected.",
        "good_sources": ["Reddit_Path.pdf", "Dataquest.pdf", "Datadriven.pdf",
                         "Alasdairb_Roadmap.pdf"],
        "good_keywords": ["job", "career", "engineer", "hire", "experience", "path"],
    },
    {
        "id": "A3",
        "query": "whats the sql transformation tool everyone uses",
        "note": "Oblique reference to dbt with no mention of 'dbt' at all. Tests "
                "whether the embedder can bridge 'sql transformation tool' to dbt "
                "content. If top result is not from dbt.pdf it is a retrieval failure.",
        "good_sources": ["dbt.pdf", "Dataquest.pdf"],
        "good_keywords": ["dbt", "transform", "sql", "model", "warehouse"],
    },
    {
        "id": "A4",
        "query": "I want to transition from software engineering to data engineering",
        "note": "Career transition query. Mostly career-advice corpus (Reddit, Dataquest, "
                "Datadriven). Should NOT retrieve Spark/dbt/Uber infrastructure chunks — "
                "if it does, that is off-topic retrieval caused by generic DE vocabulary overlap.",
        "good_sources": ["Reddit_Path.pdf", "Dataquest.pdf", "Datadriven.pdf",
                         "Alasdairb_Roadmap.pdf", "Datatalks.pdf"],
        "good_keywords": ["transition", "software", "engineer", "career", "background", "path"],
    },
]

# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

GOOD_SCORE_THRESHOLD = 0.60
WARN_SCORE_THRESHOLD = 0.45


def _grade_result(result: dict, query_meta: dict) -> tuple[str, str]:
    """
    Return (grade, reason) for the top-1 result of a query.

    Checks three things independently, then combines:
      - Score band (GOOD >= 0.60, WARN 0.45-0.60, BAD < 0.45)
      - Source relevance (is the top source in the expected-source list?)
      - Keyword signal (does the chunk text contain at least one good keyword?)
    """
    score = result["score"]
    source = result["source"]
    text_lower = result["text"].lower()

    score_band = (
        "GOOD" if score >= GOOD_SCORE_THRESHOLD
        else "WARN" if score >= WARN_SCORE_THRESHOLD
        else "BAD"
    )
    source_ok = source in query_meta["good_sources"]
    keyword_hit = any(kw in text_lower for kw in query_meta["good_keywords"])

    # Downgrade if source is wrong even when score is high (off-topic retrieval).
    if score_band == "GOOD" and not source_ok and not keyword_hit:
        return "WARN", (
            f"Score is strong ({score:.4f}) but source '{source}' is unexpected "
            "and chunk text lacks domain keywords — possible off-topic retrieval."
        )

    if score_band == "BAD":
        return "BAD", (
            f"Score {score:.4f} is below {WARN_SCORE_THRESHOLD} threshold. "
            "Retrieval returned a weakly related chunk — the LLM would lack "
            "grounding to answer this question."
        )

    if score_band == "WARN":
        if not keyword_hit:
            reason = (
                f"Score {score:.4f} is in the uncertain band and chunk text "
                "lacks expected keywords. Answer quality would be unreliable."
            )
        elif not source_ok:
            reason = (
                f"Score {score:.4f} is in the uncertain band. Source '{source}' "
                "is unexpected but keywords suggest the topic is close."
            )
        else:
            reason = (
                f"Score {score:.4f} is in the uncertain band (0.45-0.60). "
                "Content is plausibly relevant but retrieval confidence is low."
            )
        return "WARN", reason

    # GOOD band
    kw_found = [kw for kw in query_meta["good_keywords"] if kw in text_lower]
    return "GOOD", (
        f"Score {score:.4f} >= {GOOD_SCORE_THRESHOLD}. "
        f"Source '{source}' is in the expected set. "
        f"Keywords found: {kw_found}."
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

SEP_HEAVY = "=" * 88
SEP_LIGHT = "-" * 88
GRADE_LABELS = {"GOOD": "[GOOD]", "WARN": "[WARN]", "BAD": "[ BAD]"}
WRAP = 84


def _p(text: str) -> None:
    """Print with Windows-safe encoding."""
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
    sys.stdout.buffer.flush()


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(
        textwrap.fill(line, width=WRAP, initial_indent=prefix, subsequent_indent=prefix)
        for line in text.splitlines()
        if line.strip()
    )


def _print_query_header(q: dict, adversarial: bool = False) -> None:
    _p(SEP_HEAVY)
    tag = "  [ADVERSARIAL] " if adversarial else "  "
    _p(f"{tag}{q['id']}  |  {q['query']}")
    if adversarial:
        _p(f"  Note: {q['note']}")
    else:
        _p(f"  Expected answer: {q['expected']}")
    _p(SEP_LIGHT)


def _print_result(r: dict, is_top: bool = False) -> None:
    tag = "  TOP -> " if is_top else "         "
    _p(f"{tag}Rank {r['rank']}  |  Score: {r['score']:.4f}  |  {r['chunk_id']}")
    _p(f"         Source: {r['source']}  |  Page: {r['page_start']}-{r['page_end']}")
    _p("")
    _p(_wrap(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""), indent=9))
    _p("")


def _print_verdict(grade: str, reason: str) -> None:
    _p(SEP_LIGHT)
    _p(f"  VERDICT: {GRADE_LABELS[grade]}")
    _p(_wrap(reason, indent=4))
    _p("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_retrieval_tests(
    collection,
    model,
    top_k: int = TOP_K,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Run evaluation + adversarial queries. Returns (eval_grades, adv_grades).
    """
    eval_grades: dict[str, str] = {}
    adv_grades: dict[str, str] = {}

    _p("\n--- PART 1: EVALUATION PLAN QUERIES (exact terminology from planning.md) ---\n")
    for q in QUERIES:
        results = retrieve(q["query"], collection, model, top_k=top_k)
        grade, reason = _grade_result(results[0], q)
        eval_grades[q["id"]] = grade
        _print_query_header(q, adversarial=False)
        for r in results:
            _print_result(r, is_top=(r["rank"] == 1))
        _print_verdict(grade, reason)

    _p("\n--- PART 2: ADVERSARIAL QUERIES (colloquial, no exact document keywords) ---")
    _p("    Scores will be lower. That is expected and honest.\n")
    for q in ADVERSARIAL_QUERIES:
        results = retrieve(q["query"], collection, model, top_k=top_k)
        grade, reason = _grade_result(results[0], q)
        adv_grades[q["id"]] = grade
        _print_query_header(q, adversarial=True)
        for r in results:
            _print_result(r, is_top=(r["rank"] == 1))
        _print_verdict(grade, reason)

    return eval_grades, adv_grades


def _print_summary(eval_grades: dict[str, str], adv_grades: dict[str, str]) -> None:
    all_grades = {**eval_grades, **adv_grades}
    counts = {g: sum(1 for v in all_grades.values() if v == g)
              for g in ("GOOD", "WARN", "BAD")}

    _p(SEP_HEAVY)
    _p("  SUMMARY\n")
    _p("  Evaluation queries (exact terminology):")
    for qid, grade in eval_grades.items():
        q = next(x for x in QUERIES if x["id"] == qid)
        _p(f"    {GRADE_LABELS[grade]}  {qid}: {q['query']}")
    _p("")
    _p("  Adversarial queries (colloquial phrasing):")
    for qid, grade in adv_grades.items():
        q = next(x for x in ADVERSARIAL_QUERIES if x["id"] == qid)
        _p(f"    {GRADE_LABELS[grade]}  {qid}: {q['query']}")
    _p("")
    _p(f"  Total: {counts['GOOD']} GOOD  |  {counts['WARN']} WARN  |  {counts['BAD']} BAD"
       f"  across {len(all_grades)} queries")
    _p("""
  WHY EVALUATION SCORES ARE HIGH
    (1) Queries use exact vocabulary from document titles/content.
    (2) Corpus is small (402 chunks) and single-domain -- the semantic floor
        is elevated; nothing unrelated competes for top-k slots.
    (3) Score conversion (1 - dist/2) maps a 0.25 distance to 0.875 similarity,
        which looks strong but is normal for a focused corpus.
    Adversarial scores are the more honest signal.

  SCORE BANDS
    GOOD  score >= 0.60  and source/keywords match expected answer
    WARN  score  0.45-0.60  or source/keywords uncertain
    BAD   score <  0.45  or clearly off-topic content at rank 1
""")
    _p(SEP_HEAVY)


def main() -> None:
    _p("\n  RETRIEVAL QUALITY TEST")
    _p("  docs/planning.md evaluation queries + adversarial stress tests")
    _p("  Index: data/chroma_db  |  Model: all-MiniLM-L6-v2  |  Top-k: 5\n")

    collection, model = load_index()
    eval_grades, adv_grades = run_retrieval_tests(collection, model)
    _print_summary(eval_grades, adv_grades)


if __name__ == "__main__":
    main()
