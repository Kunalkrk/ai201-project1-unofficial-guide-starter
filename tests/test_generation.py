"""
Grounded generation test for the Unofficial Guide RAG pipeline.

The core question for each test:
    Could this response have come from anywhere other than the retrieved chunks?

If yes — even if the answer is factually correct — it is a grounding failure.
A response is grounded if and only if its claims can be traced back to the text
that was actually retrieved and passed into the prompt. Correct answers from the
LLM's training data are just as much a failure as wrong ones.

Three test cases
----------------
T1  What is dbt used for?
    Covered clearly in dbt.pdf. Expect a traceable, specific answer.

T2  What should a data engineering beginner learn first?
    Covered across Dataquest.pdf, Datadriven.pdf, Alasdairb_Roadmap.pdf.
    Expect SQL/Python/databases — traceable to multiple sources.

T3  How does TCP/IP networking work?
    Completely outside the corpus. No document covers network protocols.
    Expect the exact refusal phrase. Any substantive answer is a grounding failure.

Grounding verification method
------------------------------
For each sentence in the answer we compute its Jaccard word-overlap with the
combined text of all retrieved chunks. Sentences with overlap below
TRACE_THRESHOLD (0.12) are flagged as potentially ungrounded — they contain
words not present in the retrieved context.

We also scan the answer for HALLUCINATION_SIGNALS: hedging phrases that are
characteristic of an LLM drawing on training knowledge rather than the provided
context ("generally", "typically", "research shows", etc.).

Neither check is a perfect detector, but together they provide a transparent,
reproducible signal. A sentence that scores below the trace threshold AND
contains a hallucination signal is the clearest warning sign.

Run:
    python tests/test_generation.py
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from groq import Groq

from src.embed import load_index
from src.generate import NO_CONTEXT_REPLY, generate_answer

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum Jaccard word-overlap for a sentence to be considered traceable.
# 0.12 = at least ~1 in 8 content words must appear in the retrieved chunks.
TRACE_THRESHOLD = 0.12

# Phrases that suggest the LLM is drawing on training knowledge.
HALLUCINATION_SIGNALS = [
    "typically", "generally", "usually", "in general", "it is common",
    "commonly", "most engineers", "most data engineers", "it is known",
    "research shows", "studies show", "experts say", "it is worth noting",
    "it's worth noting", "it is important to note", "it's important to note",
    "many engineers", "best practice", "industry standard", "widely used",
    "often considered", "often regarded", "it is recommended", "i recommend",
    "you should generally", "as a rule",
]

WRAP = 88

# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "T1",
        "query": "What is dbt used for?",
        "expect_refusal": False,
        "good_sources": ["dbt.pdf"],
        "must_contain_keywords": ["transform", "dbt", "sql", "warehouse", "model"],
        "note": (
            "Well-covered in dbt.pdf. A grounded answer must mention SQL transformation "
            "and warehouse concepts. Any answer about dbt that does not trace to the "
            "retrieved chunks is using LLM training knowledge."
        ),
    },
    {
        "id": "T2",
        "query": "What should a data engineering beginner focus on learning first?",
        "expect_refusal": False,
        "good_sources": ["Dataquest.pdf", "Datadriven.pdf", "Alasdairb_Roadmap.pdf"],
        "must_contain_keywords": ["sql", "python"],
        "note": (
            "Covered across career-advice sources. The answer should name SQL and Python "
            "specifically — those appear in the retrieved chunks. A generic answer like "
            "'focus on fundamentals and problem-solving' would score well with users but "
            "is not traceable to the corpus and is therefore a grounding failure."
        ),
    },
    {
        "id": "T3",
        "query": "How does TCP/IP networking work?",
        "expect_refusal": True,
        "good_sources": [],
        "must_contain_keywords": [],
        "note": (
            "Completely outside the corpus. No document covers network protocols. "
            "The only correct response is the exact refusal phrase. Any substantive "
            "answer — even a technically accurate one about TCP/IP — is a grounding "
            "failure because it comes from the model's training data, not the retrieved context."
        ),
    },
]

# ---------------------------------------------------------------------------
# Grounding analysis
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lowercase words of 3+ characters, stripped of punctuation."""
    return {
        w.lower().strip(".,;:!?\"'()-")
        for w in text.split()
        if len(w.strip(".,;:!?\"'()-")) >= 3
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on '. ', '! ', '? '."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _chunk_vocabulary(chunks: list[dict]) -> set[str]:
    """Union of all content words across all retrieved chunks."""
    vocab: set[str] = set()
    for c in chunks:
        vocab |= _tokenise(c["text"])
    return vocab


def analyse_grounding(answer: str, chunks: list[dict]) -> dict:
    """
    Check whether each sentence in `answer` is traceable to `chunks`.

    Returns a dict with:
        sentences        — list of (sentence, overlap_score, hallucination_flags, traced)
        traced_count     — sentences with overlap >= TRACE_THRESHOLD
        untraced_count   — sentences below the threshold
        hallucination_count — sentences with at least one hallucination signal
        overall_grade    — "PASS", "WARN", or "FAIL"
        grounding_ratio  — traced_count / total sentences
    """
    vocab = _chunk_vocabulary(chunks)
    sentences = _split_sentences(answer)
    results = []

    for sent in sentences:
        sent_tokens = _tokenise(sent)
        overlap = _jaccard(sent_tokens, vocab)
        flags = [sig for sig in HALLUCINATION_SIGNALS if sig in sent.lower()]
        traced = overlap >= TRACE_THRESHOLD
        results.append({
            "sentence": sent,
            "overlap": round(overlap, 3),
            "hallucination_flags": flags,
            "traced": traced,
        })

    total = len(results)
    traced = sum(1 for r in results if r["traced"])
    untraced = total - traced
    hal_count = sum(1 for r in results if r["hallucination_flags"])
    ratio = traced / total if total else 0.0

    if total == 0:
        grade = "FAIL"
    elif ratio >= 0.80 and hal_count == 0:
        grade = "PASS"
    elif ratio >= 0.60 or hal_count <= 1:
        grade = "WARN"
    else:
        grade = "FAIL"

    return {
        "sentences": results,
        "traced_count": traced,
        "untraced_count": untraced,
        "hallucination_count": hal_count,
        "overall_grade": grade,
        "grounding_ratio": round(ratio, 3),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

SEP_HEAVY = "=" * WRAP
SEP_LIGHT = "-" * WRAP
GRADE_LABELS = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}


def _p(text: str) -> None:
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
    sys.stdout.buffer.flush()


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return "\n".join(
        textwrap.fill(line, width=WRAP - 2, initial_indent=prefix,
                      subsequent_indent=prefix)
        for line in text.splitlines() if line.strip()
    )


def _print_case_header(tc: dict) -> None:
    _p(SEP_HEAVY)
    _p(f"  {tc['id']}  |  {tc['query']}")
    refusal_tag = "  [expects: REFUSAL]" if tc["expect_refusal"] else ""
    _p(f"  Expected sources: {tc['good_sources'] or '(none — out-of-domain)'}{refusal_tag}")
    _p(f"  Must-contain keywords: {tc['must_contain_keywords'] or '(none)'}")
    _p(SEP_LIGHT)
    _p(_wrap(f"NOTE: {tc['note']}", indent=2))
    _p("")


def _print_retrieval(chunks: list[dict]) -> None:
    _p("  RETRIEVED CHUNKS")
    for c in chunks:
        _p(f"    Rank {c['rank']}  score {c['score']:.4f}  {c['chunk_id']}")
    _p("")


def _print_answer(answer: str, sources: str) -> None:
    _p("  ANSWER")
    _p(_wrap(answer, indent=4))
    _p("")
    _p("  " + sources.replace("\n", "\n  "))
    _p("")


def _print_grounding_analysis(analysis: dict, answer: str, expect_refusal: bool) -> None:
    _p(SEP_LIGHT)
    _p("  GROUNDING ANALYSIS")
    _p("")

    if expect_refusal:
        is_refusal = answer.strip() == NO_CONTEXT_REPLY.strip()
        if is_refusal:
            _p("    [PASS]  Correct refusal — exact phrase returned, no LLM-generated content.")
        else:
            _p("    [FAIL]  Expected refusal but got a substantive answer.")
            _p("            This answer came from LLM training data, not the retrieved context.")
        _p("")
        return

    ratio_pct = round(analysis["grounding_ratio"] * 100)
    _p(f"    Grounding ratio: {analysis['traced_count']} / "
       f"{analysis['traced_count'] + analysis['untraced_count']} sentences "
       f"traceable ({ratio_pct}%)  |  threshold: Jaccard >= {TRACE_THRESHOLD}")
    _p(f"    Hallucination signals found: {analysis['hallucination_count']}")
    _p(f"    Overall grade: {GRADE_LABELS[analysis['overall_grade']]}")
    _p("")

    _p("    Sentence-by-sentence breakdown:")
    for r in analysis["sentences"]:
        status = "traced " if r["traced"] else "UNTRACED"
        flags = f"  << signals: {r['hallucination_flags']}" if r["hallucination_flags"] else ""
        _p(f"      [{status}]  overlap={r['overlap']:.3f}  "
           f"{r['sentence'][:70]}{'...' if len(r['sentence']) > 70 else ''}{flags}")
    _p("")


def _print_verdict(tc: dict, answer: str, chunks: list[dict], analysis: dict) -> None:
    _p(SEP_LIGHT)
    _p("  VERDICT")
    _p("")

    if tc["expect_refusal"]:
        passed = answer.strip() == NO_CONTEXT_REPLY.strip()
        grade = "PASS" if passed else "FAIL"
        if passed:
            _p(f"    {GRADE_LABELS[grade]}  Refusal phrase matched exactly.")
            _p("          The system correctly identified there was no relevant context.")
        else:
            _p(f"    {GRADE_LABELS[grade]}  Grounding failure — model answered from training knowledge.")
            _p(_wrap(
                "The system prompt needs tightening. The LLM ignored the grounding rules "
                "and generated content not present in the retrieved context.",
                indent=10
            ))
        _p("")
        return

    # Check keyword coverage
    answer_lower = answer.lower()
    missing_kw = [kw for kw in tc["must_contain_keywords"] if kw not in answer_lower]
    source_match = any(c["source"] in tc["good_sources"] for c in chunks)
    grade = analysis["overall_grade"]

    _p(f"    {GRADE_LABELS[grade]}")

    if not source_match:
        _p(_wrap(
            f"WARNING: Top results did not come from expected sources {tc['good_sources']}. "
            "The answer may be drawing on less relevant chunks.",
            indent=6
        ))

    if missing_kw:
        _p(_wrap(
            f"WARNING: Answer is missing expected keywords: {missing_kw}. "
            "The retrieved context covers these terms — their absence may indicate "
            "the answer is too generic or not grounded in the specific content.",
            indent=6
        ))

    if grade == "PASS":
        _p("      All answer sentences are traceable to the retrieved chunks.")
        _p("      No hallucination signals detected.")
    elif grade == "WARN":
        _p(_wrap(
            f"{analysis['untraced_count']} sentence(s) had low overlap with retrieved chunks. "
            "Review the untraced sentences above — they may be accurate paraphrases "
            "(acceptable) or they may be LLM-generated additions (grounding failure).",
            indent=6
        ))
    else:
        _p(_wrap(
            f"GROUNDING FAILURE: {analysis['untraced_count']} untraced sentence(s) and "
            f"{analysis['hallucination_count']} hallucination signal(s). "
            "A significant portion of this answer cannot be traced to the retrieved context. "
            "Tighten the system prompt or raise MIN_SCORE_THRESHOLD.",
            indent=6
        ))
    _p("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_generation_tests(collection, embed_model, groq_client: Groq) -> dict[str, str]:
    grades: dict[str, str] = {}

    for tc in TEST_CASES:
        answer, sources, chunks = generate_answer(
            tc["query"], collection, embed_model, groq_client
        )
        analysis = analyse_grounding(answer, chunks) if not tc["expect_refusal"] else {}

        _print_case_header(tc)
        _print_retrieval(chunks)
        _print_answer(answer, sources)
        _print_grounding_analysis(analysis, answer, tc["expect_refusal"])
        _print_verdict(tc, answer, chunks, analysis)

        if tc["expect_refusal"]:
            grades[tc["id"]] = "PASS" if answer.strip() == NO_CONTEXT_REPLY.strip() else "FAIL"
        else:
            grades[tc["id"]] = analysis.get("overall_grade", "FAIL")

    return grades


def _print_summary(grades: dict[str, str]) -> None:
    counts = {g: sum(1 for v in grades.values() if v == g)
              for g in ("PASS", "WARN", "FAIL")}
    _p(SEP_HEAVY)
    _p("  SUMMARY\n")
    for tid, grade in grades.items():
        tc = next(t for t in TEST_CASES if t["id"] == tid)
        _p(f"    {GRADE_LABELS[grade]}  {tid}: {tc['query']}")
    _p("")
    _p(f"  {counts['PASS']} PASS  |  {counts['WARN']} WARN  |  {counts['FAIL']} FAIL"
       f"  (of {len(grades)} tests)")
    _p(f"""
  GROUNDING REMINDER
    A PASS means answer sentences are traceable to retrieved chunks via word
    overlap AND no hallucination-signal phrases were detected.
    A WARN means some sentences had low overlap — may be valid paraphrases or
    may be LLM additions; inspect the sentence breakdown above.
    A FAIL on T3 (refusal test) is the most critical failure: it means the
    system answered from training data rather than the retrieved context.

  TRACE METHOD
    Jaccard word-overlap between each answer sentence and the combined
    vocabulary of the retrieved chunks. Threshold: {TRACE_THRESHOLD}.
    This catches verbatim and near-verbatim borrowing. It does NOT catch
    fluent paraphrases that preserve meaning without sharing words — those
    require human review of the sentence breakdown.
""")
    _p(SEP_HEAVY)


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        _p("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)

    _p("\n  GROUNDED GENERATION TEST")
    _p("  Core question: could this response have come from anywhere other")
    _p("  than the retrieved chunks? If yes, it is a grounding failure.\n")

    collection, embed_model = load_index()
    grades = run_generation_tests(collection, embed_model, groq_client)
    _print_summary(grades)


if __name__ == "__main__":
    main()
