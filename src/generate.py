"""
Generation + Gradio interface for the Unofficial Guide RAG pipeline.

Pipeline stage covered here (see docs/planning.md):
    6. Generation  ->  generate_answer()  (Groq llama-3.3-70b-versatile)

Full flow:
    user query
        -> retrieve()        top-5 chunks from ChromaDB   (src/embed.py)
        -> build_prompt()    numbered context + strict grounding instructions
        -> Groq API          llama-3.3-70b-versatile generates answer body ONLY
        -> format_sources()  sources appended from retrieval metadata (NOT from LLM)
        -> Gradio UI         displays answer + sources

Grounding contract
------------------
The system prompt enforces three hard constraints:
  1. Answer using ONLY the numbered context blocks provided.
  2. Never introduce facts, names, dates, or claims not present in the context.
  3. If the answer is not in the context, respond with the exact refusal phrase.

Source attribution is 100% deterministic: format_sources() reads chunk_id, source,
and page from the retrieve() result dict. The LLM generates the answer body only —
it never writes citation text.

Usage
-----
    # Launch Gradio interface (opens browser automatically):
    python src/generate.py

    # One-shot CLI answer (no UI):
    python src/generate.py --query "What is dbt used for?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embed import load_index, retrieve  # noqa: E402

load_dotenv(Path(__file__).parent.parent / ".env")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
NO_CONTEXT_REPLY = "I don't have enough information in the provided sources."

# Minimum similarity score to pass a chunk to the LLM.
# Chunks below this threshold are dropped before prompting to avoid injecting
# weakly related content that could mislead the model.
MIN_SCORE_THRESHOLD = 0.50

# ----------------------------------------------------------------------------
# 1. Prompt construction
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a grounded question-answering assistant for a Data Engineering knowledge base.

STRICT RULES — you must follow all of these without exception:

1. Answer ONLY using the numbered context blocks provided below the user question.
2. Do NOT use any external knowledge, training data, or information not present in the context.
3. Do NOT invent, infer, or extrapolate facts, names, tools, dates, or claims beyond what the context explicitly states.
4. Do NOT generate citations, source names, or document references — sources are handled separately.
5. If the provided context does not contain enough information to answer the question, respond with exactly this phrase and nothing else:
   I don't have enough information in the provided sources.
6. Keep your answer concise and directly responsive to the question.
7. You may quote or paraphrase the context, but do not blend it with outside knowledge.

These rules are system-level constraints, not suggestions. Violating them produces an incorrect answer.\
"""


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Construct the messages list for the Groq chat completion call.

    Each retrieved chunk is inserted as a numbered context block:

        [Context 1 | ApacheSpark.pdf | page 1]
        Apache Spark is a unified analytics engine ...

        [Context 2 | Dataquest.pdf | page 18]
        ...

    Numbering lets the model reference specific blocks in its answer without
    generating citation text (that is handled by format_sources() instead).

    Returns a messages list: [{"role": "system", ...}, {"role": "user", ...}]
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Context {i} | {chunk['source']} | page {chunk['page_start']}-{chunk['page_end']}]"
        context_blocks.append(f"{header}\n{chunk['text']}")

    context_section = "\n\n".join(context_blocks)

    user_message = (
        f"Question: {query}\n\n"
        f"Context:\n{context_section}\n\n"
        "Answer using only the context above. "
        f"If the context does not answer the question, respond with exactly: "
        f'"{NO_CONTEXT_REPLY}"'
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


# ----------------------------------------------------------------------------
# 2. Source formatting (deterministic — never touches LLM output)
# ----------------------------------------------------------------------------

def format_sources(chunks: list[dict]) -> str:
    """
    Build the Sources section from retrieval metadata.

    Source attribution is derived entirely from the retrieve() result dict —
    the LLM has no role in generating this text. Each line is constructed
    from chunk_id, source (filename), page range, and similarity score.

    Example output:
        Sources:
        - ApacheSpark.pdf  |  page 1-1  |  score 0.8766  (chunk 0)
        - Dataquest.pdf    |  page 21-21  |  score 0.8018  (chunk 33)
    """
    if not chunks:
        return "Sources:\n  (none)"

    lines = ["Sources:"]
    for chunk in chunks:
        # chunk_id format: "ApacheSpark.pdf::chunk0000"
        chunk_num = chunk["chunk_id"].split("chunk")[-1].lstrip("0") or "0"
        lines.append(
            f"  - {chunk['source']}"
            f"  |  page {chunk['page_start']}-{chunk['page_end']}"
            f"  |  score {chunk['score']:.4f}"
            f"  (chunk {chunk_num})"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 3. Generation
# ----------------------------------------------------------------------------

def generate_answer(
    query: str,
    collection,
    embed_model,
    groq_client: Groq,
    top_k: int = TOP_K,
) -> tuple[str, str, list[dict]]:
    """
    Full RAG generation: retrieve -> prompt -> Groq -> format.

    Returns (answer_text, sources_text, raw_chunks) so the UI can display
    answer and sources separately and tests can inspect raw_chunks.

    Steps:
      1. Retrieve top_k chunks from ChromaDB.
      2. Filter chunks below MIN_SCORE_THRESHOLD (too weakly related).
      3. If no chunks pass the filter, return the refusal phrase immediately
         without making a Groq API call.
      4. Build the grounded prompt (numbered context blocks).
      5. Call Groq; extract answer text from the response.
      6. Format sources deterministically from metadata (not from LLM output).
    """
    # Step 1: retrieve
    chunks = retrieve(query, collection, embed_model, top_k=top_k)

    # Step 2: filter low-confidence chunks
    filtered = [c for c in chunks if c["score"] >= MIN_SCORE_THRESHOLD]

    # Step 3: no usable context
    if not filtered:
        return NO_CONTEXT_REPLY, "Sources:\n  (none — no chunks met the relevance threshold)", chunks

    # Step 4: build prompt
    messages = build_prompt(query, filtered)

    # Step 5: call Groq
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,      # deterministic — grounded answers must not vary
        max_tokens=1024,
    )
    answer = response.choices[0].message.content.strip()

    # Step 6: deterministic sources (built from metadata, not from LLM text)
    sources = format_sources(filtered)

    return answer, sources, filtered


# ----------------------------------------------------------------------------
# 4. Formatted output
# ----------------------------------------------------------------------------

def format_response(answer: str, sources: str) -> str:
    """
    Combine answer and sources into the final display string.

    Format:
        Answer:
        <answer text>

        Sources:
        - doc.pdf  |  page 1-1  |  score 0.88  (chunk 0)
        ...
    """
    return f"Answer:\n{answer}\n\n{sources}"


# ----------------------------------------------------------------------------
# 5. Gradio interface
# ----------------------------------------------------------------------------

def build_ui(collection, embed_model, groq_client: Groq):
    """
    Build and return a Gradio Blocks UI.

    Layout:
      - Text input:  user question
      - Button:      Ask
      - Text output: Answer section (grounded, LLM-generated)
      - Text output: Sources section (deterministic, from metadata)
    """
    import gradio as gr

    def respond(query: str) -> tuple[str, str]:
        query = query.strip()
        if not query:
            return "Please enter a question.", ""
        answer, sources, _ = generate_answer(query, collection, embed_model, groq_client)
        return answer, sources

    with gr.Blocks(title="Data Engineering Unofficial Guide") as ui:
        gr.Markdown(
            "## Data Engineering Unofficial Guide\n"
            "Ask anything about data engineering careers, tools, and learning paths.\n"
            "Answers are grounded strictly in the retrieved source documents."
        )

        with gr.Row():
            query_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What skills should I learn first as a data engineering beginner?",
                lines=2,
                scale=4,
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1)

        with gr.Row():
            answer_box = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False,
                scale=3,
            )
            sources_box = gr.Textbox(
                label="Sources",
                lines=8,
                interactive=False,
                scale=2,
            )

        # Example questions drawn from the evaluation plan in planning.md
        gr.Examples(
            examples=[
                "Is there one data engineering roadmap?",
                "What skills should beginners learn first?",
                "What is Apache Spark used for?",
                "What is dbt used for?",
                "What project is recommended for aspiring data engineers?",
            ],
            inputs=query_box,
        )

        ask_btn.click(fn=respond, inputs=query_box, outputs=[answer_box, sources_box])
        query_box.submit(fn=respond, inputs=query_box, outputs=[answer_box, sources_box])

    return ui


# ----------------------------------------------------------------------------
# 6. CLI + entry point
# ----------------------------------------------------------------------------

def _require_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        print(
            "ERROR: GROQ_API_KEY is not set.\n"
            "Add it to your .env file:\n"
            "  GROQ_API_KEY=your_key_here"
        )
        sys.exit(1)
    return key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG generation layer — Groq + Gradio interface."
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a single query from the CLI without launching the UI.",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--no-ui", action="store_true",
        help="Suppress the Gradio UI even when --query is not provided.",
    )
    args = parser.parse_args()

    api_key = _require_api_key()
    groq_client = Groq(api_key=api_key)
    collection, embed_model = load_index()

    if args.query:
        # CLI mode: print answer + sources, no UI
        answer, sources, chunks = generate_answer(
            args.query, collection, embed_model, groq_client, top_k=args.top_k
        )
        print(format_response(answer, sources))
        return

    if args.no_ui:
        print("Index loaded. Run with --query <text> to ask a question.")
        return

    # Default: launch Gradio
    ui = build_ui(collection, embed_model, groq_client)
    ui.launch()


if __name__ == "__main__":
    main()
