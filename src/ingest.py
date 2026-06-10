"""
Document ingestion + chunking for the Unofficial Guide RAG pipeline.

Pipeline stages covered here (see docs/planning.md):
    1. Document Ingestion  -> load_pdfs() / extract_text()  (pdfplumber)
    2. Chunking            -> chunk_documents()             (paragraph-aware, token-budgeted)

Output is a list of Chunk objects ready for the embedding stage
(sentence-transformers all-MiniLM-L6-v2) and the ChromaDB vector store.
Each chunk carries the raw text plus scalar-only metadata that ChromaDB accepts.

Chunking strategy (from docs/planning.md):
    chunk size = 500 tokens, overlap = 100 tokens, paragraph-aware splitting preferred.

IMPORTANT CAVEAT:
    all-MiniLM-L6-v2 has max_seq_length = 256. Chunks longer than 256 tokens are
    SILENTLY TRUNCATED by the embedder before the vector is produced -- the tail of
    the chunk never contributes to retrieval. We default to the spec's 500/100 but
    emit a warning when chunk_size exceeds the model limit so the tradeoff is visible.
    Consider --chunk-size 256 --overlap 50 to match the model exactly.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pdfplumber
from transformers import AutoTokenizer
from transformers import logging as hf_logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ingest")

# Silence the "Token indices sequence length is longer than 512" notice: we
# intentionally tokenize long paragraphs to measure them before splitting.
hf_logging.set_verbosity_error()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MAX_TOKENS = 256          # all-MiniLM-L6-v2 hard truncation limit
DEFAULT_CHUNK_SIZE = 200        # fits all-MiniLM-L6-v2 (256 max) with no truncation
DEFAULT_OVERLAP = 50            # ~25% overlap to preserve cross-boundary context
DEFAULT_DOC_DIR = Path(__file__).parent.parent / "documents"

# Sentence boundary used only when a single paragraph is larger than chunk_size.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------

@dataclass
class Page:
    """One extracted PDF page, in reading order."""
    source: str          # file name, e.g. "ApacheSpark.pdf"
    page: int            # 1-based page number
    text: str


@dataclass
class Chunk:
    """A retrieval unit ready for embedding + ChromaDB."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------

def load_pdfs(doc_dir: Path = DEFAULT_DOC_DIR) -> list[Path]:
    """Return all PDF paths in the documents folder, sorted for determinism."""
    if not doc_dir.exists():
        raise FileNotFoundError(f"Documents folder not found: {doc_dir}")
    pdfs = sorted(doc_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No .pdf files found in {doc_dir}")
    log.info("Found %d PDF(s) in %s", len(pdfs), doc_dir)
    return pdfs


# ----------------------------------------------------------------------------
# 2. Extraction
# ----------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> list[Page]:
    """
    Extract text from a PDF page-by-page using pdfplumber.

    Pages are iterated in file order, so multi-page reading order is preserved.
    Each page becomes one Page record tagged with its 1-based page number, so we
    can attribute chunks back to a page (or page range) later.
    """
    pages: list[Page] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = clean_text(raw)
            if cleaned:
                pages.append(Page(source=pdf_path.name, page=i, text=cleaned))
    if not pages:
        log.warning("No extractable text in %s (scanned/image PDF?)", pdf_path.name)
    return pages


# ----------------------------------------------------------------------------
# 3. Cleaning
# ----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize raw PDF text while preserving paragraph structure.

    - Repair words hyphenated across a line break ("engineer-\\ning" -> "engineering").
    - Collapse single newlines (PDF soft-wraps) into spaces.
    - Keep blank lines as paragraph separators.
    - Strip common web-export chrome (cookie/nav lines) and collapse whitespace.
    """
    if not text:
        return ""

    # Repair hyphenation at line ends: "word-\nnext" -> "wordnext"
    text = re.sub(r"-\n(?=\w)", "", text)

    # Normalize line endings, then preserve paragraph breaks (blank lines) while
    # treating single newlines as soft wraps that should become spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)

    cleaned_paragraphs: list[str] = []
    for para in paragraphs:
        # Soft-wrap newlines -> spaces, collapse runs of whitespace.
        para = re.sub(r"\s*\n\s*", " ", para)
        para = re.sub(r"[ \t]{2,}", " ", para).strip()
        if _is_chrome(para):
            continue
        if para:
            cleaned_paragraphs.append(para)

    return "\n\n".join(cleaned_paragraphs)


_CHROME_PATTERNS = re.compile(
    r"^(skip to (main )?content|cookie|we use cookies|accept all|"
    r"subscribe|sign in|log in|share this|menu|search|navigation)\b",
    re.IGNORECASE,
)


def _is_chrome(para: str) -> bool:
    """Heuristic: drop short boilerplate lines common in web-exported PDFs."""
    return bool(_CHROME_PATTERNS.match(para)) and len(para) < 60


# ----------------------------------------------------------------------------
# Tokenization (uses the real embedding-model tokenizer)
# ----------------------------------------------------------------------------

class TokenCounter:
    """Counts tokens with the same tokenizer the embedder uses, so 'tokens'
    means exactly what all-MiniLM-L6-v2 sees."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self._tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def tail(self, text: str, n_tokens: int) -> str:
        """Return the last ~n_tokens of `text` as a clean substring of the original.

        Uses the fast tokenizer's offset mapping to find where the last n_tokens
        begin, then slices the *original* text (snapped forward to a word
        boundary). This avoids WordPiece artifacts ("##s") and respaced
        punctuation that decoding token ids would introduce, keeping the overlap
        prefix readable and a faithful substring of the source chunk."""
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        if len(offsets) <= n_tokens:
            return text
        start = offsets[-n_tokens][0]
        # If we landed mid-word, advance to the next whitespace for a clean start.
        if start > 0 and not text[start - 1].isspace():
            nxt = text.find(" ", start)
            if nxt != -1:
                start = nxt + 1
        return text[start:].strip()

    def hard_split(self, text: str, max_tokens: int) -> list[str]:
        """Last-resort split of a too-long sentence into <= max_tokens windows."""
        ids = self._tok.encode(text, add_special_tokens=False)
        return [
            self._tok.decode(ids[i : i + max_tokens]).strip()
            for i in range(0, len(ids), max_tokens)
        ]


# ----------------------------------------------------------------------------
# 4. Chunking
# ----------------------------------------------------------------------------

@dataclass
class _Block:
    """A paragraph-sized unit guaranteed to fit within chunk_size."""
    text: str
    page: int
    tokens: int


def _split_paragraphs(pages: Iterable[Page]) -> list[tuple[str, int]]:
    """Flatten pages into (paragraph_text, page_number) tuples in reading order."""
    out: list[tuple[str, int]] = []
    for pg in pages:
        for para in pg.text.split("\n\n"):
            para = para.strip()
            if para:
                out.append((para, pg.page))
    return out


def _to_blocks(
    paragraphs: list[tuple[str, int]], counter: TokenCounter, max_block: int
) -> list[_Block]:
    """
    Turn paragraphs into blocks that each fit inside max_block tokens.

    max_block is the *content* budget (chunk_size - overlap), so that a single
    block plus the overlap prefix still fits within chunk_size. Paragraphs over
    the budget are split on sentence boundaries, and any single sentence still
    over the limit is hard-split by tokens.
    """
    blocks: list[_Block] = []
    for para, page in paragraphs:
        n = counter.count(para)
        if n <= max_block:
            blocks.append(_Block(para, page, n))
            continue

        # Oversized paragraph: pack sentences up to the budget.
        buf, buf_tokens = [], 0
        for sent in _SENTENCE_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            st = counter.count(sent)
            if st > max_block:
                # Flush buffer, then hard-split this giant sentence.
                if buf:
                    blocks.append(_Block(" ".join(buf), page, buf_tokens))
                    buf, buf_tokens = [], 0
                for piece in counter.hard_split(sent, max_block):
                    blocks.append(_Block(piece, page, counter.count(piece)))
            elif buf_tokens + st > max_block:
                blocks.append(_Block(" ".join(buf), page, buf_tokens))
                buf, buf_tokens = [sent], st
            else:
                buf.append(sent)
                buf_tokens += st
        if buf:
            blocks.append(_Block(" ".join(buf), page, buf_tokens))
    return blocks


def chunk_documents(
    pages: list[Page],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    counter: TokenCounter | None = None,
) -> list[Chunk]:
    """
    Split extracted pages into <=chunk_size-token, paragraph-aware chunks where
    consecutive chunks share ~overlap tokens.

    Strategy (sliding window over paragraph blocks):
      - Greedily pack whole paragraph blocks into a chunk until the next block
        would exceed the budget -> forward boundaries fall between paragraphs
        whenever possible (paragraph-aware).
      - Overlap is taken at the TOKEN level: each new chunk begins with the last
        ~overlap tokens of the previous chunk's text. This guarantees real
        overlap even when paragraphs are large (a paragraph-boundary-only carry
        would otherwise yield zero overlap for single-paragraph chunks).
      - Blocks are pre-sized to chunk_size - overlap so that prefix + one block
        never exceeds chunk_size (and thus never the model's truncation limit).

    Metadata per chunk: source, page (primary), page_start, page_end,
    token_count, chunk_index.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if counter is None:
        counter = TokenCounter()
    if chunk_size > MODEL_MAX_TOKENS:
        log.warning(
            "chunk_size=%d exceeds %s max_seq_length=%d -> the embedder will "
            "TRUNCATE each chunk to %d tokens. Retrieval will ignore the tail. "
            "Use --chunk-size %d to match the model.",
            chunk_size, EMBED_MODEL, MODEL_MAX_TOKENS, MODEL_MAX_TOKENS,
            MODEL_MAX_TOKENS,
        )

    if not pages:
        return []
    source = pages[0].source
    content_budget = chunk_size - overlap
    blocks = _to_blocks(_split_paragraphs(pages), counter, content_budget)

    chunks: list[Chunk] = []
    carry_text = ""          # token-level overlap prefix from the previous chunk
    carry_page: int | None = None
    i, n = 0, len(blocks)
    while i < n:
        carry_tokens = counter.count(carry_text) if carry_text else 0

        # Pack blocks [i:j) on top of the carried overlap (always take >= 1 block).
        cur: list[_Block] = []
        used = carry_tokens
        j = i
        while j < n and (not cur or used + blocks[j].tokens <= chunk_size):
            cur.append(blocks[j])
            used += blocks[j].tokens
            j += 1

        body = "\n\n".join(b.text for b in cur)
        text = f"{carry_text}\n\n{body}" if carry_text else body
        pages_in_chunk = ([carry_page] if carry_page is not None else []) + [b.page for b in cur]
        idx = len(chunks)
        chunks.append(
            Chunk(
                id=f"{source}::chunk{idx:04d}",
                text=text,
                metadata={
                    "source": source,
                    "page": pages_in_chunk[0],
                    "page_start": min(pages_in_chunk),
                    "page_end": max(pages_in_chunk),
                    "token_count": counter.count(text),
                    "chunk_index": idx,
                },
            )
        )

        if j >= n:
            break

        # Carry the tail of THIS chunk forward as the next chunk's overlap prefix.
        carry_text = counter.tail(text, overlap)
        carry_page = cur[-1].page
        i = j  # forward boundary advances past consumed blocks

    return chunks


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def build_chunks(
    doc_dir: Path = DEFAULT_DOC_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Full ingestion: load -> extract -> clean -> chunk, across all PDFs."""
    counter = TokenCounter()
    all_chunks: list[Chunk] = []
    for pdf_path in load_pdfs(doc_dir):
        pages = extract_text(pdf_path)
        chunks = chunk_documents(pages, chunk_size, overlap, counter)
        log.info("%-32s %3d page(s) -> %3d chunk(s)", pdf_path.name, len(pages), len(chunks))
        all_chunks.extend(chunks)
    log.info("TOTAL: %d chunks from %s", len(all_chunks), doc_dir.name)
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and chunk PDFs for the RAG pipeline.")
    parser.add_argument("--doc-dir", type=Path, default=DEFAULT_DOC_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "data" / "chunks.json",
                        help="Where to write chunks as JSON for the embedding stage.")
    args = parser.parse_args()

    chunks = build_chunks(args.doc_dir, args.chunk_size, args.overlap)
    args.out.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %d chunks -> %s", len(chunks), args.out)


if __name__ == "__main__":
    main()
