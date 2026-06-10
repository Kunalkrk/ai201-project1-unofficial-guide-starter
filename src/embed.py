"""
Embedding, vector storage, and retrieval for the Unofficial Guide RAG pipeline.

Pipeline stages covered here (see docs/planning.md):
    3. Embedding Generation  -> embed_chunks()      (sentence-transformers all-MiniLM-L6-v2)
    4. Vector Store          -> store_in_chroma()   (ChromaDB, local persistence, cosine)
    5. Retrieval             -> retrieve()          (cosine similarity, top-k = 5)

Usage:
    # Build (or rebuild) the index from data/chunks.json:
    python src/embed.py --index

    # Query the index:
    python src/embed.py --query "What skills should a data engineer learn first?"

    # Both at once (index then query):
    python src/embed.py --index --query "What is Apache Spark used for?"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("embed")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_chunks"
DEFAULT_TOP_K = 5
BATCH_SIZE = 64

CHUNKS_FILE = Path(__file__).parent.parent / "data" / "chunks.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"


# ----------------------------------------------------------------------------
# 1. Load pre-chunked documents
# ----------------------------------------------------------------------------

def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    """
    Load the JSON produced by ingest.py.

    Each entry is {"id": str, "text": str, "metadata": dict}.
    All metadata values are scalars — ChromaDB requires this.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}\n"
            "Run 'python src/ingest.py' first to generate it."
        )
    chunks = json.loads(path.read_text(encoding="utf-8"))
    log.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


# ----------------------------------------------------------------------------
# 2. Embedding
# ----------------------------------------------------------------------------

def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """
    Generate one embedding vector per chunk using all-MiniLM-L6-v2.

    The model produces 384-dimensional, L2-normalized vectors. Batching keeps
    memory use predictable regardless of corpus size.

    Returns a list of float lists in the same order as `chunks`.
    """
    texts = [c["text"] for c in chunks]
    log.info("Embedding %d chunks in batches of %d ...", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit vectors -> cosine = dot product
        convert_to_numpy=True,
    )

    log.info("Done. Embedding shape: %s", embeddings.shape)
    return embeddings.tolist()


# ----------------------------------------------------------------------------
# 3. Vector store
# ----------------------------------------------------------------------------

def _get_client(chroma_dir: Path = CHROMA_DIR) -> chromadb.PersistentClient:
    """
    Open (or create) a ChromaDB persistent store at chroma_dir.

    PersistentClient writes to disk automatically — no explicit .persist()
    call needed. The directory is created if it does not exist.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def store_in_chroma(
    chunks: list[dict],
    embeddings: list[list[float]],
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Upsert chunks + embeddings into a ChromaDB collection.

    Collection is configured for cosine similarity (hnsw:space = cosine).
    ChromaDB's cosine mode stores HNSW index entries and computes
    `distance = 1 - cosine_similarity` at query time (range [0, 2];
    lower = more similar). retrieve() converts this to a 0-1 score.

    Uses upsert() so re-running is idempotent: existing IDs are updated,
    new ones are inserted, nothing raises a duplicate-key error.

    What gets stored per chunk:
        ids        — the chunk ID string (e.g. "ApacheSpark.pdf::chunk0003")
        embeddings — 384-dim float vector
        documents  — the raw chunk text (returned in retrieval results)
        metadatas  — source, page, page_start, page_end, token_count, chunk_index
    """
    client = _get_client(chroma_dir)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Upsert in batches to keep memory use predictable.
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        log.info("  upserted chunks %d-%d / %d", start + 1, end, total)

    stored = collection.count()
    log.info("Collection '%s' now contains %d vectors.", collection_name, stored)
    return collection


# ----------------------------------------------------------------------------
# 4. Retrieval
# ----------------------------------------------------------------------------

def retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Embed `query` and return the top-k most similar chunks from ChromaDB.

    How ChromaDB retrieves:
        1. The query text is embedded with the same model used during indexing.
        2. ChromaDB searches its HNSW index using cosine distance
           (distance = 1 - cosine_similarity).
        3. Results are returned sorted by ascending distance (best match first).

    The raw ChromaDB distance is converted to a similarity score:
        similarity = 1 - (distance / 2)
    This maps [0, 2] distance to [1.0, 0.0] similarity, where 1.0 = identical.

    Returns a list of dicts (length <= top_k), each containing:
        rank       — 1-based result position
        score      — cosine similarity [0, 1]; higher = more relevant
        text       — the chunk text (what the LLM will receive as context)
        source     — originating PDF filename
        page       — page number (primary)
        page_start — start of page range if chunk spans pages
        page_end   — end of page range if chunk spans pages
        chunk_id   — the full chunk identifier
    """
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns lists-of-lists (one inner list per query).
    # We pass one query, so index [0] unwraps the single-query wrapper.
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    for rank, (chunk_id, text, meta, dist) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        similarity = 1.0 - (dist / 2.0)
        retrieved.append(
            {
                "rank": rank,
                "score": round(similarity, 4),
                "text": text,
                "source": meta.get("source", ""),
                "page": meta.get("page", ""),
                "page_start": meta.get("page_start", ""),
                "page_end": meta.get("page_end", ""),
                "chunk_id": chunk_id,
            }
        )

    return retrieved


# ----------------------------------------------------------------------------
# Convenience: load collection + model once for interactive/generation use
# ----------------------------------------------------------------------------

def load_index(
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBED_MODEL_NAME,
) -> tuple[chromadb.Collection, SentenceTransformer]:
    """
    Load an already-built index and the embedding model.

    Use this in the generation stage (Milestone 5) to avoid re-indexing:

        collection, model = load_index()
        results = retrieve("What is dbt?", collection, model)
    """
    client = _get_client(chroma_dir)
    collection = client.get_collection(name=collection_name)
    model = SentenceTransformer(model_name)
    log.info(
        "Loaded collection '%s' (%d vectors).", collection_name, collection.count()
    )
    return collection, model


# ----------------------------------------------------------------------------
# Build index orchestration
# ----------------------------------------------------------------------------

def build_index(
    chunks_file: Path = CHUNKS_FILE,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBED_MODEL_NAME,
) -> tuple[chromadb.Collection, SentenceTransformer]:
    """
    Full indexing pipeline: load chunks -> embed -> store in ChromaDB.

    Returns the collection and model so they can be used immediately for
    retrieval without re-loading.
    """
    chunks = load_chunks(chunks_file)
    model = SentenceTransformer(model_name)
    embeddings = embed_chunks(chunks, model)
    collection = store_in_chroma(chunks, embeddings, chroma_dir, collection_name)
    return collection, model


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _safe_print(text: str) -> None:
    """Print text, replacing characters unsupported by the terminal encoding."""
    import sys
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
    sys.stdout.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunks into ChromaDB and/or query the index."
    )
    parser.add_argument(
        "--index", action="store_true",
        help="Build (or rebuild) the ChromaDB index from data/chunks.json.",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query string to retrieve top-k chunks.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chunks-file", type=Path, default=CHUNKS_FILE)
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME)
    args = parser.parse_args()

    if not args.index and not args.query:
        parser.print_help()
        return

    if args.index:
        collection, model = build_index(
            args.chunks_file, args.chroma_dir, args.collection
        )
    else:
        log.info("Loading existing index (skipping --index).")
        collection, model = load_index(args.chroma_dir, args.collection)

    if args.query:
        results = retrieve(args.query, collection, model, top_k=args.top_k)
        _safe_print(f'\nQuery: "{args.query}"')
        _safe_print(f"Top {len(results)} results:\n")
        sep = "-" * 80
        for r in results:
            _safe_print(sep)
            _safe_print(f"  Rank {r['rank']}  |  Score: {r['score']:.4f}  |  {r['chunk_id']}")
            _safe_print(f"  Source: {r['source']}  |  Page: {r['page_start']}-{r['page_end']}")
            _safe_print("")
            words, line = r["text"].split(), []
            for word in words:
                if sum(len(w) + 1 for w in line) + len(word) > 76:
                    _safe_print("  " + " ".join(line))
                    line = [word]
                else:
                    line.append(word)
            if line:
                _safe_print("  " + " ".join(line))
            _safe_print("")
        _safe_print(sep)


if __name__ == "__main__":
    main()
