import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
import argparse
import json

from src.indexing.chunk_builder import ChunkBuilderFactory, SimpleChunkBuilder, TableAwareChunkBuilder
from src.indexing.filing_chunk_orchestrator import FilingChunkOrchestrator


COLLECTION_NAME = "rag"
logging.basicConfig(level=logging.INFO)

def build_chroma(chunks, chroma_path="db"):
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing_ids = set(collection.get(include=[]).get("ids", []))
    added = 0
    skipped = 0

    for idx, chunk in enumerate(chunks):
        chunk_idx = chunk.get("chunk_index")
        id_suffix = chunk_idx if chunk_idx is not None else idx
        doc_id = f"{chunk.get('year')}_{chunk.get('section', 'unknown')}_{id_suffix}"
        if doc_id in existing_ids:
            skipped += 1
            continue

        summary = chunk.get("summary") or ""
        page_context = chunk.get("page_context") or ""
        page_section = chunk.get("page_section") or ""
        hybrid_embedding_text = "\n".join([p for p in [summary, page_context, page_section] if p]).strip()
        if not hybrid_embedding_text:
            hybrid_embedding_text = chunk.get("text", "")
        metadata = {
            "year": int(chunk.get("year") or 0),
            "section": chunk.get("section") or "",
            "table_row_count": int(chunk.get("table_row_count") or 0),
            "chunk_nature": chunk.get("chunk_nature") or "",
            "summary": summary,
            "page_number": int(chunk.get("page_number") or 0),
            "page_context": page_context,
            "page_section": page_section,
            "page_subsection": chunk.get("page_subsection") or "",
            "page_subsubsection": chunk.get("page_subsubsection") or "",
            "hybrid_embedding_text": hybrid_embedding_text,
        }

        try:
            collection.add(
                ids=[doc_id],
                documents=[chunk.get("text", "")],
                metadatas=[metadata],
            )
            existing_ids.add(doc_id)
            added += 1
            if added % 50 == 0:
                logging.info(f"Progress: added={added}, skipped={skipped}")
        except Exception as exc:
            logging.error(f"Failed to add chunk {doc_id}: {exc}")
            raise

    logging.info(f"ChromaDB done. Added {added} chunks, skipped {skipped} existing.")


def parse_args():
    parser = argparse.ArgumentParser(description="Index chunks into ChromaDB.")
    parser.add_argument("--max-filings", type=int, default=6)
    parser.add_argument("--max-chunk-chars", type=int, default=2000)
    parser.add_argument("--chunk-overlap-chars", type=int, default=400)
    parser.add_argument("--chroma-path", default="db", help="Path to the ChromaDB index")
    parser.add_argument("--chunker", choices=["simple", "table-aware"], default="table-aware", help="Chunk builder type")
    return parser.parse_args()


def main():
    args = parse_args()

    chunk_builder = ChunkBuilderFactory.get_chunk_builder(args.chunker)
    chunk_orchestrator = FilingChunkOrchestrator(chunk_builder=chunk_builder, company_cik="0000005272", filing_form="10-K")
    chunks = chunk_orchestrator.build(max_filings=args.max_filings, 
                                      max_chunk_chars=args.max_chunk_chars, 
                                      chunk_overlap_chars=args.chunk_overlap_chars)
    chunk_stats = chunk_orchestrator.summarize_chunks(chunks)

    logging.info(f"Built {len(chunks)} chunks")

    logging.info("Chunk stats:")
    logging.info(json.dumps(chunk_stats, indent=2))

    # save chunk stats for reference
    stats_path = Path(args.chroma_path) / "chunk_stats.json"
    with open(stats_path, "w") as f:
        json.dump(chunk_stats, f)

    build_chroma(chunks, chroma_path=args.chroma_path)


if __name__ == "__main__":
    main()
