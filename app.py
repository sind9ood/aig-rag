import streamlit as st
import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from pathlib import Path

from src.config.variable_paths import VARIABLE_PATHS
from src.config.runtime_aliases import build_retrieval_queries
from src.retrieval.extraction import extract_from_metadata
from src.retrieval.retrieve_docs import TableAwareRetriever

VARIABLE_ORDER = list(VARIABLE_PATHS.keys())

CHROMA_PATH = "db"
COLLECTION_NAME = "rag"


def _reconstruct_chunks(collection):
    """Rebuild in-memory chunk dicts from ChromaDB for BM25 and year discovery."""
    result = collection.get(include=["documents", "metadatas"])
    chunks = []
    for text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        summary = str(meta.get("summary") or "").strip()
        page_context = str(meta.get("page_context") or "").strip()
        page_section = str(meta.get("page_section") or "").strip()
        hybrid_embedding_text = str(meta.get("hybrid_embedding_text") or "").strip()
        if not hybrid_embedding_text:
            hybrid_embedding_text = "\n".join([p for p in [summary, page_context, page_section] if p]).strip() or (text or "")
        chunks.append({
            "text": text or "",
            "year": meta.get("year"),
            "section": meta.get("section"),
            "summary": summary,
            "page_context": page_context,
            "page_section": page_section,
            "hybrid_embedding_text": hybrid_embedding_text,
            "table_row_count": meta.get("table_row_count", 0),
            "chunk_nature": meta.get("chunk_nature"),
            "matched_variables": [v for v in meta.get("matched_variables", "").split(",") if v],
            "table_rows": [],
        })
    return chunks


@st.cache_resource(show_spinner=False)
def load_retriever_from_cache():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    chroma_collection = client.get_or_create_collection(name=COLLECTION_NAME)
    chunks = _reconstruct_chunks(chroma_collection)
    retriever = TableAwareRetriever(chunks=chunks, chroma_collection=chroma_collection)
    return chunks, retriever


def available_years(chunks):
    years = sorted({c.get("year") for c in chunks if c.get("year") is not None}, reverse=True)
    if not years:
        return []

    min_year = min(years) - 2
    max_year = max(years)
    return list(range(max_year, min_year - 1, -1))


def extract_variable_for_year(retriever, variable_name, target_year, scope="all"):
    config = VARIABLE_PATHS[variable_name]
    base_query = config.query_template.format(year=target_year-1)
    query_bundle = build_retrieval_queries(variable_name, base_query, target_year=target_year-1)
    bm25_query = query_bundle.get("bm25_query", "")
    embedding_query = query_bundle.get("embedding_query", "")

    docs = retriever.retrieve(
        variable_name=variable_name,
        query=embedding_query,
        target_year=target_year,
        top_k=config.top_k,
        bm25_query=bm25_query,
        embedding_query=embedding_query,
    )

    result = extract_from_metadata(docs, variable_name, target_year, max_docs=config.max_docs)
    retrieved_docs = result.get("retrieved_docs", [])
    supporting_docs = result.get("supporting_docs", [])
    return {
        "value": result.get("value") or result.get("raw_response") or "N/A",
        "retrieved_docs": retrieved_docs,
        "supporting_docs": supporting_docs,
        "base_query": base_query,
        "bm25_query": bm25_query,
        "embedding_query": embedding_query,
    }


def format_retrieved_docs(retrieved_docs):
    if not retrieved_docs:
        return "N/A"

    parts = []
    for doc in retrieved_docs[:3]:
        excerpt = doc.get("excerpt", "").replace("\n", " ").strip()
        # Only show excerpt, not full text or other fields
        parts.append(f"{excerpt[:120]}")
    return "\n".join(parts)


def main():
    load_dotenv()
    st.set_page_config(page_title="AIG 10-K Variable Viewer", layout="wide")
    st.title("AIG 10-K Variable Viewer")
    st.caption("Select a year to retrieve source documents and summarize them with OpenRouter.")

    if not Path(CHROMA_PATH).exists():
        st.error("ChromaDB index not found.")
        st.code(".venv/bin/python index_chunks_to_chroma.py")
        st.caption("Run the command above once, then re-run this app.")
        return

    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("Missing OPENROUTER_API_KEY. Add it to your .env file.")
        return

    with st.spinner("Loading prebuilt chunks and retrieval index..."):
        chunks, retriever = load_retriever_from_cache()

    if not chunks:
        st.error("ChromaDB is present but contains no indexed chunks.")
        st.code(".venv/bin/python index_chunks_to_chroma.py --max-filings 2")
        st.caption("Run the command above to populate the collection, then re-run this app.")
        return

    years = available_years(chunks)
    if not years:
        st.error("No filing years were found in chunk metadata.")
        return

    selected_year = st.selectbox("Target year", years, index=0)

    st.caption("Path config is loaded from src/config/variable_paths.py")

    if st.button("Run extraction", type="primary"):
        rows = []
        query_rows = []
        for variable_name in VARIABLE_ORDER:
            config = VARIABLE_PATHS[variable_name]
            result = extract_variable_for_year(retriever, variable_name, selected_year)
            query_rows.append({
                "Variable": config.display_name,
                "BM25 Query": result["bm25_query"],
                "Embedding Query": result["embedding_query"],
            })
            rows.append(
                {
                    "Variable": config.display_name,
                    "Value": result.get("value") or "N/A",
                    "Supporting Docs": result.get("supporting_docs", []),
                    "Retrieved Docs": format_retrieved_docs(result["retrieved_docs"]),
                }
            )

        st.subheader("Retrieval Queries")
        st.dataframe(query_rows, hide_index=True, use_container_width=True)
        st.subheader(f"Results for {selected_year}")
        st.dataframe(rows, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()