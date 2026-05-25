import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import os
import logging

BM25_SCORE_WEIGHT = float(os.getenv("BM25_SCORE_WEIGHT", os.getenv("BM25_RRF_WEIGHT", "1.0")))
EMBEDDING_SCORE_WEIGHT = float(os.getenv("EMBEDDING_SCORE_WEIGHT", os.getenv("EMBEDDING_RRF_WEIGHT", "1.0")))


class BaseRetriever:

    def __init__(self, chunks, chroma_collection=None):
        self.chunks = [dict(chunk) for chunk in chunks]
        self.chroma_collection = chroma_collection
        logging.info(f"Total chunks: {len(self.chunks)}")

    def _target_scope_pool(self, target_year):
        return [c for c in self.chunks if c.get("year") is not None and target_year == c["year"]]

    def retrieve(self, variable_name, query, target_year, top_k=5, **kwargs):
        raise NotImplementedError()

    def _normalize_score_map(self, score_map):
        if not score_map:
            return {}
        
        values = list(score_map.values())
        min_v, max_v = min(values), max(values)
        span = max_v - min_v
        return {k: (1.0 if v > 0 else 0.0) if span <= 1e-12 else (v - min_v) / span for k, v in score_map.items()}


class BM25Retriever(BaseRetriever):
    def retrieve(self, variable_name, query, target_year, top_k=5, **kwargs):
        pool = self._target_scope_pool(target_year)
        if not pool:
            return []
        
        tokenized = [(c.get("text", "")).split() for c in pool]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores((query or "").split())
        score_map = {i: float(score) for i, score in enumerate(scores)}
        norm = self._normalize_score_map(score_map)
        ranked_idx = sorted(norm.keys(), key=lambda i: norm.get(i, 0.0), reverse=True)[:top_k]
        ranked = []

        for _, idx in enumerate(ranked_idx, start=1):
            item = dict(pool[idx])
            item["bm25_score"] = round(norm.get(idx, 0.0), 6)
            ranked.append(item)

        return ranked
    

class EmbeddingRetriever(BaseRetriever):
    def retrieve(self, variable_name, query, target_year, top_k=5, query_embedding=None, **kwargs):

        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ranked = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            item = dict(meta or {})
            item["text"] = doc
            item["embedding_score"] = dist
            ranked.append(item)

        return ranked


class HybridRetriever(BaseRetriever):
    def retrieve(self, variable_name, query, target_year, top_k=5, bm25_query=None, embedding_query=None, table_bonus=True, **kwargs):
        pool = self._target_scope_pool(target_year)
        if not pool:
            return []
        bm25_q = bm25_query or query

        # BM25
        tokenized = [(c.get("text", "")).split() for c in pool]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores((bm25_q or "").split())
        bm25_map = {i: float(score) for i, score in enumerate(bm25_scores)}
        bm25_norm = self._normalize_score_map(bm25_map)

        # Embedding (ChromaDB)
        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=len(pool),
            include=["metadatas", "distances"]
        )

        emb_map = {}
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for meta, dist in zip(metas, dists):
            idx = meta.get("chunk_index") if isinstance(meta, dict) else None
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(pool):
                emb_map[idx] = float(dist)
        emb_norm = self._normalize_score_map(emb_map)

        candidate_indices = sorted(set(bm25_norm.keys()) | set(emb_norm.keys()))
        fused_scores = {}

        for idx in candidate_indices:
            if table_bonus and pool[idx].get("chunk_nature") == "table_main":
                bm25_weight = BM25_SCORE_WEIGHT * 2.0
            else:
                bm25_weight = BM25_SCORE_WEIGHT

            bm25_component = bm25_weight * bm25_norm.get(idx, 0.0)
            emb_component = EMBEDDING_SCORE_WEIGHT * emb_norm.get(idx, 0.0)
            fused_scores[idx] = bm25_component + emb_component

        ranked_idx = sorted(candidate_indices, key=lambda i: fused_scores.get(i, 0.0), reverse=True)[:top_k]
        ranked = []
        for final_rank, idx in enumerate(ranked_idx, start=1):
            item = dict(pool[idx])
            item["hybrid_embedding_score"] = round(fused_scores.get(idx, 0.0), 6)
            item["hybrid_embedding_rank"] = final_rank
            item["bm25_score"] = round(bm25_norm.get(idx, 0.0), 6)
            item["embedding_score"] = round(emb_norm.get(idx, 0.0), 6)
            ranked.append(item)

        return ranked


class RetrieverFactory:
    @staticmethod
    def get_retriever(retriever_type, chunks, chroma_collection=None):
        if retriever_type == "bm25":
            return BM25Retriever(chunks, chroma_collection)
        elif retriever_type == "embedding":
            return EmbeddingRetriever(chunks, chroma_collection)
        elif retriever_type == "hybrid":
            return HybridRetriever(chunks, chroma_collection)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")