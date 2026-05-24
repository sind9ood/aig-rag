import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import os


BM25_SCORE_WEIGHT = float(os.getenv("BM25_SCORE_WEIGHT", os.getenv("BM25_RRF_WEIGHT", "1.0")))
EMBEDDING_SCORE_WEIGHT = float(os.getenv("EMBEDDING_SCORE_WEIGHT", os.getenv("EMBEDDING_RRF_WEIGHT", "1.0")))

class TableAwareRetriever:
    def __init__(self, chunks, chroma_collection=None):
        self.chunks = [dict(chunk) for chunk in chunks]
        self.chroma_collection = chroma_collection
        import logging
        logging.info(f"Total chunks: {len(self.chunks)}")

    def _target_scope_pool(self, target_year):
        # Always use all chunks in the target year window
        return [
            c for c in self.chunks
            if c.get("year") is not None and target_year == c["year"]
        ]

    def _bm25_score_map(self, pool, query, text_field="text"):
        if not pool:
            return {}

        tokenized = [(c.get(text_field, "") or c.get("text", "")).split() for c in pool]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores((query or "").split())
        return {i: float(score) for i, score in enumerate(scores)}

    def _embedding_similarity_map(self, pool, query, text_field="text"):
        if not pool:
            return {}
        if not (query or "").strip():
            return {}

        # Local ephemeral collection keeps this ranking independent from the persistent index schema.
        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        col = client.get_or_create_collection(name="rank")

        ids = [f"d_{i}" for i in range(len(pool))]
        docs = [c.get(text_field, "") or c.get("text", "") for c in pool]
        metas = [{"idx": i} for i in range(len(pool))]
        col.add(ids=ids, documents=docs, metadatas=metas)

        result = col.query(
            query_texts=[query],
            n_results=len(pool),
            include=["metadatas", "distances"],
        )

        meta_rows = result.get("metadatas", [[]])[0] or []
        distance_rows = result.get("distances", [[]])[0] or []

        similarity_map = {}
        for meta, distance in zip(meta_rows, distance_rows):
            idx = meta.get("idx") if isinstance(meta, dict) else None
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue

            if 0 <= idx < len(pool):
                d = float(distance)
                similarity_map[idx] = 1.0 / (1.0 + max(d, 0.0))

        return similarity_map

    def _normalize_score_map(self, score_map):
        if not score_map:
            return {}

        values = list(score_map.values())
        min_v = min(values)
        max_v = max(values)
        span = max_v - min_v
        if span <= 1e-12:
            return {k: (1.0 if v > 0 else 0.0) for k, v in score_map.items()}
        return {k: (v - min_v) / span for k, v in score_map.items()}

    def _rank_hybrid_embedding_pool(self, pool, bm25_query, embedding_query, top_k):
        if not pool:
            return []

        bm25_raw = self._bm25_score_map(pool, bm25_query, text_field="text")
        emb_similarity_raw = self._embedding_similarity_map(pool, embedding_query, text_field="hybrid_embedding_text")

        bm25_norm = self._normalize_score_map(bm25_raw)
        emb_norm = self._normalize_score_map(emb_similarity_raw)

        candidate_indices = sorted(set(bm25_norm.keys()) | set(emb_norm.keys()))
        fused_scores = {}
        for idx in candidate_indices:
            # add table_main bonus if applicable
            if pool[idx].get("chunk_nature") == "table_main":
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

    def retrieve(
        self,
        variable_name,
        query,
        target_year,
        top_k=5,
        bm25_query=None,
        embedding_query=None,
    ):
        _ = variable_name
        pool = self._target_scope_pool(target_year)

        if not pool:
            return []

        bm25_q = bm25_query or query
        embedding_q = embedding_query or query
        initial_k = max(top_k * 4, 24)
        ranked = self._rank_hybrid_embedding_pool(pool, bm25_q, embedding_q, initial_k)
        return ranked[:top_k]