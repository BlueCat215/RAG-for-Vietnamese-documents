"""Module Cross-Encoder Reranker cho tiếng Việt (thanhtantran/Vietnamese_Reranker)."""

import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class VietnameseReranker:
    def __init__(self, model_name: str = "thanhtantran/Vietnamese_Reranker", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        print(f"[*] Khởi tạo Reranker Model: {model_name} trên {device}...")
        self.model = CrossEncoder(model_name, max_length=512, device=device)

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_k: int = 5,
        alpha: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        Chấm điểm từng cặp (query, chunk_text) và xếp hạng lại bằng Soft Re-ranking:
            soft_score = alpha * rerank_raw + (1 - alpha) * dense_score
        """
        if not candidate_chunks:
            return []

        pairs = [(query, c["text"]) for c in candidate_chunks]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        reranked_chunks = []
        for chunk, score in zip(candidate_chunks, raw_scores):
            c_copy = dict(chunk)
            r_score = float(score)
            d_score = float(chunk.get("dense_score", 0.0))

            soft_score = alpha * r_score + (1.0 - alpha) * d_score if d_score > 0.0 else r_score

            c_copy["rerank_raw"] = r_score
            c_copy["dense_score"] = d_score
            c_copy["rerank_score"] = float(soft_score)
            reranked_chunks.append(c_copy)

        reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        final_results = []
        for rank, c in enumerate(reranked_chunks[:top_k]):
            c["rank"] = rank + 1
            c["retriever"] = "reranked"
            final_results.append(c)

        return final_results
