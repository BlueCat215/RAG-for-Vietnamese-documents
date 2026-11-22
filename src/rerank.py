"""Module Cross-Encoder Reranker cho tiếng Việt (thanhtantran/Vietnamese_Reranker) đã chuẩn hóa Sigmoid."""

import numpy as np
import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


def sigmoid(x: float) -> float:
    """Hàm kích hoạt Sigmoid đưa logit thô của Cross-Encoder về khoảng xác suất [0, 1]."""
    return float(1.0 / (1.0 + np.exp(-float(x))))


class VietnameseReranker:
    def __init__(self, model_name: str = "thanhtantran/Vietnamese_Reranker", device: str = "cpu"):
        self.model_name = model_name
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        print(f"[*] Khởi tạo Reranker Model: {model_name} trên {self.device}...")
        self.model = CrossEncoder(model_name, max_length=512, device=self.device)

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_k: int = 5,
        alpha: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        Chấm điểm từng cặp (query, chunk_text) và xếp hạng lại bằng Soft Re-ranking chuẩn hóa:
            r_prob = sigmoid(r_raw)
            soft_score = alpha * r_prob + (1 - alpha) * dense_score
        Đảm bảo cả r_prob và dense_score đều nằm trong [0, 1] trước khi kết hợp lồi.
        """
        if not candidate_chunks:
            return []

        pairs = [(query, c["text"]) for c in candidate_chunks]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        reranked_chunks = []
        for chunk, score in zip(candidate_chunks, raw_scores):
            c_copy = dict(chunk)
            r_raw = float(score)
            r_prob = sigmoid(r_raw)
            d_score = float(chunk.get("dense_score", 0.0))

            # Kết hợp hai phân phối xác suất đã chuẩn hóa [0, 1]
            if d_score > 0.0:
                soft_score = alpha * r_prob + (1.0 - alpha) * d_score
            else:
                soft_score = r_prob

            c_copy["rerank_raw"] = r_raw
            c_copy["rerank_prob"] = r_prob
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
