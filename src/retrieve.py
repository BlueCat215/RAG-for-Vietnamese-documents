"""
Module truy xuất (Retrieval) cho tài liệu tiếng Việt.
Hỗ trợ BM25 Baseline, Dense Retrieval (FAISS) và Hybrid Search (Weighted RRF).
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
import numpy as np


# ==============================================================================
# Tokenizer cho BM25 tiếng Việt
# ==============================================================================
def tokenize_vietnamese(text: str, mode: str = "dual") -> List[str]:
    """
    Bộ tách từ tiếng Việt tối ưu cho BM25:
    - 'syllable': Tách theo âm tiết đơn lẻ.
    - 'word': Tách từ ghép qua Pyvi / Underthesea.
    - 'dual': Kết hợp âm tiết + từ ghép (tăng Precision mà vẫn bảo toàn Recall).
    """
    text = text.lower().strip()
    syllable_tokens = [w.strip(".,;:?!\"'()[]{}«»/\\-") for w in text.split()]
    syllable_tokens = [w for w in syllable_tokens if w]

    if mode == "syllable":
        return syllable_tokens

    word_tokens = []
    try:
        from pyvi import ViTokenizer
        segmented = ViTokenizer.tokenize(text)
        word_tokens = [w.strip(".,;:?!\"'()[]{}«»/\\-") for w in segmented.split()]
        word_tokens = [w for w in word_tokens if w]
    except ImportError:
        try:
            from underthesea import word_tokenize
            segmented = word_tokenize(text, format="text")
            word_tokens = [w.strip(".,;:?!\"'()[]{}«»/\\-") for w in segmented.split()]
            word_tokens = [w for w in word_tokens if w]
        except ImportError:
            return syllable_tokens

    if mode == "word":
        return word_tokens

    if mode == "dual":
        compounds = [w for w in word_tokens if "_" in w]
        return syllable_tokens + compounds

    return syllable_tokens


# ==============================================================================
# Lớp BM25 Retriever
# ==============================================================================
class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75, mode: str = "dual"):
        self.chunks = chunks
        self.mode = mode
        self.corpus_tokens = [
            tokenize_vietnamese(c["text"], mode=self.mode)
            for c in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Truy xuất top-k chunk bằng BM25."""
        query_tokens = tokenize_vietnamese(query, mode=self.mode)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            chunk["bm25_score"] = float(scores[idx])
            chunk["rank"] = rank + 1
            chunk["retriever"] = "bm25"
            results.append(chunk)

        return results


# ==============================================================================
# Hợp nhất xếp hạng RRF (Reciprocal Rank Fusion)
# ==============================================================================
def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    weights: Optional[List[float]] = None,
    k: int = 20,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Hợp nhất nhiều danh sách kết quả (BM25 + Dense) bằng Weighted RRF:
        RRF_score(d) = sum(w_r * 1 / (k + rank_r(d)))
    Bảo toàn dense_score để phục vụ Soft Re-ranking.
    """
    if weights is None or len(weights) != len(results_list):
        weights = [1.0] * len(results_list)

    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for results, w in zip(results_list, weights):
        for rank, item in enumerate(results):
            c_id = item["chunk_id"]
            if c_id not in chunk_map:
                chunk_map[c_id] = dict(item)

            if item.get("retriever") == "dense":
                chunk_map[c_id]["dense_score"] = float(item.get("score", 0.0))
            elif "dense_score" not in chunk_map[c_id] and "dense_score" in item:
                chunk_map[c_id]["dense_score"] = float(item["dense_score"])

            if item.get("retriever") == "bm25":
                chunk_map[c_id]["bm25_score"] = float(item.get("score", 0.0))
            elif "bm25_score" not in chunk_map[c_id] and "bm25_score" in item:
                chunk_map[c_id]["bm25_score"] = float(item["bm25_score"])

            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (w / (k + (rank + 1)))

    # Sắp xếp theo điểm RRF giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    final_results = []
    for rank, (c_id, score) in enumerate(sorted_items):
        res = dict(chunk_map[c_id])
        res["score"] = score
        res["rank"] = rank + 1
        res["retriever"] = "hybrid_rrf"
        final_results.append(res)

    return final_results
