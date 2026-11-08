"""
Module đánh giá định lượng tầng truy hồi (Retrieval Evaluation) - NLP-21.

Tính toán các chỉ số học thuật chuẩn:
- Recall@K (K=1, 3, 5, 10): Tỷ lệ câu hỏi tìm thấy ít nhất 1 đoạn bằng chứng trong top K.
- MRR (Mean Reciprocal Rank): Điểm trung bình nghịch đảo thứ hạng xuất hiện của đoạn bằng chứng đầu tiên.
- nDCG@K: Đánh giá chất lượng xếp hạng có trọng số theo vị trí.
- Phân tách đo riêng cho: BM25, Dense Vector, Hybrid RRF, và Hybrid + Reranker (Ablation Study).
"""

import os
import sys
import math

# Đảm bảo import được module 'src' khi chạy trực tiếp từ thư mục eval
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Cấu hình UTF-8 cho Windows console để tránh UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.utils import load_config, load_jsonl
from src.pipeline import RAGPipeline


def is_hit(chunk: Dict[str, Any], evidences: List[Dict[str, Any]]) -> bool:
    """
    Kiểm tra xem chunk truy hồi được có chứa bằng chứng chuẩn hay không.
    So khớp theo doc_id và (article hoặc nội dung quote).
    """
    if not evidences:
        return False

    chunk_doc = chunk.get("doc_id", "")
    chunk_text = chunk.get("text", "").lower()
    chunk_heading = chunk.get("heading", "").lower()

    for ev in evidences:
        ev_doc = ev.get("doc_id", "")
        # Kiểm tra trùng doc_id
        if ev_doc and ev_doc != chunk_doc:
            continue

        ev_art = (ev.get("article") or "").lower()
        ev_quote = (ev.get("quote") or "").lower()

        # Nếu trùng Điều trong heading
        if ev_art and ev_art in chunk_heading:
            return True

        # Nếu trích dẫn nằm trong nội dung chunk
        if ev_quote and (ev_quote in chunk_text or chunk_text in ev_quote):
            return True

    return False


def calculate_retrieval_metrics(
    retrieved_results: List[List[Dict[str, Any]]],
    ground_truths: List[List[Dict[str, Any]]],
    k_list: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Tính Recall@K, MRR, nDCG@10 cho toàn bộ tập câu hỏi có đáp án.
    """
    recalls = {k: [] for k in k_list}
    mrr_list = []
    ndcg_list = []

    for chunks, evidences in zip(retrieved_results, ground_truths):
        if not evidences: # Bỏ qua câu không có đáp án khi đo retrieval hit
            continue

        # Tìm các vị trí hit (1-based index)
        hit_ranks = []
        for rank, c in enumerate(chunks, 1):
            if is_hit(c, evidences):
                hit_ranks.append(rank)

        # Tính Recall@K
        for k in k_list:
            hit_at_k = any(r <= k for r in hit_ranks)
            recalls[k].append(1.0 if hit_at_k else 0.0)

        # Tính MRR
        if hit_ranks:
            first_hit = min(hit_ranks)
            mrr_list.append(1.0 / first_hit)
        else:
            mrr_list.append(0.0)

        # Tính nDCG@10 chuẩn xác theo lý thuyết IR:
        # IDCG phải được tính từ min(|Rel|, K) chunk đúng ở các thứ hạng lý tưởng đầu tiên (1, 2, ..., n_ideal)
        hits_at_10 = [r for r in hit_ranks if r <= 10]
        if hits_at_10:
            dcg = sum(1.0 / math.log2(r + 1) for r in hits_at_10)
            # Số chunk đúng tối đa có thể xếp vào top 10
            n_ideal = min(max(len(hit_ranks), len(evidences)), 10)
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
            ndcg_list.append(min(1.0, dcg / idcg) if idcg > 0 else 0.0)
        else:
            ndcg_list.append(0.0)

    results = {}
    for k in k_list:
        results[f"Recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
    results["MRR"] = float(np.mean(mrr_list)) if mrr_list else 0.0
    results["nDCG@10"] = float(np.mean(ndcg_list)) if ndcg_list else 0.0

    return results


def run_evaluation(
    split: str = "dev",
    config_path: str = "config/config.yaml"
):
    qa_file = f"data/qa/{split}.jsonl"
    output_csv = f"results/retrieval_ablation_{split}.csv"
    default_csv = "results/retrieval_ablation.csv"

    print("=" * 60)
    print(f"🔬 CHẠY ĐÁNH GIÁ ĐỊNH LƯỢNG TẦNG TRUY HỒI (SPLIT: {split.upper()})")
    print("=" * 60)

    qa_data = load_jsonl(qa_file)
    if not qa_data:
        print(f"[!] Không tìm thấy dữ liệu QA tại {qa_file}")
        return

    # Lọc câu hỏi có đáp án
    eval_samples = [q for q in qa_data if q.get("answerable", True) and q.get("evidence")]
    print(f"[*] Tổng số câu hỏi kiểm thử: {len(eval_samples)}")

    pipeline = RAGPipeline(config_path=config_path)
    # Tầng đánh giá truy hồi chỉ cần BM25, Vector Index và Reranker; tắt nạp trước LLM để chạy nhanh và tiết kiệm VRAM
    pipeline.initialize(preload_llm=False)

    experiments = [
        {"name": "BM25 (Baseline)", "method": "bm25", "use_rerank": False},
        {"name": "Dense Vector", "method": "dense", "use_rerank": False},
        {"name": "Hybrid (RRF)", "method": "hybrid", "use_rerank": False},
        {"name": "Hybrid + Reranker (Full)", "method": "hybrid", "use_rerank": True},
    ]

    all_metrics = []

    for exp in experiments:
        name = exp["name"]
        print(f"\n[*] Đang chạy thí nghiệm: {name}...")
        retrieved_all = []
        gt_all = []

        for sample in eval_samples:
            q = sample["question"]
            evs = sample["evidence"]
            chunks, _ = pipeline.retrieve(
                query=q,
                method=exp["method"],
                top_k=10,
                use_rerank=exp["use_rerank"]
            )
            retrieved_all.append(chunks)
            gt_all.append(evs)

        metrics = calculate_retrieval_metrics(retrieved_all, gt_all, k_list=[1, 3, 5, 10])
        metrics["Method"] = name
        all_metrics.append(metrics)
        print(f"    -> Recall@1: {metrics['Recall@1']:.3f} | Recall@5: {metrics['Recall@5']:.3f} | MRR: {metrics['MRR']:.3f}")

    # Xuất bảng tổng hợp kết quả
    df = pd.DataFrame(all_metrics)
    cols = ["Method", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@10"]
    df = df[cols]

    os.makedirs("results", exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if split == "dev":
        df.to_csv(default_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("📊 BẢNG KẾT QUẢ SO SÁNH PHƯƠNG PHÁP (ABLATION STUDY)")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n[+] Đã lưu bảng số liệu khoa học vào: {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đánh giá tầng truy hồi RAG Tiếng Việt (NLP-21)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="Tập dữ liệu cần đo (dev hoặc test)")
    args = parser.parse_args()

    run_evaluation(split=args.split)
