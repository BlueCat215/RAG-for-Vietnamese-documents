"""
Module đánh giá định lượng tầng truy hồi (Retrieval Evaluation) - NLP-21.

Các tính năng học thuật nâng cao:
- Recall@K (K = 1, 2, 3, 5, 10): Tỷ lệ tìm thấy ít nhất 1 chunk bằng chứng.
- MRR (Mean Reciprocal Rank): Điểm nghịch đảo thứ hạng chunk trúng đầu tiên.
- nDCG@10: Đánh giá xếp hạng phân cấp chuẩn xác (đảm bảo nghiêm ngặt trong [0, 1]).
- Latency (ms): Thời gian truy hồi trung bình và p95 (95th percentile).
- Phân tích tách nhóm (Category Breakdown):
  + Nhóm Lexical / Numeric (từ khóa trực tiếp, số hiệu, Điều khoản)
  + Nhóm Paraphrase / Colloquial (khẩu ngữ sinh viên, đo năng lực ngữ nghĩa)
- Bootstrap 95% Confidence Interval (CI) qua 1000 lần lấy mẫu lại có hoàn lại.
- So sánh cặp (Paired Bootstrap Test): Đo mức độ cải thiện delta và p-value có ý nghĩa thống kê.
"""

import os
import sys
import math
import time
import argparse
from typing import List, Dict, Any, Tuple

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


def calculate_per_query_metrics(
    retrieved_chunks: List[Dict[str, Any]],
    evidences: List[Dict[str, Any]],
    k_list: List[int] = [1, 2, 3, 5, 10]
) -> Dict[str, float]:
    """Tính toán chi tiết các metric cho từng câu hỏi đơn lẻ."""
    hit_ranks = []
    for rank, c in enumerate(retrieved_chunks, 1):
        if is_hit(c, evidences):
            hit_ranks.append(rank)

    q_metrics = {}
    for k in k_list:
        q_metrics[f"Recall@{k}"] = 1.0 if any(r <= k for r in hit_ranks) else 0.0

    q_metrics["MRR"] = (1.0 / min(hit_ranks)) if hit_ranks else 0.0

    # nDCG@10 chuẩn xác
    hits_at_10 = [r for r in hit_ranks if r <= 10]
    if hits_at_10:
        dcg = sum(1.0 / math.log2(r + 1) for r in hits_at_10)
        n_ideal = min(max(len(hit_ranks), len(evidences)), 10)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
        q_metrics["nDCG@10"] = min(1.0, dcg / idcg) if idcg > 0 else 0.0
    else:
        q_metrics["nDCG@10"] = 0.0

    return q_metrics


def bootstrap_ci(scores: List[float], n_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Tính 95% Confidence Interval bằng Bootstrap resampling."""
    if not scores or len(scores) < 2:
        return 0.0, 0.0
    arr = np.array(scores)
    boot_means = []
    rng = np.random.RandomState(42)
    n = len(arr)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        boot_means.append(np.mean(arr[idx]))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return lower, upper


def paired_bootstrap_test(scores_a: List[float], scores_b: List[float], n_bootstraps: int = 1000) -> Tuple[float, float, float]:
    """Kiểm định so sánh cặp (Paired Bootstrap Test): Đo delta và p-value."""
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    diff = arr_a - arr_b
    observed_delta = float(np.mean(diff))

    rng = np.random.RandomState(42)
    boot_deltas = []
    n = len(diff)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        boot_deltas.append(np.mean(diff[idx]))

    # p-value: Tỷ lệ mẫu có delta <= 0 (nếu kỳ vọng A > B)
    p_val = float(np.mean(np.array(boot_deltas) <= 0.0))
    lower = float(np.percentile(boot_deltas, 2.5))
    upper = float(np.percentile(boot_deltas, 97.5))
    return observed_delta, p_val, (lower, upper)


def run_evaluation(
    split: str = "dev",
    config_path: str = "config/config.yaml"
):
    qa_file = f"data/qa/{split}.jsonl"
    output_csv = f"results/retrieval_ablation_{split}.csv"
    breakdown_csv = f"results/retrieval_breakdown_{split}.csv"

    print("=" * 75)
    print(f"🔬 ĐÁNH GIÁ ĐỊNH LƯỢNG TẦNG TRUY HỒI & SO SÁNH PHƯƠNG PHÁP ({split.upper()})")
    print("=" * 75)

    qa_data = load_jsonl(qa_file)
    if not qa_data:
        print(f"[!] Không tìm thấy dữ liệu QA tại {qa_file}")
        return

    # Lọc câu hỏi có đáp án
    eval_samples = [q for q in qa_data if q.get("answerable", True) and q.get("evidence")]
    print(f"[*] Tổng số câu hỏi kiểm thử có đáp án: {len(eval_samples)} câu")

    pipeline = RAGPipeline(config_path=config_path)
    pipeline.initialize(preload_llm=False)

    experiments = [
        {"name": "BM25 (Baseline)", "method": "bm25", "use_rerank": False},
        {"name": "Dense Vector", "method": "dense", "use_rerank": False},
        {"name": "Hybrid (Weighted RRF)", "method": "hybrid", "use_rerank": False},
        {"name": "Hybrid + Reranker (Đề xuất)", "method": "hybrid", "use_rerank": True},
    ]

    k_list = [1, 2, 3, 5, 10]
    exp_results = {}

    for exp in experiments:
        name = exp["name"]
        print(f"\n[*] Đang chạy thí nghiệm: {name}...")
        per_query_data = []
        latencies = []

        for sample in eval_samples:
            q = sample["question"]
            evs = sample["evidence"]
            p_level = sample.get("paraphrase_level", "lexical-overlap")

            t0 = time.time()
            chunks, timings = pipeline.retrieve(
                query=q,
                method=exp["method"],
                top_k=10,
                use_rerank=exp["use_rerank"]
            )
            lat_ms = (time.time() - t0) * 1000
            latencies.append(lat_ms)

            q_met = calculate_per_query_metrics(chunks, evs, k_list=k_list)
            q_met["qid"] = sample["qid"]
            q_met["paraphrase_level"] = p_level
            q_met["latency_ms"] = lat_ms
            per_query_data.append(q_met)

        exp_results[name] = {
            "per_query": per_query_data,
            "latencies": latencies
        }

    # ==============================================================================
    # 1. BẢNG KẾT QUẢ TỔNG THỂ (OVERALL ABLATION + BOOTSTRAP CI + LATENCY)
    # ==============================================================================
    summary_rows = []
    for exp in experiments:
        name = exp["name"]
        pq = exp_results[name]["per_query"]
        lats = exp_results[name]["latencies"]

        row = {"Method": name}
        for k in k_list:
            row[f"Recall@{k}"] = round(float(np.mean([x[f"Recall@{k}"] for x in pq])), 4)
        mrr_val = float(np.mean([x["MRR"] for x in pq]))
        row["MRR"] = round(mrr_val, 4)
        mrr_ci_low, mrr_ci_high = bootstrap_ci([x["MRR"] for x in pq])
        row["MRR (95% CI)"] = f"[{mrr_ci_low:.3f}, {mrr_ci_high:.3f}]"

        row["nDCG@10"] = round(float(np.mean([x["nDCG@10"] for x in pq])), 4)
        row["Latency_mean_ms"] = round(float(np.mean(lats)), 1)
        row["Latency_p95_ms"] = round(float(np.percentile(lats, 95)), 1)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    os.makedirs("results", exist_ok=True)
    df_summary.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print("📊 BẢNG 1: TỔNG HỢP SO SÁNH PHƯƠNG PHÁP (OVERALL RETRIEVAL ABLATION)")
    print("=" * 75)
    cols_display = ["Method", "Recall@1", "Recall@2", "Recall@3", "Recall@5", "Recall@10", "MRR", "MRR (95% CI)", "nDCG@10", "Latency_mean_ms", "Latency_p95_ms"]
    print(df_summary[cols_display].to_string(index=False))
    print(f"[+] Đã lưu vào: {output_csv}")

    # ==============================================================================
    # 2. BẢNG PHÂN TÍCH TÁCH NHÓM (BREAKDOWN BY QUESTION TYPE)
    # ==============================================================================
    breakdown_rows = []
    categories = [
        ("lexical-overlap", "Nhóm Từ khóa / Số liệu (Lexical / Numeric)"),
        ("paraphrase", "Nhóm Khẩu ngữ tự nhiên (Paraphrase / Colloquial)")
    ]

    for cat_key, cat_name in categories:
        for exp in experiments:
            name = exp["name"]
            pq_cat = [x for x in exp_results[name]["per_query"] if x["paraphrase_level"] == cat_key]
            if not pq_cat:
                continue
            b_row = {
                "Category": cat_name,
                "Sample Count": len(pq_cat),
                "Method": name,
                "Recall@1": round(float(np.mean([x["Recall@1"] for x in pq_cat])), 4),
                "Recall@3": round(float(np.mean([x["Recall@3"] for x in pq_cat])), 4),
                "Recall@5": round(float(np.mean([x["Recall@5"] for x in pq_cat])), 4),
                "MRR": round(float(np.mean([x["MRR"] for x in pq_cat])), 4),
                "nDCG@10": round(float(np.mean([x["nDCG@10"] for x in pq_cat])), 4),
            }
            breakdown_rows.append(b_row)

    df_breakdown = pd.DataFrame(breakdown_rows)
    df_breakdown.to_csv(breakdown_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print("📊 BẢNG 2: PHÂN TÍCH TÁCH THEO LOẠI CÂU HỎI (CATEGORY BREAKDOWN)")
    print("=" * 75)
    print(df_breakdown.to_string(index=False))
    print(f"[+] Đã lưu vào: {breakdown_csv}")

    # ==============================================================================
    # 3. KIỂM ĐỊNH SO SÁNH CẶP (PAIRED BOOTSTRAP TEST: RERANKER VS OTHERS)
    # ==============================================================================
    print("\n" + "=" * 75)
    print("🔬 BẢNG 3: KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ SO SÁNH CẶP (PAIRED TEST TRÊN MRR)")
    print("=" * 75)
    target_exp = "Hybrid + Reranker (Đề xuất)"
    target_mrrs = [x["MRR"] for x in exp_results[target_exp]["per_query"]]

    for base_exp in ["BM25 (Baseline)", "Dense Vector", "Hybrid (Weighted RRF)"]:
        base_mrrs = [x["MRR"] for x in exp_results[base_exp]["per_query"]]
        delta, p_val, ci_diff = paired_bootstrap_test(target_mrrs, base_mrrs)
        signif = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else "ns")
        print(f"[*] {target_exp} vs {base_exp}:")
        print(f"    -> Δ MRR: +{delta:.4f} | 95% CI of Δ: [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}] | p-value: {p_val:.4f} ({signif})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá tầng truy hồi RAG Tiếng Việt (NLP-21)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="Tập dữ liệu cần đo (dev hoặc test)")
    args = parser.parse_args()

    run_evaluation(split=args.split)
