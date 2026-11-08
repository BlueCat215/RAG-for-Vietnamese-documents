"""
Module đánh giá định lượng tầng sinh (Generation Evaluation) - NLP-21.

So sánh 4 nhánh cốt lõi (Ablation Study theo chuẩn học thuật RAG):
- B0: No-Retrieval (LLM thuần) - Không đưa ngữ cảnh tài liệu vào prompt.
- B1: BM25 -> LLM - Truy hồi bằng từ khóa chính xác BM25.
- B2: Dense -> LLM - Truy hồi bằng vector embedding ngữ nghĩa (Vietnamese_Embedding).
- B3: Hybrid + Soft Reranker -> LLM (Phương pháp đề xuất) - Kết hợp BM25 + Dense RRF + Reranker.

Các chỉ số đo lường học thuật:
- Accuracy / F1 Overlap: Độ trùng khớp nội dung trả lời với đáp án chuẩn.
- Groundedness / Faithfulness: Tỷ lệ các mệnh đề/từ vựng có căn cứ trích dẫn từ tài liệu.
- Citation Rate: Tỷ lệ câu trả lời có gắn mã trích dẫn [n] chính xác.
- Refusal Accuracy: Tỷ lệ từ chối đúng đối với câu hỏi ngoài phạm vi (answerable = False).
- Over-Refusal Rate: Tỷ lệ từ chối nhầm đối với câu hỏi có trong tài liệu (answerable = True).
- Latency (ms): Thời gian phản hồi chi tiết (truy hồi, rerank, sinh văn bản).
"""

import os
import sys
import re
import time
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

# Đảm bảo import được module 'src'
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

from src.utils import load_config, load_jsonl
from src.pipeline import RAGPipeline


def normalize_vietnamese_text(text: str) -> List[str]:
    """Tách từ đơn giản và loại bỏ ký tự đặc biệt phục vụ đo F1 overlap."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    return tokens


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Tính F1 overlap giữa câu trả lời sinh ra và đáp án chuẩn."""
    pred_tokens = normalize_vietnamese_text(prediction)
    gt_tokens = normalize_vietnamese_text(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(gt_tokens))
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def compute_groundedness(prediction: str, contexts: List[Dict[str, Any]]) -> float:
    """
    Tính Faithfulness / Groundedness:
    Tỷ lệ từ mang thông tin trong câu trả lời xuất hiện trong các đoạn ngữ cảnh trích dẫn.
    """
    if not contexts:
        return 0.0

    context_text = " ".join([c.get("text", "") for c in contexts]).lower()
    pred_tokens = normalize_vietnamese_text(prediction)
    if not pred_tokens:
        return 0.0

    hits = sum(1 for t in set(pred_tokens) if t in context_text)
    return hits / len(set(pred_tokens))


def check_is_refused(text: str) -> bool:
    """Kiểm tra xem câu trả lời có phải là một thông báo từ chối hay không."""
    refusal_keywords = [
        "không tìm thấy thông tin",
        "tài liệu không đề cập",
        "tài liệu không cung cấp",
        "không có thông tin này",
        "không có thông tin trong tài liệu"
    ]
    t = text.lower()
    return any(k in t for k in refusal_keywords) and len(text.strip()) < 200


def run_generation_evaluation(
    split: str = "dev",
    max_samples: int = None,
    config_path: str = "config/config.yaml"
):
    qa_file = f"data/qa/{split}.jsonl"
    print("=" * 70)
    print(f"🤖 ĐÁNH GIÁ ĐỊNH LƯỢNG TẦNG SINH & SO SÁNH BASELINE (SPLIT: {split.upper()})")
    print("=" * 70)

    qa_data = load_jsonl(qa_file)
    if not qa_data:
        print(f"[!] Không tìm thấy tập câu hỏi tại: {qa_file}")
        return

    if max_samples and max_samples < len(qa_data):
        qa_data = qa_data[:max_samples]
        print(f"[*] Chế độ rút gọn: Đánh giá trên {max_samples} câu đầu tiên.")
    else:
        print(f"[*] Tổng số câu hỏi kiểm thử: {len(qa_data)} câu.")

    # Khởi tạo Pipeline hoàn chỉnh (bao gồm nạp LLM vào GPU)
    pipeline = RAGPipeline(config_path=config_path)
    pipeline.initialize(preload_llm=True)

    experiments = [
        {"name": "B0: No-Retrieval (LLM thuần)", "method": "no_retrieval", "use_rerank": False},
        {"name": "B1: BM25 -> LLM", "method": "bm25", "use_rerank": False},
        {"name": "B2: Dense -> LLM", "method": "dense", "use_rerank": False},
        {"name": "B3: Hybrid + Reranker (Đề xuất)", "method": "hybrid", "use_rerank": True},
    ]

    all_predictions = []
    summary_records = []

    for exp in experiments:
        exp_name = exp["name"]
        method = exp["method"]
        use_rerank = exp["use_rerank"]
        print(f"\n[*] Đang chạy thí nghiệm: {exp_name}...")

        f1_scores = []
        groundedness_scores = []
        citation_flags = []
        refusal_true_positives = []
        over_refusal_flags = []
        latencies = []

        for sample in qa_data:
            qid = sample["qid"]
            question = sample["question"]
            ground_truth = sample["answer"]
            answerable = sample.get("answerable", True)

            # Thực thi pipeline
            res = pipeline.query(
                question=question,
                method=method,
                top_k=4,
                use_rerank=use_rerank
            )

            answer = res.get("answer", "")
            contexts = res.get("contexts", [])
            lat = res.get("latency_ms", {}).get("total_ms", 0.0)
            latencies.append(lat)

            # Đánh giá tính từ chối
            is_refused = res.get("refused", False) or check_is_refused(answer)

            if not answerable:
                # Câu ngoài phạm vi: nếu từ chối là đúng (True Positive)
                refusal_true_positives.append(1.0 if is_refused else 0.0)
                f1 = 1.0 if is_refused else 0.0
                g_score = 1.0 if is_refused else 0.0
                has_cite = False
                is_over_refusal = False
            else:
                # Câu có đáp án trong tài liệu:
                if is_refused:
                    # Bị từ chối nhầm (Over-refusal)
                    is_over_refusal = True
                    over_refusal_flags.append(1.0)
                    f1 = 0.0
                    g_score = 0.0
                    has_cite = False
                else:
                    is_over_refusal = False
                    over_refusal_flags.append(0.0)
                    f1 = compute_token_f1(answer, ground_truth)
                    g_score = compute_groundedness(answer, contexts) if contexts else 0.0
                    has_cite = bool(re.search(r"\[\d+\]", answer))

                f1_scores.append(f1)
                groundedness_scores.append(g_score)
                citation_flags.append(1.0 if has_cite else 0.0)

            all_predictions.append({
                "split": split,
                "method": exp_name,
                "qid": qid,
                "question": question,
                "answerable": answerable,
                "ground_truth": ground_truth,
                "generated_answer": answer,
                "is_refused": is_refused,
                "over_refused": is_over_refusal,
                "f1_score": round(f1, 4),
                "groundedness": round(g_score, 4),
                "has_citation": has_cite,
                "latency_ms": round(lat, 1)
            })

        # Tổng kết chỉ số cho phương pháp
        avg_f1 = np.mean(f1_scores) * 100 if f1_scores else 0.0
        avg_groundedness = np.mean(groundedness_scores) * 100 if groundedness_scores else 0.0
        citation_rate = np.mean(citation_flags) * 100 if citation_flags else 0.0
        refusal_acc = np.mean(refusal_true_positives) * 100 if refusal_true_positives else 0.0
        over_refusal_rate = np.mean(over_refusal_flags) * 100 if over_refusal_flags else 0.0
        avg_lat = np.mean(latencies) if latencies else 0.0

        summary_records.append({
            "Method": exp_name,
            "Accuracy F1 (%)": round(avg_f1, 2),
            "Faithfulness (%)": round(avg_groundedness, 2),
            "Citation Rate (%)": round(citation_rate, 2),
            "Refusal Acc (%)": round(refusal_acc, 2),
            "Over-Refusal (%)": round(over_refusal_rate, 2),
            "Avg Latency (ms)": round(avg_lat, 1)
        })

        print(f"    -> F1: {avg_f1:.1f}% | Faithfulness: {avg_groundedness:.1f}% | Refusal Acc: {refusal_acc:.1f}% | Over-Refusal: {over_refusal_rate:.1f}%")

    # Lưu kết quả chi tiết từng câu
    os.makedirs("results", exist_ok=True)
    pred_df = pd.DataFrame(all_predictions)
    pred_path = f"results/generation_predictions_{split}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # Lưu bảng tóm tắt
    sum_df = pd.DataFrame(summary_records)
    sum_path = f"results/generation_summary_{split}.csv"
    sum_df.to_csv(sum_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("📊 BẢNG TỔNG HỢP SO SÁNH CHẤT LƯỢNG TẦNG SINH & BASELINES")
    print("=" * 70)
    print(sum_df.to_string(index=False))
    print(f"\n[+] Đã lưu chi tiết dự đoán từng câu vào: {pred_path}")
    print(f"[+] Đã lưu bảng tổng hợp báo cáo vào:     {sum_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá tầng sinh RAG Tiếng Việt (NLP-21)")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="Tập dữ liệu cần đo (dev hoặc test)")
    parser.add_argument("--max_samples", type=int, default=None, help="Giới hạn số câu hỏi (để chạy smoke test nhanh)")
    args = parser.parse_args()

    run_generation_evaluation(split=args.split, max_samples=args.max_samples)
