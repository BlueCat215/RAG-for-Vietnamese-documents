"""
Script vẽ đồ thị thực nghiệm định lượng phục vụ báo cáo khoa học và luận văn (NLP-21).
Đọc dữ liệu từ thư mục results/ và xuất các biểu đồ chất lượng cao (300 DPI) vào results/figures/.
"""

import os
import sys
import re

# Cấu hình UTF-8 cho Windows console
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plotting_style():
    """Thiết lập phong cách hiển thị chuẩn cho báo cáo học thuật."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans", "Helvetica"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.autolayout": True,
        "figure.dpi": 300,
    })


def parse_ci(ci_str: str) -> tuple[float, float]:
    """Phân tích chuỗi khoảng tin cậy '[0.756, 0.887]' thành (lower, upper)."""
    match = re.search(r"\[([\d\.]+),\s*([\d\.]+)\]", str(ci_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


def plot_fig1_recall_curve(df: pd.DataFrame, out_dir: str):
    """Biểu đồ 1: Đường cong hội tụ Recall@K (K = 1, 2, 3, 5, 10)."""
    k_cols = ["Recall@1", "Recall@2", "Recall@3", "Recall@5", "Recall@10"]
    k_vals = [1, 2, 3, 5, 10]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    markers = ["o", "s", "D", "^"]
    linestyles = ["--", "-.", "-", ":"]

    for i, row in df.iterrows():
        method_name = row["Method"]
        # Đơn giản hóa tên hiển thị
        label = method_name.replace(" (Baseline)", "").replace(" (Đề xuất)", "")
        recalls = [row[c] * 100 for c in k_cols]

        ax.plot(
            k_vals,
            recalls,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2.2,
            markersize=7,
            label=f"{label} (R@1={row['Recall@1']*100:.1f}%)"
        )

    ax.set_title("Đường cong hội tụ Recall@K trên tập Kiểm thử (Test Set - 84 câu)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Ngưỡng K (Số đoạn văn bản lấy ra)", fontsize=11, labelpad=8)
    ax.set_ylabel("Tỷ lệ tìm thấy bằng chứng - Recall@K (%)", fontsize=11, labelpad=8)
    ax.set_xticks(k_vals)
    ax.set_ylim(55, 100)
    ax.legend(title="Phương pháp", fontsize=9.5, title_fontsize=10, loc="lower right", framealpha=0.95)

    out_path = os.path.join(out_dir, "fig1_recall_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Biểu đồ 1: {out_path}")


def plot_fig2_mrr_ndcg(df: pd.DataFrame, out_dir: str):
    """Biểu đồ 2: So sánh MRR (có Error Bar 95% CI) và nDCG@10."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    methods = [m.replace(" (Baseline)", "").replace(" (Đề xuất)", "") for m in df["Method"]]
    x = np.arange(len(methods))
    width = 0.35

    mrr_vals = df["MRR"].values
    ndcg_vals = df["nDCG@10"].values

    # Tính toán độ dài râu sai số CI
    yerr_lower = []
    yerr_upper = []
    for i, row in df.iterrows():
        low, up = parse_ci(row.get("MRR (95% CI)", ""))
        if low > 0 and up > 0:
            yerr_lower.append(mrr_vals[i] - low)
            yerr_upper.append(up - mrr_vals[i])
        else:
            yerr_lower.append(0.0)
            yerr_upper.append(0.0)
    yerr = [yerr_lower, yerr_upper]

    rects1 = ax.bar(
        x - width / 2,
        mrr_vals,
        width,
        yerr=yerr,
        capsize=5,
        color="#2b5c8f",
        label="MRR (Kèm 95% Bootstrap CI)",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    rects2 = ax.bar(
        x + width / 2,
        ndcg_vals,
        width,
        color="#e28743",
        label="nDCG@10 (Đánh giá xếp hạng chuẩn)",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    # Hiển thị số liệu trên đỉnh cột
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_title("So sánh Chất lượng Xếp hạng: MRR và nDCG@10 (Tập Test)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Điểm số [0.0 - 1.0]", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.5, 0.98)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    out_path = os.path.join(out_dir, "fig2_mrr_ndcg.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Biểu đồ 2: {out_path}")


def plot_fig3_category_breakdown(df_bd: pd.DataFrame, out_dir: str):
    """Biểu đồ 3: Phân tích tách nhóm: Tra cứu Từ khóa vs Khẩu ngữ đời sống."""
    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Tách 2 nhóm
    categories = df_bd["Category"].unique()
    lex_df = df_bd[df_bd["Category"].str.contains("Lexical|Từ khóa", case=False, na=False)]
    para_df = df_bd[df_bd["Category"].str.contains("Paraphrase|Khẩu ngữ", case=False, na=False)]

    methods = [m.replace(" (Baseline)", "").replace(" (Đề xuất)", "") for m in lex_df["Method"]]
    x = np.arange(len(methods))
    width = 0.35

    lex_mrr = lex_df["MRR"].values
    para_mrr = para_df["MRR"].values

    rects1 = ax.bar(
        x - width / 2,
        lex_mrr,
        width,
        color="#388e3c",
        label="Nhóm Từ khóa / Số liệu (39 câu)",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.88
    )

    rects2 = ax.bar(
        x + width / 2,
        para_mrr,
        width,
        color="#7b1fa2",
        label="Nhóm Khẩu ngữ tự nhiên (45 câu)",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.88
    )

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax.set_title("Năng lực Truy hồi theo Loại câu hỏi: Từ khóa chuẩn vs Khẩu ngữ", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Điểm MRR", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.5, 0.98)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    out_path = os.path.join(out_dir, "fig3_category_breakdown.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Biểu đồ 3: {out_path}")


def plot_fig4_latency_tradeoff(df: pd.DataFrame, out_dir: str):
    """Biểu đồ 4: Đánh đổi Chất lượng (MRR) và Độ trễ (Latency ms) trên thang Log."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    methods = [m.replace(" (Baseline)", "").replace(" (Đề xuất)", "") for m in df["Method"]]
    latencies = df["Latency_mean_ms"].values
    mrrs = df["MRR"].values

    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    markers = ["o", "s", "D", "^"]

    for i in range(len(methods)):
        m = methods[i]
        x_val = latencies[i]
        y_val = mrrs[i]

        ax.scatter(
            x_val,
            y_val,
            color=colors[i],
            marker=markers[i],
            s=160,
            edgecolors="black",
            linewidth=1.2,
            zorder=4,
            label=f"{m} ({x_val:.1f}ms, MRR={y_val:.3f})"
        )

        # Chú thích vị trí
        offset_x = 1.15
        offset_y = 0.005
        if "Weighted RRF" in m:
            ax.annotate(
                f"{m}\n(Điểm tối ưu: Nhanh hơn 61x)",
                xy=(x_val, y_val),
                xytext=(x_val * 0.45, y_val + 0.022),
                arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
                fontsize=9,
                fontweight="bold",
                color="#1b5e20",
                ha="center"
            )
        elif "Reranker" in m:
            ax.annotate(
                f"{m}\n(9,320ms - Nặng CPU)",
                xy=(x_val, y_val),
                xytext=(x_val * 0.4, y_val - 0.03),
                arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
                fontsize=8.5,
                color="#b71c1c",
                ha="center"
            )

    ax.set_xscale("log")
    ax.set_title("Đánh đổi giữa Độ trễ (Latency ms) và Độ chính xác (MRR)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Thời gian phản hồi trung bình (ms, Thang log)", fontsize=11, labelpad=8)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=11, labelpad=8)
    ax.set_ylim(0.70, 0.88)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    out_path = os.path.join(out_dir, "fig4_latency_tradeoff.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Biểu đồ 4: {out_path}")


def plot_fig5_rag_vs_no_rag(df_gen: pd.DataFrame, out_dir: str):
    """Biểu đồ 5: So sánh Không dùng RAG (B0) vs Có dùng RAG (B1, B2, B3)."""
    fig, ax = plt.subplots(figsize=(9.2, 5.5))

    methods = [
        "B0: Không RAG\n(LLM thuần)",
        "B1: RAG\n(BM25)",
        "B2: RAG\n(Dense)",
        "B3: RAG Đề xuất\n(Hybrid + Rerank)"
    ]
    x = np.arange(len(methods))
    width = 0.25

    f1_vals = df_gen["Accuracy F1 (%)"].values
    faith_vals = df_gen["Faithfulness (%)"].values
    cite_vals = df_gen["Citation Rate (%)"].values

    rects1 = ax.bar(
        x - width,
        f1_vals,
        width,
        color="#3b75af",
        label="Độ chính xác F1 Overlap (%)",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    rects2 = ax.bar(
        x,
        faith_vals,
        width,
        color="#2ca02c",
        label="Độ trung thực / Bằng chứng (Faithfulness %)",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    rects3 = ax.bar(
        x + width,
        cite_vals,
        width,
        color="#ff7f0e",
        label="Tỷ lệ trích dẫn nguồn (Citation Rate %)",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    # Hiển thị số liệu trên đỉnh cột
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # Đánh dấu chú thích làm nổi bật luận điểm khoa học
    ax.annotate(
        "B0 (Không RAG):\n0% Bằng chứng\n(Ảo giác 100%)",
        xy=(0, 17),
        xytext=(0.15, 45),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#b71c1c",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#ef9a9a", lw=1)
    )

    ax.set_title("So sánh Đột phá: Khi KHÔNG DÙNG RAG (B0) vs CÓ DÙNG RAG (B1 - B3)", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Tỷ lệ phần trăm (%)", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 118)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    out_path = os.path.join(out_dir, "fig5_rag_vs_no_rag.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Biểu đồ 5: {out_path}")


def main():
    setup_plotting_style()
    out_dir = "results/figures"
    os.makedirs(out_dir, exist_ok=True)

    test_ablation_file = "results/retrieval_ablation_test.csv"
    test_breakdown_file = "results/retrieval_breakdown_test.csv"
    gen_summary_file = "results/generation_summary_dev.csv"

    if not os.path.exists(test_ablation_file) or not os.path.exists(test_breakdown_file):
        print(f"[!] Lỗi: Không tìm thấy file dữ liệu thực nghiệm trong results/")
        return

    df_test = pd.read_csv(test_ablation_file)
    df_bd = pd.read_csv(test_breakdown_file)

    print("=" * 60)
    print("[*] BAT DAU VE BO BIEU DO BAO CAO KHOA HOC (300 DPI)")
    print("=" * 60)

    plot_fig1_recall_curve(df_test, out_dir)
    plot_fig2_mrr_ndcg(df_test, out_dir)
    plot_fig3_category_breakdown(df_bd, out_dir)
    plot_fig4_latency_tradeoff(df_test, out_dir)

    if os.path.exists(gen_summary_file):
        df_gen = pd.read_csv(gen_summary_file)
        plot_fig5_rag_vs_no_rag(df_gen, out_dir)

    print("=" * 60)
    print(f"[+] Hoan tat! Tat ca anh bieu do da duoc luu tai: {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
