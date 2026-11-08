"""
Script chạy thử nghiệm hỏi đáp dòng lệnh (CLI Interactive Test)
Dùng để kiểm tra nhanh RAG Pipeline sau khi đã build index xong.
"""

import sys
import io
import re

# Đảm bảo UTF-8 toàn diện trên môi trường Windows
if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.pipeline import RAGPipeline
from src import logger as rag_logger


def annotate_answer(answer_text: str, sources: list) -> str:
    """Gắn chú giải vị trí nguồn (Điều/Khoản, Trang, Tài liệu) vào cuối các câu trong câu trả lời."""
    if not sources:
        return answer_text

    cmap = {}
    for s in sources:
        cid = s.get("citation_id")
        if cid is None:
            continue
        doc = s.get("doc_id", "Tài liệu").replace(".txt", "").replace(".docx", "").replace(".pdf", "")
        clause = s.get("clause", "")
        heading = s.get("heading", "")
        p_start = s.get("page_start", 1)
        p_end = s.get("page_end", p_start)
        page_str = f"tr. {p_start}" if p_start == p_end else f"tr. {p_start}-{p_end}"

        loc_items = []
        if clause:
            loc_items.append(clause)
        elif heading:
            loc_items.append(heading)
        loc_items.append(page_str)
        loc_items.append(f"tài liệu: {doc}")

        cmap[cid] = f"[{cid}: {' | '.join(loc_items)}]"

    def replacer(m):
        n = int(m.group(1))
        if n in cmap:
            return f" [{cmap[n]}]"
        return m.group(0)

    annotated = re.sub(r"\[(\d+)\]", replacer, answer_text)
    if not re.search(r"\[\d+\]", answer_text) and sources:
        refs = [cmap[s["citation_id"]] for s in sources[:3] if s.get("citation_id") in cmap]
        if refs:
            annotated += "\n\n(Căn cứ tham chiếu: " + " • ".join(refs) + ")"

    return annotated


def main():
    rag_logger.setup_logger(log_dir="logs")
    print("=" * 60)
    print("💬 HỆ THỐNG HỎI ĐÁP QUY CHẾ ĐÀO TẠO ĐẠI HỌC (CLI TEST)")
    print("=" * 60)

    pipeline = RAGPipeline(config_path="config/config.yaml")
    try:
        pipeline.initialize()
    except Exception as e:
        print(f"[!] Lỗi khi nạp pipeline: {e}")
        print("    => Bạn đã chạy lệnh 'python run_indexing.py' để dựng chỉ mục chưa?")
        return

    print("\n[+] Sẵn sàng! Gõ câu hỏi của bạn (hoặc gõ 'q' để thoát):")
    print("-" * 60)

    while True:
        try:
            query = input("\n유 Đặt câu hỏi: ").strip()
            if not query:
                continue

            if not sys.stdin.isatty():
                print(query)

            if query.lower() in ["q", "exit", "quit"]:
                print("Tạm biệt!")
                break

            print("[*] Đang truy xuất dữ liệu & sinh câu trả lời...")
            res = pipeline.query(query, method="hybrid", top_k=5, use_rerank=True)

            print("\n(•‿•) Trợ lý trả lời:")
            display_text = annotate_answer(res["answer"], res.get("sources", []))
            print(display_text)

            lat = res["latency_ms"]
            print(f"\n⏱️ Thời gian phản hồi: {lat.get('total_ms', 0):.0f}ms (Retrieve: {lat.get('retrieve_ms', 0):.0f}ms, Rerank: {lat.get('rerank_ms', 0):.0f}ms, LLM: {lat.get('generate_ms', 0):.0f}ms)")

            if res.get("sources"):
                print("\n📚 Nguồn trích dẫn chi tiết:")
                for s in res["sources"]:
                    loc = s.get("clause") or s.get("heading") or ""
                    print(f"  [{s['citation_id']}] {s['doc_id']} - {loc} (Trang {s.get('page_start', 1)}, Điểm: {s.get('score', 0):.3f})")
            elif res.get("refused"):
                print(f"  ℹ️ Lý do từ chối: {res.get('refusal_reason', 'Không đủ dữ liệu trong context')}")

            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break
        except Exception as e:
            print(f"[!] Có lỗi xảy ra trong quá trình xử lý: {e}")


if __name__ == "__main__":
    main()
