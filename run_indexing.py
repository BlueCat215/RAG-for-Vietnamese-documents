"""
Script chạy toàn bộ quy trình tiền xử lý và lập chỉ mục:
Raw Data (PDF/DOCX/TXT) -> Extract -> Clean -> Chunk (S3) -> FAISS Indexing (GPU) -> Lưu đĩa
"""

import os
import sys
import glob
import time

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

from src.utils import load_config
from src.extract import extract_document
from src.chunking import chunk_structural, save_chunks_to_jsonl
from src.index import VectorIndex


def run_pipeline_indexing(config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)
    raw_dir = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    index_dir = cfg["paths"]["index_dir"]
    emb_cfg = cfg["models"]["embedding"]
    chunk_cfg = cfg["chunking"]

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 BƯỚC 1: TRÍCH XUẤT VÀ LÀM SẠCH VĂN BẢN (EXTRACT & CLEAN)")
    print("=" * 60)

    # Lấy toàn bộ file PDF, DOCX, TXT trong data/raw/
    supported_exts = ["*.pdf", "*.docx", "*.txt"]
    file_paths = []
    for ext in supported_exts:
        file_paths.extend(glob.glob(os.path.join(raw_dir, ext)))

    if not file_paths:
        print(f"[!] LỖI: Không tìm thấy tài liệu nào trong '{raw_dir}'!")
        print("    Vui lòng copy các file .pdf, .docx, hoặc .txt vào data/raw/ trước khi chạy.")
        return

    print(f"[*] Tìm thấy {len(file_paths)} tài liệu. Bắt đầu xử lý...")
    all_pages_data = []

    for fpath in file_paths:
        fname = os.path.basename(fpath)
        print(f"  -> Đang trích xuất: {fname}...")
        try:
            pages = extract_document(fpath)
            all_pages_data.extend(pages)
            print(f"     [+] Thành công: {len(pages)} trang/đoạn.")
        except Exception as e:
            print(f"     [!] Lỗi khi xử lý {fname}: {e}")

    print("\n" + "=" * 60)
    print("🚀 BƯỚC 2: PHÂN ĐOẠN THEO CẤU TRÚC ĐIỀU/KHOẢN (CHUNKING S3)")
    print("=" * 60)
    t0 = time.time()
    chunks = chunk_structural(all_pages_data, max_tokens=chunk_cfg.get("max_tokens", 512))
    print(f"[+] Đã tạo ra tổng cộng {len(chunks)} chunks (Thời gian: {time.time() - t0:.2f}s).")

    # Lưu chunks ra data/processed/
    chunks_output_path = os.path.join(processed_dir, "chunks_structural.jsonl")
    save_chunks_to_jsonl(chunks, chunks_output_path)
    print(f"[+] Đã lưu danh sách chunk vào: {chunks_output_path}")

    print("\n" + "=" * 60)
    print("🚀 BƯỚC 3: TÍNH VECTOR EMBEDDING VÀ DỰNG FAISS INDEX")
    print("=" * 60)
    idx = VectorIndex(model_name=emb_cfg["name"], device=emb_cfg.get("device", "cuda"))
    idx.build_index(
        chunks=chunks,
        batch_size=emb_cfg.get("batch_size", 8),
        output_dir=index_dir
    )

    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT XÂY DỰNG INDEX THÀNH CÔNG!")
    print(f"   Dữ liệu sẵn sàng tại: {index_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_indexing()
