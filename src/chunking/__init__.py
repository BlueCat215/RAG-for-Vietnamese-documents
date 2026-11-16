"""
Package Chunking cho tài liệu pháp quy tiếng Việt (NLP-21).

Cung cấp đầy đủ các chiến lược cắt đoạn:
- chunk_structural: Cắt theo Chương / Điều / Khoản (S3 - chiến lược chính)
- chunk_fixed_size: Cắt cố định theo số từ (S1 - baseline)
- chunk_recursive: Cắt đệ quy theo dấu phân cách (S2)
- save_chunks_to_jsonl: Lưu danh sách chunk ra file JSONL an toàn

src/chunking/
├── __init__.py       # Package entry-point: xuất các hàm gọi chung (chunk_structural, chunk_fixed_size, v.v.)
├── utils.py          # Các hàm tiện ích: NFC, sửa font Đ/Ð, bisect trang, clean header/footer
├── fixed.py          # Chiến lược S1: Cắt cố định theo số từ (Fixed Token Baseline)
├── recursive.py      # Chiến lược S2: Cắt đệ quy theo đoạn / câu
└── structural.py     # Chiến lược S3: Cắt cấu trúc Chương / Điều / Khoản (Cốt lõi)
"""

import os
import json
from typing import List, Dict, Any

from src.chunking.fixed import chunk_fixed_size
from src.chunking.recursive import chunk_recursive
from src.chunking.structural import chunk_structural
from src.chunking.utils import (
    normalize_legal_text,
    estimate_token_count,
    get_page_number,
    make_chunk,
)


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """Lưu chunks ra JSONL an toàn, tự tạo thư mục cha nếu chưa có."""
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


__all__ = [
    "chunk_structural",
    "chunk_fixed_size",
    "chunk_recursive",
    "save_chunks_to_jsonl",
    "normalize_legal_text",
    "estimate_token_count",
    "get_page_number",
    "make_chunk",
]
