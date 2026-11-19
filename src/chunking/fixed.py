"""
Chiến lược 1 (S1): Cắt cố định theo số từ (Fixed Token Baseline).
"""

import re
from typing import List, Dict, Any
from src.chunking.utils import group_by_doc, build_doc_text, get_page_number, make_chunk


def chunk_fixed_size(
    pages_data: List[Dict[str, Any]], chunk_size: int = 300, overlap: int = 50
) -> List[Dict[str, Any]]:
    """Cắt cố định theo số từ trên chuỗi liên tục của tài liệu."""
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc_pages in group_by_doc(pages_data).items():
        full_text, page_starts, pages = build_doc_text(doc_pages)
        spans = [m.span() for m in re.finditer(r"\S+", full_text)]
        n = len(spans)
        if n == 0:
            continue

        step = max(1, chunk_size - overlap)
        idx = 0
        for i in range(0, n, step):
            j = min(i + chunk_size, n)
            chunk_text = " ".join(full_text[a:b] for a, b in spans[i:j])
            p_start = get_page_number(spans[i][0], page_starts, pages)
            p_end = get_page_number(spans[j - 1][1] - 1, page_starts, pages)
            heading = (
                f"Trang {p_start}" if p_start == p_end else f"Trang {p_start} - {p_end}"
            )
            all_chunks.append(
                make_chunk(
                    doc_id,
                    "s1",
                    idx,
                    chunk_text,
                    p_start,
                    p_end,
                    heading,
                    None,
                    "fixed",
                )
            )
            idx += 1
            if j >= n:
                break
    return all_chunks
