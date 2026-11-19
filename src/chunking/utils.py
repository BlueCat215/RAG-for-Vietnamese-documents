"""
Các hàm tiện ích nền tảng cho việc phân đoạn văn bản pháp quy:
- Chuẩn hóa văn bản pháp luật, Unicode NFC, sửa lỗi font PDF (Ð -> Đ).
- Định danh chunk (hash) & ước lượng token tiếng Việt.
- Tra cứu số trang O(log N) bằng bisect.
- Lọc bỏ header/footer lặp trang và ghép văn bản hoàn chỉnh.
"""

import re
import math
import bisect
import hashlib
import unicodedata
from typing import List, Dict, Any, Tuple, Optional


def normalize_legal_text(text: Optional[str]) -> str:
    """Chuẩn hóa NFC, sửa lỗi font 'Ð'/'ð' từ PDF, chuẩn hóa xuống dòng."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("Ð", "Đ").replace("ð", "đ")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    return text


def hash_text(text: str) -> str:
    """Mã băm ngắn định danh chunk."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def estimate_token_count(text: str) -> int:
    """Ước lượng số token cho tiếng Việt (~1.3 token / âm tiết)."""
    return int(len(text.strip().split()) * 1.3)


def get_page_number(char_pos: int, page_starts: List[int], pages: List[int]) -> int:
    """Tìm số trang theo vị trí ký tự bằng bisect O(log N)."""
    idx = bisect.bisect_right(page_starts, char_pos) - 1
    idx = max(0, min(idx, len(pages) - 1))
    return pages[idx]


def trim_span(src: str, s: int, e: int) -> Tuple[int, int]:
    """Bỏ khoảng trắng hai đầu nhưng vẫn giữ offset tuyệt đối."""
    while s < e and src[s].isspace():
        s += 1
    while e > s and src[e - 1].isspace():
        e -= 1
    return s, e


def make_chunk(doc_id: str, tag: str, idx: int, text: str, page_start: int, page_end: int,
               heading: str, clause: Optional[str], strategy: str) -> Dict[str, Any]:
    """Tạo schema chuẩn cho một chunk."""
    return {
        "chunk_id": f"{doc_id}_{tag}_{idx:04d}_{hash_text(text)}",
        "doc_id": doc_id,
        "page_start": page_start,
        "page_end": page_end,
        "heading": heading,
        "clause": clause,
        "text": text,
        "strategy": strategy,
        "estimated_tokens": estimate_token_count(text),
    }


def group_by_doc(pages_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Gom nhóm danh sách trang theo doc_id."""
    docs: Dict[str, List[Dict[str, Any]]] = {}
    for p in pages_data:
        docs.setdefault(p["doc_id"], []).append(p)
    return docs


_PAGE_NUM_RE = re.compile(
    r"^(?:Trang\s+)?\d{1,4}(?:\s*/\s*\d{1,4})?$", re.IGNORECASE)
_STRUCT_LINE_RE = re.compile(r"^(?:Điều|Chương|Mục)\b")


def clean_page_texts(doc_pages: List[Dict[str, Any]], min_pages: int = 4,
                     ratio: float = 0.6, max_len: int = 100) -> List[str]:
    """Loại bỏ header/footer lặp lại ở >= ratio số trang và số trang đơn thuần."""
    texts = [normalize_legal_text(p.get("text")) for p in doc_pages]
    repeated = set()
    if len(texts) >= min_pages:
        counts: Dict[str, int] = {}
        for t in texts:
            for ln in {x.strip() for x in t.split("\n") if x.strip()}:
                if len(ln) <= max_len:
                    counts[ln] = counts.get(ln, 0) + 1
        threshold = max(2, math.ceil(len(texts) * ratio))
        repeated = {ln for ln, c in counts.items()
                    if c >= threshold and not _STRUCT_LINE_RE.match(ln)}

    cleaned = []
    for t in texts:
        kept = []
        for ln in t.split("\n"):
            s = ln.strip()
            if s and (s in repeated or _PAGE_NUM_RE.match(s)):
                continue
            kept.append(ln)
        cleaned.append("\n".join(kept))
    return cleaned


def build_doc_text(doc_pages: List[Dict[str, Any]], clean: bool = True) -> Tuple[str, List[int], List[int]]:
    """Ghép các trang thành một chuỗi liên tục kèm mảng offset trang."""
    doc_pages = sorted(doc_pages, key=lambda x: x["page"])
    texts = clean_page_texts(doc_pages) if clean else [
        normalize_legal_text(p.get("text")) for p in doc_pages]

    parts, page_starts, pages = [], [], []
    pos = 0
    for p, t in zip(doc_pages, texts):
        seg = t + "\n\n"
        page_starts.append(pos)
        pages.append(p["page"])
        parts.append(seg)
        pos += len(seg)
    return "".join(parts), page_starts, pages
