"""
Chiến lược 2 (S2): Cắt đệ quy theo dấu phân cách (\n\n -> \n -> câu -> từ).
"""

from typing import List, Dict, Any, Tuple, Optional
from src.chunking.utils import group_by_doc, build_doc_text, get_page_number, trim_span, make_chunk

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive_split_spans(src: str, lo: int = 0, hi: Optional[int] = None, max_chars: int = 1200,
                          overlap_chars: int = 150, seps: Optional[List[str]] = None) -> List[Tuple[int, int]]:
    """Cắt src[lo:hi] đệ quy theo seps, trả về danh sách offset tuyệt đối (start, end)."""
    if hi is None:
        hi = len(src)
    if seps is None:
        seps = _SEPARATORS
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    if hi - lo <= max_chars:
        return [(lo, hi)] if src[lo:hi].strip() else []

    if not seps:
        step = max(1, max_chars - overlap_chars)
        out, s = [], lo
        while s < hi:
            e = min(s + max_chars, hi)
            out.append((s, e))
            if e >= hi:
                break
            s += step
        return out

    sep = seps[0]
    pieces: List[Tuple[int, int]] = []
    pos = lo
    while pos < hi:
        idx = src.find(sep, pos, hi)
        if idx == -1:
            pieces.append((pos, hi))
            break
        pieces.append((pos, idx + len(sep)))
        pos = idx + len(sep)

    spans: List[Tuple[int, int]] = []

    def flush(group: List[Tuple[int, int]]) -> None:
        if not group:
            return
        s, e = group[0][0], group[-1][1]
        if e - s > max_chars:
            spans.extend(recursive_split_spans(
                src, s, e, max_chars, overlap_chars, seps[1:]))
        elif src[s:e].strip():
            spans.append((s, e))

    cur: List[Tuple[int, int]] = []
    cur_len = 0
    for ps, pe in pieces:
        plen = pe - ps
        if cur and cur_len + plen > max_chars:
            flush(cur)
            carry: List[Tuple[int, int]] = []
            carry_len = 0
            for piece in reversed(cur):
                ln = piece[1] - piece[0]
                if carry_len + ln <= overlap_chars:
                    carry.insert(0, piece)
                    carry_len += ln
                else:
                    break
            if carry_len + plen > max_chars:
                carry, carry_len = [], 0
            cur, cur_len = carry, carry_len
        cur.append((ps, pe))
        cur_len += plen
    flush(cur)
    return spans


def chunk_recursive(pages_data: List[Dict[str, Any]], max_chars: int = 1200, overlap_chars: int = 150) -> List[Dict[str, Any]]:
    """Cắt tài liệu đệ quy theo đoạn/câu."""
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc_pages in group_by_doc(pages_data).items():
        full_text, page_starts, pages = build_doc_text(doc_pages)
        idx = 0
        for s, e in recursive_split_spans(full_text, 0, len(full_text), max_chars, overlap_chars):
            s, e = trim_span(full_text, s, e)
            if s >= e:
                continue
            p_start = get_page_number(s, page_starts, pages)
            p_end = get_page_number(e - 1, page_starts, pages)
            heading = f"Trang {p_start}" if p_start == p_end else f"Trang {p_start} - {p_end}"
            all_chunks.append(make_chunk(
                doc_id, "s2", idx, full_text[s:e], p_start, p_end, heading, None, "recursive"))
            idx += 1
    return all_chunks
