"""
Chiến lược 3 (S3 - Structural): Phân đoạn theo cấu trúc Chương / Điều / Khoản văn bản pháp quy.
"""

import re
import bisect
from typing import List, Dict, Any, Tuple, Optional
from src.chunking.utils import (
    group_by_doc,
    build_doc_text,
    get_page_number,
    trim_span,
    make_chunk,
    estimate_token_count,
)
from src.chunking.recursive import chunk_recursive, recursive_split_spans

_ARTICLE_RE = re.compile(
    r"^[ \t]*(Điều[ \t]+\d+[a-zđ]*\.[ \t]*[^\n]*)", re.MULTILINE)
_CHAPTER_RE = re.compile(
    r"^[ \t]*(Chương[ \t]+(?:[IVXLCDM]+|\d+)\b[^\n]{0,100})(?:\n[ \t]*([^\n\.\:0-9]{3,100}))?[ \t]*$",
    re.MULTILINE,
)
_CLAUSE_RE = re.compile(r"^[ \t]*(\d{1,2})\.[ \t]+", re.MULTILINE)
_RECIPIENT_RE = re.compile(r"^[ \t]*Nơi nhận[ \t]*:", re.MULTILINE)


def _find_chapters(full_text: str) -> List[Tuple[int, str]]:
    chapters = []
    for m in _CHAPTER_RE.finditer(full_text):
        first = m.group(1).strip()
        rest = re.sub(r"^Chương\s+\S+", "", first).strip(" :-.–")
        if rest.endswith((".", ",", ";")) or " này" in rest or " quy định tại " in rest:
            continue
        label = first
        if m.group(2):
            label += " - " + m.group(2).strip()
        chapters.append((m.start(1), label))
    return chapters


def _find_articles(full_text: str) -> List["re.Match[str]"]:
    raw = list(_ARTICLE_RE.finditer(full_text))
    kept = []
    for i, m in enumerate(raw):
        if i + 1 >= len(raw):
            kept.append(m)
            continue
        if len(full_text[m.end(1):raw[i + 1].start(1)].strip()) >= 15:
            kept.append(m)
    return kept


def _find_clauses(full_text: str, lo: int, hi: int) -> List["re.Match[str]"]:
    out, expected = [], 1
    for m in _CLAUSE_RE.finditer(full_text, lo, hi):
        if int(m.group(1)) == expected:
            out.append(m)
            expected += 1
    return out


def _chunk_structural_doc(doc_id: str, doc_pages: List[Dict[str, Any]], max_tokens: int,
                          split_chars: int, split_overlap: int) -> List[Dict[str, Any]]:
    full_text, page_starts, pages = build_doc_text(doc_pages)
    chapters = _find_chapters(full_text)
    chapter_starts = [c[0] for c in chapters]
    articles = _find_articles(full_text)

    if not articles:
        fb = chunk_recursive(doc_pages)
        for c in fb:
            c["strategy"] = "structural_fallback_recursive"
        return fb

    chunks: List[Dict[str, Any]] = []
    counter = 0

    def span_pages(s: int, e: int) -> Tuple[int, int]:
        return (get_page_number(s, page_starts, pages),
                get_page_number(max(s, e - 1), page_starts, pages))

    def emit(text: str, p_start: int, p_end: int, heading: str, clause: Optional[str], strategy: str) -> None:
        nonlocal counter
        chunks.append(make_chunk(doc_id, "s3", counter, text,
                      p_start, p_end, heading, clause, strategy))
        counter += 1

    def emit_pieces(prefix: str, lo: int, hi: int, heading: str, clause: Optional[str], strategy: str) -> None:
        budget = max(300, split_chars - len(prefix))
        for s, e in recursive_split_spans(full_text, lo, hi, budget, split_overlap):
            s, e = trim_span(full_text, s, e)
            if s >= e:
                continue
            ps, pe = span_pages(s, e)
            body = full_text[s:e]
            emit(f"{prefix}\n{body}" if prefix else body,
                 ps, pe, heading, clause, strategy)

    # 1. Phần mở đầu trước Điều 1
    pre_s, pre_e = trim_span(full_text, 0, articles[0].start(1))
    if pre_e - pre_s >= 40:
        pre_text = full_text[pre_s:pre_e]
        heading = "Căn cứ & Quy định chung"
        if estimate_token_count(pre_text) <= max_tokens:
            ps, pe = span_pages(pre_s, pre_e)
            emit(pre_text, ps, pe, heading, None, "structural_preamble")
        else:
            emit_pieces("", pre_s, pre_e, heading, None, "structural_preamble")

    # 2. Duyệt từng Điều
    for i, m in enumerate(articles):
        art_start = m.start(1)
        title_end = m.end(1)
        next_art_start = articles[i +
                                  1].start(1) if i + 1 < len(articles) else len(full_text)
        art_end = next_art_start

        ch_idx = bisect.bisect_right(chapter_starts, art_start)
        if ch_idx < len(chapter_starts) and chapter_starts[ch_idx] < next_art_start:
            art_end = chapter_starts[ch_idx]
        if i + 1 >= len(articles):
            rm = _RECIPIENT_RE.search(full_text, title_end)
            if rm and rm.start() < art_end:
                art_end = rm.start()

        article_title = m.group(1).strip()
        current_chapter = chapters[ch_idx -
                                   1][1] if ch_idx - 1 >= 0 else "Quy định chung"
        parent_header = f"{current_chapter} - {article_title}"

        art_s, art_e = trim_span(full_text, art_start, art_end)
        article_body = full_text[art_s:art_e]
        p_start, p_end = span_pages(art_s, art_e)

        # 2a. Điều vừa vặn
        whole_text = f"{current_chapter}\n{article_body}"
        if estimate_token_count(whole_text) <= max_tokens:
            emit(whole_text, p_start, p_end, parent_header, None, "structural")
            continue

        # 2b. Điều dài -> tách theo Khoản
        clause_matches = _find_clauses(full_text, title_end, art_end)
        if clause_matches:
            for c_i, cm in enumerate(clause_matches):
                c_start = title_end if c_i == 0 else cm.start(1)
                c_end = clause_matches[c_i + 1].start(
                    1) if c_i + 1 < len(clause_matches) else art_end
                s, e = trim_span(full_text, c_start, c_end)
                if s >= e:
                    continue
                clause_label = f"Khoản {cm.group(1)}"
                prefix = f"{parent_header} > {clause_label}"
                candidate = f"{prefix}\n{full_text[s:e]}"
                if estimate_token_count(candidate) <= max_tokens:
                    ps, pe = span_pages(s, e)
                    emit(candidate, ps, pe, parent_header,
                         clause_label, "structural_clause")
                else:
                    emit_pieces(prefix, s, e, parent_header,
                                clause_label, "structural_clause_split")
        else:
            emit_pieces(parent_header, title_end, art_end,
                        parent_header, None, "structural_split")

    return chunks


def chunk_structural(pages_data: List[Dict[str, Any]], max_tokens: int = 512,
                     split_chars: int = 1200, split_overlap: int = 100) -> List[Dict[str, Any]]:
    """Cắt văn bản theo cấu trúc Chương / Điều / Khoản."""
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, doc_pages in group_by_doc(pages_data).items():
        all_chunks.extend(_chunk_structural_doc(
            doc_id, doc_pages, max_tokens, split_chars, split_overlap))
    return all_chunks
