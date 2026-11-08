"""
Module logging tập trung cho RAG Pipeline.
Ghi log ra file (logs/rag_debug_<timestamp>.txt) và stderr đồng thời.
Dùng cho debug terminal riêng biệt theo dõi từng bước xử lý.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional


_rag_logger: Optional[logging.Logger] = None
_log_file_path: Optional[str] = None


def setup_logger(log_dir: str = "logs", log_to_console: bool = False) -> logging.Logger:
    """
    Khởi tạo logger, ghi toàn bộ dữ liệu debug ra file txt.
    Cửa sổ terminal bên trái (LOG / log_viewer.py) sẽ đọc từ file này để hiển thị.
    Terminal chính (hoi_dap / Streamlit) giữ sạch, không in debug rác.
    """
    global _rag_logger, _log_file_path

    if _rag_logger is not None:
        return _rag_logger

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file_path = os.path.join(log_dir, f"rag_debug_{timestamp}.txt")

    logger = logging.getLogger("rag_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Handler ghi ra file (cửa sổ LOG sẽ đọc realtime từ file này)
    fh = logging.FileHandler(_log_file_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Chỉ in ra console nếu được yêu cầu cụ thể (mặc định False để terminal tương tác sạch sẽ)
    if log_to_console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    _rag_logger = logger
    logger.info(f"Logger khởi động. File log: {_log_file_path}")
    return logger


def get_logger() -> logging.Logger:
    """Lấy logger hiện tại, tự động khởi tạo nếu chưa có."""
    if _rag_logger is None:
        return setup_logger()
    return _rag_logger


def get_log_file_path() -> Optional[str]:
    return _log_file_path


def log_step(step: int, title: str, data: Any = None):
    """Ghi một bước xử lý có đánh số vào log."""
    logger = get_logger()
    sep = "─" * 60
    logger.debug(f"\n{sep}")
    logger.debug(f"[STEP {step}] {title}")
    if data is not None:
        if isinstance(data, str):
            logger.debug(data)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                logger.debug(f"  [{i+1}] {item}")
        elif isinstance(data, dict):
            for k, v in data.items():
                logger.debug(f"  {k}: {v}")
        else:
            logger.debug(str(data))


def log_query_start(query: str, method: str, top_k: int, use_rerank: bool):
    logger = get_logger()
    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info(f"QUERY MỚI: {query!r}")
    logger.info(f"Cấu hình: method={method}, top_k={top_k}, use_rerank={use_rerank}")
    logger.info(sep)


def log_bm25_results(candidates: List[Dict[str, Any]]):
    logger = get_logger()
    logger.debug(f"[BM25] Trả về {len(candidates)} ứng viên:")
    for c in candidates:
        logger.debug(
            f"  Rank {c.get('rank','-')} | score={c.get('bm25_score', c.get('score',0)):.4f} "
            f"| {c.get('doc_id','')} / {c.get('heading','')[:60]}"
        )


def log_dense_results(candidates: List[Dict[str, Any]]):
    logger = get_logger()
    logger.debug(f"[Dense] Trả về {len(candidates)} ứng viên:")
    for c in candidates:
        logger.debug(
            f"  Rank {c.get('rank','-')} | score={c.get('dense_score', c.get('score',0)):.4f} "
            f"| {c.get('doc_id','')} / {c.get('heading','')[:60]}"
        )


def log_hybrid_results(candidates: List[Dict[str, Any]]):
    logger = get_logger()
    logger.debug(f"[Hybrid RRF] Sau gộp: {len(candidates)} ứng viên:")
    for c in candidates:
        logger.debug(
            f"  Rank {c.get('rank','-')} | rrf={c.get('score',0):.6f} "
            f"| dense={c.get('dense_score',0):.4f} | bm25={c.get('bm25_score',0):.4f} "
            f"| {c.get('doc_id','')} / {c.get('heading','')[:50]}"
        )


def log_rerank_results(candidates: List[Dict[str, Any]]):
    logger = get_logger()
    logger.debug(f"[Reranker] Sau xếp hạng lại: {len(candidates)} chunk cuối cùng:")
    for c in candidates:
        logger.debug(
            f"  Rank {c.get('rank','-')} | soft={c.get('rerank_score',0):.4f} "
            f"| raw={c.get('rerank_raw',0):.4f} | dense={c.get('dense_score',0):.4f} "
            f"| {c.get('doc_id','')} / {c.get('heading','')[:50]}"
        )


def log_refusal_check(max_score: float, threshold: float, refused: bool):
    logger = get_logger()
    status = "TỪ CHỐI (out-of-domain)" if refused else "CHẤP NHẬN"
    logger.debug(
        f"[Refusal Gate] max_score={max_score:.4f}, threshold={threshold:.4f} → {status}"
    )


def log_prompt(prompt: str, token_estimate: int):
    logger = get_logger()
    logger.debug(f"[Prompt → LLM] ~{token_estimate} ký tự")
    # Chỉ in 500 ký tự đầu để tránh spam log
    preview = prompt[:500].replace("\n", "↵") + ("..." if len(prompt) > 500 else "")
    logger.debug(f"  Preview: {preview}")


def log_llm_output(raw: str, cleaned: str, citations: List[int], hallucinated: List[int]):
    logger = get_logger()
    logger.debug(f"[LLM Output Raw] ({len(raw)} ký tự): {raw[:300].replace(chr(10), '↵')}...")
    logger.debug(f"[LLM Output Cleaned] ({len(cleaned)} ký tự)")
    logger.debug(f"[Citations] Hợp lệ: {citations} | Bịa đặt: {hallucinated}")


def log_latency(timings: Dict[str, float]):
    logger = get_logger()
    logger.info(
        f"[Latency] retrieve={timings.get('retrieve_ms',0):.0f}ms | "
        f"rerank={timings.get('rerank_ms',0):.0f}ms | "
        f"generate={timings.get('generate_ms',0):.0f}ms | "
        f"total={timings.get('total_ms',0):.0f}ms"
    )
