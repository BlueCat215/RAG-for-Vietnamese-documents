"""Giao diện Web Hỏi Đáp Quy Chế Đào Tạo Đại Học (RAG Tiếng Việt - NLP-21)."""

import os
import subprocess
import sys
import re
import streamlit as st
from src.pipeline import RAGPipeline
from src.utils import load_config
from src import logger as rag_logger


st.set_page_config(
    page_title="Hỏi Đáp Quy Chế Đào Tạo (RAG Tiếng Việt)",
    page_icon="🎓",
    layout="wide"
)


# ── Khởi tạo Logger và spawn Debug Terminal ─────────────────────────────────
def _init_debug_terminal():
    """Khởi động logger và mở một terminal riêng hiển thị log realtime."""
    if st.session_state.get("_debug_terminal_spawned"):
        return
    log = rag_logger.setup_logger(log_dir="logs")
    log_path = rag_logger.get_log_file_path()
    if log_path and sys.platform == "win32":
        try:
            subprocess.Popen(
                ["cmd", "/k", f"python log_viewer.py --file \"{log_path}\""],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=os.getcwd()
            )
        except Exception as e:
            log.warning(f"Không mở được debug terminal: {e}")
    st.session_state["_debug_terminal_spawned"] = True


_init_debug_terminal()


# ── Pipeline ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline():
    """Khởi tạo và cache RAG Pipeline một lần duy nhất."""
    p = RAGPipeline(config_path="config/config.yaml")
    p.initialize()
    return p


# ── Giao diện chính ──────────────────────────────────────────────────────────
st.title("🎓 Trợ Lý Hỏi Đáp Quy Chế Đào Tạo Đại Học")
st.caption("Hệ thống RAG tiếng Việt hỗ trợ sinh viên tra cứu quy chế, học phí, học lại và tốt nghiệp")

with st.sidebar:
    st.header("⚙️ Cấu hình truy xuất")
    retriever_mode = st.selectbox(
        "Phương thức tìm kiếm:",
        options=["hybrid", "dense", "bm25"],
        format_func=lambda x: {
            "hybrid": "Hybrid Search (BM25 + Dense RRF)",
            "dense": "Dense Vector Search (Ngữ nghĩa)",
            "bm25": "BM25 Baseline (Từ khóa)"
        }[x]
    )
    use_rerank = st.checkbox("Bật Reranker (Cross-Encoder)", value=True)
    top_k = st.slider("Số lượng đoạn trích dẫn (Top-K):", min_value=1, max_value=8, value=4)

    st.divider()
    st.header("📁 Tài liệu trong hệ thống")
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    if os.path.exists(raw_dir):
        files = [f for f in os.listdir(raw_dir) if f.endswith(('.pdf', '.docx', '.txt'))]
        for f in files:
            st.text(f"📄 {f}")
    else:
        st.info("Chưa có tài liệu.")

    st.divider()
    log_path = rag_logger.get_log_file_path()
    if log_path and os.path.exists(log_path):
        st.caption(f"📋 Log: `{os.path.basename(log_path)}`")

try:
    with st.spinner("Đang nạp mô hình và chỉ mục..."):
        pipeline = get_pipeline()
except Exception as e:
    st.error(f"Lỗi khởi tạo Pipeline: {e}. Vui lòng kiểm tra xem bạn đã chạy 'python run_indexing.py' chưa.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Bạn cần tra cứu thông tin gì về quy chế đào tạo, đăng ký học phần, tính điểm rèn luyện hay điều kiện miễn giảm học phí?"}
    ]


# ── Hàm tiện ích ─────────────────────────────────────────────────────────────
def _build_citation_map(sources: list) -> dict:
    """Tạo dict ánh xạ số trích dẫn → chú giải chi tiết vị trí và nguồn tài liệu."""
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

        annotation = " | ".join(loc_items)
        cmap[cid] = {
            "annotation": annotation,
            "short": f"[{cid}: {annotation}]",
            "doc": doc
        }
    return cmap


def render_answer_with_inline_citations(answer_text: str, sources: list) -> str:
    """
    Kèm chú giải vị trí và nguồn tài liệu ngay cuối các câu/mệnh đề trong nội dung trả lời.
    Thay thế [n] thành chú giải: 📌 *[n: Điều/Khoản | Trang | Tài liệu]*
    """
    if not sources:
        return answer_text

    cmap = _build_citation_map(sources)

    def replacer(m):
        n = int(m.group(1))
        if n in cmap:
            return f" 📌 *{cmap[n]['short']}*"
        return m.group(0)

    annotated = re.sub(r"\[(\d+)\]", replacer, answer_text)

    # Nếu câu trả lời không chứa mã [n] nào nhưng có sources hợp lệ, bổ sung chú giải căn cứ ở cuối
    if not re.search(r"\[\d+\]", answer_text) and "📌" not in annotated and sources:
        notes = []
        for s in sources[:3]:
            cid = s.get("citation_id")
            if cid in cmap:
                notes.append(f"📌 *{cmap[cid]['short']}*")
        if notes:
            annotated += "\n\n**Căn cứ nguồn tài liệu:** " + " • ".join(notes)

    return annotated


def display_sources(sources: list, expanded: bool = True):
    """Hiển thị chi tiết căn cứ và đoạn văn bản nguyên văn RAG đã truy hồi."""
    if not sources:
        return
    with st.expander("📚 Xem nguồn trích dẫn & căn cứ pháp lý", expanded=expanded):
        for s in sources:
            cid = s.get("citation_id", "?")
            doc = s.get("doc_id", "Tài liệu")
            heading = s.get("heading", "")
            clause = s.get("clause", "")
            p_start = s.get("page_start", 1)
            p_end = s.get("page_end", p_start)
            page_str = f"Trang {p_start}" if p_start == p_end else f"Trang {p_start} - {p_end}"
            score = s.get("score")
            score_str = f" • Điểm phù hợp: `{score:.3f}`" if score is not None else ""

            st.markdown(f"#### `[{cid}]` {doc}")
            sub_headers = []
            if clause:
                sub_headers.append(f"**{clause}**")
            if heading:
                sub_headers.append(f"*{heading}*")
            if sub_headers:
                st.markdown(" ➔ ".join(sub_headers))

            st.caption(f"📍 {page_str}{score_str}")

            chunk_text = s.get("text") or s.get("snippet", "")
            with st.container(border=True):
                st.markdown(f"**Nội dung văn bản:**\n\n{chunk_text}")
            st.divider()


# ── Hiển thị lịch sử chat ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            display_sources(msg["sources"], expanded=False)


# ── Xử lý câu hỏi mới ───────────────────────────────────────────────────────
if prompt := st.chat_input("Nhập câu hỏi của bạn (ví dụ: điều kiện tốt nghiệp, thủ tục học lại, miễn giảm học phí...):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu quy chế và tổng hợp câu trả lời..."):
            res = pipeline.query(
                question=prompt,
                method=retriever_mode,
                top_k=top_k,
                use_rerank=use_rerank
            )
            answer_text = res["answer"]
            sources = res.get("sources", [])
            lat = res.get("latency_ms", {})

            # Log LLM output vào debug terminal
            gen_result = res.get("citations_valid", [])
            hal_result = res.get("citations_hallucinated", [])
            rag_logger.get_logger().info(
                f"[Answer] {len(answer_text)} ký tự | citations={gen_result} | hallucinated={hal_result}"
            )

            # Hiển thị câu trả lời với inline citation được làm giàu
            display_text = render_answer_with_inline_citations(answer_text, sources)
            st.markdown(display_text)

            st.caption(
                f"⏱️ Phản hồi trong {lat.get('total_ms', 0):.0f}ms "
                f"(Truy xuất: {lat.get('retrieve_ms', 0):.0f}ms, "
                f"Rerank: {lat.get('rerank_ms', 0):.0f}ms, "
                f"LLM: {lat.get('generate_ms', 0):.0f}ms)"
            )

            if sources:
                display_sources(sources, expanded=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": display_text,
                "sources": sources
            })
