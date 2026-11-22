import os
import subprocess
import fitz  # PyMuPDF
import docx
from typing import List, Dict, Any
from src.clean import clean_vietnamese_text


def _table_to_markdown(table: docx.table.Table) -> str:
    """Chuyển đổi bảng docx thành định dạng Markdown Table để bảo toàn cấu trúc dòng/cột."""
    if not table.rows:
        return ""

    rows_data = []
    for row in table.rows:
        row_cells = [
            c.text.strip().replace("\n", " ").replace("|", "/") for c in row.cells
        ]
        rows_data.append(row_cells)

    if not rows_data:
        return ""

    n_cols = max(len(r) for r in rows_data)
    normalized_rows = []
    for r in rows_data:
        if len(r) < n_cols:
            r = r + [""] * (n_cols - len(r))
        normalized_rows.append(r)

    header = "| " + " | ".join(normalized_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * n_cols) + " |"
    body = [
        "| " + " | ".join(r) + " |" for r in normalized_rows[1:]
    ] if len(normalized_rows) > 1 else []

    return "\n".join([header, separator] + body)


def extract_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Trích xuất văn bản từ tệp PDF theo từng trang.
    Ưu tiên sử dụng pymupdf4llm để bóc tách cấu trúc bảng biểu và Markdown chất lượng cao.
    Nếu thất bại, tự động fallback về PyMuPDF chuẩn.
    """
    doc_id = os.path.basename(file_path)
    pages_data = []

    try:
        import pymupdf4llm
        md_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        for idx, chunk in enumerate(md_chunks):
            raw_text = chunk.get("text", "")
            cleaned_text = clean_vietnamese_text(raw_text)
            is_empty_or_scan = len(cleaned_text.strip()) < 40

            pages_data.append({
                "doc_id": doc_id,
                "page": idx + 1,
                "text": cleaned_text,
                "char_count": len(cleaned_text),
                "needs_ocr": is_empty_or_scan
            })
    except Exception as e:
        # Fallback về trích xuất text thuần bằng fitz
        doc = fitz.open(file_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            raw_text = page.get_text("text")
            cleaned_text = clean_vietnamese_text(raw_text)
            is_empty_or_scan = len(cleaned_text.strip()) < 40

            pages_data.append({
                "doc_id": doc_id,
                "page": page_idx + 1,
                "text": cleaned_text,
                "char_count": len(cleaned_text),
                "needs_ocr": is_empty_or_scan
            })
        doc.close()

    return pages_data


def extract_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    Trích xuất từ DOCX bảo toàn thứ tự tự nhiên của cả đoạn văn (Paragraph)
    và bảng biểu (Table) dưới định dạng Markdown.
    """
    doc_id = os.path.basename(file_path)
    doc = docx.Document(file_path)

    full_blocks = []
    for child in doc.element.body:
        if child.tag.endswith("p"):
            p = docx.text.paragraph.Paragraph(child, doc)
            t = p.text.strip()
            if t:
                full_blocks.append(t)
        elif child.tag.endswith("tbl"):
            tbl = docx.table.Table(child, doc)
            md_tbl = _table_to_markdown(tbl)
            if md_tbl:
                full_blocks.append(md_tbl)

    raw_text = "\n\n".join(full_blocks)
    cleaned_text = clean_vietnamese_text(raw_text)

    return [{
        "doc_id": doc_id,
        "page": 1,
        "text": cleaned_text,
        "char_count": len(cleaned_text),
        "needs_ocr": False
    }]


def extract_from_doc(file_path: str) -> List[Dict[str, Any]]:
    """
    Trích xuất từ tệp .doc nhị phân cổ điển (OLE CFBF).
    Tự động chuyển đổi sang .docx qua PowerShell Word COM trên Windows nếu chưa có file .docx.
    """
    doc_id = os.path.basename(file_path)
    docx_path = file_path + "x"

    if not os.path.exists(docx_path):
        abs_in = os.path.abspath(file_path).replace("'", "''")
        abs_out = os.path.abspath(docx_path).replace("'", "''")
        ps_script = f"""
        $ErrorActionPreference = 'Stop'
        $word = New-Object -ComObject Word.Application
        $word.Visible = $false
        try {{
            $doc = $word.Documents.Open('{abs_in}')
            $doc.SaveAs2('{abs_out}', 16)
            $doc.Close()
        }} finally {{
            $word.Quit()
        }}
        """
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                check=True,
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi khi chuyển đổi file .doc sang .docx ({file_path}): {e}")

    if os.path.exists(docx_path):
        res = extract_from_docx(docx_path)
        for r in res:
            r["doc_id"] = doc_id
        return res
    else:
        raise FileNotFoundError(f"Không tìm thấy file chuyển đổi .docx tương ứng cho {file_path}")


def extract_from_txt(file_path: str) -> List[Dict[str, Any]]:
    """Trích xuất từ file TXT hỗ trợ mã hóa UTF-8."""
    doc_id = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            raw_text = f.read()

    cleaned_text = clean_vietnamese_text(raw_text)

    return [{
        "doc_id": doc_id,
        "page": 1,
        "text": cleaned_text,
        "char_count": len(cleaned_text),
        "needs_ocr": False
    }]


def extract_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Hàm entry point tự động nhận diện đuôi file để gọi hàm trích xuất tương ứng.
    Hỗ trợ .pdf, .docx, .doc, .txt.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_from_pdf(file_path)
    elif ext == ".docx":
        return extract_from_docx(file_path)
    elif ext == ".doc":
        return extract_from_doc(file_path)
    elif ext == ".txt":
        return extract_from_txt(file_path)
    else:
        raise ValueError(
            f"Định dạng {ext} chưa được hỗ trợ. Chỉ hỗ trợ .pdf, .docx, .doc, .txt")
