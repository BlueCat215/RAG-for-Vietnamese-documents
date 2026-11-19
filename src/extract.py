import os
import fitz  # PyMuPDF
import docx
from typing import List, Dict, Any
from src.clean import clean_vietnamese_text
# Module trích xuất tài liệu từ file PDF, DOCX, TXT kèm metadata trang và cảnh báo file scan


def extract_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Trích xuất văn bản từ tệp PDF theo từng trang bằng PyMuPDF.
    Phát hiện và cảnh báo nếu trang bị scan (ít ký tự).
    """
    doc_id = os.path.basename(file_path)
    pages_data = []

    doc = fitz.open(file_path)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text("text")
        cleaned_text = clean_vietnamese_text(raw_text)

        # Cảnh báo nếu trang có ít hơn 40 ký tự (khả năng cao là trang scan/ảnh)
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
    Trích xuất từ DOCX.
    DOCX không phân trang rõ ràng, nhóm thành các đoạn logic dựa trên Heading.
    """
    doc_id = os.path.basename(file_path)
    doc = docx.Document(file_path)

    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    raw_text = "\n".join(full_text)
    cleaned_text = clean_vietnamese_text(raw_text)

    return [{
        "doc_id": doc_id,
        "page": 1,
        "text": cleaned_text,
        "char_count": len(cleaned_text),
        "needs_ocr": False
    }]


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
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_from_pdf(file_path)
    elif ext == ".docx":
        return extract_from_docx(file_path)
    elif ext == ".txt":
        return extract_from_txt(file_path)
    else:
        raise ValueError(
            f"Định dạng {ext} chưa được hỗ trợ. Chỉ hỗ trợ .pdf, .docx, .txt")
