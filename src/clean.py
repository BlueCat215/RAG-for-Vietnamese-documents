"""Module làm sạch văn bản tiếng Việt: chuẩn hóa Unicode NFC, xử lý gãy dòng."""

import re
import unicodedata
def normalize_unicode_nfc(text: str) -> str:
    """
    Chuẩn hóa chuỗi văn bản về Unicode NFC (dựng sẵn).
    Rất quan trọng với tiếng Việt vì bàn phím gõ có thể sinh mã tổ hợp (NFD).
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def remove_control_characters(text: str) -> str:
    """Loại bỏ ký tự điều khiển lạ, zero-width spaces và soft-hyphens."""
    if not text:
        return ""
    # Giữ lại các ký tự xuống dòng (\n, \r, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xad", "")
    return text


def clean_repeated_spaces(text: str) -> str:
    """Chuẩn hóa nhiều khoảng trắng liên tiếp nhưng giữ định dạng xuống dòng."""
    lines = text.split("\n")
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(cleaned_lines)


def fix_broken_linebreaks(text: str) -> str:
    """
    Nối lại các dòng bị ngắt giữa câu (đặc thù thường gặp khi trích xuất text từ PDF).
    Nếu dòng không kết thúc bằng dấu ngắt câu (. : ; ? !)
    và dòng kế tiếp KHÔNG bắt đầu bằng tiêu đề (Điều, Chương, Mục, 1., a)), thì nối lại.
    """
    lines = text.split("\n")
    if not lines:
        return ""

    merged_lines = []
    buffer = ""

    header_pattern = re.compile(
        r"^(Chương|Điều|Mục|Phần|\d+\.|\([a-zđ\d]+\)|[a-zđ]\))\s+",
        re.IGNORECASE
    )

    for line in lines:
        line_str = line.strip()
        if not line_str:
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            continue

        if not buffer:
            buffer = line_str
        else:
            # Kiểm tra xem buffer có kết thúc bằng dấu chấm câu không
            ends_with_punct = buffer[-1] in ".:;?!"
            is_new_section = bool(header_pattern.match(line_str))

            if not ends_with_punct and not is_new_section:
                # Nối dòng
                buffer = f"{buffer} {line_str}"
            else:
                merged_lines.append(buffer)
                buffer = line_str

    if buffer:
        merged_lines.append(buffer)

    return "\n".join(merged_lines)


def clean_vietnamese_text(text: str) -> str:
    """
    Pipeline làm sạch tổng thể:
    1. Chuẩn hóa NFC
    2. Xóa ký tự điều khiển
    3. Nối các dòng gãy của PDF
    4. Chuẩn hóa khoảng trắng
    Lưu ý: KHÔNG hạ lowercase, KHÔNG xóa dấu câu, số hay các từ viết tắt.
    """
    if not text:
        return ""
    text = normalize_unicode_nfc(text)
    text = remove_control_characters(text)
    text = fix_broken_linebreaks(text)
    text = clean_repeated_spaces(text)
    return text.strip()