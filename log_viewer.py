"""
Log Viewer: Mở một terminal riêng hiển thị realtime từng bước xử lý RAG.
Chạy độc lập: python log_viewer.py
app.py sẽ tự spawn script này khi khởi động.
"""

import os
import sys
import time
import glob
import argparse


def find_latest_log(log_dir: str = "logs") -> str | None:
    """Tìm file log mới nhất trong thư mục logs/."""
    pattern = os.path.join(log_dir, "rag_debug_*.txt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def tail_file(filepath: str, poll_interval: float = 0.3):
    """Đọc và in nội dung file log liên tục (giống `tail -f`)."""
    # Cấu hình UTF-8 cho Windows console
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 70)
    print("  RAG PIPELINE - DEBUG TERMINAL")
    print(f"  Đang theo dõi: {filepath}")
    print("  Nhấn Ctrl+C để thoát")
    print("=" * 70)
    print()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        # Nhảy đến cuối file để chỉ in nội dung mới
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(poll_interval)


def wait_for_log(log_dir: str = "logs", timeout: float = 30.0) -> str | None:
    """Chờ file log mới xuất hiện trong thư mục logs/."""
    print(f"[*] Đang chờ file log trong '{log_dir}'...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        path = find_latest_log(log_dir)
        if path:
            return path
        time.sleep(0.5)
    return None


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Debug Terminal")
    parser.add_argument("--log-dir", default="logs", help="Thư mục chứa file log")
    parser.add_argument("--file", default=None, help="Đường dẫn file log cụ thể")
    args = parser.parse_args()

    if args.file and os.path.exists(args.file):
        log_path = args.file
    else:
        log_path = find_latest_log(args.log_dir)
        if not log_path:
            log_path = wait_for_log(args.log_dir)

    if not log_path:
        print(f"[!] Không tìm thấy file log. Hãy chạy app.py trước.")
        sys.exit(1)

    try:
        tail_file(log_path)
    except KeyboardInterrupt:
        print("\n[*] Debug terminal đã đóng.")


if __name__ == "__main__":
    main()
