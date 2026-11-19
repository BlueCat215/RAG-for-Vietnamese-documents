"""
Script kiem tra moi truong va phan cung (Smoke Test) cho du an RAG tieng Viet.
"""

import sys
import os

# Ep stdout ho tro UTF-8 tren Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace")


def run_smoke_test():
    print("=" * 60)
    print("[*] SMOKE TEST - KIEM TRA PHAN CUNG & MOI TRUONG DU AN RAG")
    print("=" * 60)

    # 1. Python version
    print(f"[+] Python Version: {sys.version.split()[0]}")

    # 2. PyTorch & CUDA
    try:
        import torch
        print(f"[+] PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"[+] CUDA Available: {cuda_available}")

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(
                0).total_memory / (1024 ** 3)
            print(f"[+] GPU Name: {gpu_name}")
            print(f"[+] Tong dung luong VRAM: {total_vram:.2f} GB")

            if total_vram <= 4.5:
                print("    -> Phat hien GPU ~4GB VRAM (RTX 3050 hoac tuong duong).")
                print(
                    "    -> He thong se tu dong kich hoat Quantization 4-bit cho LLM va Reranker CPU.")
        else:
            print(
                "    [!] Canh bao: PyTorch dang chay tren CPU (chua nhan GPU CUDA).")
    except ImportError:
        print("    [!] PyTorch chua duoc cai dat.")

    # 3. BitsAndBytes (4-bit quantization)
    print("-" * 60)
    print("[*] Kiem tra thu vien BitsAndBytes (4-bit quantization)...")
    try:
        import bitsandbytes as bnb
        print("[+] bitsandbytes: Da cai dat thanh cong!")
    except ImportError:
        print(
            "    [!] Chua co bitsandbytes. Tren Windows co the cai qua: pip install bitsandbytes")

    # 4. Sentence-Transformers & FAISS
    print("-" * 60)
    print("[*] Kiem tra Sentence-Transformers & FAISS...")
    try:
        import sentence_transformers
        print(
            f"[+] sentence-transformers: {sentence_transformers.__version__}")
    except ImportError:
        print("    [!] Chua cai dat sentence-transformers.")

    try:
        import faiss
        print(f"[+] FAISS: {faiss.__version__}")
    except ImportError:
        print("    [!] Chua cai dat faiss-cpu.")

    # 5. Xu ly tieng Viet
    print("-" * 60)
    print("[*] Kiem tra thu vien xu ly tieng Viet...")
    try:
        import underthesea
        print(f"[+] underthesea: {underthesea.__version__}")
    except ImportError:
        print("    [!] Chua cai dat underthesea.")

    print("=" * 60)
    print("[+] HOAN TAT KIEM TRA SO BO MOI TRUONG!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
