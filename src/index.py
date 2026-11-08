"""Module xây dựng và quản lý chỉ mục FAISS Vector Store."""

import os
import gc
import json
import torch
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


class VectorIndex:
    def __init__(self, model_name: str = "AITeamVN/Vietnamese_Embedding", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.index = None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 8, output_dir: str = "data/index"):
        """
        Tạo vector embeddings cho danh sách chunks và lưu vào FAISS Index.
        Sau khi chạy xong sẽ giải phóng VRAM GPU để LLM sử dụng.
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        print(f"[*] Đang tải Embedding Model: {self.model_name} trên thiết bị: {self.device}...")
        model = SentenceTransformer(self.model_name, device=self.device)

        print(f"[*] Bắt đầu tính toán vector cho {len(texts)} chunks (batch_size={batch_size})...")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        dim = embeddings.shape[1]
        print(f"[*] Khởi tạo FAISS IndexFlatIP (kích thước vector: {dim})...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[+] Đã giải phóng bộ nhớ GPU sau khi dựng Index.")

        self.save(output_dir)

    def save(self, output_dir: str):
        """Lưu file faiss.index và chunks_metadata.json."""
        os.makedirs(output_dir, exist_ok=True)
        index_file = os.path.join(output_dir, "faiss.index")
        meta_file = os.path.join(output_dir, "chunks_metadata.json")

        faiss.write_index(self.index, index_file)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"[+] Đã lưu chỉ mục thành công vào: {output_dir}")

    def load(self, index_dir: str):
        """Nạp chỉ mục đã lưu từ trước."""
        index_file = os.path.join(index_dir, "faiss.index")
        meta_file = os.path.join(index_dir, "chunks_metadata.json")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            raise FileNotFoundError(f"Không tìm thấy dữ liệu index trong {index_dir}")

        self.index = faiss.read_index(index_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"[+] Đã tải chỉ mục gồm {len(self.chunks)} chunks từ {index_dir}")

    def search(self, query: str, model: SentenceTransformer, top_k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm ngữ nghĩa (Dense Search) bằng mô hình vector embedding."""
        if self.index is None:
            raise ValueError("Index chưa được nạp hoặc khởi tạo!")

        q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(q_vec.astype(np.float32), top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["dense_score"] = float(score)
            chunk["rank"] = rank + 1
            chunk["retriever"] = "dense"
            results.append(chunk)

        return results
