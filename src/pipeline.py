"""
Pipeline RAG End-to-End cho tài liệu tiếng Việt (NLP-21).

Tích hợp toàn bộ luồng:
Query -> (BM25 / Dense / Hybrid RRF) -> Re-ranking mềm (Soft Re-ranking) -> Refusal Gate (Ngưỡng 0.05) -> LLM Generation -> Output + Nguồn

Xử lý ngoại lệ và tối ưu:
1. Cơ chế từ chối 2 lớp (Two-tier Refusal):
   - Lớp 1 (Truy hồi & Lọc thô): Ngưỡng tau = 0.05 lọc ngay các câu hỏi hoàn toàn lạc đề (out-of-domain).
     Không gọi LLM -> Tiết kiệm 100% thời gian sinh và VRAM.
   - Lớp 2 (Đọc hiểu & Đối chiếu mềm): LLM tự đối chiếu ngữ nghĩa giữa khẩu ngữ đời sống
     của người dùng với thuật ngữ văn bản hành chính/pháp lý trong tài liệu.
2. Tận dụng biểu diễn ngữ nghĩa đa chiều của AITeamVN/Vietnamese_Embedding kết hợp Cross-Encoder.
3. Quản lý tài nguyên tối ưu:
   - Embedding query chạy trên CPU (0MB VRAM).
   - Reranker chạy trên CPU (0MB VRAM).
   - LLM chiếm dụng GPU với BitsAndBytes 4-bit NF4.
4. Đo đạc độ trễ chi tiết (retrieve_ms, rerank_ms, generate_ms, total_ms).
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

from src.utils import load_config
from src.retrieve import BM25Retriever, reciprocal_rank_fusion
from src.index import VectorIndex
from src.rerank import VietnameseReranker
from src.generate import VietnameseLLMGenerator


class RAGPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.paths = self.config["paths"]
        self.models_cfg = self.config["models"]
        self.retrieval_cfg = self.config["retrieval"]

        # 1. Khởi tạo FAISS Index
        emb_cfg = self.models_cfg["embedding"]
        self.vector_index = VectorIndex(model_name=emb_cfg["name"], device=emb_cfg.get("device", "cuda"))
        
        # 2. Khởi tạo embedding model cho truy vấn (chạy trên CPU để tiết kiệm VRAM cho LLM)
        self.query_encoder = None
        self.emb_model_name = emb_cfg["name"]

        # 3. Khởi tạo BM25 (sẽ nạp sau khi index được load)
        self.bm25_retriever = None

        # 4. Khởi tạo Reranker (chạy trên CPU)
        rerank_cfg = self.models_cfg["reranker"]
        self.reranker = VietnameseReranker(
            model_name=rerank_cfg["name"],
            device=rerank_cfg.get("device", "cpu")
        )
        self.top_n_rerank = rerank_cfg.get("top_n_to_rerank", 10)
        self.top_k_final = rerank_cfg.get("top_k_final", 5)

        # 5. Khởi tạo LLM Generator (chạy GPU 4-bit)
        llm_cfg = self.models_cfg["llm"]
        self.generator = VietnameseLLMGenerator(
            model_name=llm_cfg["name"],
            quantization=llm_cfg.get("quantization", "4bit"),
            device=llm_cfg.get("device", "cuda"),
            max_new_tokens=llm_cfg.get("max_new_tokens", 600),
            temperature=llm_cfg.get("temperature", 0.1)
        )

        # Ngưỡng từ chối tau (mặc định 0.05 phù hợp phân phối phi tuyến tính Cross-Encoder)
        self.refusal_threshold = float(self.retrieval_cfg.get("refusal_threshold", 0.05))
        self.is_ready = False

    def initialize(self, index_dir: Optional[str] = None, preload_llm: bool = True):
        """
        Nạp chỉ mục và chuẩn bị sẵn sàng cho truy vấn.
        preload_llm: Nếu True, nạp trước LLM vào GPU ngay từ đầu để sẵn sàng phản hồi ngay từ câu đầu tiên.
        """
        idx_dir = index_dir or self.paths["index_dir"]
        print(f"[*] Đang nạp chỉ mục từ: {idx_dir}...")
        self.vector_index.load(idx_dir)

        # Khởi tạo mô hình encode câu hỏi trên CPU
        print(f"[*] Nạp Query Encoder ({self.emb_model_name}) trên CPU...")
        self.query_encoder = SentenceTransformer(self.emb_model_name, device="cpu")

        # Khởi tạo BM25 trên tập chunks đã nạp với cơ chế Dual Tokenization
        bm25_cfg = self.retrieval_cfg.get("bm25", {})
        bm25_mode = bm25_cfg.get("mode", "dual")
        print(f"[*] Khởi tạo BM25 (chế độ: {bm25_mode}) trên {len(self.vector_index.chunks)} chunks...")
        self.bm25_retriever = BM25Retriever(
            self.vector_index.chunks,
            k1=bm25_cfg.get("k1", 1.5),
            b=bm25_cfg.get("b", 0.75),
            mode=bm25_mode
        )

        # Nạp trước LLM vào GPU
        if preload_llm:
            print(f"[*] Đang nạp trước LLM Generator vào thiết bị: {self.generator.device}...")
            self.generator.load_model()
            print("[+] LLM đã nạp sẵn trong VRAM GPU!")

        self.is_ready = True
        print("[+] Pipeline RAG đã sẵn sàng phục vụ 100%!")

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 5,
        use_rerank: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Truy xuất các đoạn văn liên quan nhất:
        - method: 'bm25' | 'dense' | 'hybrid'
        - use_rerank: True | False (dùng để chạy thí nghiệm ablation)
        Tận dụng biểu diễn ngữ nghĩa đa chiều của AITeamVN/Vietnamese_Embedding và Re-ranking mềm.
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline chưa được initialize()!")

        timings = {}
        t0 = time.time()

        # 1. Tìm kiếm theo phương pháp
        n_candidates = self.top_n_rerank if use_rerank else top_k
        if method == "bm25":
            candidates = self.bm25_retriever.search(query, top_k=n_candidates)
        elif method == "dense":
            candidates = self.vector_index.search(query, self.query_encoder, top_k=n_candidates)
        elif method == "hybrid":
            bm25_results = self.bm25_retriever.search(query, top_k=n_candidates)
            dense_results = self.vector_index.search(query, self.query_encoder, top_k=n_candidates)
            candidates = reciprocal_rank_fusion(
                [bm25_results, dense_results],
                k=self.retrieval_cfg.get("rrf", {}).get("k", 60),
                top_k=n_candidates
            )
        else:
            raise ValueError(f"Phương thức không hợp lệ: {method}")

        timings["retrieve_ms"] = (time.time() - t0) * 1000

        # 2. Xếp hạng lại bằng Re-ranking mềm (Soft Re-ranking)
        if use_rerank and candidates:
            t1 = time.time()
            final_chunks = self.reranker.rerank(query, candidates, top_k=top_k)
            timings["rerank_ms"] = (time.time() - t1) * 1000
        else:
            final_chunks = candidates[:top_k]
            timings["rerank_ms"] = 0.0

        return final_chunks, timings

    def query(
        self,
        question: str,
        method: str = "hybrid",
        top_k: int = 5,
        use_rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Hàm xử lý truy vấn hỏi đáp hoàn chỉnh từ đầu đến cuối (End-to-End).
        """
        question = question.strip()
        if not question:
            return {
                "question": question,
                "answer": "Vui lòng nhập câu hỏi hợp lệ.",
                "refused": True,
                "sources": [],
                "latency_ms": {}
            }

        # Baseline B0: Không truy hồi (No Retrieval) - Đo năng lực thuần của LLM khi không có RAG
        if method == "no_retrieval":
            t0 = time.time()
            gen_res = self.generator.generate_without_retrieval(question)
            gen_ms = (time.time() - t0) * 1000
            return {
                "question": question,
                "answer": gen_res["answer"],
                "refused": gen_res["refused"],
                "sources": [],
                "contexts": [],
                "latency_ms": {
                    "retrieve_ms": 0.0,
                    "rerank_ms": 0.0,
                    "generate_ms": gen_ms,
                    "total_ms": gen_ms
                }
            }

        # Bước 1: Truy xuất và xếp hạng
        contexts, timings = self.retrieve(question, method=method, top_k=top_k, use_rerank=use_rerank)

        # Bước 2: Kiểm tra ngưỡng từ chối Lớp 1 (Refusal Gate)
        # Nếu điểm cao nhất thấp hơn ngưỡng tau (0.05) -> Dừng ngay (Out-of-domain)
        max_score = 0.0
        if contexts:
            # Ưu tiên lấy rerank_score (đã qua Soft Re-ranking), ngược lại lấy score của retriever
            max_score = contexts[0].get("rerank_score", contexts[0].get("score", 0.0))

        if not contexts or max_score < self.refusal_threshold:
            return {
                "question": question,
                "answer": "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.",
                "refused": True,
                "refusal_reason": f"Điểm tương đồng cao nhất ({max_score:.3f}) dưới ngưỡng từ chối ({self.refusal_threshold}).",
                "sources": [],
                "contexts": contexts,
                "latency_ms": {**timings, "generate_ms": 0.0, "total_ms": sum(timings.values())}
            }

        # Bước 3: Gọi LLM sinh câu trả lời (Lớp 2 - LLM tự đọc hiểu, suy luận ngữ nghĩa và đối chiếu)
        t2 = time.time()
        gen_res = self.generator.generate(question, contexts)
        timings["generate_ms"] = (time.time() - t2) * 1000
        timings["total_ms"] = sum(timings.values())

        # Bước 4: Ánh xạ mã trích dẫn sang thông tin nguồn chi tiết (Bằng chứng RAG)
        sources = []
        cited_indices = set(gen_res.get("citations", []))
        
        if cited_indices:
            for cite_idx in sorted(list(cited_indices)):
                if 1 <= cite_idx <= len(contexts):
                    c = contexts[cite_idx - 1]
                    sources.append({
                        "citation_id": cite_idx,
                        "doc_id": c.get("doc_id"),
                        "page_start": c.get("page_start"),
                        "page_end": c.get("page_end"),
                        "heading": c.get("heading"),
                        "clause": c.get("clause"),
                        "score": c.get("rerank_score", c.get("score")),
                        "text": c.get("text", ""),
                        "snippet": c.get("text", "")[:250] + "..."
                    })
        else:
            # Fallback: Nếu LLM diễn đạt theo tên Điều thay vì gắn mã [n],
            # vẫn cung cấp đầy đủ các đoạn ngữ cảnh Top-K để người dùng đối chiếu căn cứ
            for idx, c in enumerate(contexts, 1):
                sources.append({
                    "citation_id": idx,
                    "doc_id": c.get("doc_id"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "heading": c.get("heading"),
                    "clause": c.get("clause"),
                    "score": c.get("rerank_score", c.get("score")),
                    "text": c.get("text", ""),
                    "snippet": c.get("text", "")[:250] + "..."
                })

        return {
            "question": question,
            "answer": gen_res["answer"],
            "refused": gen_res["refused"],
            "sources": sources,
            "contexts": contexts,
            "citations_valid": gen_res["citations"],
            "citations_hallucinated": gen_res["hallucinated_citations"],
            "latency_ms": timings
        }
