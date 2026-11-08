# Hệ Thống Hỏi Đáp Truy Hồi Tăng Cường Sinh (RAG) Cho Tài Liệu Tiếng Việt

<p align="right"> Ngôn ngữ: <b>Tiếng Việt</b> | <a style="font-weight: bold;" href="../../README.md">Tiếng Anh</a> </p>

## 1. Khái Niệm Tổng Quan & Kiến Trúc

- **RAG (Retrieval-Augmented Generation / Truy hồi Tăng cường Sinh):** Kỹ thuật kết hợp giữa việc tìm kiếm (truy hồi) các đoạn văn bản liên quan từ kho tài liệu nội bộ và việc sử dụng mô hình ngôn ngữ lớn (LLM) để tổng hợp ra câu trả lời chính xác, giúp hạn chế hiện tượng bịa đặt thông tin.
- **Kho tài liệu miền hẹp (Domain-Specific Corpus):** Tập hợp các văn bản chuyên ngành (pháp luật, tài chính, văn bản doanh nghiệp, v.v.) được giới hạn phạm vi để làm căn cứ tri thức cho hệ thống RAG.
- **Agentic RAG:** Kiến trúc RAG nâng cao trong đó mô hình đóng vai trò như một tác nhân (agent) có khả năng tự lập kế hoạch, truy hồi đa bước, gọi công cụ phụ trợ (tools) và kiểm tra tính đầy đủ của ngữ cảnh trước khi trả lời.

---

## 2. Tiền Xử Lý Dữ Liệu & Đặc Thù Ngôn Ngữ Tiếng Việt

- **Tách từ tiếng Việt (Word Segmentation):** Do tiếng Việt là ngôn ngữ đơn lập với các từ ghép gồm nhiều âm tiết rời rạc cách nhau bởi khoảng trắng (ví dụ: _"học sinh"_, _"ngân hàng"_), việc tách từ nhằm liên kết các âm tiết thành một từ hoàn chỉnh (ví dụ: _"học_sinh"_) giúp mô hình nhúng hiểu đúng ngữ nghĩa.
  - **Các công cụ tách từ phổ biến:** **PyVi** (tốc độ rất nhanh nhưng độ chính xác vừa phải), **Underthesea** (độ chính xác cao, cân bằng tốt), **VnCoreNLP** (dựa trên luật/CRF với độ chính xác cao), và **coccoc-tokenizer** (viết bằng C++ cho hiệu năng vượt trội).
- **Tokenizer Premium (Sự phình to số lượng Token):** Hệ thống dấu thanh phức tạp khiến các bộ chia từ (tokenizer) chuẩn phương Tây phân rã văn bản tiếng Việt thành số lượng token lớn gấp 1.4 đến 4.5 lần so với tiếng Anh, làm gia tăng chi phí và độ trễ.
- **Chia đoạn văn bản (Chunking):** Việc chia tài liệu dài thành các đoạn nhỏ vừa vặn (thường từ 256–512 tokens) với một độ đè phủ (overlap khoảng 10–20%) để đưa vào mô hình nhúng.
  - **Word-Aware Chunking:** Chiến lược phân đoạn thông minh bảo toàn ranh giới từ ghép tiếng Việt, đảm bảo điểm cắt không làm đứt đôi giữa các âm tiết của một từ.

---

## 3. Tầng Truy Hồi (Retrieval Layer) & Không Gian Vectơ

- **Bi-Encoder:** Kiến trúc mô hình hai nhánh mã hóa câu hỏi và các đoạn văn bản một cách độc lập thành các vectơ cố định trong không gian nhiều chiều.
- **Dense Retrieval (Truy hồi dày / Biểu trưng):** Phương pháp tìm kiếm dựa trên sự tương đồng ngữ nghĩa giữa các vectơ (ví dụ dùng Cosine Similarity) bằng các mô hình nhúng như `Vietnamese_Embedding`, `bkai-foundation-models/vietnamese-bi-encoder` hoặc `bge-m3`.
- **Sparse Retrieval (Truy hồi thưa / Từ khóa):** Phương pháp tìm kiếm dựa trên tần suất xuất hiện từ khóa chính xác như thuật toán **BM25** hoặc các mô hình trọng số token thưa.
- **Hybrid Search (Truy hồi lai):** Giải pháp kết hợp song song cả Dense Retrieval (hiểu ngữ nghĩa, từ đồng nghĩa) và Sparse Retrieval (bắt chính xác thuật ngữ hiếm, số hiệu, mã định danh) thông qua công thức cộng trọng số hoặc giải thuật **RRF (Reciprocal Rank Fusion)**.
- **Multi-Vector / Late Interaction (ColBERT):** Cơ chế giữ lại toàn bộ tập vectơ của từng token thay vì nén thành 1 vectơ duy nhất, giúp tính điểm tương tác muộn chi tiết hơn nhưng tốn tài nguyên hơn.
- **Vector Database (Cơ sở dữ liệu Vectơ):** Hệ thống lưu trữ và tìm kiếm các vectơ không gian chuyên dụng với tốc độ cao như FAISS, Chroma, Weaviate, Qdrant.

---

## 4. Tầng Tái Xếp Hạng (Reranking Layer)

- **Cross-Encoder / Reranker:** Mô hình cho câu hỏi và đoạn văn bản đi qua cùng một lớp Self-Attention toàn cục để đánh giá trực tiếp mối quan hệ logic và mức độ liên quan chi tiết giữa hai bên.
- **Vietnamese_Reranker:** Mô hình Reranker chuyên dụng cho tiếng Việt (được tinh chỉnh từ `bge-reranker-v2-m3`), thường được áp dụng ở giai đoạn thứ hai để xếp hạng lại Top 30–50 ứng viên thành Top 3–5 đoạn tối ưu nhất.
- **Khai thác Mẫu Tiêu Cực (Negative Mining):** Chiến lược lựa chọn các mẫu sai khi huấn luyện mô hình truy hồi:
  - **Hard Negatives:** Mẫu tiêu cực rất giống câu đúng, dễ làm mô hình nhầm lẫn.
  - **Semi-Hard Negatives:** Mẫu tiêu cực có độ tương đồng vừa phải, giúp cân bằng dữ liệu và tối ưu hiệu năng cho Cross-Encoder.

---

## 5. Tầng Sinh Văn Bản & Mô Hình Ngôn Ngữ Lớn (LLM)

- **LLM tiếng Việt (Large Language Model):** Mô hình ngôn ngữ lớn đóng vai trò tổng hợp ngữ cảnh được truy hồi thành câu trả lời tự nhiên (ví dụ: `SeaLLM-7B-v2`, `Vistral-7B`, `VinaLLaMA`, `Qwen2.5`, `GPT-4o`).
- **System Prompt & Rào chắn (Guardrails):** Mẫu hướng dẫn hệ thống chỉ định rõ vai trò, bắt buộc mô hình chỉ dùng thông tin trong ngữ cảnh được cung cấp, trích dẫn nguồn cụ thể và phải tự động từ chối (refusal) nếu thiếu bằng chứng.
- **Ảo giác (Hallucination):** Hiện tượng LLM tự bịa đặt thông tin không có trong tài liệu nguồn.
- **Độ bền vững phương ngữ (Dialect Robustness):** Khả năng ứng phó của mô hình trước sự biến đổi từ vựng, đại từ và ngữ khí giữa các phương ngữ tiếng Việt (Bắc, Trung, Nam).

---

## 6. Khung Đánh Giá Hệ Thống (Evaluation Metrics & Frameworks)

- **Chỉ số Đánh giá Truy hồi (Retrieval Metrics):**
  - **Recall@K:** Tỷ lệ tài liệu thực sự liên quan xuất hiện trong Top-K kết quả.
  - **MRR (Mean Reciprocal Rank):** Chỉ số đo thứ hạng vị trí của tài liệu liên quan đầu tiên tìm thấy.
  - **nDCG@K:** Đánh giá chất lượng xếp hạng dựa trên độ liên quan có phân cấp của các đoạn văn bản.
  - **Exist@m:** Tỷ lệ câu hỏi mà hệ thống tìm thấy ít nhất 1 đoạn chứa đáp án đúng nằm trong Top-m ứng viên.
- **Bộ chỉ số RAGAS / RAG Triad:**
  - **Context Relevance (Độ liên quan ngữ cảnh):** Đo tỷ lệ thông tin hữu ích trong các đoạn được truy hồi so với lượng văn bản dư thừa.
  - **Answer Faithfulness / Groundedness (Độ trung thực của câu trả lời):** Đo tỷ lệ các phát biểu trong câu trả lời có căn cứ trực tiếp từ ngữ cảnh trích xuất (kiểm tra ảo giác).
  - **Answer Relevance (Độ liên quan câu trả lời):** Đo mức độ phù hợp giữa câu trả lời tạo ra và mục đích ban đầu của câu hỏi.
