# Day 22 Lab — Reflection & Analysis

**Họ và Tên:** Võ Thiên Phú
**Mã Học Viên:** 2A202600446

---

## 1. Tổng quan Lab

Lab xây dựng một production-grade RAG pipeline với đầy đủ observability (LangSmith), prompt versioning (Prompt Hub), automated evaluation (RAGAS), và output safety (Guardrails AI).

---

## 2. Kết quả RAGAS Evaluation

### V1 vs V2 Comparison Table

| Metric | V1 (Concise) | V2 (Structured) | Winner |
|--------|-------------|-----------------|--------|
| **faithfulness** | **0.9539** | 0.8675 | V1 |
| answer_relevancy | 0.8886 | **0.9025** | V2 |
| context_recall | **0.9600** | 0.9400 | V1 |
| context_precision | **0.9367** | 0.9133 | V1 |
| **Mean** | **0.9348** | **0.9058** | V1 |

- **Faithfulness target (>= 0.8):** V1 = 0.9539 ✅ | V2 = 0.8675 ✅
- **Overall winner:** Prompt V1 (3/4 metrics)

---

## 3. Phân tích: Tại sao V1 thắng?

### 3.1 Faithfulness — V1 (0.9539) vs V2 (0.8675)

Prompt V1 yêu cầu câu trả lời ngắn gọn (2-4 câu), tập trung vào việc dùng **CHỈ** context được cung cấp. Điều này buộc model:

- Trả lời ngắn gọn, ít cơ hội hallucinate thông tin ngoài context
- Giảm thiểu chi tiết không có trong retrieved passages
- Tập trung vào factual content thay vì suy diễn

Prompt V2 yêu cầu 3-5 câu với cấu trúc rõ ràng, nhưng việc viết "clear, well-organized answer" khiến model có xu hướng **bổ sung thông tin tự nhiên** — dẫn đến nhiều claims hơn, và một số claims không hoàn toàn có trong context.

**Kết luận:** Constraints ngắn gọn của V1 giúp giảm hallucination, tăng faithfulness.

### 3.2 Answer Relevancy — V2 (0.9025) vs V1 (0.8886)

V2 thắng ở answer relevancy vì:

- Prompt V2 yêu cầu "Identify the key facts relevant to the question" — giúp model tập trung đúng vào câu hỏi
- Structured format (instructions 1-4) khuyến khích model trả lời **đúng trọng tâm** câu hỏi
- Với câu hỏi phức tạp (như "How does RAGAS compute faithfulness?"), V2 cho câu trả lời có tổ chức hơn

### 3.3 Context Recall & Precision — V1 thắng cả hai

- **Context Recall (V1: 0.96 vs V2: 0.94):** Cả hai dùng cùng vectorstore và retriever (k=3), nên sự khác biệt nhỏ. Prompt V1 ngắn gọn hơn nên model ít bỏ sót thông tin quan trọng trong context.
- **Context Precision (V1: 0.9367 vs V2: 0.9133):** V1 tập trung vào "answer using ONLY the provided context" — giảm việc sử dụng thông tin nhiễu từ context không liên quan.

### 3.4 Tổng hợp

| Khía cạnh | V1 tốt hơn | V2 tốt hơn |
|-----------|-----------|-----------|
| Faithfulness | ✅ Giảm hallucination | |
| Answer Relevancy | | ✅ Trả lời đúng trọng tâm |
| Context Recall | ✅ Ít bỏ sót context | |
| Context Precision | ✅ Ít nhiễu | |
| Readability | | ✅ Có cấu trúc rõ ràng |

**Khuyến nghị:** Nếu ưu tiên **factual accuracy** (không hallucinate), dùng **V1**. Nếu ưu tiên **user experience** (câu trả lời dễ đọc, có tổ chức), dùng **V2**. Có thể kết hợp: dùng V2 cho câu hỏi phức tạp, V1 cho câu hỏi đơn giản.

---

## 4. Nhận xét về các bước thực hiện

### Step 1 — LangSmith RAG Pipeline
- RAG pipeline hoạt động ổn định với 50 câu hỏi
- FAISS vector search với chunk_size=500, overlap=50 cho kết quả retrieval tốt
- Tất cả 50 traces được gửi lên LangSmith thành công

### Step 2 — Prompt Hub & A/B Routing
- Push/pull prompts lên LangSmith Prompt Hub hoạt động đúng
- Deterministic routing (MD5 hash) đảm bảo reproducibility
- Tỷ lệ split 38%/62% (19 V1, 31 V2) — gần 50/50

### Step 3 — RAGAS Evaluation
- V1 đạt faithfulness = 0.9539 — vượt mục tiêu 0.8 rất nhiều
- V2 đạt faithfulness = 0.8675 — cũng vượt mục tiêu
- Cả 4 metrics được tính đầy đủ

### Step 4 — Guardrails AI
- PII Detector hoạt động tốt với 4 loại PII (email, phone, SSN, credit card)
- JSON Formatter auto-repair thành công với fences, single quotes, trailing commas
- Không có lỗi khi chạy

---

## 5. Kết luận

- **Prompt V1 (concise)** là lựa chọn tốt hơn cho RAG pipeline vì minimize hallucination và maximize faithfulness
- **LangSmith Prompt Hub** là công cụ hiệu quả để quản lý và version prompts
- **RAGAS** cung cấp đánh giá objective giúp so sánh các prompt versions một cách systematic
- **Guardrails AI** giúp bảo vệ LLM outputs khỏi PII leakage và malformed outputs
