# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Nguyễn Mai Phương]
**Nhóm:** [C401 - B1]
**Ngày:** [10/04/2026]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (gần 1.0) nghĩa là hai text embedding vectors có hướng gần giống nhau trong không gian vector, tức là hai đoạn văn bản mang ý nghĩa ngữ nghĩa tương tự nhau. Giá trị càng gần 1 thì nội dung càng liên quan chặt chẽ về mặt ngữ nghĩa.

**Ví dụ HIGH similarity:**
- Sentence A: "Python là ngôn ngữ lập trình phổ biến cho machine learning."
- Sentence B: "Python được sử dụng rộng rãi trong lĩnh vực trí tuệ nhân tạo."
- Tại sao tương đồng: Cả hai câu đều nói về Python trong ngữ cảnh AI/ML, chia sẻ cùng chủ đề và ngữ nghĩa tương tự dù dùng từ khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Python là ngôn ngữ lập trình phổ biến cho machine learning."
- Sentence B: "Hôm nay thời tiết ở Hà Nội rất đẹp."
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác nhau (lập trình vs thời tiết), không có sự trùng lặp về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo góc giữa hai vectors mà không phụ thuộc vào độ dài (magnitude) của chúng, nên hai đoạn văn cùng ý nghĩa nhưng khác độ dài vẫn cho similarity cao. Euclidean distance bị ảnh hưởng bởi magnitude, dẫn đến kết quả sai lệch khi so sánh embeddings có norm khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Phép tính:*
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
>
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks` — tăng từ 23 lên 25 chunks. Overlap nhiều hơn giúp bảo toàn ngữ cảnh tại các ranh giới chunk, tránh mất thông tin khi một câu hoặc ý tưởng bị cắt ngang giữa hai chunk liền kề, từ đó cải thiện chất lượng retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Giáo trình triết học Mác - Lênin

**Tại sao nhóm chọn domain này?**
> Đây là môn học bắt buộc với toàn bộ sinh viên đại học tại Việt Nam, nhưng tài liệu thường dày và trừu tượng, khiến sinh viên khó tra cứu nhanh khi ôn thi. Một RAG chatbot trên domain này cho phép sinh viên đặt câu hỏi tự nhiên như 'Vật chất là gì theo Lenin?' và nhận câu trả lời trích dẫn đúng chương, đúng nguồn — thay vì phải lật từng trang giáo trình. Ngoài ra, nội dung giáo trình có tính ổn định cao (ít thay đổi theo năm), rất phù hợp để xây dựng và đánh giá một hệ thống RAG mà không lo dữ liệu bị lỗi thời.

Dưới đây là form đã được tổng hợp từ file `Giao-trinh.md`:

---

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Giáo trình Triết học Mác - Lênin | Đại học quốc gia Hà Nội - Trung tâm thư viện và trí thức số | ~683,585 | source="Giáo trình Triết học Mác-Lênin", level="Đại học", audience="Khối ngành ngoài lý luận chính trị" |

---

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `chapter` | int | `2` | Lọc theo chương khi người dùng hỏi về một phần cụ thể (VD: "Câu hỏi về chương 2") |
| `chapter_title` | string | `"Chủ nghĩa duy vật biện chứng"` | Giúp LLM trích dẫn đúng tiêu đề chương trong câu trả lời |
| `topic` | string | `"vật chất, ý thức, phép biện chứng"` | Hỗ trợ semantic filter theo chủ đề triết học cụ thể |
| `source` | string | `"Giáo trình Triết học Mác-Lênin, Bộ GD&ĐT"` | Cho phép cite nguồn chính xác trong câu trả lời RAG |
| `chunk_index` | int | `14` | Giúp reconstruct ngữ cảnh xung quanh chunk được retrieve (lấy chunk liền kề) |
| `audience` | string | `"Khối ngành ngoài lý luận chính trị"` | Điều chỉnh độ phức tạp câu trả lời nếu hệ thống phục vụ nhiều đối tượng |


---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Giáo trình Triết học (684,722 chars) | FixedSizeChunker (`fixed_size`) | 1370 | 499.8 | No (5%) — cắt giữa từ/câu, mất ngữ cảnh |
| Giáo trình Triết học | SentenceChunker (`by_sentences`) | 1610 | 423.3 | Yes (100%) — luôn cắt tại ranh giới câu |
| Giáo trình Triết học | RecursiveChunker (`recursive`) | 2029 | 335.5 | Partial (30%) — ưu tiên `\n\n` nhưng fallback cắt ngang |

### Strategy Của Tôi

**Loại:** Custom — `ParentChildChunker` (Small-to-Big)

**Mô tả cách hoạt động:**
> Bước 1: Tách document thành **parent chunks** dựa trên heading regex — nhận diện các mẫu tiêu đề giáo trình: `CHƯƠNG`, `**I. …**`, `**1\. …**`, `***a. …***`. Mỗi parent là một section/subsection hoàn chỉnh. Bước 2: Mỗi parent được chia nhỏ thành **child chunks** bằng `SentenceChunker` (3 câu/chunk). Nếu child nào vượt 500 ký tự, dùng `FixedSizeChunker` cắt lại với overlap=50. Bước 3: Khi retrieval, embed và search trên child (nhỏ, chính xác), nhưng trả về parent_content (lớn, đầy đủ ngữ cảnh) cho LLM.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Giáo trình triết học có cấu trúc phân cấp rõ ràng (Chương → Mục → Tiểu mục → a/b/c), mỗi tiểu mục là một đơn vị ngữ nghĩa hoàn chỉnh. Small-to-Big khai thác cấu trúc này: child chunk nhỏ (avg 319 chars) giúp matching chính xác thuật ngữ triết học, còn parent chunk lớn cung cấp đủ ngữ cảnh để LLM trả lời đúng — tránh trường hợp retrieve đúng câu nhưng thiếu context giải thích.

**Code snippet (nếu custom):**
```python
from src.chunking import ParentChildChunker

chunker = ParentChildChunker(child_sentences=3, child_max_chars=500)
children = chunker.chunk(text)
# => 71 parent sections, 2220 child chunks
# => avg child: 319 chars (embed), avg parent: 26258 chars (LLM context)

# Each child dict:
# {
#   "child_id": "p2_c0",
#   "parent_id": "p2",
#   "content": "child text for embedding...",
#   "parent_content": "full section text for LLM...",
#   "heading": "a. Nguồn gốc của triết học"
# }
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Giáo trình Triết học | SentenceChunker (best baseline) | 1610 | 423.3 | Tốt — giữ ranh giới câu, nhưng mất ngữ cảnh section |
| Giáo trình Triết học | ParentChildChunker | 2220 child / 71 parent | 319 (child) | Tốt hơn — child nhỏ giúp match chính xác, parent lớn cung cấp context đầy đủ cho LLM |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | ParentChildChunker (Small-to-Big) | 8 | Child nhỏ (319 chars) match chính xác thuật ngữ; parent lớn giữ ngữ cảnh section cho LLM; 4/5 queries relevant top-3 | Parent quá lớn (avg 26K chars) có thể vượt context window LLM; heading regex chỉ hoạt động tốt với giáo trình có format chuẩn |
| Chu Thị Ngọc Huyền | Sentence Chunking | 8 | Bảo toàn ngữ cảnh logic của lập luận triết học bằng cách tôn trọng ranh giới câu, giúp RAG retrieval cao hơn | Chunk size nhỏ hơn (422 vs 500 chars) có thể bỏ lỡ context nếu lập luận triết học kéo dài trên nhiều câu |
| Hứa Quang Linh | AgenticChunker | 9 | Tự phát hiện ranh giới chủ đề bằng embedding; mỗi chunk mang đủ ngữ cảnh 1 khái niệm triết học | Chunk lớn (avg ~4K chars) có thể chiếm nhiều context window; chạy chậm hơn (~97s trên 684K chars) |
| Tôi Chu Bá Tuấn Anh | RecursiveChunker| 8.5| Cân bằng tốt giữa ngữ nghĩa và độ dài; giữ được cấu trúc tài liệu (paragraph/sentence); retrieval ổn định| Phụ thuộc heuristic nên đôi khi split chưa tối ưu; có thể tạo chunk rời rạc; cần tuning chunk size và overlap thêm|
| Nguyễn Thị Tuyết | RecursiveChunker | 8.0 | Giữ ngữ cảnh tốt, ít cắt ngang đoạn | Cần tinh chỉnh thêm theo chương |
| Nguyễn Văn Lĩnh          | SentenceChunker(3) | 8.5                   | Giữ ngữ pháp, retrieval scores cao (0.3-0.5) | Tăng số lượng chunk (1610 vs 300), có thể chậm retrieval |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> AgenticChunker của Hứa Quang Linh đạt retrieval score cao nhất (9/10) vì nó tự động phát hiện ranh giới chủ đề bằng embedding, phù hợp với giáo trình triết học — nơi mỗi khái niệm (vật chất, ý thức, phép biện chứng…) cần được giữ nguyên trong một chunk để LLM trả lời chính xác. Tuy nhiên, nếu cân nhắc chi phí vận hành (tốc độ, khả năng mở rộng), ParentChildChunker là lựa chọn thực tế hơn vì đạt 8/10 nhưng chạy nhanh hơn nhiều và khai thác trực tiếp cấu trúc phân cấp sẵn có của giáo trình mà không cần gọi embedding model khi chunking.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex lookbehind `(?<=[.!?])(?:\s|\n)` để tách câu — detect ranh giới câu sau dấu `.`, `!`, `?` theo sau bởi khoảng trắng hoặc newline. Xử lý edge case: text rỗng trả về `[]`, câu không có dấu kết thúc vẫn được giữ nguyên (không bị mất). Sau khi tách, gom mỗi `max_sentences_per_chunk` câu thành một chunk, strip whitespace thừa.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy thử lần lượt từng separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Với mỗi separator, split text thành parts rồi gom lại sao cho mỗi chunk không vượt `chunk_size`. Nếu chunk con vẫn quá lớn, đệ quy với separator tiếp theo. Base case: text đã nhỏ hơn `chunk_size` thì trả về nguyên — hoặc hết separator thì force-split theo `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Dual-backend: ưu tiên ChromaDB nếu có (`collection.add` với ids, documents, embeddings, metadatas), nếu không thì fallback sang in-memory list. Khi search, tính dot product giữa query embedding và tất cả stored embeddings, sort descending theo score, trả về top-k. ChromaDB backend dùng `collection.query` và convert distance sang score bằng `1.0 - distance`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter **trước** khi search: in-memory dùng list comprehension lọc metadata match tất cả key-value pairs, rồi chạy similarity trên tập đã lọc; ChromaDB truyền `where` clause trực tiếp. `delete_document` tìm tất cả record có `metadata.doc_id == doc_id`, xóa khỏi store (in-memory dùng list comprehension loại bỏ, ChromaDB dùng `collection.delete`), trả về `True/False` dựa trên có record nào bị xóa không.

### KnowledgeBaseAgent

**`answer`** — approach:
> RAG pattern 3 bước: (1) Search top-k chunks liên quan từ `EmbeddingStore`, (2) Ghép nội dung các chunks thành context block (nối bằng `\n\n`), (3) Build prompt có cấu trúc `"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"` rồi gọi `llm_fn(prompt)`. Context được inject trực tiếp vào prompt, không cần template engine.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED
============================= 42 passed in 0.11s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Triết học là hệ thống quan điểm lý luận chung nhất về thế giới." | "Triết học nghiên cứu những quy luật phổ biến của tự nhiên, xã hội và tư duy." | high | 0.5986 | Yes — high (cùng định nghĩa triết học) |
| 2 | "Chủ nghĩa duy vật cho rằng vật chất có trước, quyết định ý thức." | "Chủ nghĩa duy tâm cho rằng ý thức có trước, quyết định vật chất." | low | 0.9829 | No — thực tế rất high! |
| 3 | "Phép biện chứng duy vật nghiên cứu mối liên hệ phổ biến và sự phát triển." | "Hôm nay trời mưa rất to ở thành phố Hồ Chí Minh." | low | 0.3529 | Yes — low (khác domain hoàn toàn) |
| 4 | "Thực tiễn là tiêu chuẩn của chân lý." | "Nhận thức phải được kiểm nghiệm qua hoạt động thực tiễn." | high | 0.6065 | Yes — high (cùng chủ đề thực tiễn-nhận thức) |
| 5 | "Nguồn gốc của triết học gắn liền với sự phát triển của tư duy trừu tượng." | "Hình thái kinh tế xã hội là phạm trù của chủ nghĩa duy vật lịch sử." | low | 0.5708 | Partial — trung bình, cùng domain nhưng khác chủ đề |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 2 bất ngờ nhất: duy vật và duy tâm là hai quan điểm đối lập nhau nhưng cosine similarity lên tới 0.9829. Điều này cho thấy embeddings biểu diễn chủ đề và cấu trúc ngữ nghĩa (cả hai câu cùng nói về mối quan hệ vật chất–ý thức, dùng từ vựng gần giống nhau) chứ không mã hóa tính đối lập logic. Đây là hạn chế quan trọng của embedding models: chúng không phân biệt được "A quyết định B" vs "B quyết định A" — cần kết hợp thêm metadata filter hoặc reranker để xử lý.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Triết học là gì? | Triết học là hệ thống tri thức lý luận chung nhất về thế giới và vị trí con người trong thế giới đó. |
| 2 | Vấn đề cơ bản của triết học gồm những mặt nào? | Gồm mặt bản thể luận (vật chất - ý thức cái nào có trước) và mặt nhận thức luận (con người có khả năng nhận thức thế giới hay không). |
| 3 | Vai trò của thực tiễn đối với nhận thức là gì? | Thực tiễn là cơ sở, động lực, mục đích và tiêu chuẩn kiểm tra chân lý của nhận thức. |
| 4 | Phép biện chứng duy vật nhấn mạnh điều gì? | Nhấn mạnh sự vận động, phát triển và mối liên hệ phổ biến của sự vật hiện tượng. |
| 5 | Sự khác nhau giữa chủ nghĩa duy vật và duy tâm là gì? | Duy vật coi vật chất có trước, quyết định ý thức; duy tâm coi ý thức/tinh thần có trước. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Triết học là gì? | "Triết học là hạt nhân lý luận của thế giới quan. Triết học Mác-Lênin đem lại thế giới quan duy vật biện chứng…" | 0.6600 | Partial — nói về vai trò triết học, chưa trả lời trực tiếp "triết học là gì" | Agent trích dẫn triết học = hạt nhân thế giới quan, đúng hướng nhưng thiếu định nghĩa chính thức |
| 2 | Vấn đề cơ bản của triết học gồm những mặt nào? | "Vấn đề cơ bản của triết học… trước khi giải quyết các vấn đề cụ thể, nó buộc phải giải quyết…" | 0.7605 | Yes — đúng section "Nội dung vấn đề cơ bản của triết học" | Agent trích đúng phần bản thể luận và nhận thức luận |
| 3 | Vai trò của thực tiễn đối với nhận thức là gì? | "Vai trò của thực tiễn đối với nhận thức:     Thực tiễn là cơ sở, động lực của nhận thức…" | 0.7683 | Yes — match chính xác heading + nội dung | Agent trả lời đầy đủ: cơ sở, động lực, mục đích, tiêu chuẩn kiểm tra chân lý |
| 4 | Phép biện chứng duy vật nhấn mạnh điều gì? | "Ph.Ăngghen đòi hỏi tư duy khoa học phải thấy sự thống nhất giữa biện chứng khách quan và chủ quan…" | 0.7092 | Yes — nói đúng về biện chứng duy vật, dẫn đến khái niệm phép biện chứng | Agent trích dẫn biện chứng khách quan–chủ quan, đúng nội dung |
| 5 | Sự khác nhau giữa chủ nghĩa duy vật và duy tâm là gì? | "Mác và Ăngghen đấu tranh chống chủ nghĩa duy tâm, phê phán chủ nghĩa duy vật siêu hình…" | 0.6396 | Partial — nói về cuộc đấu tranh duy vật vs duy tâm nhưng chưa define rõ sự khác biệt | Agent thiếu định nghĩa trực tiếp, chỉ mô tả bối cảnh lịch sử |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> AgenticChunker của Quang Linh cho thấy việc dùng embedding để tự động phát hiện ranh giới chủ đề có thể đạt retrieval score cao hơn (9/10) so với rule-based approach. Đồng thời, cách RecursiveChunker của Tuấn Anh đạt 8.5/10 chứng minh rằng tuning tốt các heuristic đơn giản cũng cho kết quả cạnh tranh mà không cần thêm complexity.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> 

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ giảm kích thước parent chunk bằng cách tách ở mức heading sâu hơn (sub-subsection) để parent_content không quá lớn (hiện avg 26K chars — dễ vượt context window LLM). Ngoài ra, tôi sẽ thêm metadata chapter/topic vào mỗi child chunk để hỗ trợ filtered search, và thử kết hợp reranker để xử lý trường hợp embedding không phân biệt được các khái niệm đối lập (như duy vật vs duy tâm, Pair 2 trong Section 5).

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **81 / 100** |
