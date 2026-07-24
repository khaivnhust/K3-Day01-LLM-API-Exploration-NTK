# K3 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng placeholder mặc định bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Ở temperature 0.0 và 0.5, model chọn một sự thật "an toàn", phổ biến (món phở) và diễn đạt khá mạch lạc, ít lặp lại giữa các lần chạy. Khi tăng lên 1.0 và 1.5, câu trả lời bắt đầu chọn các chủ đề đa dạng và bất ngờ hơn hẳn (chữ Quốc ngữ, bánh mì), câu văn cũng "sáng tạo"/bay bổng hơn nhưng đôi khi kém chặt chẽ hoặc lạc đề nhẹ. Nhìn chung nhiệt độ càng cao thì độ đa dạng và tính ngẫu nhiên của output càng tăng, đổi lại độ ổn định/độ chính xác giảm.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Khoảng 0.0–0.3. Chatbot hỗ trợ khách hàng cần trả lời nhất quán, đúng chính sách/quy trình công ty, tránh "bịa" thông tin hay đổi giọng điệu thất thường giữa các lượt hỏi giống nhau nhiệt độ thấp giúp output ổn định và dễ kiểm soát chất lượng hơn là sáng tạo.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Tổng output/ngày = 10.000 × 3 × 350 = 10.500.000 token = 10.500 nghìn token.
> Chi phí GPT-4o = 10.500 × 0.010 = **105 USD/ngày**.
> Chi phí GPT-4o-mini = 10.500 × 0.0006 = **6.3 USD/ngày**.
> → GPT-4o đắt hơn mini khoảng **16.7 lần** (= 0.010/0.0006) cho cùng workload.
> GPT-4o đáng dùng khi task cần suy luận phức tạp, độ chính xác cao (phân tích hợp đồng, code review, tư vấn y tế/pháp lý). Mini phù hợp cho các tác vụ đơn giản, khối lượng lớn như trả lời FAQ, phân loại intent, tóm tắt ngắn nơi chất lượng "đủ tốt" đã đáp ứng được nhu cầu.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Bản "giáo viên tiểu học" dùng câu chuyện ẩn dụ (cuốn sổ thần kỳ của lớp học), từ vựng đơn giản, xưng hô "con/cô", có emoji-style liệt kê thân thiện và kết bằng câu hỏi gợi mở ("Con thấy thú vị không nào?"). Bản "chuyên gia tài chính" dùng thuật ngữ kỹ thuật (Distributed Ledger Technology, Merkle Root, Nonce, Proof of Work/Stake, mã hóa khóa công khai), cấu trúc theo mục I/II/III rõ ràng như tài liệu chuyên môn, dài và chi tiết hơn hẳn. System prompt không chỉ đổi giọng văn mà còn đổi cả chiều sâu nội dung, mức độ thuật ngữ và cấu trúc trình bày chứng tỏ persona định hình toàn bộ chiến lược trả lời của model chứ không chỉ văn phong bề mặt.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Với đoạn văn 148 từ: tiktoken đếm được 159 token, trong khi ước lượng `số từ / 0.75` cho ra 197.3 token  chênh khoảng **19.4%** (ước lượng theo công thức "0.75 từ ≈ 1 token" cao hơn số thật, vì công thức đó được hiệu chỉnh cho tiếng Anh chứ không phải tiếng Việt). Tiếng Việt có dấu thanh và nhiều âm tiết đơn lẻ (từ đơn) không khớp với các "subword" phổ biến mà tokenizer học được chủ yếu từ dữ liệu tiếng Anh, nên nhiều khi một từ tiếng Việt bị tách thành 2–3 token thay vì 1, khiến số token thực tế dao động khó đoán so với công thức ước lượng thô dựa trên số từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất trong các giao diện chat tương tác trực tiếp với người dùng (chatbot, trợ lý ảo), vì nó cho phép người dùng thấy phản hồi xuất hiện ngay lập tức thay vì chờ toàn bộ câu trả lời (có thể mất vài giây đến hàng chục giây với output dài) điều này giảm cảm giác "đứng hình" và giúp trải nghiệm mượt mà hơn, đặc biệt quan trọng khi max_tokens lớn. Ngược lại, non-streaming phù hợp hơn khi hệ thống cần xử lý toàn bộ response trước khi dùng ví dụ parse JSON có cấu trúc, chạy qua bước kiểm duyệt/hậu xử lý trước khi hiển thị, gọi API trong pipeline batch không có người dùng trực tiếp theo dõi, hoặc khi cần retry toàn bộ request nếu có lỗi giữa chừng (stream dở dang khó rollback).

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff giãn thời gian chờ ngày càng dài giữa các lần retry (0.1s → 0.2s → 0.4s...), giúp giảm áp lực lên server đang quá tải dần dần thay vì giữ nguyên tần suất tấn công cho server thêm thời gian để phục hồi trước mỗi lần thử lại tiếp theo. Nếu hàng nghìn client cùng dùng delay cố định giống nhau (ví dụ luôn chờ đúng 1 giây), tất cả sẽ đồng loạt retry cùng lúc theo từng nhịp 1 giây, tạo ra các đợt "thundering herd" liên tục dội vào server đang quá tải, khiến nó không bao giờ có cơ hội hồi phục và có thể làm tình trạng quá tải kéo dài hoặc tệ hơn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> System prompt: *"Bạn là trợ giảng thân thiện của khóa học AI, trả lời ngắn gọn (tối đa 3-4 câu) bằng tiếng Việt, chỉ giải thích sâu hơn khi người dùng yêu cầu."*
> Lý do chọn "trả lời ngắn gọn": trợ lý CLI hiển thị qua terminal, câu trả lời dài dòng vừa tốn token/chi phí vừa làm loãng thông tin quan trọng, ép model đi thẳng vào trọng tâm giúp trải nghiệm dùng nhanh và rẻ hơn. Lý do chỉ định "bằng tiếng Việt": tránh model tự chuyển sang tiếng Anh giữa chừng (hay gặp với model đa ngôn ngữ khi câu hỏi có từ mượn/thuật ngữ tiếng Anh), đảm bảo tính nhất quán cho người dùng mục tiêu là học viên Việt Nam.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> Hạn chế lớn nhất là history chỉ giữ 3 lượt gần nhất (6 message) nếu người dùng hỏi lại điều đã trao đổi từ lượt thứ 4 trở về trước, trợ lý sẽ "quên" hoàn toàn ngữ cảnh đó, gây trải nghiệm khó chịu trong các phiên chat dài. Cải thiện cụ thể: thêm bước tóm tắt, mỗi khi history sắp vượt quá giới hạn, gọi thêm một lần API để nén các lượt cũ thành 1 đoạn tóm tắt ngắn (2-3 câu), lưu đoạn tóm tắt đó làm một message hệ thống bổ sung đặt ngay sau persona, rồi mới cắt history chi tiết như hiện tại. Cách này giữ được ngữ cảnh dài hạn ở dạng nén mà không làm phình to chi phí token mỗi lượt gọi.

---

## Danh Sách Kiểm Tra Nộp Bài

- [X] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [X] Cả 4 checkpoint pytest đều pass
- [X] Tất cả 9 câu trong file này đã được trả lời
- [X] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
