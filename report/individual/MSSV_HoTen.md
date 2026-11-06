# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603734
- **Họ và tên:** Nguyễn Tấn Dũng
- **SEED** 3166

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

====================================================================================================
BẢNG TỔNG HỢP SO SÁNH 4 THẾ HỆ AI
----------------------------------------------------------------------------------------------------
Phương pháp                     | F1 (test) | F1 (SUB 40) | p95 (ms/tin) | USD / 1k tin
----------------------------------------------------------------------------------------------------
1. BoW + LogReg (tự code)       |   0.903   |    0.919    |    0.070 ms  |    $0.000
2. BoW + MLP (deep learning)    |   0.944   |    0.889    |    0.325 ms  |    $0.000
3a. Encoder (MiniLM) + kNN      |   0.905   |    0.950    |   89.316 ms  |    $0.000
3b. Encoder (MiniLM) + LogReg   |   0.984   |    0.974    |   22.136 ms  |    $0.000
3c. Encoder (MiniLM) + MLP head |   0.979   |    0.974    |   41.310 ms  |    $0.000
4a. LLM nhỏ zero-shot           |     —     |    0.909    | 3110.430 ms  |    $0.007
4b. LLM lớn zero-shot           |     —     |    0.976    | 1214.269 ms  |    $0.065
4c. LLM nhỏ few-shot            |     —     |    0.930    | 1261.563 ms  |    $0.018
====================================================================================================

## 3. Dự đoán và thực tế


1. **`tradeoff` (Biên Pareto và SLA):**
   - *Dự đoán:* Nhóm mô hình BoW và Encoder+LogReg sẽ đạt chuẩn SLA 50 ms. BoW+LogReg và Encoder+LogReg sẽ nằm trên đường Pareto.
   - *Thực tế:* Khớp chính xác với dự đoán. Cả hai mô hình đều nằm trên đường biên Pareto, trong đó BoW+LogReg đạt tốc độ cao nhất (p95 ~0.07 ms), còn Encoder+LogReg tối ưu nhất về F1 trong nhóm đạt SLA (F1_SUB = 0.974, p95 ~22.1 ms).
   - *Nguyên nhân lệch (hoặc khớp):* Các mô hình hồi quy tuyến tính gọn nhẹ trên vector đặc trưng chất lượng cao luôn cân bằng tốt nhất giữa tốc độ suy luận và năng lực phân loại, không bị độ trễ ẩn như mạng nhiều tầng hay API.

2. **`dl_curve` (Đường cong huấn luyện MLP):**
   - *Dự đoán:* Val loss sẽ giảm nhanh ở vài epoch đầu rồi tăng trở lại do overfitting; F1 chỉ nhỉnh hơn nhẹ hoặc tương đương Logistic Regression vì feature BoW không có ngữ cảnh.
   - *Thực tế:* Điểm val loss đạt cực tiểu rất sớm ở epoch thứ 6 rồi bắt đầu tăng dần, thể hiện rõ hiện tượng overfit; F1 test đạt 0.944 (so với 0.903 của BoW+LogReg), nhưng trên tập SUB thì F1 thậm chí giảm xuống 0.889.
   - *Nguyên nhân lệch:* Bộ dữ liệu nhỏ với feature túi từ rời rạc khiến việc tăng thêm tham số ẩn (128k tham số) chủ yếu dẫn đến ghi nhớ dữ liệu huấn luyện thay vì học được quy luật tổng quát mới.

3. **`llm_models` (Model lớn vs model nhỏ zero-shot):**
   - *Dự đoán:* Model lớn có F1 cao hơn một chút (~0.95 vs ~0.90) và đắt hơn 2–5 lần; độ trễ của LLM cao hơn hàng chục lần so với Embedding.
   - *Thực tế:* Model lớn `gpt-6.1-terra` đạt F1 = 0.976 so với `gpt-6-luna` đạt 0.909 (chênh lệch đúng như dự đoán), chi phí đắt hơn gấp 10 lần ($0.065 vs $0.007/1k tin), độ trễ p95 lên tới 1.2–3.1 giây (chậm hơn hàng chục nghìn lần so với BoW).
   - *Nguyên nhân lệch:* Độ trễ thực tế bị khuếch đại do mạng viễn thông và cơ chế sinh token tuần tự qua mạng, chứng minh LLM khó có thể đứng độc lập ở hệ thống thời gian thực cao.

## 4. Phân tích 50 case khó

- **Mô hình tụt điểm nhiều nhất:** `1. BoW + LogReg` tụt mạnh nhất từ F1 = 0.919 (test thường) xuống chỉ còn **F1 = 0.286** trên 50 case khó (Accuracy chỉ 50%). Mô hình tụt nặng nhất ở nhóm **Né từ khoá (Obfuscation)** và **Spam giọng tự nhiên (Natural spam)**.
- **Hai case cụ thể và cơ chế sai:**
  1. **Case #30 (Né từ khoá):** `"F R E E  e n t r y  into our weekly draw, t x t  WIN  to 80086"` (Nhãn thật: `spam`).
     - *Cơ chế sai:* Thuật toán tách từ (tokenize) theo khoảng trắng làm các từ khóa `F`, `R`, `E`, `E`, `t`, `x`, `t` trở thành các token đơn ký tự nằm ngoài từ điển hoặc có trọng số bằng 0. Mô hình BoW hoàn toàn mất dấu hiệu nhận diện đặc trưng và đoán nhầm là `ham`.
  2. **Case #21 (Thao túng / lừa đảo):** `"Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"` (Nhãn thật: `spam`).
     - *Cơ chế sai:* Kẻ lừa đảo đóng vai người thân, dùng từ ngữ thân mật và tình cảm (`Mum`, `friend`, `explain later`), hoàn toàn không chứa từ rác truyền thống (`win`, `prize`, `claim`). Các từ trong câu đều có trọng số âm (kéo về ham) trong từ điển, khiến BoW gán nhãn `ham` với độ tin cậy rất cao.

## 5. Thử thách cá nhân

- **Đề bài (#5):** Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.
- **Cách làm:**
  - Thiết kế 2 tin spam né từ khóa: 1 tin lừa đảo việc làm lương cao (`$45/hr`) và 1 tin mạo danh bưu điện yêu cầu nộp lệ phí hải quan (`customs fee £1.50`).
  - Thiết kế 2 tin ham: 1 tin rủ bạn bè ăn uống có cụm `feel free` và 1 tin trao đổi về trận chung kết bóng đá có từ `win`.
  - Đưa 4 tin vào pipeline trích xuất đặc trưng (BoW và vector MiniLM đã chuẩn hóa) rồi dự đoán qua 2 bộ phân loại tương ứng.
- **Kết quả thực nghiệm:**
  - BoW + LogReg: đúng **2/4 tin** (50%).
  - Embedding (MiniLM) + LogReg: đúng **3/4 tin** (75%).
- **Giải thích:**
  - *Tin 1 (Spam việc làm):* BoW đoán nhầm `ham` vì không có từ khóa rác quen thuộc; Embedding hiểu được cấu trúc ngữ nghĩa chào mời thu nhập cao nên phân loại đúng `spam`.
  - *Tin 2 (Spam lệ phí hải quan):* Cả hai đều đoán nhầm `ham` vì văn phong giao dịch ngắn gọn, lịch sự đã kéo vector ngữ nghĩa của MiniLM lệch sang cụm tin giao dịch thông thường.
  - *Tin 3 (Ham rủ ăn uống):* Cả hai đoán đúng `ham`. Embedding hiểu ngữ cảnh thân mật `feel free`; BoW không bị lệch vì có các đại từ/liên từ mang trọng số âm bù lại cho chữ `free`.
  - *Tin 4 (Ham thể thao):* Cả hai đoán đúng `ham`. Embedding hiểu ngữ cảnh thể thao học đường; BoW được các từ ngữ cá nhân kéo nhãn về an toàn.

## 6. Phương án cho sản phẩm thật

- **Yêu cầu bài toán:** Lọc 10 triệu SMS/ngày (trung bình ~116 tin/giây), yêu cầu độ trễ **p95 < 50 ms/tin**.
- **Lựa chọn phương án:** Sử dụng **`3b. Encoder (MiniLM) + LogReg`** làm giải pháp nòng cốt hoặc triển khai theo **kiến trúc Router 2 tầng**.
- **Dẫn chứng số liệu:**
  - Mô hình `3b. Encoder + LogReg` đạt độ trễ **p95 = 22.14 ms** (thỏa mãn điều kiện < 50 ms), thông lượng đạt **1.018 tin/giây** (gấp 9 lần yêu cầu trung bình của hệ thống) và F1 đạt mức cao nhất trong các mô hình chạy cục bộ (**0.984**).
  - Ngược lại, nếu dùng 100% LLM: độ trễ p95 lên tới 1.2–3.1 giây (vi phạm nặng nề SLA) và làm phát sinh chi phí từ **$3,150/tháng** (`gpt-6-luna`) đến **$31,500/tháng** (`gpt-6.1-terra`).
- **Đánh đổi:**
  - Chấp nhận tốn thêm tài nguyên bộ nhớ RAM/GPU để nạp mô hình encoder MiniLM (~90MB) và độ trễ tăng từ micro-giây (của BoW) lên mức ~22 ms (vẫn nằm trong vùng an toàn của SLA).
  - Đổi lại, hệ thống có khả năng khái quát hóa ngữ nghĩa xuất sắc, triệt tiêu gần như hoàn toàn tình trạng bắt nhầm các tin nhắn thông thường chứa từ khóa nhạy cảm.

## 7. AI log và điều đã học
### AI log
- **Phần 4 — Cấu hình:**
  - *Tôi hỏi:* Sửa lỗi `AuthenticationError 401: Incorrect API key provided` khi chuyển sang OpenAI API.
  - *AI trả lời:* Giải thích do phiên Colab vẫn lưu `OPENAI_API_KEY` từ key Gemini cũ; hướng dẫn đọc lại trực tiếp từ Colab Secrets và xóa `OPENAI_BASE_URL`.
  - *Tôi sửa/kiểm chứng:* Đã kiểm tra lại secret name, cập nhật đúng key OpenAI và gọi thử hàm `call_llm` thành công.
- **Phần 4 — Triển khai hàm API:**
  - *Tôi hỏi:* Hướng dẫn code các hàm `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch`.
  - *AI trả lời:* Viết code chi tiết với các tham số `reasoning_effort="none"` và `max_completion_tokens`.
  - *Tôi sửa/kiểm chứng:* Phát hiện giá trị `max_tokens: 3` bị cạn token dẫn đến lỗi trả về rỗng (`invalid=40`), đã tự chỉnh sửa tăng lên `max_tokens: 10` để trích xuất đầy đủ nhãn phân loại.
- **Phần 5 — Thử thách cá nhân:**
  - *Tôi hỏi:* Gợi ý 4 mẫu tin nhắn thử thách và code so sánh BoW vs Embedding+LogReg.
  - *AI trả lời:* Đưa ra các mẫu câu tình huống thực tế và khung chạy thử nghiệm.
  - *Tôi sửa/kiểm chứng:* Tự chạy thực tế trên notebook, ghi nhận kết quả và trực tiếp phân tích nguyên nhân từng trường hợp đúng/sai.

### Điều đã học được
Hôm nay tôi hiểu rằng không phải cứ mô hình nhiều tham số hay phức tạp (MLP, LLM) là sẽ đem lại hiệu quả vượt trội trong môi trường thực tế. Với dữ liệu dạng túi từ (BoW), mạng nơ-ron sâu không cải thiện được nhiều do đặc trưng thiếu ngữ cảnh, trong khi lại rất dễ bị overfit. Một mô hình biểu diễn ngữ nghĩa cô đọng được pre-train tốt (như MiniLM) kết hợp cùng một tầng phân loại tuyến tính đơn giản chính là điểm cân bằng hoàn hảo nhất giữa chất lượng, tốc độ và chi phí triển khai hệ sinh thái AI.