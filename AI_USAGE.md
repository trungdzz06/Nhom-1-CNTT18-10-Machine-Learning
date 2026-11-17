# Báo Cáo Khai Báo Sử Dụng Công Cụ AI (AI Usage Declaration)

**Học phần:** Nhập môn Học máy (FIT4013) - Năm học 2026–2027  
**Giảng viên hướng dẫn:** ThS. Lê Thị Thùy Trang  
**Đề tài 1:** Phân loại cảm xúc từ phản hồi của sinh viên (Student Feedback Sentiment Classification)  

Tuân thủ quy định về Liêm chính học thuật và Sử dụng công cụ AI được quy định tại Mục 2.3 và Mục 2.5 của tài liệu Hướng dẫn thực hiện bài tập lớn, nhóm sinh viên công khai và minh bạch toàn bộ việc sử dụng công cụ trí tuệ nhân tạo (AI) trong quá trình thực hiện đề tài như sau:

---

## 1. Công cụ AI được sử dụng
- **Tên công cụ:** Antigravity AI Assistant (Google DeepMind) / Mô hình ngôn ngữ lớn Gemini.
- **Môi trường sử dụng:** Hỗ trợ phát triển trong IDE.

---

## 2. Mục đích và Phạm vi sử dụng
1. **Khám phá và thiết kế cấu trúc dự án:**
   - Hỗ trợ xây dựng cấu trúc repository chuẩn mực khoa học (modular architecture: `src/`, `data/`, `notebooks/`, `figures/`, `models/`, `report/`).
   - Gợi ý phân chia các hàm xử lý độc lập nhằm bảo đảm tính tái lập (reproducibility) và chống rò rỉ dữ liệu (data leakage prevention).
2. **Hỗ trợ lập trình và kiểm tra cú pháp (Syntax & Code Refactoring):**
   - Viết các hàm tiền xử lý chuỗi ký tự bằng biểu thức chính quy (regular expressions).
   - Xây dựng pipeline Scikit-Learn và thực hiện `GridSearchCV` trên 5-fold Stratified Cross-Validation.
   - Hỗ trợ vẽ biểu đồ nhiệt ma trận nhầm lẫn (Confusion Matrix Heatmaps) và biểu đồ so sánh các chỉ số bằng Matplotlib và Seaborn.
   - Xây dựng giao diện web demo trực quan với Streamlit (`app.py`).
3. **Soạn thảo và định dạng tài liệu học thuật bằng LaTeX:**
   - Hỗ trợ template chuẩn định dạng bài báo cáo học thuật bằng LaTeX (công thức toán học cho Định lý Bayes, TF-IDF, chỉ số Gini, khoảng cách Cosine, bảng số liệu và danh mục tài liệu tham khảo BibTeX).

---

## 3. Trách nhiệm và Kiểm chứng của Nhóm sinh viên
- Nhóm sinh viên trực tiếp định hình bài toán, lựa chọn thuật toán theo đúng phạm vi kiến thức đã học trong học phần: **Multinomial Naive Bayes**, **Cây quyết định (Decision Tree)**, **K-Nearest Neighbors (KNN)**.
- Toàn bộ kết quả thực nghiệm, ma trận nhầm lẫn, bảng chỉ số (Accuracy, Macro-F1, Weighted-F1, Precision, Recall) và các phân tích lỗi (Error Analysis) đều được chạy thực tế 100% trên tập dữ liệu được giao (`student_feedback_dataset.csv`).
- Toàn bộ thành viên trong nhóm hiểu rõ cơ chế toán học của từng thuật toán, quy trình tiền xử lý, cấu trúc mã nguồn và sẵn sàng giải trình toàn bộ nội dung trong buổi bảo vệ vấn đáp trước Giảng viên.
