# BÀI TẬP LỚN HỌC PHẦN NHẬP MÔN HỌC MÁY (FIT4013)
## ĐỀ TÀI 1: PHÂN LOẠI CẢM XÚC TỪ PHẢN HỒI CỦA SINH VIÊN (STUDENT FEEDBACK SENTIMENT CLASSIFICATION)

- **Giảng viên hướng dẫn:** ThS. Lê Thị Thùy Trang
- **Năm học:** 2026–2027
- **Phạm vi thuật toán được sử dụng:**
  - **Mô hình bắt buộc:** Multinomial Naive Bayes (MNB) cho đặc trưng đếm (Bag-of-Words) và TF-IDF.
  - **Mô hình so sánh:** Cây quyết định (Decision Tree Classifier) và K-Láng giềng gần nhất (K-Nearest Neighbors - KNN).
  - **Mô hình mốc chuẩn:** Dummy Classifier (Majority Class Baseline).

---

## 1. Giới thiệu bài toán và Mục tiêu
Đề tài tập trung giải quyết bài toán Phân loại cảm xúc (Sentiment Analysis) từ phản hồi học thuật của sinh viên. Ý kiến đóng góp của người học là nguồn thông tin then chốt giúp các cơ sở giáo dục đại học nâng cao chất lượng giảng dạy và cơ sở vật chất.

Hệ thống phân loại mỗi phản hồi văn bản (`feedback_text`) thành 1 trong 3 nhóm cảm xúc:
- **Tích cực (Positive):** 489 mẫu (~40.0%)
- **Trung tính (Neutral):** 368 mẫu (~30.1%)
- **Tiêu cực (Negative):** 366 mẫu (~29.9%)

### Thách thức chính của bài toán:
1. Độ dài văn bản ngắn (trung bình ~11.5 từ/câu), mật độ từ vựng thưa thớt (sparsity).
2. Sự xuất hiện của yếu tố châm biếm/mỉa mai (`sarcasm_flag == 1`), làm đảo ngược ý nghĩa cảm xúc bề mặt của từ ngữ.
3. Không gian từ vựng có độ chồng lấn (overlap) lớn giữa lớp trung tính và hai lớp phân cực.

---

## 2. Cấu trúc thư mục dự án

```text
.
├── data/
│   ├── student_feedback_dataset.csv      # Dữ liệu phản hồi sinh viên gốc (1,223 dòng, 11 cột)
│   └── error_analysis.csv                # Tập các mẫu kiểm thử dự đoán sai kèm xác suất và phân tích
├── figures/                              # Tất cả biểu đồ trực quan hóa EDA và kết quả đánh giá (DPI 300)
│   ├── class_distribution.png            # Phân phối nhãn mục tiêu (Target Distribution)
│   ├── text_length_distribution.png      # Phân phối độ dài từ và ký tự theo nhãn
│   ├── department_feedback_sentiment.png # Tỷ lệ cảm xúc theo Khoa và Loại phản hồi
│   ├── sarcasm_impact.png                # Tác động của cờ châm biếm tới nhãn
│   ├── top_words_per_sentiment.png       # Top 12 từ xuất hiện nhiều nhất ở từng lớp
│   ├── vectorizer_comparison.png         # So sánh hiệu năng BoW vs TF-IDF (Unigram, Bigram, min_df)
│   ├── model_comparison_metrics.png      # Biểu đồ so sánh Accuracy, Macro-F1, Weighted-F1 các mô hình
│   ├── confusion_matrix_baseline.png     # Ma trận nhầm lẫn của mô hình Baseline
│   ├── confusion_matrix_naive_bayes.png  # Ma trận nhầm lẫn của Multinomial Naive Bayes
│   ├── confusion_matrix_decision_tree.png# Ma trận nhầm lẫn của Cây quyết định
│   └── confusion_matrix_knn.png          # Ma trận nhầm lẫn của K-Nearest Neighbors
├── models/                               # Lưu trữ mô hình và vectorizer đã huấn luyện tối ưu (.pkl)
│   ├── best_vectorizer.pkl               # TF-IDF Vectorizer đã fit trên tập Train
│   ├── best_multinomial_nb.pkl           # Mô hình MultinomialNB tối ưu (alpha=0.5)
│   ├── best_decision_tree.pkl            # Mô hình Decision Tree tối ưu (entropy, unpruned)
│   └── best_knn.pkl                      # Mô hình KNN tối ưu (k=9, cosine distance)
├── notebooks/
│   └── BTL_Topic1_Student_Feedback_Sentiment.ipynb # Notebook thực thi toàn bộ pipeline có chú giải
├── report/
│   ├── report.tex                        # Báo cáo học thuật định dạng LaTeX đầy đủ 6 chương
│   └── references.bib                    # Danh mục tài liệu tham khảo chuẩn BibTeX
├── src/                                  # Mã nguồn các module tái sử dụng
│   ├── __init__.py
│   ├── eda.py                            # Module khám phá và trực quan hóa dữ liệu
│   ├── preprocessing.py                  # Module tiền xử lý văn bản (chống rò rỉ dữ liệu)
│   ├── features.py                       # Module trích xuất đặc trưng BoW & TF-IDF
│   ├── models.py                         # Module khởi tạo và tinh chỉnh siêu tham số mô hình
│   └── evaluate.py                       # Module tính toán độ đo, vẽ đồ thị và phân tích lỗi
├── app.py                                # Giao diện Web tương tác Streamlit (Demo - Điểm thưởng)
├── main.py                               # Script chạy toàn bộ pipeline tự động từ đầu đến cuối
├── requirements.txt                      # Danh sách các thư viện cần thiết
├── AI_USAGE.md                           # Khai báo minh bạch việc sử dụng công cụ AI
└── README.md                             # Hướng dẫn chi tiết dự án
```

---

## 3. Hướng dẫn cài đặt và chạy chương trình

### 3.1 Cài đặt môi trường
Yêu cầu Python $\ge 3.9$. Khuyến nghị khởi tạo môi trường ảo:
```bash
# Tạo môi trường ảo (tùy chọn)
python -m venv venv
venv\Scripts\activate   # Trên Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 3.2 Chạy toàn bộ Pipeline học máy
Để chạy quy trình từ EDA, tiền xử lý, huấn luyện các mô hình, xuất biểu đồ và phân tích lỗi:
```bash
python main.py
```
Toàn bộ biểu đồ sẽ được tự động lưu vào thư mục `figures/`, mô hình lưu vào `models/` và kết quả phân tích lỗi lưu tại `data/error_analysis.csv`.

### 3.3 Khởi chạy ứng dụng Web Demo (Streamlit - Điểm thưởng)
Ứng dụng cho phép người dùng trực tiếp nhập câu phản hồi bất kỳ của sinh viên, lựa chọn thuật toán (MultinomialNB, Decision Tree, KNN), nhận dự đoán tức thời kèm phân phối xác suất và xem toàn bộ biểu đồ thực nghiệm:
```bash
streamlit run app.py
```

### 3.4 Xem Jupyter Notebook
```bash
jupyter notebook notebooks/BTL_Topic1_Student_Feedback_Sentiment.ipynb
```

---

## 4. Tóm tắt kết quả thực nghiệm

### 4.1 So sánh các phương pháp biểu diễn đặc trưng (trên MultinomialNB, $\alpha=1.0$)
| Kỹ thuật biểu diễn | Kích thước từ vựng (Vocab) | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: |
| **BoW (Unigram)** | 767 | 0.3918 | 0.3780 |
| **BoW (Unigram + Bigram)** | 8,601 | 0.3918 | 0.3409 |
| **TF-IDF (Unigram)** | 767 | **0.4000** | 0.3605 |
| **TF-IDF (Unigram + Bigram)**| 8,601 | 0.3959 | 0.2545 |
| **TF-IDF (min_df=2)** | 767 | **0.4000** | 0.3605 |
| **TF-IDF (Sublinear TF)** | 767 | **0.4000** | 0.3608 |

*Nhận xét:* Biểu diễn **Unigram** (cả BoW và TF-IDF) vượt trội rõ rệt so với **Bigram**. Do kích thước tập dữ liệu huấn luyện (978 mẫu) tương đối khiêm tốn, việc mở rộng sang Bigram làm tăng số chiều từ vựng lên gấp hơn 11 lần (8,601 đặc trưng), dẫn tới hiện tượng ma trận thưa trầm trọng và phân tán xác suất của mô hình.

---

### 4.2 So sánh hiệu năng các mô hình trên tập kiểm thử (Test Set - 245 mẫu)
| Mô hình | Siêu tham số tối ưu | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Majority Class)** | Chiếm đa số (Positive) | 0.4000 | 0.1333 | 0.3333 | 0.1905 | 0.2286 |
| **Multinomial Naive Bayes** | $\alpha = 0.5$ | 0.3878 | **0.3693** | **0.3648** | **0.3570** | **0.3705** |
| **Decision Tree** | `entropy`, `depth=None` | 0.3429 | 0.3362 | 0.3356 | 0.3357 | 0.3428 |
| **K-Nearest Neighbors** | $k = 9$, `cosine`, `distance`| 0.3265 | 0.3159 | 0.3182 | 0.3154 | 0.3215 |

### 4.3 Những phát hiện quan trọng:
1. **Multinomial Naive Bayes** là mô hình hiệu quả và cân bằng nhất giữa các lớp (Macro-F1 đạt 0.3570, vượt trội hoàn toàn so với Baseline 0.1905). Dù Accuracy của Baseline là 0.4000, Baseline chỉ học vẹt nhãn đa số và có Macro-Recall/Precision cực kỳ yếu kém trên các lớp còn lại.
2. **K-NN và Cây quyết định** bị ảnh hưởng mạnh bởi hiện tượng "Lời nguyền số chiều" (Curse of Dimensionality) trong không gian văn bản thưa. Khoảng cách giữa các vector tài liệu bị co cụm khiến K-NN khó tìm láng giềng đại diện chính xác, trong khi Decision Tree dễ bị phân nhánh quá sâu và học nhiễu.
3. **Phân tích lỗi (Error Analysis):**
   - Sự nhầm lẫn tập trung chủ yếu ở nhãn **Neutral** khi bị phân loại nhầm thành Positive (42 mẫu) hoặc Negative (20 mẫu).
   - Yếu tố **châm biếm (Sarcasm)** làm tăng tỷ lệ dự đoán sai từ **59.41%** (ở câu bình thường) lên đến **69.77%** (ở câu có châm biếm), do nghĩa cảm xúc thực tế bị đảo ngược so với từ ngữ bề mặt.

---

## 5. Minh bạch sử dụng AI và Liêm chính học thuật
Chi tiết khai báo sử dụng công cụ AI phục vụ bài tập lớn được trình bày minh bạch trong tệp [AI_USAGE.md](AI_USAGE.md). Nhóm sinh viên cam kết hoàn toàn làm chủ sản phẩm, nắm vững cơ chế toán học của các thuật toán và sẵn sàng phản biện trong buổi bảo vệ vấn đáp.
