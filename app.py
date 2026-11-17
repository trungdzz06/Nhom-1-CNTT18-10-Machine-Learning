"""
Ứng dụng Web Demo (Streamlit): Phân loại cảm xúc từ phản hồi của sinh viên
Đề tài 1 - Học phần Nhập môn Học máy (FIT4013)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import clean_text

st.set_page_config(
    page_title="Phân loại Cảm xúc Phản hồi Sinh viên | FIT4013",
    page_icon="🎓",
    layout="wide"
)

# Đường dẫn tài nguyên
MODELS_DIR = "models"
FIGURES_DIR = "figures"

@st.cache_resource
def load_all_models():
    vec_path = os.path.join(MODELS_DIR, "best_vectorizer.pkl")
    nb_path = os.path.join(MODELS_DIR, "best_multinomial_nb.pkl")
    dt_path = os.path.join(MODELS_DIR, "best_decision_tree.pkl")
    knn_path = os.path.join(MODELS_DIR, "best_knn.pkl")
    
    vec = joblib.load(vec_path) if os.path.exists(vec_path) else None
    nb = joblib.load(nb_path) if os.path.exists(nb_path) else None
    dt = joblib.load(dt_path) if os.path.exists(dt_path) else None
    knn = joblib.load(knn_path) if os.path.exists(knn_path) else None
    return vec, nb, dt, knn

vec, nb_model, dt_model, knn_model = load_all_models()

# Header
st.title("🎓 Hệ thống Phân loại Cảm xúc Phản hồi Sinh viên")
st.markdown("""
**Học phần:** Nhập môn Học máy (FIT4013) — **Đề tài 1:** Phân loại cảm xúc từ phản hồi của sinh viên  
**Giảng viên hướng dẫn:** ThS. Lê Thị Thùy Trang  
*Ứng dụng giao diện Demo hỗ trợ dự đoán cảm xúc trực quan và phân tích dữ liệu.*
""")
st.divider()

# Sidebar
st.sidebar.header("⚙️ Cấu hình mô hình")
model_choice = st.sidebar.selectbox(
    "Chọn thuật toán phân loại:",
    ("Multinomial Naive Bayes (Khuyến nghị)", "Decision Tree (Cây quyết định)", "K-Nearest Neighbors (KNN)")
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Giới hạn thuật toán")
st.sidebar.info("""
Mô hình được xây dựng nghiêm ngặt dựa trên kiến thức học phần:
- **Multinomial Naive Bayes** (Cốt lõi cho dữ liệu văn bản/TF-IDF)
- **Cây quyết định (Decision Tree)**
- **K-Nearest Neighbors (KNN)**
""")

tab1, tab2, tab3 = st.tabs(["🚀 Dự đoán trực tiếp", "📊 Khám phá dữ liệu (EDA)", "📈 So sánh hiệu năng mô hình"])

with tab1:
    st.subheader("Nhập phản hồi sinh viên (Tiếng Anh)")
    
    example_prompts = [
        "Select an example...",
        "The professor was extremely helpful and the course materials were excellent!",
        "Poor classroom facilities, broken air conditioner and very noisy environment.",
        "The course covers the syllabus content within the allocated weekly hours.",
        "Great job explaining complex algorithms clearly and patiently.",
        "Terrible experience, difficult exams without sufficient preparation or guidance."
    ]
    selected_example = st.selectbox("Chọn câu mẫu thử nghiệm:", example_prompts)
    
    initial_text = "" if selected_example == example_prompts[0] else selected_example
    user_input = st.text_area("Nội dung phản hồi (feedback_text):", value=initial_text, height=120)
    
    col_btn, col_clear = st.columns([1, 5])
    predict_clicked = col_btn.button("🔍 Dự đoán cảm xúc", type="primary")
    
    if predict_clicked or user_input:
        if not user_input.strip():
            st.warning("Vui lòng nhập văn bản phản hồi.")
        elif vec is None:
            st.error("Chưa tìm thấy mô hình đã huấn luyện trong thư mục `models/`. Vui lòng chạy `python main.py` trước.")
        else:
            # Chọn model
            if "Naive Bayes" in model_choice:
                active_model = nb_model
                model_name_str = "Multinomial Naive Bayes"
            elif "Decision Tree" in model_choice:
                active_model = dt_model
                model_name_str = "Decision Tree"
            else:
                active_model = knn_model
                model_name_str = "K-Nearest Neighbors"
                
            # Tiền xử lý
            cleaned = clean_text(user_input)
            vec_input = vec.transform([cleaned])
            
            prediction = active_model.predict(vec_input)[0]
            
            # Tính xác suất nếu có
            if hasattr(active_model, "predict_proba"):
                probs = active_model.predict_proba(vec_input)[0]
                classes = list(active_model.classes_)
            else:
                probs = None
                classes = ["negative", "neutral", "positive"]
                
            st.markdown("### Kết quả dự đoán:")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if prediction == "positive":
                    st.success("### 🎉 Cảm xúc: TÍCH CỰC (Positive)")
                elif prediction == "negative":
                    st.error("### ⚠️ Cảm xúc: TIÊU CỰC (Negative)")
                else:
                    st.warning("### ⚖️ Cảm xúc: TRUNG TÍNH (Neutral)")
                    
                st.markdown(f"**Mô hình sử dụng:** `{model_name_str}`")
                st.markdown(f"**Văn bản sau tiền xử lý:** `{cleaned}`")
                
            with col2:
                if probs is not None:
                    st.markdown("**Phân bố xác suất dự đoán (Probability Distribution):**")
                    prob_df = pd.DataFrame({
                        "Cảm xúc": [c.capitalize() for c in classes],
                        "Xác suất": probs
                    })
                    st.bar_chart(prob_df.set_index("Cảm xúc"))
                    for c, p in zip(classes, probs):
                        st.write(f"- **{c.capitalize()}**: `{p*100:.2f}%`")
                else:
                    st.info("Mô hình này không hỗ trợ hiển thị phân phối xác suất xác thực.")

with tab2:
    st.subheader("Trực quan hóa và khám phá dữ liệu (EDA)")
    st.markdown("Bộ dữ liệu gồm 1,223 mẫu phản hồi sinh viên phân bố trong 3 nhóm cảm xúc.")
    
    eda_images = [
        ("Phân phối các nhãn cảm xúc", "class_distribution.png"),
        ("Độ dài văn bản phản hồi theo nhãn", "text_length_distribution.png"),
        ("Tỷ lệ cảm xúc theo Khoa và Loại phản hồi", "department_feedback_sentiment.png"),
        ("Ảnh hưởng của yếu tố châm biếm (Sarcasm)", "sarcasm_impact.png"),
        ("Top từ khóa xuất hiện nhiều nhất theo từng lớp", "top_words_per_sentiment.png")
    ]
    
    for title, fname in eda_images:
        fpath = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(fpath):
            st.markdown(f"#### 📌 {title}")
            st.image(fpath, use_container_width=True)
            st.markdown("---")

with tab3:
    st.subheader("Kết quả đánh giá và so sánh mô hình")
    st.markdown("""
    Đánh giá độc lập trên tập kiểm thử **Test Set (20% - 245 mẫu)** với các độ đo chuẩn:
    `Accuracy`, `Precision`, `Recall`, `Macro-F1`, `Weighted-F1`.
    """)
    
    comp_img = os.path.join(FIGURES_DIR, "model_comparison_metrics.png")
    if os.path.exists(comp_img):
        st.image(comp_img, caption="Biểu đồ so sánh hiệu năng các mô hình", use_container_width=True)
        
    vec_img = os.path.join(FIGURES_DIR, "vectorizer_comparison.png")
    if os.path.exists(vec_img):
        st.image(vec_img, caption="So sánh các kỹ thuật biểu diễn đặc trưng (BoW vs TF-IDF)", use_container_width=True)
        
    st.markdown("#### Ma trận nhầm lẫn (Confusion Matrices)")
    cm_cols = st.columns(3)
    cms = [
        ("Multinomial Naive Bayes", "confusion_matrix_naive_bayes.png", 0),
        ("Decision Tree", "confusion_matrix_decision_tree.png", 1),
        ("K-Nearest Neighbors", "confusion_matrix_knn.png", 2)
    ]
    for mname, fname, col_i in cms:
        p = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(p):
            with cm_cols[col_i]:
                st.markdown(f"**{mname}**")
                st.image(p, use_container_width=True)
