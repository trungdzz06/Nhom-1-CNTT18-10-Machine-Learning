"""
Main Pipeline: Đề tài 1 - Phân loại cảm xúc từ phản hồi của sinh viên (Student Feedback Sentiment Classification)
Quy trình thực hiện hoàn chỉnh:
1. Nạp dữ liệu và kiểm tra cấu trúc
2. Phân tích khám phá dữ liệu (EDA) & Trực quan hóa
3. Tiền xử lý dữ liệu và phân tách Train/Test có kiểm soát (Stratified, random_state=42) chống Data Leakage
4. Đánh giá Baseline (Majority Class)
5. Thử nghiệm và so sánh các phương pháp biểu diễn đặc trưng (Bag-of-Words vs TF-IDF, unigram vs bigram)
6. Huấn luyện, tinh chỉnh siêu tham số và so sánh các thuật toán đã học:
   - Multinomial Naive Bayes (MNB)
   - Cây quyết định (Decision Tree Classifier)
   - K-Nearest Neighbors (KNN Classifier)
7. Đánh giá đa chiều trên tập kiểm thử (Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Confusion Matrix)
8. Phân tích lỗi chi tiết (Error Analysis)
9. Lưu trữ mô hình phục vụ triển khai Demo
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report

from src.eda import perform_eda
from src.preprocessing import clean_text
from src.features import build_vectorizer_configurations, get_vectorizer
from src.models import (
    get_baseline_model, get_hyperparameter_grids, save_model
)
from src.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_model_comparison,
    plot_vectorizer_comparison, analyze_errors
)

def run_pipeline(data_path: str = "data/student_feedback_dataset.csv", figures_dir: str = "figures", models_dir: str = "models"):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("BƯỚC 1 & 2: KHÁM PHÁ VÀ PHÂN TÍCH DỮ LIỆU (EDA)")
    print("=" * 70)
    df = perform_eda(data_path=data_path, output_dir=figures_dir)
    
    print("\n" + "=" * 70)
    print("BƯỚC 3: PHÂN CHIA DỮ LIỆU HUẤN LUYỆN / KIỂM THỬ (TRAIN/TEST SPLIT)")
    print("=" * 70)
    X = df['feedback_text']
    y = df['sentiment_label']
    classes = sorted(list(y.unique()))
    
    # Chia 80% train, 20% test có phân tầng (stratify) và cố định random_state=42
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Tập huấn luyện (Train set): {len(X_train)} mẫu (80%)")
    print(f"Tập kiểm thử (Test set):    {len(X_test)} mẫu (20%)")
    print("Phân phối nhãn trên tập Train:\n", y_train.value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    print("Phân phối nhãn trên tập Test:\n", y_test.value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    
    # Baseline
    print("\n" + "=" * 70)
    print("BƯỚC 4: THỰC THI MÔ HÌNH BASELINE (MỐC CHUẨN SO SÁNH)")
    print("=" * 70)
    baseline = get_baseline_model()
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    base_metrics = compute_metrics(y_test, y_pred_baseline)
    base_metrics["Model"] = "Baseline (Majority Class)"
    print("Kết quả mô hình Baseline (Majority Class):")
    for k, v in base_metrics.items():
        if k != "Model":
            print(f" - {k:18s}: {v:.4f}")
    plot_confusion_matrix(
        y_test, y_pred_baseline, labels=classes,
        title="Confusion Matrix - Baseline (Majority Class)",
        save_path=os.path.join(figures_dir, "confusion_matrix_baseline.png")
    )
    
    # Thử nghiệm đặc trưng với MultinomialNB
    print("\n" + "=" * 70)
    print("BƯỚC 5: SO SÁNH CÁC PHƯƠNG PHÁP BIỂU DIỄN ĐẶC TRƯNG (BoW vs TF-IDF)")
    print("=" * 70)
    vec_configs = build_vectorizer_configurations()
    vec_results = []
    
    for v_name, vec in vec_configs.items():
        # Fit trên tập Train, transform trên tập Train và Test để chống data leakage
        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)
        
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB(alpha=1.0)
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        
        m = compute_metrics(y_test, preds)
        m["Representation"] = v_name
        m["Vocabulary Size"] = X_tr.shape[1]
        vec_results.append(m)
        print(f"-> {v_name:26s} | Vocab: {X_tr.shape[1]:5d} | Acc: {m['Accuracy']:.4f} | Macro F1: {m['Macro F1']:.4f}")
        
    vec_df = pd.DataFrame(vec_results)
    plot_vectorizer_comparison(vec_df, os.path.join(figures_dir, "vectorizer_comparison.png"))
    
    # Lựa chọn biểu diễn TF-IDF unigram và BoW unigram để huấn luyện các thuật toán
    print("\n" + "=" * 70)
    print("BƯỚC 6: TINH CHỈNH SIÊU THAM SỐ VÀ SO SÁNH CÁC THUẬT TOÁN ĐÃ HỌC")
    print("=" * 70)
    
    # Dùng TF-IDF Vectorizer chuẩn hóa
    tfidf_vec = get_vectorizer(method="tfidf", ngram_range=(1, 1))
    X_train_vec = tfidf_vec.fit_transform(X_train)
    X_test_vec = tfidf_vec.transform(X_test)
    
    # Lưu vectorizer
    joblib.dump(tfidf_vec, os.path.join(models_dir, "best_vectorizer.pkl"))
    
    param_grids = get_hyperparameter_grids()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    trained_models = {"Baseline": baseline}
    all_metrics = [base_metrics]
    
    # 1. Multinomial Naive Bayes
    print("\n--- [1] Multinomial Naive Bayes (GridSearchCV 5-fold CV) ---")
    gs_nb = GridSearchCV(param_grids["MultinomialNB"]["model"], param_grids["MultinomialNB"]["param_grid"], cv=skf, scoring="f1_macro", n_jobs=-1)
    gs_nb.fit(X_train_vec, y_train)
    best_nb = gs_nb.best_estimator_
    trained_models["MultinomialNB"] = best_nb
    print(f"Tham số tối ưu MultinomialNB: {gs_nb.best_params_} (Best CV Macro F1: {gs_nb.best_score_:.4f})")
    y_pred_nb = best_nb.predict(X_test_vec)
    nb_metrics = compute_metrics(y_test, y_pred_nb)
    nb_metrics["Model"] = "Multinomial Naive Bayes"
    all_metrics.append(nb_metrics)
    plot_confusion_matrix(
        y_test, y_pred_nb, labels=classes,
        title=f"Confusion Matrix - Multinomial Naive Bayes (alpha={gs_nb.best_params_['alpha']})",
        save_path=os.path.join(figures_dir, "confusion_matrix_naive_bayes.png")
    )
    save_model(best_nb, os.path.join(models_dir, "best_multinomial_nb.pkl"))
    
    # 2. Decision Tree Classifier
    print("\n--- [2] Decision Tree (GridSearchCV 5-fold CV) ---")
    gs_dt = GridSearchCV(param_grids["DecisionTree"]["model"], param_grids["DecisionTree"]["param_grid"], cv=skf, scoring="f1_macro", n_jobs=-1)
    gs_dt.fit(X_train_vec, y_train)
    best_dt = gs_dt.best_estimator_
    trained_models["DecisionTree"] = best_dt
    print(f"Tham số tối ưu Decision Tree: {gs_dt.best_params_} (Best CV Macro F1: {gs_dt.best_score_:.4f})")
    y_pred_dt = best_dt.predict(X_test_vec)
    dt_metrics = compute_metrics(y_test, y_pred_dt)
    dt_metrics["Model"] = "Decision Tree"
    all_metrics.append(dt_metrics)
    plot_confusion_matrix(
        y_test, y_pred_dt, labels=classes,
        title=f"Confusion Matrix - Decision Tree (max_depth={gs_dt.best_params_['max_depth']})",
        save_path=os.path.join(figures_dir, "confusion_matrix_decision_tree.png")
    )
    save_model(best_dt, os.path.join(models_dir, "best_decision_tree.pkl"))
    
    # 3. K-Nearest Neighbors Classifier
    print("\n--- [3] K-Nearest Neighbors (GridSearchCV 5-fold CV) ---")
    gs_knn = GridSearchCV(param_grids["KNN"]["model"], param_grids["KNN"]["param_grid"], cv=skf, scoring="f1_macro", n_jobs=-1)
    gs_knn.fit(X_train_vec, y_train)
    best_knn = gs_knn.best_estimator_
    trained_models["KNN"] = best_knn
    print(f"Tham số tối ưu KNN: {gs_knn.best_params_} (Best CV Macro F1: {gs_knn.best_score_:.4f})")
    y_pred_knn = best_knn.predict(X_test_vec)
    knn_metrics = compute_metrics(y_test, y_pred_knn)
    knn_metrics["Model"] = "K-Nearest Neighbors"
    all_metrics.append(knn_metrics)
    plot_confusion_matrix(
        y_test, y_pred_knn, labels=classes,
        title=f"Confusion Matrix - KNN (k={gs_knn.best_params_['n_neighbors']}, metric={gs_knn.best_params_['metric']})",
        save_path=os.path.join(figures_dir, "confusion_matrix_knn.png")
    )
    save_model(best_knn, os.path.join(models_dir, "best_knn.pkl"))
    
    # Tổng kết bảng kết quả
    print("\n" + "=" * 70)
    print("BƯỚC 7: TỔNG HỢP VÀ SO SÁNH HIỆU NĂNG MÔ HÌNH TRÊN TEST SET")
    print("=" * 70)
    metrics_df = pd.DataFrame(all_metrics)
    columns_order = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    print(metrics_df[columns_order].to_string(index=False))
    plot_model_comparison(metrics_df, os.path.join(figures_dir, "model_comparison_metrics.png"))
    
    # Phân tích lỗi (Error Analysis)
    print("\n" + "=" * 70)
    print("BƯỚC 8: PHÂN TÍCH LỖI DỰ ĐOÁN (ERROR ANALYSIS)")
    print("=" * 70)
    probs_nb = best_nb.predict_proba(X_test_vec)
    error_df = analyze_errors(
        raw_texts=X_test,
        y_true=y_test,
        y_pred=y_pred_nb,
        probs=probs_nb,
        class_labels=best_nb.classes_,
        metadata_df=meta_test,
        n_samples=15
    )
    print(f"Tổng số mẫu dự đoán sai của MultinomialNB trên Test set: {len(error_df)} / {len(X_test)} ({len(error_df)/len(X_test)*100:.2f}%)")
    error_csv_path = os.path.join("data", "error_analysis.csv")
    error_df.to_csv(error_csv_path, index=False, encoding='utf-8')
    print(f"-> Đã xuất danh sách mẫu sai ra: {error_csv_path}")
    
    print("\nPhân tích nhầm lẫn theo cặp lớp (Actual -> Predicted):")
    confused_pairs = error_df.groupby(['actual', 'predicted']).size().reset_index(name='count')
    print(confused_pairs.sort_values(by='count', ascending=False).to_string(index=False))
    
    print("\nẢnh hưởng của yếu tố châm biếm (Sarcasm) tới tỷ lệ dự đoán sai:")
    sarcasm_err = pd.crosstab(meta_test['sarcasm_flag'], y_test == y_pred_nb, normalize='index') * 100
    sarcasm_err.columns = ['Sai (Error %)', 'Đúng (Correct %)']
    sarcasm_err.index = ['Không châm biếm (0)', 'Châm biếm (1)']
    print(sarcasm_err.round(2))
    
    print("\nTrích xuất 5 ví dụ dự đoán sai tiêu biểu:")
    for i, r in error_df.head(5).iterrows():
        print(f"\nVí dụ #{i+1}:")
        print(f" - Văn bản phản hồi: \"{r['text']}\"")
        print(f" - Nhãn thực tế:     {r['actual']} (Cảm xúc: {r.get('emotion_tag', 'N/A')}, Sarcasm: {r.get('sarcasm_flag', 'N/A')})")
        print(f" - Nhãn dự đoán:     {r['predicted']} (Độ tự tin: {r['confidence']:.2f})")
        print(f" - Xác suất các lớp: Negative={r.get('prob_negative',0):.2f}, Neutral={r.get('prob_neutral',0):.2f}, Positive={r.get('prob_positive',0):.2f}")
        
    print("\n" + "=" * 70)
    print("HOÀN TẤT TOÀN BỘ QUY TRÌNH HỌC MÁY!")
    print("=" * 70)
    return metrics_df, vec_df, error_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy pipeline Đề tài 1: Phân loại cảm xúc phản hồi sinh viên")
    parser.add_argument("--data", type=str, default="data/student_feedback_dataset.csv", help="Đường dẫn file dữ liệu CSV")
    args = parser.parse_args()
    run_pipeline(data_path=args.data)
