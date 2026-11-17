"""
Module: Models
Chức năng: Khởi tạo, huấn luyện và tìm kiếm siêu tham số cho các mô hình học máy:
1. Baseline: DummyClassifier (Majority Class)
2. Multinomial Naive Bayes (MultinomialNB) - Mô hình cốt lõi theo hướng dẫn
3. Cây quyết định (DecisionTreeClassifier) - Thuật toán đã học dùng so sánh
4. K-Nearest Neighbors (KNeighborsClassifier) - Thuật toán đã học dùng so sánh
"""

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def get_baseline_model():
    """
    Mô hình baseline làm mốc chuẩn so sánh: Dự đoán lớp chiếm đa số (Majority Class).
    """
    return DummyClassifier(strategy="most_frequent")

def get_default_models():
    """
    Tập các mô hình với cấu hình mặc định hoặc tiêu chuẩn ban đầu.
    """
    return {
        "Baseline (Majority)": DummyClassifier(strategy="most_frequent"),
        "MultinomialNB (default)": MultinomialNB(alpha=1.0),
        "Decision Tree (default)": DecisionTreeClassifier(random_state=42),
        "KNN (k=5, cosine)": KNeighborsClassifier(n_neighbors=5, metric="cosine")
    }

def get_hyperparameter_grids():
    """
    Định nghĩa không gian tìm kiếm siêu tham số (Hyperparameter Search Space)
    cho từng thuật toán đã học trong môn học.
    """
    grids = {
        "MultinomialNB": {
            "model": MultinomialNB(),
            "param_grid": {
                "alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "metric": ["cosine", "euclidean"],
                "weights": ["uniform", "distance"]
            }
        }
    }
    return grids

def tune_model(model_name: str, X_train, y_train, cv_splits: int = 5, scoring: str = "f1_macro"):
    """
    Thực hiện Grid Search với Stratified K-Fold CV để tìm siêu tham số tối ưu trên tập Train.
    """
    grids = get_hyperparameter_grids()
    if model_name not in grids:
        raise ValueError(f"Không tìm thấy cấu hình tham số cho mô hình: {model_name}")
        
    cfg = grids[model_name]
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=cfg["model"],
        param_grid=cfg["param_grid"],
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def save_model(model, file_path: str):
    """Lưu mô hình ra file pkl"""
    joblib.dump(model, file_path)

def load_model(file_path: str):
    """Nạp mô hình từ file pkl"""
    return joblib.load(file_path)
