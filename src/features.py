"""
Module: Feature Extraction & Representation
Chức năng: Vector hóa văn bản theo Bag-of-Words (BoW) và TF-IDF với nhiều cấu hình n-gram, min_df, max_features.
Tuân thủ nguyên tắc học máy: Vectorizer chỉ được 'fit' trên tập Train và 'transform' trên tập Test để tránh Data Leakage.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.preprocessing import clean_text

def get_vectorizer(
    method: str = "tfidf",
    ngram_range: tuple = (1, 1),
    min_df: int = 1,
    max_df: float = 1.0,
    max_features: int = None,
    sublinear_tf: bool = False
):
    """
    Khởi tạo vectorizer dựa trên cấu hình yêu cầu.
    - method: 'count' (Bag-of-Words) hoặc 'tfidf' (TF-IDF)
    - ngram_range: (1, 1) là unigram, (1, 2) là unigram + bigram
    - min_df: tần số xuất hiện tối thiểu của từ trong tài liệu
    - max_df: ngưỡng tần số tài liệu tối đa
    - max_features: số lượng đặc trưng tối đa được chọn
    - sublinear_tf: áp dụng công thức 1 + log(tf) thay vì tf thông thường (chỉ áp dụng cho TF-IDF)
    """
    method = method.lower()
    if method == "count" or method == "bow":
        return CountVectorizer(
            preprocessor=clean_text,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features
        )
    elif method == "tfidf":
        return TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf
        )
    else:
        raise ValueError(f"Phương pháp vector hóa '{method}' không được hỗ trợ. Chọn 'count' hoặc 'tfidf'.")

def build_vectorizer_configurations():
    """
    Tạo danh sách các cấu hình thử nghiệm biểu diễn đặc trưng theo yêu cầu của bài tập lớn.
    """
    configs = {
        "BoW (Unigram)": get_vectorizer(method="count", ngram_range=(1, 1)),
        "BoW (Unigram + Bigram)": get_vectorizer(method="count", ngram_range=(1, 2)),
        "TF-IDF (Unigram)": get_vectorizer(method="tfidf", ngram_range=(1, 1)),
        "TF-IDF (Unigram + Bigram)": get_vectorizer(method="tfidf", ngram_range=(1, 2)),
        "TF-IDF (min_df=2)": get_vectorizer(method="tfidf", ngram_range=(1, 1), min_df=2),
        "TF-IDF (Sublinear TF)": get_vectorizer(method="tfidf", ngram_range=(1, 1), sublinear_tf=True),
    }
    return configs
