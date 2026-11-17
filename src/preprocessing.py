"""
Module: Preprocessing
Chức năng: Tiền xử lý văn bản phản hồi sinh viên.
- Chuẩn hóa chữ thường (lowercase)
- Loại bỏ URL, ký tự đặc biệt, dấu câu
- Chuẩn hóa khoảng trắng thừa
- Lọc stop words (bảo lưu các từ mang sắc thái phủ định quan trọng như 'not', 'no', 'never')
- Hạn chế Data Leakage: Các hàm xử lý văn bản hoạt động độc lập trên từng chuỗi văn bản.
"""

import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Tập stopwords tùy chỉnh: giữ lại các từ phủ định quan trọng cho bài toán phân loại cảm xúc
NEGATION_WORDS = {'not', 'no', 'nor', 'never', 'neither', 'hardly', 'scarcely', 'barely', 'cannot', 'couldn\'t', 'didn\'t', 'doesn\'t', 'isn\'t', 'wasn\'t', 'won\'t', 'wouldn\'t'}
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

def clean_text(text: str, remove_stopwords: bool = True, keep_negations: bool = True) -> str:
    """
    Tiền xử lý chuỗi văn bản:
    1. Chuyển thành chữ thường
    2. Loại bỏ URL, email, tag html
    3. Loại bỏ dấu câu và ký tự số/đặc biệt
    4. Chuẩn hóa khoảng trắng
    5. Loại bỏ stop words (tùy chọn)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Chuyển thành chữ thường
    text = text.lower()
    
    # 2. Xóa URL nếu có
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 3. Xóa thẻ HTML nếu có
    text = re.sub(r'<.*?>', ' ', text)
    
    # 4. Thay thế dấu câu bằng khoảng trắng
    text = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', text)
    
    # 5. Loại bỏ ký tự đặc biệt ngoài bảng chữ cái tiếng Anh
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 6. Tách từ và lọc stop words
    words = text.split()
    if remove_stopwords:
        stop_set = CUSTOM_STOP_WORDS if keep_negations else ENGLISH_STOP_WORDS
        words = [w for w in words if w not in stop_set and len(w) > 1]
    else:
        words = [w for w in words if len(w) > 1]
        
    return " ".join(words)

def preprocess_corpus(texts, remove_stopwords: bool = True, keep_negations: bool = True):
    """
    Tiền xử lý toàn bộ danh sách hoặc Series văn bản.
    """
    return [clean_text(t, remove_stopwords=remove_stopwords, keep_negations=keep_negations) for t in texts]
