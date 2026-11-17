"""
Module: Exploratory Data Analysis (EDA)
Chức năng: Khám phá, thống kê mô tả và trực quan hóa phân phối dữ liệu phản hồi sinh viên.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from src.preprocessing import clean_text

# Cấu hình phong cách biểu đồ chuẩn học thuật
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def perform_eda(data_path: str = "data/student_feedback_dataset.csv", output_dir: str = "figures"):
    """
    Thực hiện phân tích khám phá dữ liệu toàn diện và lưu các biểu đồ vào thư mục output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    print("=" * 60)
    print("KHÁM PHÁ VÀ PHÂN TÍCH DỮ LIỆU (EDA)")
    print("=" * 60)
    print(f"Tổng số lượng mẫu: {len(df)}")
    print(f"Số lượng cột đặc trưng: {df.shape[1]}")
    print(f"Danh sách các cột: {list(df.columns)}")
    print("\nKiểm tra giá trị thiếu (Missing values):")
    print(df.isnull().sum())
    
    print("\nKiểm tra dữ liệu trùng lặp trong feedback_text:")
    duplicates = df['feedback_text'].duplicated().sum()
    print(f"Số lượng mẫu trùng lặp: {duplicates}")
    
    # Tính toán độ dài văn bản (theo số từ và số ký tự)
    df['word_count'] = df['feedback_text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['feedback_text'].apply(lambda x: len(str(x)))
    
    print("\nThống kê mô tả độ dài văn bản (số từ):")
    print(df.groupby('sentiment_label')['word_count'].describe())
    
    print("\nPhân phối nhãn cảm xúc (Sentiment Label):")
    label_counts = df['sentiment_label'].value_counts()
    for lbl, count in label_counts.items():
        print(f" - {lbl:8s}: {count:4d} ({count / len(df) * 100:.2f}%)")
        
    palette = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
    
    # 1. Trực quan hóa phân phối nhãn mục tiêu
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = sns.barplot(
        x=label_counts.index, 
        y=label_counts.values, 
        palette=[palette[k] for k in label_counts.index],
        ax=ax,
        edgecolor='black',
        linewidth=1
    )
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(f'{int(height)}\n({height/len(df)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title("Phân phối nhãn cảm xúc phản hồi của sinh viên (Target Distribution)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Nhãn cảm xúc", fontsize=11)
    ax.set_ylabel("Số lượng mẫu", fontsize=11)
    ax.set_ylim(0, max(label_counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ: {os.path.join(output_dir, 'class_distribution.png')}")
    
    # 2. Trực quan hóa phân phối độ dài văn bản theo nhãn
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(
        data=df, x='word_count', hue='sentiment_label', 
        palette=palette, multiple='stack', kde=True, ax=axes[0],
        edgecolor='white', bins=15
    )
    axes[0].set_title("Histogram độ dài câu (Số từ)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Số lượng từ / phản hồi", fontsize=11)
    axes[0].set_ylabel("Tần suất", fontsize=11)
    
    sns.boxplot(
        data=df, x='sentiment_label', y='word_count', 
        palette=palette, ax=axes[1], boxprops=dict(alpha=0.8)
    )
    axes[1].set_title("Boxplot độ dài câu theo từng nhãn", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Nhãn cảm xúc", fontsize=11)
    axes[1].set_ylabel("Số lượng từ", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "text_length_distribution.png"), dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ: {os.path.join(output_dir, 'text_length_distribution.png')}")
    
    # 3. Phân bố theo Khoa và Loại phản hồi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dept_ct = pd.crosstab(df['department'], df['sentiment_label'], normalize='index') * 100
    dept_ct.plot(kind='bar', stacked=True, color=[palette['negative'], palette['neutral'], palette['positive']], ax=axes[0], edgecolor='black')
    axes[0].set_title("Tỷ lệ cảm xúc theo Khoa / Viện (Department)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Khoa", fontsize=11)
    axes[0].set_ylabel("Tỷ lệ (%)", fontsize=11)
    axes[0].legend(title="Cảm xúc", loc='upper right')
    axes[0].tick_params(axis='x', rotation=45)
    
    fb_ct = pd.crosstab(df['feedback_type'], df['sentiment_label'], normalize='index') * 100
    fb_ct.plot(kind='bar', stacked=True, color=[palette['negative'], palette['neutral'], palette['positive']], ax=axes[1], edgecolor='black')
    axes[1].set_title("Tỷ lệ cảm xúc theo Loại phản hồi (Feedback Type)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Loại phản hồi", fontsize=11)
    axes[1].set_ylabel("Tỷ lệ (%)", fontsize=11)
    axes[1].legend(title="Cảm xúc", loc='upper right')
    axes[1].tick_params(axis='x', rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "department_feedback_sentiment.png"), dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ: {os.path.join(output_dir, 'department_feedback_sentiment.png')}")
    
    # 4. Phân tích yếu tố châm biếm (Sarcasm Flag)
    fig, ax = plt.subplots(figsize=(7, 5))
    sarcasm_ct = pd.crosstab(df['sarcasm_flag'], df['sentiment_label'], normalize='index') * 100
    sarcasm_ct.index = ['Không châm biếm (0)', 'Có châm biếm (1)']
    sarcasm_ct.plot(kind='bar', color=[palette['negative'], palette['neutral'], palette['positive']], ax=ax, edgecolor='black', width=0.6)
    ax.set_title("Phân bố cảm xúc theo nhãn châm biếm (Sarcasm Flag)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Tỷ lệ (%) trong từng nhóm", fontsize=11)
    ax.set_xlabel("Nhóm phản hồi", fontsize=11)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Cảm xúc")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "sarcasm_impact.png"), dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ: {os.path.join(output_dir, 'sarcasm_impact.png')}")
    
    # 5. Top các từ phổ biến nhất theo từng nhóm cảm xúc
    cleaned_texts = [clean_text(t) for t in df['feedback_text']]
    df['cleaned_text'] = cleaned_texts
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, target in enumerate(['negative', 'neutral', 'positive']):
        subset = df[df['sentiment_label'] == target]
        all_words = " ".join(subset['cleaned_text']).split()
        top_words = Counter(all_words).most_common(12)
        if top_words:
            w_names, w_counts = zip(*top_words)
            sns.barplot(x=list(w_counts), y=list(w_names), ax=axes[idx], color=palette[target], edgecolor='black')
            axes[idx].set_title(f"Top 12 từ khóa: {target.upper()}", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Tần suất xuất hiện", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "top_words_per_sentiment.png"), dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ: {os.path.join(output_dir, 'top_words_per_sentiment.png')}")
    
    return df

if __name__ == "__main__":
    perform_eda()
