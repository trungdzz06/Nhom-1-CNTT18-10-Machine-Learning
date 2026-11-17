"""
Module: Evaluation and Error Analysis
Chức năng: Đánh giá hiệu năng mô hình, vẽ biểu đồ so sánh, ma trận nhầm lẫn (Confusion Matrix)
và phân tích lỗi dự đoán (Error Analysis) chuyên sâu.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'

def compute_metrics(y_true, y_pred):
    """
    Tính toán đầy đủ các độ đo chuẩn:
    Accuracy, Macro Precision, Weighted Precision, Macro Recall, Weighted Recall, Macro F1, Weighted F1.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def plot_confusion_matrix(y_true, y_pred, labels, title: str, save_path: str):
    """
    Vẽ ma trận nhầm lẫn chuẩn hóa theo tỷ lệ thực tế (Recall) và số lượng mẫu.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
            
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        cbar=True, linewidths=0.5, linecolor='gray'
    )
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Nhãn dự đoán (Predicted)", fontsize=11)
    ax.set_ylabel("Nhãn thực tế (Ground Truth)", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(results_df: pd.DataFrame, save_path: str):
    """
    Vẽ biểu đồ cột so sánh Accuracy, Macro-F1 và Weighted-F1 giữa các mô hình.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    plot_df = results_df[['Model', 'Accuracy', 'Macro F1', 'Weighted F1']].set_index('Model')
    plot_df.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', width=0.7)
    
    ax.set_title("So sánh hiệu năng giữa các mô hình đã học (Test Set)", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Giá trị độ đo", fontsize=11)
    ax.set_xlabel("Mô hình", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', rotation=25)
    ax.legend(loc='upper right', frameon=True)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0,
                        xytext=(0, 2), textcoords='offset points')
                        
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

def plot_vectorizer_comparison(vec_results_df: pd.DataFrame, save_path: str):
    """
    Vẽ biểu đồ so sánh ảnh hưởng của phương pháp biểu diễn đặc trưng (BoW vs TF-IDF, unigram vs bigram).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    plot_df = vec_results_df[['Representation', 'Accuracy', 'Macro F1']].set_index('Representation')
    plot_df.plot(kind='bar', ax=ax, color=['#3498db', '#e67e22'], edgecolor='black', width=0.6)
    
    ax.set_title("Ảnh hưởng của phương pháp biểu diễn đặc trưng tới MultinomialNB", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Giá trị độ đo", fontsize=11)
    ax.set_xlabel("Cấu hình biểu diễn đặc trưng", fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.tick_params(axis='x', rotation=30)
    ax.legend(["Accuracy", "Macro F1"], loc='upper right', frameon=True)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9,
                        xytext=(0, 2), textcoords='offset points')
                        
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

def analyze_errors(raw_texts, y_true, y_pred, probs=None, class_labels=None, metadata_df=None, n_samples=10):
    """
    Trích xuất và phân tích định tính các trường hợp mô hình dự đoán sai (Error Analysis).
    """
    errors = []
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    raw_texts_list = list(raw_texts)
    
    indices = [i for i in range(len(y_true_list)) if y_true_list[i] != y_pred_list[i]]
    
    for idx in indices:
        item = {
            "index": idx,
            "text": raw_texts_list[idx],
            "actual": y_true_list[idx],
            "predicted": y_pred_list[idx],
        }
        if probs is not None and class_labels is not None:
            prob_dict = {f"prob_{lbl}": round(float(probs[idx][l_idx]), 3) for l_idx, lbl in enumerate(class_labels)}
            item.update(prob_dict)
            item["confidence"] = round(float(np.max(probs[idx])), 3)
            
        if metadata_df is not None:
            row_meta = metadata_df.iloc[idx]
            for col in ['sarcasm_flag', 'emotion_tag', 'feedback_type', 'department']:
                if col in row_meta:
                    item[col] = row_meta[col]
                    
        errors.append(item)
        
    error_df = pd.DataFrame(errors)
    return error_df
