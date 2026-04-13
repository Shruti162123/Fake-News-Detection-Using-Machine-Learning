import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _save_plot(output_dir, file_name):
    """Helper to save each figure with high quality."""
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches="tight")
    plt.close()


def generate_research_graphs(df, y_test, y_pred, y_prob, model, vectorizer, output_dir="graphs"):
    """
    Generate multiple graphs for a research paper.
    All graphs are saved as PNG files inside output_dir.
    """
    sns.set_style("whitegrid")

    # 1) Class Distribution
    plt.figure(figsize=(6, 4))
    class_counts = df["label"].value_counts().sort_index()
    plt.bar(["Fake (0)", "Real (1)"], class_counts.values, color=["#ff6b6b", "#4dabf7"])
    plt.title("Class Distribution")
    plt.xlabel("News Class")
    plt.ylabel("Count")
    _save_plot(output_dir, "01_class_distribution.png")

    # 2) Content Length Distribution by Class
    temp_df = df.copy()
    temp_df["content_length"] = temp_df["content"].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=temp_df,
        x="content_length",
        hue="label",
        bins=40,
        kde=True,
        palette={0: "#ff6b6b", 1: "#4dabf7"},
        alpha=0.5,
    )
    plt.title("Content Length Distribution by Class")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    _save_plot(output_dir, "02_content_length_distribution.png")

    # 3) Boxplot of Content Length by Class
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=temp_df, x="label", y="content_length", palette=["#ff6b6b", "#4dabf7"])
    plt.xticks([0, 1], ["Fake (0)", "Real (1)"])
    plt.title("Content Length Boxplot by Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Words")
    _save_plot(output_dir, "03_content_length_boxplot.png")

    # 4) Confusion Matrix (Raw)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.title("Confusion Matrix (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    _save_plot(output_dir, "04_confusion_matrix_counts.png")

    # 5) Confusion Matrix (Normalized)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    _save_plot(output_dir, "05_confusion_matrix_normalized.png")

    # 6) ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="#1c7ed6", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    _save_plot(output_dir, "06_roc_curve.png")

    # 7) Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#f08c00", linewidth=2, label=f"AUC = {pr_auc:.4f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    _save_plot(output_dir, "07_precision_recall_curve.png")

    # 8) Prediction Probability Distribution by True Class
    y_test_arr = np.array(y_test)
    plt.figure(figsize=(7, 4))
    sns.histplot(y_prob[y_test_arr == 0], color="#ff6b6b", bins=30, label="True Fake (0)", alpha=0.6)
    sns.histplot(y_prob[y_test_arr == 1], color="#4dabf7", bins=30, label="True Real (1)", alpha=0.6)
    plt.title("Predicted Probability Distribution")
    plt.xlabel("Predicted Probability of Real News")
    plt.ylabel("Frequency")
    plt.legend()
    _save_plot(output_dir, "08_probability_distribution.png")

    # 9) Top Positive TF-IDF Features (Real Class)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_real_idx = np.argsort(coefs)[-20:]
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names[top_real_idx], coefs[top_real_idx], color="#4dabf7")
    plt.title("Top 20 Features Supporting Real News")
    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("Feature")
    _save_plot(output_dir, "09_top_features_real.png")

    # 10) Top Negative TF-IDF Features (Fake Class)
    top_fake_idx = np.argsort(coefs)[:20]
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names[top_fake_idx], coefs[top_fake_idx], color="#ff6b6b")
    plt.title("Top 20 Features Supporting Fake News")
    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("Feature")
    _save_plot(output_dir, "10_top_features_fake.png")

    print(f"\nSaved 10 graphs successfully in: {output_dir}")
