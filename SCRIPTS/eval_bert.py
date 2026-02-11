import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# Configuration
INPUT_CSV = "DATA/cleaned_corona_data.csv"
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
OUTPUT_DIR = "OUTPUTS"
BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nUsing device: {DEVICE}")

# Load Dataset
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["OriginalTweet", "Sentiment"])
texts = df["OriginalTweet"].astype(str).tolist()
truth_labels = df["Sentiment"].astype(str).tolist()

labels_order = [
    "Extremely Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Extremely Positive"
]

print(f"Total samples: {len(texts)}")

# Load Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# Label Mapping (Stars → Dataset Labels)
star_to_sentiment = {
    "1 star": "Extremely Negative",
    "2 stars": "Negative",
    "3 stars": "Neutral",
    "4 stars": "Positive",
    "5 stars": "Extremely Positive",
}

# Inference with tqdm
all_preds = []
all_probs = []
num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(
    range(0, len(texts), BATCH_SIZE),
    total=num_batches,
    desc="Running BERT Inference"
):
    batch_texts = texts[i:i+BATCH_SIZE]

    inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred_indices = torch.argmax(probs, dim=1).tolist()
    id2label = model.config.id2label
    batch_preds = [star_to_sentiment[id2label[idx]] for idx in pred_indices]
    all_preds.extend(batch_preds)
    all_probs.extend(probs.cpu().tolist())

# Metrics
accuracy = accuracy_score(truth_labels, all_preds)
report_dict = classification_report(
    truth_labels,
    all_preds,
    labels=labels_order,
    output_dict=True
)

cm = confusion_matrix(truth_labels, all_preds, labels=labels_order)
print(f"\nAccuracy: {accuracy:.4f}")

# Save classification report to CSV
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(f"{OUTPUT_DIR}/bert_classification_report.csv")

# Save confusion matrix to CSV
pd.DataFrame(cm, index=labels_order, columns=labels_order) \
    .to_csv(f"{OUTPUT_DIR}/bert_confusion_matrix.csv")

# 1. Confusion Matrix
plt.figure()
plt.imshow(cm)
plt.xticks(range(len(labels_order)), labels_order, rotation=45)
plt.yticks(range(len(labels_order)), labels_order)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_confusion_matrix.png")
plt.close()

# 2. Normalized Confusion Matrix
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure()
plt.imshow(cm_norm)
plt.xticks(range(len(labels_order)), labels_order, rotation=45)
plt.yticks(range(len(labels_order)), labels_order)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_normalized_confusion_matrix.png")
plt.close()

# 3. True Label Distribution
true_counts = pd.Series(truth_labels).value_counts().reindex(labels_order)
plt.figure()
true_counts.plot(kind="bar")
plt.title("True Label Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_true_distribution.png")
plt.close()

# 4. Predicted Label Distribution
pred_counts = pd.Series(all_preds).value_counts().reindex(labels_order)
plt.figure()
pred_counts.plot(kind="bar")
plt.title("Predicted Label Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_predicted_distribution.png")
plt.close()

# 5. Precision per Class
precision_scores = [report_dict[label]["precision"] for label in labels_order]
plt.figure()
plt.bar(labels_order, precision_scores)
plt.title("Precision per Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_precision_per_class.png")
plt.close()

# 6. Recall per Class
recall_scores = [report_dict[label]["recall"] for label in labels_order]
plt.figure()
plt.bar(labels_order, recall_scores)
plt.title("Recall per Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_recall_per_class.png")
plt.close()

# 7. F1 Score per Class
f1_scores = [report_dict[label]["f1-score"] for label in labels_order]
plt.figure()
plt.bar(labels_order, f1_scores)
plt.title("F1 Score per Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_f1_per_class.png")
plt.close()

# 8. ROC Curves (One-vs-Rest)
y_true_bin = label_binarize(truth_labels, classes=labels_order)
y_score = np.array(all_probs)

for i, label in enumerate(labels_order):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve - {label} (AUC = {roc_auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bert_roc_{label.replace(' ', '_')}.png")
    plt.close()

print("\nAll evaluation figures and CSV files saved to OUTPUTS/")
print("Pipeline complete.")