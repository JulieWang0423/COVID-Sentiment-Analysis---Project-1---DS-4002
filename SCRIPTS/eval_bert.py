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

# Inference
all_preds = []
all_probs = []
num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(range(0, len(texts), BATCH_SIZE),
              total=num_batches,
              desc="Running BERT Inference"):

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

# Save reports
pd.DataFrame(report_dict).transpose().to_csv(
    f"{OUTPUT_DIR}/bert_classification_report.csv"
)

pd.DataFrame(cm, index=labels_order, columns=labels_order).to_csv(
    f"{OUTPUT_DIR}/bert_confusion_matrix.csv"
)


# Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.colorbar()
plt.xticks(range(len(labels_order)), labels_order, rotation=45)
plt.yticks(range(len(labels_order)), labels_order)
threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i,
                 f"{cm[i, j]}",
                 ha="center",
                 va="center",
                 color=color,
                 fontsize=10)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_confusion_matrix.png", dpi=300)
plt.close()

# Normalized Confusion Matrix
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
plt.figure(figsize=(8, 6))
plt.imshow(cm_norm)
plt.colorbar()
plt.xticks(range(len(labels_order)), labels_order, rotation=45)
plt.yticks(range(len(labels_order)), labels_order)
threshold = cm_norm.max() / 2

for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        color = "white" if cm_norm[i, j] > threshold else "black"
        plt.text(j, i,
                 f"{cm_norm[i, j]:.2f}",
                 ha="center",
                 va="center",
                 color=color,
                 fontsize=10)

plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_normalized_confusion_matrix.png", dpi=300)
plt.close()

# True Label Distribution
true_counts = pd.Series(truth_labels).value_counts().reindex(labels_order)
plt.figure()
true_counts.plot(kind="bar")
plt.title("True Label Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_true_distribution.png")
plt.close()

# Predicted Label Distribution
pred_counts = pd.Series(all_preds).value_counts().reindex(labels_order)
plt.figure()
pred_counts.plot(kind="bar")
plt.title("Predicted Label Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_predicted_distribution.png")
plt.close()

# Precision / Recall / F1
precision_scores = [report_dict[label]["precision"] for label in labels_order]
recall_scores = [report_dict[label]["recall"] for label in labels_order]
f1_scores = [report_dict[label]["f1-score"] for label in labels_order]
plt.figure()
plt.bar(labels_order, precision_scores)
plt.title("Precision per Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_precision_per_class.png")
plt.close()

plt.figure()
plt.bar(labels_order, recall_scores)
plt.title("Recall per Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_recall_per_class.png")
plt.close()

plt.figure()
plt.bar(labels_order, f1_scores)
plt.title("F1 Score per Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_f1_per_class.png")
plt.close()

# 8. ROC Curves (One-vs-Rest)
from itertools import cycle

y_true_bin = label_binarize(truth_labels, classes=labels_order)
y_score = np.array(all_probs)

plt.figure(figsize=(8, 6))

colors = cycle(["blue", "green", "red", "purple", "orange"])

for i, (label, color) in enumerate(zip(labels_order, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve (One-vs-Rest)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/bert_multiclass_roc.png", dpi=300)
plt.close()

