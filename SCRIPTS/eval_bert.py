import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT_CSV = "DATA/cleaned_corona_data.csv" 
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  # pretrained 5-way sentiment model

# Load data
df = pd.read_csv(INPUT_CSV)
# Expect df["text"] and df["label"] as strings, e.g. "positive", "neutral", etc.
texts = df["text"].astype(str).tolist()
truth_labels = df["label"].astype(str).tolist()

# Load prebuilt sentiment model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # inference mode

# Tokenize all text (batch)
inputs = tokenizer(
    texts,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

# Inference: get logits
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

# Convert to predicted class indices
pred_idx = torch.argmax(logits, dim=1).tolist()

# Convert predicted indices → string labels using model config
# For nlptown model, id2label is something like {"0":"1 star", ..., "4":"5 stars"}
id2label = model.config.id2label
pred_labels = [id2label[i] for i in pred_idx]

# Now compare model predictions to ground truth
print("=== Accuracy ===")
print(accuracy_score(truth_labels, pred_labels))

print("\n=== Classification Report ===")
print(classification_report(truth_labels, pred_labels))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(truth_labels, pred_labels))
