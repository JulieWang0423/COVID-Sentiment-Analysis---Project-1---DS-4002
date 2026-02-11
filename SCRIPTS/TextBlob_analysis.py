import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

sns.set_style('whitegrid')

CLEANED_DATA_PATH = '/Users/sihewang/PycharmProjects/COVID-Sentiment-Analysis---Project-1---DS-4002/DATA/deep_cleaned_corona_data.csv'
TEXT_COL = 'OriginalTweet'
LABEL_COL = 'Sentiment'
DATE_COL = 'TweetAt'

df = pd.read_csv(CLEANED_DATA_PATH)
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
df[LABEL_COL] = df[LABEL_COL].str.strip()

LABELS = ['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive']

#def merge_sentiment(LABELS):
    #if 'Positive' in LABELS: return 'Positive'
    #if 'Negative' in LABELS: return 'Negative'
    #return 'Neutral'

#df['Sentiment_3'] = df['Sentiment'].apply(merge_sentiment)

print(f'Loaded {len(df)} rows')
print(df[LABEL_COL].value_counts())

# Score TextBlob Polarity
df['tb_polarity'] = df[TEXT_COL].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Map polarity to 5-class labels (thresholds from Analysis Plan)

def map_sentiment(score):
    if score >= 0.5:
        return 'Extremely Positive'
    elif score >= 0.05:
        return 'Positive'
    elif score >= -0.05:
        return 'Neutral'
    elif score >= -0.5:
        return 'Negative'
    else:
        return 'Extremely Negative'

df['tb_sentiment'] = df['tb_polarity'].apply(map_sentiment)

# Evaluation metrics
y_true = df[LABEL_COL]
y_pred = df['tb_sentiment']

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro  = f1_score(y_true, y_pred, average='macro', zero_division=0)

print(f'\n{"="*50}')
print(f'Accuracy:        {accuracy:.4f}')
print(f'Macro Precision: {precision:.4f}')
print(f'Macro Recall:    {recall:.4f}')
print(f'Macro F1:        {f1_macro:.4f}')
print(f'{"="*50}\n')

print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

pd.DataFrame({
    'Model': ['TextBlob'], 'Accuracy': [accuracy],
    'Macro_Precision': [precision], 'Macro_Recall': [recall], 'Macro_F1': [f1_macro]
}).to_csv('textblob_metrics.csv', index=False)


# Confusion matrix heatmap

cm = confusion_matrix(y_true, y_pred, labels=LABELS)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABELS, yticklabels=LABELS, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('TextBlob — Confusion Matrix')

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Oranges',
            xticklabels=LABELS, yticklabels=LABELS, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('TextBlob — Normalized Confusion Matrix')

plt.tight_layout()
plt.savefig('textblob_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# Per-class F1 bar chart

per_class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
colors = ['#D32F2F', '#FF7043', '#FFC107', '#66BB6A', '#2E7D32']

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(LABELS, per_class_f1, color=colors, edgecolor='black')
for bar, val in zip(bars, per_class_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontweight='bold')
ax.axhline(y=f1_macro, color='navy', linestyle='--', label=f'Macro F1 = {f1_macro:.3f}')
ax.set_ylim(0, 1.0)
ax.set_ylabel('F1 Score')
ax.set_title('TextBlob — Per-Class F1 Score')
ax.legend()
plt.tight_layout()
plt.savefig('textblob_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.show()


# Sentiment trend over time

if DATE_COL in df.columns:
    df['date'] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors='coerce')
    daily = df.groupby('date')['tb_polarity'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index, daily.values, marker='o', linewidth=2, color='steelblue')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.fill_between(daily.index, daily.values, 0,
                    where=(daily.values >= 0), alpha=0.2, color='green')
    ax.fill_between(daily.index, daily.values, 0,
                    where=(daily.values < 0), alpha=0.2, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Avg TextBlob Polarity')
    ax.set_title('Daily Average Sentiment (TextBlob)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('textblob_sentiment_trend.png', dpi=150, bbox_inches='tight')
    plt.show()

# Error analysis

misclassified = df[df[LABEL_COL] != df['tb_sentiment']]
print(f'Misclassified: {len(misclassified)}/{len(df)} ({len(misclassified)/len(df)*100:.1f}%)\n')

error_pairs = misclassified.groupby([LABEL_COL, 'tb_sentiment']).size().reset_index(name='count')
print('Top misclassification pairs:')
print(error_pairs.sort_values('count', ascending=False).head(10).to_string(index=False))


# Save predictions for VADER/BERT comparison

output_cols = [TEXT_COL, LABEL_COL, 'tb_polarity', 'tb_sentiment']
if DATE_COL in df.columns:
    output_cols.insert(0, DATE_COL)

df[output_cols].to_csv('textblob_predictions.csv', index=False)
