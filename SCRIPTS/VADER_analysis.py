#imports
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score)
import seaborn as sns

#read in data from .csv file
df = pd.read_csv("../DATA/deep_cleaned_corona_data.csv")
df = df.dropna(subset=['OriginalTweet', 'Sentiment']).reset_index(drop=True)

#perform sentiment analysis and save file
for i, j in df.iterrows():
    
    text = str(j["OriginalTweet"])
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    
    
    score_value = (score['pos'] - score['neg'])
    
    if score_value >= 0.15:
        df.loc[i, "VADER Sentiment"] = "Extremely Positive"
    elif score_value <= -0.15:
        df.loc[i, "VADER Sentiment"] = "Extremely Negative"
    elif score_value >= 0.05:
        df.loc[i, "VADER Sentiment"] = "Positive"
    elif score_value <= -0.05:
        df.loc[i, "VADER Sentiment"] = "Negative"
    else:
        df.loc[i, "VADER Sentiment"] = "Neutral"
        
    df.loc[i, 'neg'] = score['neg']
    df.loc[i, 'neu'] = score['neu']
    df.loc[i, 'pos'] = score['pos']
    df.loc[i, 'compound'] = score['compound']
    
df.to_csv("../OUTPUTS/VADER_predictions.csv", index=False)

#calculate and save accuracy, precision, recall, and f1 score
y_true = df['Sentiment']
y_pred = df['VADER Sentiment']
labels = ['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive']

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro  = f1_score(y_true, y_pred, average='macro', zero_division=0)

print(f'Accuracy:        {accuracy:.4f}')
print(f'Macro Precision: {precision:.4f}')
print(f'Macro Recall:    {recall:.4f}')
print(f'Macro F1:        {f1_macro:.4f}')

pd.DataFrame({
    'Model': ['VADER'], 'Accuracy': [accuracy],
    'Macro_Precision': [precision], 'Macro_Recall': [recall], 'Macro_F1': [f1_macro]
}).to_csv('../OUTPUTS/VADER_metrics.csv', index=False)

#create and save confusion matricies
cm = confusion_matrix(y_true, y_pred, labels = labels)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix (VADER)')

cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Oranges',
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Normalized Confusion Matrix (VADER)')

plt.tight_layout()
plt.savefig('../OUTPUTS/VADER_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

#graph mean sentiment over time
if 'TweetAt' in df.columns:
    df['date'] = pd.to_datetime(df['TweetAt'], dayfirst=True, errors='coerce')
    daily = df.groupby('date')['compound'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index, daily.values, marker='o', linewidth=2, color='steelblue')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.fill_between(daily.index, daily.values, 0,
                    where=(daily.values >= 0), alpha=0.2, color='green')
    ax.fill_between(daily.index, daily.values, 0,
                    where=(daily.values < 0), alpha=0.2, color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Avg TextBlob Polarity')
    ax.set_title('Daily Average Sentiment (VADER)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../OUTPUTS/VADER_sentiment_trend.png', dpi=150, bbox_inches='tight')
    plt.show()
    
#graph mean f1 score across different sentiment categories
per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, per_class_f1, edgecolor='black')
for bar, val in zip(bars, per_class_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontweight='bold')
ax.axhline(y=f1_macro, color='navy', linestyle='--', label=f'Macro F1 = {f1_macro:.3f}')
ax.set_ylim(0, 1.0)
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 Score (VADER)')
ax.legend()
plt.tight_layout()
plt.savefig('../OUTPUTS/VADER_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.show()