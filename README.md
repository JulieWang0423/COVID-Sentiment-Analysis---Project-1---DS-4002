# COVID-Sentiment-Analysis---Project-1---DS-4002
Evaluation of text classifiers on COVID-19 tweets.

## Software and Platform
We used a Linux box for this project, and the models require the following packages to be installed:

```
pip install tokenizer
pip install scikit-learn
pip install pandas
pip install torch
pip install matplotlib
```

## Tree Diagram of Project
```
.
├── DATA
│   ├── Corona_NLP_test.csv
│   ├── Corona_NLP_train.csv
│   ├── Data Appendix.pdf
│   ├── cleaned_corona_data.csv
│   └── deep_cleaned_corona_data.csv
├── LICENSE
├── OUTPUTS
│   ├── README
│   ├── bert_classification_report.csv
│   ├── bert_confusion_matrix.csv
│   ├── bert_confusion_matrix.png
│   ├── bert_f1_per_class.png
│   ├── bert_multiclass_roc.png
│   ├── bert_normalized_confusion_matrix.png
│   ├── bert_precision_per_class.png
│   ├── bert_predicted_distribution.png
│   ├── bert_recall_per_class.png
│   ├── new_textblob_confusion_matrix.png
│   ├── new_textblob_metrics.csv
│   ├── new_textblob_per_class_f1.png
│   ├── new_textblob_predictions.csv
│   ├── new_textblob_sentiment_trend.png
│   ├── textblob_confusion_matrix.png
│   ├── textblob_metrics.csv
│   ├── textblob_per_class_f1.png
│   ├── textblob_predictions.csv
│   └── textblob_sentiment_trend.png
├── README.md
└── SCRIPTS
    ├── Preprocessing.ipynb
    ├── README
    ├── TextBlob_analysis.py
    └── eval_bert.py
```

## Reproduction
TextBlob 
1. Make sure to run the dependency installers above.
2. Run `python3 SCRIPTS/TextBlob_analysis.py` in terminal.
3. TextBlob is a lightweight rule-based model, so it should finish within a few minutes on any machine. No GPU is required.
4. The outputs are generated at `new_textblob_*.png` and `new_textblob_*.csv` in the `OUTPUTS` directory.

BERT (*warning:* model is very large so may take HOURS):
1. Make sure to run the dependency installers above.
2. Run `python3 SCRIPTS/eval_bert.py` in terminal and let it infer for a while.
3. You can speed this up by using UVA's HPC OOD and giving it some GPUs. This will make it much faster.
4. The outputs are generated at `bert_*.png` in the `OUTPUTS` directory.

VADER 
1. Make sure to run the dependency installers above.
2. Run `python3 SCRIPTS/VADER_analysis.py` in terminal.
3. VADER is a lightweight rule-based model, so it should finish within a few minutes on any machine. No GPU is required.
4. The outputs are generated at `VADER_*.png` and `VADER_*.csv` in the `OUTPUTS` directory.
