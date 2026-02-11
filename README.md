# COVID-Sentiment-Analysis---Project-1---DS-4002
Evaluation of text classifiers on COVID-19 tweets.

## Software and Platform
We used a Linux box for this project, and the models require the following packages to be installed:

BERT:
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
│   ├── README
│   └── cleaned_corona_data.csv
├── LICENSE
├── OUTPUTS
│   └── README
├── README.md
└── SCRIPTS
    ├── Preprocessing.ipynb
    ├── README
    └── eval_bert.py
```

## Reproduction
BERT (*warning:* model is very large so may take HOURS):
1. Make sure to run the dependency installers above.
2. Run `python3 SCRIPTS/eval_bert.py` in terminal and let it infer for a while.