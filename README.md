# Loan Default Prediction

Predicting whether a borrower will default on a loan, using supervised machine learning (binary classification).

## Dataset
- Loan default dataset with 255,347 records and 18 columns (borrower and loan details such as age, income, credit score, loan amount, interest rate and employment type).
- Target: `Default` (1 = defaulted, 0 = repaid). The classes are imbalanced: about 11.6% defaults.
- The data file is not included in this repo. Download `Loan_default.csv` from Kaggle (https://www.kaggle.com/datasets/nikhil1e9/loan-default) and place it in a folder named `data/`.

## Approach
1. Data checks and exploratory data analysis
2. Stratified 80/10/10 train, validation and test split
3. Preprocessing pipelines: median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features
4. Compared five models: Logistic Regression, KNN, Naive Bayes, Random Forest and XGBoost
5. Selected the best model on validation F1-score for the default class, then evaluated once on the held-out test set
6. Feature importance for Random Forest and XGBoost

## Results
Accuracy is not a good measure here because of the class imbalance, so models were compared on F1-score, recall and ROC-AUC for the default class.

| Model | Default F1 | Default Recall | ROC-AUC |
|---|---|---|---|
| XGBoost | 0.340 | 0.655 | 0.753 |
| Logistic Regression | 0.330 | 0.691 | 0.750 |
| Naive Bayes | 0.324 | 0.687 | 0.742 |
| Random Forest | 0.313 | 0.259 | 0.750 |
| KNN | 0.081 | 0.046 | 0.632 |

(Validation set.) The selected model, **XGBoost**, scored on the test set: F1 0.351, default recall 0.674, ROC-AUC 0.759.

## How to run
Install the libraries:

```
pip install -r requirements.txt
```

Download `Loan_default.csv` and place it in a folder named `data/`, then open `loan-default-prediction.ipynb` in Jupyter and run all cells.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost

## Author
Luvisha Gaba, B.Sc. (Hons.) Mathematical Science, University of Delhi
