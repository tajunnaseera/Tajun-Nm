
# Predicting Customer Behavior Using Machine Learning

## Overview
This project uses machine learning to predict customer churn based on behavioral and demographic features.

## Dataset
A sample dataset (`customer_behavior_dataset.csv`) includes:
- Age, Gender, Annual Income, Spending Score
- Purchase Frequency and Days Since Last Purchase
- Churn Status (Target Variable)

## Steps to Run

1. Install requirements:
```
pip install pandas scikit-learn
```

2. Place the dataset in your working directory.

3. Run the script:
```
python customer_behavior_model.py
```

## Output
The script prints model accuracy based on a random forest classifier.
