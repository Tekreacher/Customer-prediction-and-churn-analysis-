5# Customer Churn Prediction

Predicts which customers are likely to churn using Logistic Regression, Random Forest, and XGBoost, then uses an LLM (Claude) to explain each prediction and suggest retention actions.

## Setup
pip install -r requirements.txt

## Data
Put your dataset at `data/churn.csv` (needs a `Churn` column with Yes/No, e.g. the Telco Customer Churn dataset). If the file is missing, generated sample data is used.

## Train
python train.py

## LLM explanation
export ANTHROPIC_API_KEY="your-key"
python llm_explain.py 0
