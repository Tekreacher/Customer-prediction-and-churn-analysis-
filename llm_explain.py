import os
import sys
import joblib
import pandas as pd
import anthropic

from train import load_data, MODEL_PATH

MODEL_NAME = "claude-sonnet-5-5"


def explain_customer(row_index=0):
    pipe = joblib.load(MODEL_PATH)
    X, _ = load_data()
    customer = X.iloc[[row_index]]

    churn_prob = float(pipe.predict_proba(customer)[0, 1])
    details = customer.iloc[0].to_dict()

    prompt = (
        "You are a customer retention analyst.\n"
        f"Customer details: {details}\n"
        f"Model predicted churn probability: {churn_prob:.1%}\n\n"
        "In under 150 words: explain the likely reasons for this risk level "
        "and suggest 3 specific retention actions."
    )

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"Customer #{row_index} churn probability: {churn_prob:.1%}\n")
    print(response.content[0].text)


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set the ANTHROPIC_API_KEY environment variable first.")
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    explain_customer(idx)
