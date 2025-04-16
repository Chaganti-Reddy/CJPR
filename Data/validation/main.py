import pandas as pd
from utils import load_data
from metrics import evaluate_classification, calculate_fleiss_kappa

def main():
    df = load_data("xpert_annotations.csv")

    print("Computing inter-annotator agreement (Fleiss' Kappa)...")
    kappa = calculate_fleiss_kappa(df)
    print(f"Fleiss' Kappa: {kappa}")

    print("\nEvaluating GPT predictions...")
    results = evaluate_classification(df)
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value}")

    print("\nValidation complete.")

if __name__ == "__main__":
    main()
