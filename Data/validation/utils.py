import pandas as pd
from collections import Counter

def load_data(path):
    return pd.read_csv(path)

def get_majority_label(row):
    labels = [row['expert_1'], row['expert_2'], row['expert_3']]
    most_common = Counter(labels).most_common(1)
    return most_common[0][0]

def prepare_fleiss_data(df, label_set):
    """
    Returns matrix suitable for Fleiss' Kappa: each row is a case,
    each column is count of how many raters chose that label.
    """
    matrix = []
    for _, row in df.iterrows():
        label_counts = Counter([row['expert_1'], row['expert_2'], row['expert_3']])
        row_data = [label_counts.get(label, 0) for label in label_set]
        matrix.append(row_data)
    return matrix
