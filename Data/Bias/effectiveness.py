import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from fairness import FairClassifier, fairness_aware_loss
from audits import groupwise_metrics, demographic_parity
from reweight import compute_weights

# === DATA PREP ===
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bias import generate_data  # Make sure this function is available
X, y, z = generate_data()

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Helper ===
def evaluate_fairness(model, X, y_true, z, name=""):
    model.eval()
    with torch.no_grad():
        y_pred_probs = model(torch.FloatTensor(X)).numpy().flatten()
        y_pred = (y_pred_probs > 0.5).astype(int)
    group_metrics = groupwise_metrics(y_true, y_pred, z, group_names=("Urban", "Rural"))
    dp = demographic_parity(y_pred, z)
    print(f"\n{name} Model Fairness Audit")
    for k, v in group_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Demographic Parity Gap: {dp:.4f}")
    return group_metrics, dp

# === TRAIN BASELINE MODEL ===
baseline_model = FairClassifier(input_dim=X_train.shape[1], hidden_dim=64)
optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)

X_tensor = torch.FloatTensor(X_train)
y_tensor = torch.FloatTensor(y_train).unsqueeze(1)

criterion = nn.BCELoss()
for epoch in range(50):
    baseline_model.train()
    outputs = baseline_model(X_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# === EVAL BASELINE ===
baseline_metrics, baseline_dp = evaluate_fairness(baseline_model, X_test, y_test, z_test, name="Baseline")

# === TRAIN FAIR MODEL ===
fair_model = FairClassifier(input_dim=X_train.shape[1], hidden_dim=64)
optimizer_fair = torch.optim.Adam(fair_model.parameters(), lr=0.001)

z_tensor = torch.FloatTensor(z_train).unsqueeze(1)
for epoch in range(50):
    fair_model.train()
    outputs = fair_model(X_tensor)
    loss = fairness_aware_loss(outputs, y_tensor, z_tensor, lambda_param=0.5)

    optimizer_fair.zero_grad()
    loss.backward()
    optimizer_fair.step()

# === EVAL FAIR MODEL ===
fair_metrics, fair_dp = evaluate_fairness(fair_model, X_test, y_test, z_test, name="Fair")

# === COMPARE BEFORE/AFTER ===
def print_comparison(before, after, dp_before, dp_after):
    print("\n==== Fairness Metric Comparison ====")
    for metric in ['Urban_f1', 'Rural_f1', 'delta_f1']:
        print(f"{metric}: {before[metric]:.4f} → {after[metric]:.4f} (Δ: {after[metric] - before[metric]:+.4f})")
    print(f"Demographic Parity Gap: {dp_before:.4f} → {dp_after:.4f} (Δ: {dp_after - dp_before:+.4f})")

print_comparison(baseline_metrics, fair_metrics, baseline_dp, fair_dp)
