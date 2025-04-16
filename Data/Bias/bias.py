# === Imports ===
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

X, y, z = generate_data()

# === Preprocessing ===
X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=0.3, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Sample Reweighting ===
def compute_weights(y, z, underrepresented_val=1):
    return np.where(z == underrepresented_val, 1.5, 1.0)

weights = compute_weights(y_train, z_train)

# === Fair Model ===
class FairClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FairClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def fairness_aware_loss(y_pred, y_true, z, lambda_param=1.0):
    bce = nn.BCELoss()(y_pred, y_true)
    z0_mask = (z == 0).squeeze()
    z1_mask = (z == 1).squeeze()
    dp = torch.abs(y_pred[z0_mask].mean() - y_pred[z1_mask].mean())
    return bce + lambda_param * dp

# === Training ===
input_dim = X_train.shape[1]
model = FairClassifier(input_dim=input_dim, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.FloatTensor(X_train)
y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
z_tensor = torch.FloatTensor(z_train).unsqueeze(1)

for epoch in range(50):
    model.train()
    outputs = model(X_tensor)
    loss = fairness_aware_loss(outputs, y_tensor, z_tensor, lambda_param=0.5)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.4f}")

# === Evaluation: Bias Audits ===
def groupwise_metrics(y_true, y_pred, z, group_names=("Urban", "Rural")):
    results = {}
    for group_val in np.unique(z):
        mask = (z == group_val)
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask])
        results[f"{group_names[group_val]}_acc"] = acc
        results[f"{group_names[group_val]}_f1"] = f1
    results["delta_f1"] = abs(results["Urban_f1"] - results["Rural_f1"])
    return results

def demographic_parity(y_pred, z):
    return abs(np.mean(y_pred[z == 0]) - np.mean(y_pred[z == 1]))

model.eval()
with torch.no_grad():
    y_pred_probs = model(torch.FloatTensor(X_test)).numpy().flatten()
    y_pred_labels = (y_pred_probs > 0.5).astype(int)

audit = groupwise_metrics(y_test, y_pred_labels, z_test)
dp_gap = demographic_parity(y_pred_labels, z_test)

print("\nBias Audit Results:")
for k, v in audit.items():
    print(f"{k}: {v:.4f}")
print(f"Demographic Parity Gap: {dp_gap:.4f}")

# === SHAP Explainability ===
def explain_model_with_shap(model, X_sample, feature_names=None):
    def model_predict(x):
        with torch.no_grad():
            return model(torch.FloatTensor(x)).numpy()
    
    explainer = shap.Explainer(model_predict, X_sample)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=True)

print("\nExplaining with SHAP...")
explain_model_with_shap(model, X_sample=X_test[:100], feature_names=[f"f{i}" for i in range(X.shape[1])])
