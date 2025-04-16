import torch
import torch.nn as nn

class FairClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FairClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

def fairness_aware_loss(y_pred, y_true, z, lambda_param=1.0):
    bce = nn.BCELoss()(y_pred, y_true)

    # Demographic Parity term
    z0_mask = (z == 0)
    z1_mask = (z == 1)
    dp = torch.abs(y_pred[z0_mask].mean() - y_pred[z1_mask].mean())
    return bce + lambda_param * dp
