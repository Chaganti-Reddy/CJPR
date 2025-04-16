from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def demographic_parity(y_pred, z):
    return abs(np.mean(y_pred[z == 0]) - np.mean(y_pred[z == 1]))

def groupwise_metrics(y_true, y_pred, z, group_names=("Group 0", "Group 1")):
    results = {}
    for group_val in np.unique(z):
        mask = (z == group_val)
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask])
        results[f"{group_names[group_val]}_acc"] = acc
        results[f"{group_names[group_val]}_f1"] = f1
    results["delta_f1"] = abs(results[f"{group_names[0]}_f1"] - results[f"{group_names[1]}_f1"])
    return results
