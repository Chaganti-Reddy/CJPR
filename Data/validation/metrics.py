from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.inter_rater import fleiss_kappa
from utils import get_majority_label, prepare_fleiss_data

def evaluate_classification(df):
    df['majority_label'] = df.apply(get_majority_label, axis=1)

    y_true = df['majority_label']
    y_pred = df['gpt_label']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'accuracy': round(accuracy * 100, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1_score': round(f1, 2)
    }

def calculate_fleiss_kappa(df):
    label_set = sorted(list(set(df['expert_1']) | set(df['expert_2']) | set(df['expert_3'])))
    matrix = prepare_fleiss_data(df, label_set)
    kappa = fleiss_kappa(matrix)
    return round(kappa, 2)
