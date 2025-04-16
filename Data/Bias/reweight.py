from sklearn.utils.class_weight import compute_sample_weight

def compute_weights(y, z, underrepresented_val=1):
    # Assign higher weights to underrepresented group
    weights = np.where(z == underrepresented_val, 1.5, 1.0)
    return weights

# During model training (scikit-learn)
from sklearn.linear_model import LogisticRegression

def train_with_reweighting(X_train, y_train, z_train):
    sample_weights = compute_weights(y_train, z_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model
