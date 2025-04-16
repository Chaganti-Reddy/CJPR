import shap
import pandas as pd

def explain_model_with_shap(model, X_sample, feature_names):
    explainer = shap.Explainer(model.predict_proba, X_sample)
    shap_values = explainer(X_sample)
    
    # Summary Plot
    shap.summary_plot(shap_values[:,1], features=X_sample, feature_names=feature_names)
    
    # Dependence plots for most important features
    shap.plots.bar(shap_values[:, 1], max_display=10)
