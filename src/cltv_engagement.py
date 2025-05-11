import numpy as np

def calculate_engagement(df_pca, weights):
    """Calcula un score de engagement basado en componentes PCA ponderadas."""
    engagement = np.zeros(len(df_pca))
    for i, weight in enumerate(weights):
        engagement += weight * df_pca[f"PCA_{i+1}"]
    return engagement

def estimate_cltv(churn_prob, engagement, revenue_proxy=1000):
    """Estima CLTV usando probabilidad de churn y engagement."""
    cltv = (revenue_proxy * (1 - churn_prob)) * engagement
    return cltv / cltv.max()  # Normalizar