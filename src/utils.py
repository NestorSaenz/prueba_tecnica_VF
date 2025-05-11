import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def reduce_pca(df, n_components=50):
    """
    Versión ultra-robusta para tu caso específico.
    """
    # 1. Selección segura de columnas
    pc_columns = [col for col in df.columns 
                 if str(col).startswith('PC') and 
                 pd.api.types.is_numeric_dtype(df[col])]
    
    if not pc_columns:
        raise ValueError("No se encontraron columnas PC* numéricas.")
    
    # 2. Extracción segura de datos
    try:
        X = df[pc_columns].values.astype(np.float64)
    except Exception as e:
        raise ValueError(f"No se pudieron convertir datos a float64: {e}")
    
    # 3. Verificación final de datos
    if X.size == 0:
        raise ValueError("La matriz de entrada está vacía.")
    
    # 4. Aplicación de PCA con manejo de errores
    try:
        n_components = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(X)
        return pd.DataFrame(pca_features, columns=[f"PC_{i}" for i in range(n_components)])
    except Exception as e:
        raise ValueError(f"Error en PCA: {e}\nDatos shape: {X.shape}\nTipos: {X.dtype}")