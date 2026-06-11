import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Numeric variables used for modeling
MODELING_COLS = [
    "frecuencia_compras_mes",
    "ticket_promedio_usd",
    "dias_desde_ultima_compra",
    "num_categorias_distintas",
    "porcentaje_compras_con_descuento",
    "num_devoluciones_año",
    "horas_navegacion_semana",
    "num_reviews_escritos",
    "tasa_abandono_carrito"
]

def load_data(csv_path="data/customer_data.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el dataset en {csv_path}. Por favor ejecuta data_generation.py primero.")
    return pd.read_csv(csv_path)

def detect_outliers_iqr(df, cols=MODELING_COLS, factor=1.5):
    """
    Detects outliers using the IQR method.
    Returns a dictionary with details per column and indices of outliers.
    """
    outliers_info = {}
    all_outlier_indices = set()
    
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        # Outliers in this column
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_info[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "bounds": (lower_bound, upper_bound),
            "count": len(col_outliers),
            "indices": list(col_outliers.index)
        }
        all_outlier_indices.update(col_outliers.index)
        
    return outliers_info, sorted(list(all_outlier_indices))

def get_scaled_data(df, cols=MODELING_COLS):
    """
    Scales the numeric variables using StandardScaler.
    """
    X = df[cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=cols), scaler

def run_kmeans_sweep(X_scaled, k_min=2, k_max=12):
    """
    Calculates Inertia (elbow) and Silhouette Scores for a range of k.
    """
    inertias = []
    silhouettes = []
    k_values = list(range(k_min, k_max + 1))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        
    return k_values, inertias, silhouettes

def compare_scaling_effect(df, cols=MODELING_COLS, k=4):
    """
    Compares the Silhouette Score of K-Means running on raw vs scaled data.
    """
    X_raw = df[cols].copy()
    X_scaled, _ = get_scaled_data(df, cols)
    
    # Run K-Means on raw
    km_raw = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_raw = km_raw.fit_predict(X_raw)
    score_raw = silhouette_score(X_raw, labels_raw)
    
    # Run K-Means on scaled
    km_scaled = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_scaled = km_scaled.fit_predict(X_scaled)
    score_scaled = silhouette_score(X_scaled, labels_scaled)
    
    # Also calculate cross silhouette (scaled silhouette of raw clusters vs scaled clusters)
    score_raw_on_scaled = silhouette_score(X_scaled, labels_raw)
    
    return {
        "score_raw_data_evaluated_on_raw_space": score_raw,
        "score_raw_data_evaluated_on_scaled_space": score_raw_on_scaled,
        "score_scaled_data_evaluated_on_scaled_space": score_scaled
    }

def fit_kmeans(df, X_scaled, k):
    """
    Fits K-Means with selected K, returns labels and centroids in original scale.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Centroids in original scale
    # Map centroids back: inverse transform of scaler
    # Or group raw data by cluster labels and take the mean (which matches the empirical centroid)
    df_temp = df.copy()
    df_temp["cluster_kmeans"] = labels
    centroids_raw = df_temp.groupby("cluster_kmeans")[MODELING_COLS].mean()
    
    sil_score = silhouette_score(X_scaled, labels)
    
    return labels, centroids_raw, sil_score, kmeans

def fit_pca(X_scaled):
    """
    Reduces dimension to 2 components using PCA.
    Returns the PCA-transformed DataFrame and explained variance.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    return df_pca, explained_variance, pca

def evaluate_dbscan_configs(X_scaled):
    """
    Tests multiple DBSCAN configurations of eps and min_samples.
    Returns results as a list of dicts.
    """
    # 3 distinct combinations
    configs = [
        {"eps": 1.2, "min_samples": 4},
        {"eps": 1.5, "min_samples": 5},
        {"eps": 2.0, "min_samples": 5}
    ]
    
    results = []
    for config in configs:
        db = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
        labels = db.fit_predict(X_scaled)
        
        # Calculate clusters (excluding noise -1)
        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = list(labels).count(-1)
        
        # Silhouette Score (only if there is at least 2 clusters including/excluding noise)
        if n_clusters > 1:
            # Silhouette on all points
            sil = silhouette_score(X_scaled, labels)
        else:
            sil = float('nan')
            
        results.append({
            "eps": config["eps"],
            "min_samples": config["min_samples"],
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "labels": labels,
            "silhouette": sil
        })
        
    return results

def fit_dbscan(X_scaled, eps=1.5, min_samples=5):
    """
    Fits DBSCAN with specific parameters and returns labels.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return labels

if __name__ == "__main__":
    df = load_data()
    X_scaled, _ = get_scaled_data(df)
    k_vals, inertias, silhouettes = run_kmeans_sweep(X_scaled)
    
    # Gráfico Codo
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_vals, inertias, marker='o', color='#00cc88')
    ax.set_title("Método del Codo - Inercia por K")
    ax.set_xlabel("K")
    ax.set_ylabel("Inercia")
    plt.tight_layout()
    plt.savefig("grafico_codo.png", dpi=150)
    print("Guardado: grafico_codo.png")
    
    # Gráfico Silueta
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_vals, silhouettes, marker='o', color='#ff5566')
    ax.set_title("Silhouette Score por K")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig("grafico_silueta.png", dpi=150)
    print("Guardado: grafico_silueta.png")

    # Centroides en escala original (para mostrar en video)
    segment_names = {
        0: "Cliente Premium (VIP)",          # ticket ~382, recencia ~30, descuento ~9%
        1: "Cliente Inactivo (Riesgo Fuga)", # recencia ~274, frecuencia ~1.1, abandono ~79%
        2: "Cazador de Ofertas",             # descuento ~86%, ticket ~26, frecuencia ~1.6
        3: "Cliente Fiel y Activo"           # frecuencia ~15, recencia ~17, reviews ~28
    }
    labels, centroids_raw, sil_score, _ = fit_kmeans(df, X_scaled, 4)
    centroids_raw.index = centroids_raw.index.map(segment_names)
    print("\n===== TABLA DE CENTROIDES POR SEGMENTO (escala original) =====")
    print(centroids_raw.round(2).to_string())
    print(f"\nSilhouette Score K=4 (con escala): {sil_score:.4f}")

    # Resumen DBSCAN con 3 configuraciones
    print("\n===== RESULTADOS DBSCAN (3 configuraciones) =====")
    db_results = evaluate_dbscan_configs(X_scaled)
    for r in db_results:
        print(f"  eps={r['eps']} | min_samples={r['min_samples']} | "
              f"clusters={r['n_clusters']} | ruido={r['n_noise']} | "
              f"silhouette={r['silhouette']:.4f}" if not (r['silhouette'] != r['silhouette']) else
              f"  eps={r['eps']} | min_samples={r['min_samples']} | "
              f"clusters={r['n_clusters']} | ruido={r['n_noise']} | silhouette=N/A")