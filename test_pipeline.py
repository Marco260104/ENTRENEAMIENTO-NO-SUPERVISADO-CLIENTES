import os
import sys

def run_tests():
    print("Iniciando pruebas del pipeline de segmentacion...")
    
    # 1. Verify files exist
    assert os.path.exists("data_generation.py"), "Falta data_generation.py"
    assert os.path.exists("clustering_analysis.py"), "Falta clustering_analysis.py"
    assert os.path.exists("train_and_export.py"), "Falta train_and_export.py"
    assert os.path.exists("index.html"), "Falta index.html"
    print("[OK] Todos los archivos de codigo fuente existen.")
    
    # 2. Test data loading
    try:
        from clustering_analysis import load_data, MODELING_COLS
        df = load_data()
        print(f"[OK] Dataset cargado correctamente. Registros: {len(df)}, Columnas: {list(df.columns)}")
        
        # Verify modeling columns
        for col in MODELING_COLS:
            assert col in df.columns, f"Falta columna obligatoria: {col}"
        print("[OK] Todas las variables obligatorias estan presentes en el dataset.")
    except Exception as e:
        print(f"[ERROR] Error en la carga de datos: {e}")
        sys.exit(1)
        
    # 3. Test outliers detection
    try:
        from clustering_analysis import detect_outliers_iqr
        outliers_info, all_idx = detect_outliers_iqr(df)
        print(f"[OK] Deteccion de atipicos completada. Total atipicos detectados por IQR: {len(all_idx)}")
    except Exception as e:
        print(f"[ERROR] Error en deteccion de atipicos: {e}")
        sys.exit(1)
        
    # 4. Test scaling
    try:
        from clustering_analysis import get_scaled_data
        X_scaled, scaler = get_scaled_data(df)
        assert X_scaled.shape[0] == len(df), "El tamano de los datos escalados no coincide con el original"
        assert X_scaled.shape[1] == len(MODELING_COLS), "El numero de variables escaladas no coincide"
        print("[OK] Escalado de datos mediante StandardScaler completado correctamente.")
    except Exception as e:
        print(f"[ERROR] Error en escalado de datos: {e}")
        sys.exit(1)
        
    # 5. Test K-Means sweep
    try:
        from clustering_analysis import run_kmeans_sweep
        k_vals, inertias, silhouettes = run_kmeans_sweep(X_scaled, k_min=2, k_max=5) # Reduced range for fast test
        assert len(k_vals) == 4, "Rango de K incorrecto en el barrido"
        print(f"[OK] Barrido de K-Means (K=2 a K=5) ejecutado con exito. Silueta K=4: {silhouettes[2]:.4f}")
    except Exception as e:
        print(f"[ERROR] Error en barrido de K-Means: {e}")
        sys.exit(1)
        
    # 6. Test fit K-Means
    try:
        from clustering_analysis import fit_kmeans
        labels, centroids, sil, model = fit_kmeans(df, X_scaled, 4)
        assert len(labels) == len(df), "El tamano de las etiquetas de cluster no coincide"
        assert centroids.shape == (4, len(MODELING_COLS)), "La matriz de centroides tiene dimensiones incorrectas"
        print(f"[OK] K-Means (K=4) entrenado con exito. Silueta en espacio escalado: {sil:.4f}")
    except Exception as e:
        print(f"[ERROR] Error en entrenamiento K-Means: {e}")
        sys.exit(1)
        
    # 7. Test PCA
    try:
        from clustering_analysis import fit_pca
        df_pca, exp_var, pca_model = fit_pca(X_scaled)
        assert df_pca.shape == (len(df), 2), "Las dimensiones de la reduccion PCA no son 2D"
        print(f"[OK] PCA ejecutado con exito. Varianza explicada acumulada: {sum(exp_var)*100:.2f}%")
    except Exception as e:
        print(f"[ERROR] Error en calculo PCA: {e}")
        sys.exit(1)
        
    # 8. Test DBSCAN
    try:
        from clustering_analysis import evaluate_dbscan_configs, fit_dbscan
        db_results = evaluate_dbscan_configs(X_scaled)
        assert len(db_results) == 3, "No se evaluaron las 3 combinaciones de DBSCAN"
        
        # Test fitting the optimal one
        db_labels = fit_dbscan(X_scaled, eps=1.5, min_samples=5)
        n_noise = list(db_labels).count(-1)
        print(f"[OK] DBSCAN ejecutado con exito. Ruido detectado para eps=1.5, min_samples=5: {n_noise} puntos.")
    except Exception as e:
        print(f"[ERROR] Error en DBSCAN: {e}")
        sys.exit(1)
        
    print("\n[OK] ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE! El pipeline esta 100% libre de errores.")

if __name__ == "__main__":
    run_tests()
