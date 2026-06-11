import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from clustering_analysis import load_data, MODELING_COLS

def export_model_to_html(csv_path="data/customer_data.csv", html_path="index.html"):
    # 1. Load data
    df = load_data(csv_path)
    X = df[MODELING_COLS].copy()
    
    # 2. Train StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()
    
    # 3. Train K-Means (K=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled).tolist()
    centroids = kmeans.cluster_centers_.tolist()
    
    # 4. Train DBSCAN (eps=1.5, min_samples=5)
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Get scaled coordinates of core points to verify noise in JS
    core_indices = dbscan.core_sample_indices_
    core_points = X_scaled[core_indices].tolist()
    
    # 5. Train PCA (2 components)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_components = pca.components_.tolist()
    pca_mean = pca.mean_.tolist()
    
    # Organize base points for scatter plot
    base_points_data = []
    for i in range(len(df)):
        base_points_data.append({
            "id": df.loc[i, "cliente_id"],
            "x": float(X_pca[i, 0]),
            "y": float(X_pca[i, 1]),
            "cluster": int(kmeans_labels[i]),
            "is_noise": bool(dbscan_labels[i] == -1)
        })
        
    # Serialize data for HTML injection
    data_json = {
        "mean": mean,
        "scale": scale,
        "centroids": centroids,
        "pca_components": pca_components,
        "pca_mean": pca_mean,
        "core_points": core_points,
        "base_points": base_points_data
    }
    
    # 6. Generate single self-contained HTML file with premium dark styling
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Probador de Segmentación de Clientes</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at 50% 50%, #151b26, #0b0d13);
            color: #f0f2f6;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        header h1 {{
            color: #00ffcc;
            font-weight: 700;
            font-size: 2.2rem;
            text-shadow: 0 0 15px rgba(0, 255, 204, 0.3);
            margin-bottom: 10px;
        }}
        header p {{
            color: #a3b8cc;
            font-size: 1rem;
        }}
        .card {{
            background: rgba(30, 38, 51, 0.65);
            border: 1px solid rgba(0, 255, 204, 0.2);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }}
        .grid-inputs {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .grid-inputs {{
                grid-template-columns: 1fr;
            }}
        }}
        .input-group {{
            display: flex;
            flex-direction: column;
            margin-bottom: 15px;
        }}
        .input-group label {{
            font-size: 0.9rem;
            color: #a3b8cc;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }}
        .input-group label span {{
            color: #00ffcc;
            font-weight: 600;
        }}
        input[type="range"] {{
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: #2a3547;
            border-radius: 3px;
            outline: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #00ffcc;
            cursor: pointer;
            box-shadow: 0 0 8px rgba(0, 255, 204, 0.8);
        }}
        button {{
            background: linear-gradient(135deg, #00ffcc, #00b386);
            color: #0b0d13;
            border: none;
            border-radius: 8px;
            padding: 15px 30px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 255, 204, 0.2);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 255, 204, 0.4);
        }}
        .results {{
            display: none;
            margin-top: 20px;
        }}
        .result-box {{
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 5px solid #00ffcc;
            background: rgba(21, 27, 38, 0.8);
        }}
        .result-title {{
            font-weight: 600;
            font-size: 1rem;
            color: #a3b8cc;
            margin-bottom: 8px;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 10px;
            color: #fff;
        }}
        .badge-vip {{ background-color: #bcbd22; }}
        .badge-fiel {{ background-color: #1f77b4; }}
        .badge-ofertas {{ background-color: #ff7f0e; }}
        .badge-inactivo {{ background-color: #9467bd; }}
        .badge-anomaly {{ background-color: #ff3333; }}
        .badge-normal {{ background-color: #2ca02c; }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            width: 100%;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Probador de Segmentación de Clientes</h1>
            <p>Usa los sliders para simular el comportamiento de un cliente y clasificarlo en tiempo real en los modelos entrenados.</p>
        </header>

        <section class="card">
            <div class="grid-inputs">
                <div class="left-inputs">
                    <div class="input-group">
                        <label>Frecuencia de compras al mes: <span id="val_frecuencia">5.0</span></label>
                        <input type="range" id="frecuencia" min="0.5" max="20.0" value="5.0" step="0.5" oninput="updateVal('frecuencia', this.value)">
                    </div>
                    <div class="input-group">
                        <label>Ticket Promedio por compra (USD): <span id="val_ticket">100</span></label>
                        <input type="range" id="ticket" min="5" max="500" value="100" step="5" oninput="updateVal('ticket', this.value)">
                    </div>
                    <div class="input-group">
                        <label>Días desde la última compra: <span id="val_recencia">30</span></label>
                        <input type="range" id="recencia" min="1" max="365" value="30" step="1" oninput="updateVal('recencia', this.value)">
                    </div>
                    <div class="input-group">
                        <label>Categorías distintas compradas: <span id="val_categorias">5</span></label>
                        <input type="range" id="categorias" min="1" max="15" value="5" step="1" oninput="updateVal('categorias', this.value)">
                    </div>
                    <div class="input-group">
                        <label>Porcentaje de compras con descuento: <span id="val_descuento">20%</span></label>
                        <input type="range" id="descuento" min="0.0" max="1.0" value="0.2" step="0.05" oninput="updateValPct('descuento', this.value)">
                    </div>
                </div>
                <div class="right-inputs">
                    <div class="input-group">
                        <label>Devoluciones al año: <span id="val_devoluciones">1</span></label>
                        <input type="range" id="devoluciones" min="0" max="10" value="1" step="1" oninput="updateVal('devoluciones', this.value)">
                    </div>
                    <div class="input-group">
                        <label>Horas de navegación a la semana: <span id="val_horas">10.0</span></label>
                        <input type="range" id="horas" min="0.5" max="30.0" value="10.0" step="0.5" oninput="updateVal('horas', this.value)">
                    </div>
                    <div class="input-group">
                        <label>Reseñas escritas: <span id="val_reviews">5</span></label>
                        <input type="range" id="reviews" min="0" max="50" value="5" step="1" oninput="updateVal('reviews', this.value)">
                    </div>
                    <div class="input-group">
                        <label>Tasa de abandono de carrito: <span id="val_abandono">25%</span></label>
                        <input type="range" id="abandono" min="0.0" max="1.0" value="0.25" step="0.05" oninput="updateValPct('abandono', this.value)">
                    </div>
                </div>
            </div>
            <button onclick="predictCustomer()">🚀 Clasificar Cliente</button>
        </section>

        <section id="results-section" class="card results">
            <h2>Resultados del Análisis</h2>
            <div class="result-box" id="kmeans-box">
                <div class="result-title">1. Clasificación K-Means</div>
                <div id="kmeans-badge" class="badge"></div>
                <p id="kmeans-desc" style="color: #f0f2f6;"></p>
            </div>
            <div class="result-box" id="dbscan-box">
                <div class="result-title">2. Detección de Anomalías (DBSCAN)</div>
                <div id="dbscan-badge" class="badge"></div>
                <p id="dbscan-desc" style="color: #f0f2f6;"></p>
            </div>
            <div class="chart-container">
                <canvas id="pcaChart"></canvas>
            </div>
        </section>
    </div>

    <script>
        // Model Data injected from Python
        const modelData = {json.dumps(data_json)};
        
        let myChart = null;

        function updateVal(id, val) {{
            document.getElementById('val_' + id).innerText = val;
        }}

        function updateValPct(id, val) {{
            document.getElementById('val_' + id).innerText = Math.round(val * 100) + '%';
        }}

        function predictCustomer() {{
            // 1. Get input values
            const x = [
                parseFloat(document.getElementById('frecuencia').value),
                parseFloat(document.getElementById('ticket').value),
                parseFloat(document.getElementById('recencia').value),
                parseInt(document.getElementById('categorias').value),
                parseFloat(document.getElementById('descuento').value),
                parseInt(document.getElementById('devoluciones').value),
                parseFloat(document.getElementById('horas').value),
                parseInt(document.getElementById('reviews').value),
                parseFloat(document.getElementById('abandono').value)
            ];

            // 2. Scale inputs (StandardScaler math)
            let x_scaled = [];
            for (let i = 0; i < 9; i++) {{
                x_scaled.push((x[i] - modelData.mean[i]) / modelData.scale[i]);
            }}

            // 3. Predict K-Means (Euclidean distance to centroids)
            let minCentroidDist = Infinity;
            let predictedCluster = 0;
            for (let k = 0; k < 4; k++) {{
                let dist = 0;
                for (let i = 0; i < 9; i++) {{
                    dist += Math.pow(x_scaled[i] - modelData.centroids[k][i], 2);
                }}
                dist = Math.sqrt(dist);
                if (dist < minCentroidDist) {{
                    minCentroidDist = dist;
                    predictedCluster = k;
                }}
            }}

            // Define segment meta
            const segmentMeta = {{
                0: {{
                    name: "Cliente Fiel y Activo",
                    desc: "Cliente transaccional frecuente. Compra regularmente, navega mucho y escribe reseñas activamente.",
                    badgeClass: "badge-fiel",
                    borderColor: "#1f77b4"
                }},
                1: {{
                    name: "Cliente Premium (VIP)",
                    desc: "Cliente corporativo o de alto valor. Realiza compras costosas de múltiples categorías sin recurrir a descuentos.",
                    badgeClass: "badge-vip",
                    borderColor: "#bcbd22"
                }},
                2: {{
                    name: "Cazador de Ofertas",
                    desc: "Cliente sensible al precio. Compra de forma espaciada, de bajo costo y casi siempre con cupones de descuento.",
                    badgeClass: "badge-ofertas",
                    borderColor: "#ff7f0e"
                }},
                3: {{
                    name: "Cliente Inactivo (Riesgo de Fuga)",
                    desc: "Cliente frío que lleva meses sin interactuar, tiene una alta tasa de abandono de carrito y bajas visitas.",
                    badgeClass: "badge-inactivo",
                    borderColor: "#9467bd"
                }}
            }};

            const kmRes = segmentMeta[predictedCluster];
            const kmBadge = document.getElementById('kmeans-badge');
            kmBadge.className = "badge " + kmRes.badgeClass;
            kmBadge.innerText = kmRes.name;
            document.getElementById('kmeans-desc').innerText = kmRes.desc;
            document.getElementById('kmeans-box').style.borderLeftColor = kmRes.borderColor;

            // 4. Predict DBSCAN Anomaly (Core points distance check)
            // A new point is noise if distance to ALL core points is > eps (1.5)
            let minCoreDist = Infinity;
            for (let i = 0; i < modelData.core_points.length; i++) {{
                let dist = 0;
                for (let j = 0; j < 9; j++) {{
                    dist += Math.pow(x_scaled[j] - modelData.core_points[i][j], 2);
                }}
                dist = Math.sqrt(dist);
                if (dist < minCoreDist) {{
                    minCoreDist = dist;
                }}
            }}

            const isAnomaly = minCoreDist > 1.5;
            const dbBadge = document.getElementById('dbscan-badge');
            const dbDesc = document.getElementById('dbscan-desc');
            const dbBox = document.getElementById('dbscan-box');

            if (isAnomaly) {{
                dbBadge.className = "badge badge-anomaly";
                dbBadge.innerText = "⚠️ Cliente Anómalo (Ruido)";
                dbDesc.innerText = "El comportamiento ingresado se desvía drásticamente del comportamiento denso habitual (outlier).";
                dbBox.style.borderLeftColor = "#ff3333";
            }} else {{
                dbBadge.className = "badge badge-normal";
                dbBadge.innerText = "✅ Comportamiento Normal";
                dbDesc.innerText = "El comportamiento ingresado se asemeja a la densidad de los grupos estables de clientes.";
                dbBox.style.borderLeftColor = "#2ca02c";
            }}

            // 5. Calculate PCA coordinates
            let pc1 = 0;
            let pc2 = 0;
            for (let i = 0; i < 9; i++) {{
                let centered = x_scaled[i] - modelData.pca_mean[i];
                pc1 += centered * modelData.pca_components[0][i];
                pc2 += centered * modelData.pca_components[1][i];
            }}

            // Show Results Section
            document.getElementById('results-section').style.display = 'block';

            // 6. Draw / Update PCA Scatter Chart using Chart.js
            drawChart(pc1, pc2, isAnomaly, kmRes.name);
            
            // Scroll to results
            document.getElementById('results-section').scrollIntoView({{ behavior: 'smooth' }});
        }}

        function drawChart(newX, newY, isAnomaly, clusterName) {{
            const clusterColors = {{
                0: 'rgba(31, 119, 180, 0.5)',   // Fiel
                1: 'rgba(188, 189, 34, 0.5)',   // VIP
                2: 'rgba(255, 127, 14, 0.5)',   // Ofertas
                3: 'rgba(148, 103, 189, 0.5)'   // Inactivo
            }};

            // Separate datasets for base points by cluster
            let datasets = [
                {{ label: 'Cliente Fiel y Activo', data: [], backgroundColor: 'rgba(31, 119, 180, 0.4)', pointRadius: 4 }},
                {{ label: 'Cliente Premium (VIP)', data: [], backgroundColor: 'rgba(188, 189, 34, 0.4)', pointRadius: 4 }},
                {{ label: 'Cazador de Ofertas', data: [], backgroundColor: 'rgba(255, 127, 14, 0.4)', pointRadius: 4 }},
                {{ label: 'Cliente Inactivo (Riesgo)', data: [], backgroundColor: 'rgba(148, 103, 189, 0.4)', pointRadius: 4 }},
                {{ label: 'Anomalías del Dataset', data: [], backgroundColor: 'rgba(255, 51, 51, 0.8)', pointRadius: 5, pointStyle: 'rectRot' }}
            ];

            modelData.base_points.forEach(pt => {{
                if (pt.is_noise) {{
                    datasets[4].data.push({{ x: pt.x, y: pt.y, id: pt.id }});
                }} else {{
                    datasets[pt.cluster].data.push({{ x: pt.x, y: pt.y, id: pt.id }});
                }}
            }});

            // Add the new predicted customer point dataset
            datasets.push({{
                label: 'Nuevo Cliente ingresado',
                data: [{{ x: newX, y: newY, id: 'NUEVO' }}],
                backgroundColor: isAnomaly ? '#ff3366' : '#00ffcc',
                borderColor: '#ffffff',
                borderWidth: 2,
                pointRadius: 10,
                pointStyle: 'triangle',
                showLine: false
            }});

            const ctx = document.getElementById('pcaChart').getContext('2d');
            
            if (myChart) {{
                myChart.destroy();
            }}

            myChart = new Chart(ctx, {{
                type: 'scatter',
                data: {{ datasets: datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'Componente Principal 1 (PC1)', color: '#a3b8cc' }},
                            grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                            ticks: {{ color: '#a3b8cc' }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'Componente Principal 2 (PC2)', color: '#a3b8cc' }},
                            grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                            ticks: {{ color: '#a3b8cc' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#f0f2f6' }}
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return context.dataset.label + ': ' + context.raw.id + ' (PC1: ' + context.raw.x.toFixed(2) + ', PC2: ' + context.raw.y.toFixed(2) + ')';
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Panel HTML interactivo autocontenido generado exitosamente en: {html_path}")

if __name__ == "__main__":
    export_model_to_html()
