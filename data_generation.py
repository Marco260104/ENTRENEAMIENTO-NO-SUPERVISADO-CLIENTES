import pandas as pd
import numpy as np
import os
from faker import Faker

def generate_customer_data(output_path="data/customer_data.csv", num_per_archetype=100, seed=42):
    """
    Generates a synthetic e-commerce customer dataset with 4 archetypes and noise.
    Also adds a set of explicit anomalies (outliers) to test DBSCAN.
    """
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = []
    
    # Archetype 1: Cazador de Ofertas (Bargain Hunter)
    # High discount %, low frequency, low ticket, moderate recency, high cart abandonment.
    for i in range(num_per_archetype):
        rec = {
            "arquetipo": "Cazador de Ofertas",
            "frecuencia_compras_mes": np.random.normal(1.5, 0.5),
            "ticket_promedio_usd": np.random.normal(25, 8),
            "dias_desde_ultima_compra": np.random.normal(80, 25),
            "num_categorias_distintas": np.random.randint(1, 6),
            "porcentaje_compras_con_descuento": np.random.normal(0.85, 0.08),
            "num_devoluciones_año": np.random.poisson(1.0),
            "horas_navegacion_semana": np.random.normal(6, 2),
            "num_reviews_escritos": np.random.poisson(1.5),
            "tasa_abandono_carrito": np.random.normal(0.6, 0.1)
        }
        data.append(rec)
        
    # Archetype 2: Cliente Premium (VIP)
    # High ticket, low discount, low returns, moderate-high frequency, high variety.
    for i in range(num_per_archetype):
        rec = {
            "arquetipo": "Cliente Premium",
            "frecuencia_compras_mes": np.random.normal(6.0, 1.8),
            "ticket_promedio_usd": np.random.normal(380, 50),
            "dias_desde_ultima_compra": np.random.normal(25, 12),
            "num_categorias_distintas": np.random.randint(6, 16),
            "porcentaje_compras_con_descuento": np.random.normal(0.08, 0.04),
            "num_devoluciones_año": np.random.poisson(0.8),
            "horas_navegacion_semana": np.random.normal(15, 4),
            "num_reviews_escritos": np.random.poisson(12.0),
            "tasa_abandono_carrito": np.random.normal(0.15, 0.05)
        }
        data.append(rec)

    # Archetype 3: Comprador Compulsivo/Fiel (Loyal & Active)
    # Extremely high frequency, low recency, high browsing hours, moderate-high ticket, high reviews.
    for i in range(num_per_archetype):
        rec = {
            "arquetipo": "Cliente Fiel y Activo",
            "frecuencia_compras_mes": np.random.normal(15.0, 2.5),
            "ticket_promedio_usd": np.random.normal(110, 25),
            "dias_desde_ultima_compra": np.random.normal(8, 4),
            "num_categorias_distintas": np.random.randint(8, 16),
            "porcentaje_compras_con_descuento": np.random.normal(0.25, 0.08),
            "num_devoluciones_año": np.random.poisson(3.0),
            "horas_navegacion_semana": np.random.normal(22, 3),
            "num_reviews_escritos": np.random.poisson(28.0),
            "tasa_abandono_carrito": np.random.normal(0.18, 0.06)
        }
        data.append(rec)

    # Archetype 4: Cliente Inactivo / En Riesgo (Churn Risk)
    # High recency, high cart abandonment, low frequency, low variety, low browsing.
    for i in range(num_per_archetype):
        rec = {
            "arquetipo": "Cliente Inactivo/En Riesgo",
            "frecuencia_compras_mes": np.random.normal(1.0, 0.4),
            "ticket_promedio_usd": np.random.normal(45, 15),
            "dias_desde_ultima_compra": np.random.normal(270, 45),
            "num_categorias_distintas": np.random.randint(1, 4),
            "porcentaje_compras_con_descuento": np.random.normal(0.15, 0.08),
            "num_devoluciones_año": np.random.poisson(0.8),
            "horas_navegacion_semana": np.random.normal(2.0, 1.0),
            "num_reviews_escritos": np.random.poisson(0.5),
            "tasa_abandono_carrito": np.random.normal(0.8, 0.08)
        }
        data.append(rec)

    # Anomalies / Outliers (approx. 15 records)
    # 1. High returns with almost no purchases
    # 2. Extreme ticket values
    # 3. High review writer but no purchases/activity
    # 4. Spammers / Bot behavior (100% discounts, max categories, low navigation)
    anomalies = [
        # Anomaly 1: Return Spammer
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 0.5, "ticket_promedio_usd": 450, "dias_desde_ultima_compra": 90, 
         "num_categorias_distintas": 2, "porcentaje_compras_con_descuento": 0.0, "num_devoluciones_año": 10, 
         "horas_navegacion_semana": 1.2, "num_reviews_escritos": 0, "tasa_abandono_carrito": 0.95},
        # Anomaly 2: Extreme Ticket VIP
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 1.2, "ticket_promedio_usd": 950, "dias_desde_ultima_compra": 180, 
         "num_categorias_distintas": 1, "porcentaje_compras_con_descuento": 0.0, "num_devoluciones_año": 0, 
         "horas_navegacion_semana": 28.0, "num_reviews_escritos": 1, "tasa_abandono_carrito": 0.05},
        # Anomaly 3: Review Bot
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 0.5, "ticket_promedio_usd": 15, "dias_desde_ultima_compra": 350, 
         "num_categorias_distintas": 1, "porcentaje_compras_con_descuento": 0.95, "num_devoluciones_año": 1, 
         "horas_navegacion_semana": 2.5, "num_reviews_escritos": 50, "tasa_abandono_carrito": 0.9},
        # Anomaly 4: Discount Harvester
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 19.5, "ticket_promedio_usd": 8, "dias_desde_ultima_compra": 2, 
         "num_categorias_distintas": 15, "porcentaje_compras_con_descuento": 1.0, "num_devoluciones_año": 9, 
         "horas_navegacion_semana": 29.5, "num_reviews_escritos": 48, "tasa_abandono_carrito": 0.05},
        # Anomaly 5: Phantom Browser
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 0.1, "ticket_promedio_usd": 5, "dias_desde_ultima_compra": 360, 
         "num_categorias_distintas": 1, "porcentaje_compras_con_descuento": 0.5, "num_devoluciones_año": 8, 
         "horas_navegacion_semana": 29.9, "num_reviews_escritos": 0, "tasa_abandono_carrito": 0.99},
        # Anomaly 6: High Frequency, Low Browse, Low Cart Abandonment
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 20.0, "ticket_promedio_usd": 480, "dias_desde_ultima_compra": 1, 
         "num_categorias_distintas": 1, "porcentaje_compras_con_descuento": 0.9, "num_devoluciones_año": 10, 
         "horas_navegacion_semana": 0.5, "num_reviews_escritos": 45, "tasa_abandono_carrito": 0.0},
        # Anomaly 7: Idle Rich
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 0.2, "ticket_promedio_usd": 490, "dias_desde_ultima_compra": 365, 
         "num_categorias_distintas": 12, "porcentaje_compras_con_descuento": 0.0, "num_devoluciones_año": 0, 
         "horas_navegacion_semana": 0.6, "num_reviews_escritos": 0, "tasa_abandono_carrito": 0.1},
        # Add 8 more mixed anomalies
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 18.0, "ticket_promedio_usd": 495, "dias_desde_ultima_compra": 300, 
         "num_categorias_distintas": 2, "porcentaje_compras_con_descuento": 0.8, "num_devoluciones_año": 0, 
         "horas_navegacion_semana": 1.0, "num_reviews_escritos": 4, "tasa_abandono_carrito": 0.85},
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 5.0, "ticket_promedio_usd": 120, "dias_desde_ultima_compra": 120, 
         "num_categorias_distintas": 8, "porcentaje_compras_con_descuento": 0.95, "num_devoluciones_año": 10, 
         "horas_navegacion_semana": 2.5, "num_reviews_escritos": 40, "tasa_abandono_carrito": 0.75},
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 0.8, "ticket_promedio_usd": 10, "dias_desde_ultima_compra": 5, 
         "num_categorias_distintas": 1, "porcentaje_compras_con_descuento": 0.0, "num_devoluciones_año": 9, 
         "horas_navegacion_semana": 28.0, "num_reviews_escritos": 2, "tasa_abandono_carrito": 0.0},
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 19.8, "ticket_promedio_usd": 450, "dias_desde_ultima_compra": 180, 
         "num_categorias_distintas": 15, "porcentaje_compras_con_descuento": 0.05, "num_devoluciones_año": 0, 
         "horas_navegacion_semana": 0.5, "num_reviews_escritos": 50, "tasa_abandono_carrito": 0.99},
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 0.6, "ticket_promedio_usd": 490, "dias_desde_ultima_compra": 2, 
         "num_categorias_distintas": 1, "porcentaje_compras_con_descuento": 0.98, "num_devoluciones_año": 8, 
         "horas_navegacion_semana": 29.5, "num_reviews_escritos": 1, "tasa_abandono_carrito": 0.01},
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 12.0, "ticket_promedio_usd": 15, "dias_desde_ultima_compra": 350, 
         "num_categorias_distintas": 14, "porcentaje_compras_con_descuento": 0.01, "num_devoluciones_año": 10, 
         "horas_navegacion_semana": 0.8, "num_reviews_escritos": 49, "tasa_abandono_carrito": 0.95},
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 0.5, "ticket_promedio_usd": 500, "dias_desde_ultima_compra": 1, 
         "num_categorias_distintas": 15, "porcentaje_compras_con_descuento": 0.99, "num_devoluciones_año": 0, 
         "horas_navegacion_semana": 29.0, "num_reviews_escritos": 0, "tasa_abandono_carrito": 0.99},
        {"arquetipo": "Anomalía", "frecuencia_compras_mes": 19.9, "ticket_promedio_usd": 5, "dias_desde_ultima_compra": 365, 
         "num_categorias_distintas": 1, "porcentaje_compras_con_descuento": 0.0, "num_devoluciones_año": 10, 
         "horas_navegacion_semana": 0.5, "num_reviews_escritos": 50, "tasa_abandono_carrito": 0.0}
    ]
    data.extend(anomalies)
    
    df = pd.DataFrame(data)
    
    # Apply strict variable bounds and rounding to make them look authentic
    # cliente_id: unique identifier — generated with Faker for realistic data
    fake = Faker('es_MX')       # Nombres en español latinoamericano
    Faker.seed(seed)             # Misma seed => mismos IDs cada ejecución
    ids = [f"{fake.first_name()}_{fake.last_name()}_{1000+i}".replace(" ", "_") for i in range(len(df))]
    df.insert(0, "cliente_id", ids)
    
    # Clip values to legal domain bounds as defined in prompt
    df["frecuencia_compras_mes"] = df["frecuencia_compras_mes"].clip(0.5, 20.0).round(1)
    df["ticket_promedio_usd"] = df["ticket_promedio_usd"].clip(5.0, 500.0).round(2)
    df["dias_desde_ultima_compra"] = df["dias_desde_ultima_compra"].clip(1, 365).astype(int)
    df["num_categorias_distintas"] = df["num_categorias_distintas"].clip(1, 15).astype(int)
    df["porcentaje_compras_con_descuento"] = df["porcentaje_compras_con_descuento"].clip(0.0, 1.0).round(3)
    df["num_devoluciones_año"] = df["num_devoluciones_año"].clip(0, 10).astype(int)
    df["horas_navegacion_semana"] = df["horas_navegacion_semana"].clip(0.5, 30.0).round(1)
    df["num_reviews_escritos"] = df["num_reviews_escritos"].clip(0, 50).astype(int)
    df["tasa_abandono_carrito"] = df["tasa_abandono_carrito"].clip(0.0, 1.0).round(3)
    
    # Save file
    df.to_csv(output_path, index=False)
    print(f"Dataset generado exitosamente en: {output_path}")
    print(f"Total registros: {len(df)}")
    print(df.groupby("arquetipo").size())
    return df

if __name__ == "__main__":
    generate_customer_data()
