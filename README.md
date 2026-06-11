# Proyecto: Segmentación de Clientes y Detección de Anomalías

**Estudiante:** Marco Torres  
**Carrera:** Ingeniería de Software – 7mo Semestre  
**Docente:** Isaac Torres  
**Materia:** Aprendizaje No Supervisado  

---

## 📂 Estructura del Proyecto

* `data/customer_data.csv`: Dataset generado de 415 registros.
* `requirements.txt`: Dependencias del proyecto (incluye `faker` para IDs realistas).
* `data_generation.py`: Script generador de datos con [Faker](https://faker.readthedocs.io/) para nombres de clientes en español.
* `clustering_analysis.py`: Módulo de procesamiento y algoritmos (K-Means, DBSCAN, PCA).
* `train_and_export.py`: Script que entrena los modelos y genera el simulador interactivo.
* `test_pipeline.py`: Pruebas de integración del código.
* `index.html`: Simulador interactivo en HTML/JS para probar las predicciones.
* `informe.md`: Informe técnico escrito y guion del video.

---

## 🚀 Ejecución del Proyecto

Sigue estos pasos en la consola de Windows para instalar y ejecutar el proyecto:

```powershell
# 1. Instalar las dependencias
pip install -r requirements.txt

# 2. Generar el dataset de comportamiento
python data_generation.py

# 3. Entrenar modelos y exportar al panel interactivo
python train_and_export.py

# 4. Validar el pipeline
python test_pipeline.py
```

### Probar el Simulador
* **Haz doble clic en el archivo [index.html](file:///c:/Users/marco/Desktop/AprendizajeNoSupervizado/index.html)** desde tu explorador de Windows para abrir la interfaz interactiva en el navegador y probar los modelos entrenados con diferentes sliders.
