# Como ejecutar el ejemplo desde 0

### 0. Generar entorno virtual
    python -m venv .venv
    
    [Windows]
    ./venv/Scripts/activate
    
    [Linux]
    source ./venv/bin/activate

    pip install -r requirements.txt

    cd python/logic_or_rna

### 1. Generar el dataset
    py create_dataset_logic_or.py 1000

### 2. Entrenar el modelo
    py train_logic_or.py <PATH_DATASET> <TITLE> <FUNC_ACTIVATION> (opt)<N_LAYERS>

### 3. Prueba de modelo
    cd ../scripts
    py ia_auditor.py <MODEL_PATH> <SCALER_PATH> <ENCODER_PATH> <TEST_DATASET_FILE>

# Resumen 
Este proyecto trata de la implementacion de una compuerta OR la cual recibe 2 input de tension entre 0-5V, 
se definio una banda indefinida entre 1.3V y 2V para analizar como se comportaria la IA en estos casos.

Para ello se entreno una *Red Neuronal* de 2 capas, 1 oculta y 1 de salida. Los datos de entrenamiento se lograron con el script
``create_dataset_logic_or.py`` el cual permite crear datasets basados en las especificaciones anteriores.

# Datasets

Se generó un dataset de 1000 valores los cuales se utilizaran el 20% para entrenamiento y el resto para validación, y un dataset de 250 valores para volver a validar el entramiento.

# Assets

En esta seccion se encuentran los archivos adicionales como por ejemplo la *Red Neuronal* misma, las etiquetas de los encoders para luego recuperar el valor final de prediccion y los escaladores para normalizar los datos.

# Conclusiones

***Pueden variar los resultados dependiendo del dataset***

---
### 📊 REPORTE DE RENDIMIENTO DEL MODELO CON FUNCION DE ACTIVIACION SIGMOIDE

**Métricas Globales:**
- **Accuracy (Exactitud):** `0.6840`
- **F1-Score (Weighted):** `0.6818`

**Detalle por Categoría:**

```
              precision    recall  f1-score   support

        HIGH       0.72      0.65      0.68       109
  INDEFINIDO       0.65      0.65      0.65       112
         LOW       0.71      0.93      0.81        29

    accuracy                           0.68       250
   macro avg       0.69      0.74      0.71       250
weighted avg       0.68      0.68      0.68       250

```

![Matriz de confusion](plots/matrix_confusion_logic_or_sigmoid.png)

---
### 📊 REPORTE DE RENDIMIENTO DEL MODELO  CON FUNCION DE ACTIVIACION SOFTMAX

**Métricas Globales:**
- **Accuracy (Exactitud):** `0.8320`
- **F1-Score (Weighted):** `0.8284`

**Detalle por Categoría:**

```
              precision    recall  f1-score   support

        HIGH       1.00      0.66      0.80       109
  INDEFINIDO       0.74      0.96      0.84       112
         LOW       0.88      0.97      0.92        29

    accuracy                           0.83       250
   macro avg       0.87      0.86      0.85       250
weighted avg       0.87      0.83      0.83       250

```

![Matriz de confusion](plots/matrix_confusion_logic_or_softmax.png)


---


<!-- Se realizo el entrenamiento de la RN con distintas configuraciones:

|  | Sigmoid | Softmax |  Softmax 8 capas |
|--------------|--------------|--------------|--------------|
| Precisión       | 84.5%      | 64.5%       | 43.75%     | -->

