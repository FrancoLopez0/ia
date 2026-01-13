import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report, 
    accuracy_score, 
    f1_score
)
import pandas as pd
import sys

def auditor_de_modelo(model_path, scaler_path, encoder_path, X_test, y_test_labels):
    """
    Realiza una auditoría completa de un modelo cargado.
    X_test: Array (N, 2, 1) con tensiones reales.
    y_test_labels: Array (N,) con las etiquetas reales ('HIGH', etc.)
    """
    
    # 1. Carga de componentes
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    
    # 2. Preprocesamiento (Ajuste de dimensiones para Scaler y RNN)
    n_samples, n_steps, n_features = X_test.shape
    X_flat = X_test.reshape(-1, 2)  # Pasamos a 2D para el scaler
    X_scaled = scaler.transform(X_flat)
    X_final = X_scaled.reshape(-1, n_steps, n_features) # Volvemos a 3D para RNN
    
    # 3. Inferencia
    y_pred_probs = model.predict(X_final, verbose=0)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    
    # Transformar etiquetas reales a índices para comparar
    y_true_indices = encoder.transform(y_test_labels.reshape(-1, 1))
    y_true_indices = np.argmax(y_true_indices, axis=1)
    
    # Obtener nombres de las clases
    class_names = encoder.categories_[0]
    
    # 4. Cálculo de métricas
    acc = accuracy_score(y_true_indices, y_pred_indices)
    f1 = f1_score(y_true_indices, y_pred_indices, average='weighted')
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names)
    
    # 5. Visualización: Matriz de Confusión
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f"Matriz de Confusión (Accuracy: {acc:.2%})")
    plt.show()

    # 6. Salida en formato Markdown para consola
    print("\n" + "="*40)
    print("### 📊 REPORTE DE RENDIMIENTO DEL MODELO")
    print("="*40)
    print(f"\n**Métricas Globales:**")
    print(f"- **Accuracy (Exactitud):** `{acc:.4f}`")
    print(f"- **F1-Score (Weighted):** `{f1:.4f}`")
    print(f"\n**Detalle por Categoría:**\n")
    print("```")
    print(report)
    print("```")
    print("\n" + "="*40)

# --- EJEMPLO DE USO ---
# (Asegúrate de tener tus datos y_test con las etiquetas de texto reales)
# auditor_de_modelo('logic_or_model.keras', 'scaler_tension.joblib', 'encoder_labels.joblib', X_test, y_test)

if __name__ == '__main__':

    if len(sys.argv) >= 4:
        model_path = sys.argv[1]
        scaler_path = sys.argv[2]
        encoder_path = sys.argv[3]
        FILE_NAME = sys.argv[4]

        df = pd.DataFrame(pd.read_csv(FILE_NAME))
        X_test = df.iloc[:, :-1].to_numpy()
        X_test = X_test.reshape(-1, 2, 1)

        y_test_labels = df.iloc[:, -1].to_numpy()

        auditor_de_modelo(model_path, scaler_path, encoder_path, X_test, y_test_labels)
    else:
        print("py ia_auditor <model_path> <scaler_path> <encoder_path> <FILE_NAME>")