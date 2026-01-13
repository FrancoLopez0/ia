import numpy as np
import joblib
import pandas as pd

import time

from sklearn.preprocessing import StandardScaler
from keras.models import load_model

def import_dataset():
    # Importar dataset

    dataset = pd.read_csv('../../datasets/or_logic_dataset_250.csv')

    input = dataset.iloc[:, :-1].values
    output = dataset.iloc[:, -1].values

    return input, output

def scale_variables(input_test):
    # Escalado de variables
    sc = joblib.load('assets/scaler_variables.joblib') # Cargar el guardado
    input_test = sc.transform(input_test)
    joblib.dump(sc, 'assets/scaler_variables.joblib')

    return input_test

if __name__ == '__main__':

    classifier = load_model('assets/logic_or_model_sigmoid.keras')

    encoder = joblib.load('assets/encoder_labels.joblib')

    test_data, expected_output = import_dataset()

    expected_output = encoder.inverse_transform(expected_output.reshape(-1, 1))

    test_data_final = scale_variables(test_data)

    start_time = time.time()

    prediccion_raw = classifier.predict(test_data_final)
    prediccion = encoder.inverse_transform(prediccion_raw)

    end_time = time.time()

    elapsed_time = end_time - start_time

    evaluacion = classifier.evaluate(test_data_final, expected_output)

    print(f"Loss en Test: {evaluacion[0]}")
    print(f"Accuracy en Test: {evaluacion[1] * 100:.2f}%")
    print(f"Tiempo de ejecucion: {elapsed_time:.2f} segundos")