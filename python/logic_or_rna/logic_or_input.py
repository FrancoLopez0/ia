import numpy as np
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from keras.models import load_model

def scale_variables(input_test):
    # Escalado de variables
    sc = joblib.load('assets/scaler_variables.joblib') # Cargar el guardado
    input_test = sc.transform(input_test)

    return input_test

if __name__ == '__main__':
    classifier = load_model('assets/logic_or_model_sigmoid.keras')

    encoder = joblib.load('assets/encoder_labels.joblib')

    try:
        while True:
            try:
                input_vin_0 = float(input("Ingrese el valor de Vin_0: "))
                input_vin_1 = float(input("Ingrese el valor de Vin_1: "))

                if(input_vin_0 < 0 or input_vin_1 < 0):
                    print("Ingrese valores positivos...")
                    continue
                if(input_vin_0 > 5 or input_vin_1 > 5):
                    print("Ingrese valores menores a 5...")
                    continue

            except ValueError:
                print("Ingrese un numero flotante valido (0-5)...")
                continue

            test_data_final = scale_variables([[input_vin_0, input_vin_1]])
            prediccion_raw = classifier.predict(test_data_final)
            prediccion = encoder.inverse_transform(prediccion_raw)

            print(prediccion_raw)
            print(prediccion)

    except KeyboardInterrupt:
        print("\n¡Programa interrumpido por el usuario!")
    