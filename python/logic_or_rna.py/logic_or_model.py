import numpy as np
import joblib
import pandas as pd

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

    test_data_final = scale_variables(test_data)

    prediccion_raw = classifier.predict(test_data_final)
    prediccion = encoder.inverse_transform(prediccion_raw)

    done = []
    correct = 0
    incorrect = 0
    
    for predict, expected in zip(prediccion, expected_output):

        if predict == expected :
            done.append(1)
            correct += 1
        else:
            done.append(0)
            incorrect += 1

    print(correct, incorrect, correct/(correct+incorrect))