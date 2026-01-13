import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense

import joblib
import sys

TITLE = "_sigmoid_4_layers"
FILE_DATASET = ""
FUNC_ACTIVATION_FINAL = "sigmoid"
N_LAYERS = 1

def import_dataset(path = '../../datasets/or_logic_dataset_1000.csv'):
    # Importar dataset

    dataset = pd.read_csv(path)

    input = dataset.iloc[:, :-1].values
    output = dataset.iloc[:, -1].values

    return input, output

def process_dataset(input, output):
    # Codificar datos categoricos
    #   Los datos de salida pueden ser HIGH, LOW o INDEFINIDO, estos datos no pueden ser procesados numericamente
    #   por lo que se requiere codificarlos en variables dummy

    # labelencoder_output = LabelEncoder()
    # output = labelencoder_output.fit_transform(output)

    encoder = OneHotEncoder(sparse_output=False)
    output = encoder.fit_transform(output.reshape(-1, 1))

    joblib.dump(encoder, 'assets/encoder_labels' + TITLE + '.joblib')

    return input, output

def split_dataset(input, output):
    # Dividir el dataset en entrenamiento y testing
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=0)
    return input_train, input_test, output_train, output_test

def scale_variables(input_train, input_test):
    # Escalado de variables
    sc = StandardScaler()
    input_train = sc.fit_transform(input_train)
    input_test = sc.transform(input_test)
    joblib.dump(sc, 'assets/scaler_variables' + TITLE + '.joblib')

    return input_train, input_test

def init_rna():
    classifier = Sequential()

    for i in range(N_LAYERS):
        classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=2))

    classifier.add(Dense(units=3, kernel_initializer='uniform', activation=FUNC_ACTIVATION_FINAL)) 

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("py train_logic_or.py <FILE_DATASET> <TITLE> <FUNC_ACTIVATION_FINAL> (opt)<N_LAYERS>")
        sys.exit(1)

    FILE_DATASET = sys.argv[1]
    TITLE = sys.argv[2]
    FUNC_ACTIVATION_FINAL = sys.argv[3]

    if len(sys.argv) > 4:
        N_LAYERS = int(sys.argv[4])

    input, output = import_dataset(FILE_DATASET)

    input, output = process_dataset(input, output)

    input_train, input_test, output_train, output_test = split_dataset(input, output)

    input_train, input_test = scale_variables(input_train, input_test)

    classifier = init_rna()

    classifier.fit(input_train, output_train, batch_size=10, epochs=100)

    classifier.save("assets/logic_or_model" + TITLE + ".keras")

