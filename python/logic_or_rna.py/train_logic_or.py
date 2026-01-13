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

PRONOM = "_8_layers"

def import_dataset():
    # Importar dataset

    dataset = pd.read_csv('../../datasets/or_logic_dataset_1000.csv')

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

    joblib.dump(encoder, 'assets/encoder_labels' + PRONOM + '.joblib')

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
    joblib.dump(sc, 'assets/scaler_variables' + PRONOM + '.joblib')

    return input_train, input_test

def init_rna():
    classifier = Sequential()

    classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=2))
    # classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))

    classifier.add(Dense(units=3, kernel_initializer='uniform', activation='sigmoid')) 

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

if __name__ == '__main__':

    input, output = import_dataset()

    input, output = process_dataset(input, output)

    input_train, input_test, output_train, output_test = split_dataset(input, output)

    input_train, input_test = scale_variables(input_train, input_test)

    classifier = init_rna()

    classifier.fit(input_train, output_train, batch_size=10, epochs=100)

    classifier.save("assets/logic_or_model" + PRONOM + ".keras")

