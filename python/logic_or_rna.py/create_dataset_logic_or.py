import numpy as np
import pandas as pd
import sys

LOW_INPUT        = [0, 1.3]
HIGH_INPUT       = [2, 5]
UNK_INPUT        = [1.3, 2]
OUTPUT           = ["LOW", "HIGH", "INDEFINIDO"]
OPERATIONS_CASES = ["LOW OR LOW", "LOW OR HIGH", "HIGH OR LOW", "HIGH OR HIGH", "INDEFINIDO OR HIGH", "LOW OR INDEFINIDO", "INDEFINIDO OR INDEFINIDO"]

def generate_voltage_or_dataset(samples_per_case=250):
    
    data = np.array([
        ['Vin_0', "Vin_1", 'Output'],
    ])

    operation_case = np.random.choice(OPERATIONS_CASES, samples_per_case)

    for operation_case in operation_case:
        if operation_case == "LOW OR LOW":
            # Caso 0: LOW OR LOW -> 0
            v1, v2 = np.random.uniform(LOW_INPUT[0], LOW_INPUT[1]), np.random.uniform(LOW_INPUT[0], LOW_INPUT[1])
            data = np.append(data, [[v1, v2, OUTPUT[0]]], axis=0)
        if operation_case == "LOW OR HIGH":
            # Caso 1: LOW OR HIGH -> 1
            v1, v2 = np.random.uniform(LOW_INPUT[0], LOW_INPUT[1]), np.random.uniform(HIGH_INPUT[0], HIGH_INPUT[1])
            data = np.append(data, [[v1, v2, OUTPUT[1]]], axis=0)
        if operation_case == "HIGH OR LOW":
            # Caso 2: HIGH OR LOW -> 1  
            v1, v2 = np.random.uniform(HIGH_INPUT[0], HIGH_INPUT[1]), np.random.uniform(LOW_INPUT[0], LOW_INPUT[1])
            data = np.append(data, [[v1, v2, OUTPUT[1]]], axis=0)
        if operation_case == "HIGH OR HIGH":
            # Caso 3: HIGH OR HIGH -> 1
            v1, v2 = np.random.uniform(HIGH_INPUT[0], HIGH_INPUT[1]), np.random.uniform(HIGH_INPUT[0], HIGH_INPUT[1])
            data = np.append(data, [[v1, v2, OUTPUT[1]]], axis=0)
        if operation_case == "INDEFINIDO OR HIGH":
            # Caso 4: INDEFINIDO OR HIGH -> INDEFINIDO
            v1, v2 = np.random.uniform(UNK_INPUT[0], UNK_INPUT[1] ), np.random.uniform(HIGH_INPUT[0], HIGH_INPUT[1])
            data = np.append(data, [[v1, v2, OUTPUT[2]]], axis=0)
        if operation_case == "LOW OR INDEFINIDO":
            # Caso 5: LOW OR INDEFINIDO -> INDEFINIDO
            v1, v2 = np.random.uniform(LOW_INPUT[0], LOW_INPUT[1] ), np.random.uniform(UNK_INPUT[0], UNK_INPUT[1])
            data = np.append(data, [[v1, v2, OUTPUT[2]]], axis=0)
        if operation_case == "INDEFINIDO OR INDEFINIDO":
            # Caso 5: INDEFINIDO OR INDEFINIDO -> INDEFINIDO
            v1, v2 = np.random.uniform(UNK_INPUT[0], UNK_INPUT[1] ), np.random.uniform(UNK_INPUT[0], UNK_INPUT[1])
            data = np.append(data, [[v1, v2, OUTPUT[2]]], axis=0)

    return data

if __name__ == '__main__':
    print("Generando dataset...")

    if len(sys.argv) >= 2:
        samples_per_case = int(sys.argv[1])
        if len(sys.argv) >= 3:
            FILE_NAME = sys.argv[2]
        else:
            FILE_NAME = 'or_logic_dataset'
    else:
        samples_per_case = 250
        FILE_NAME = 'or_logic_dataset'

    data = generate_voltage_or_dataset(samples_per_case)
    df = pd.DataFrame(data[1:], columns=data[0])
    df.to_csv(FILE_NAME + "_" + str(samples_per_case) + ".csv", index=False, encoding='utf-8')