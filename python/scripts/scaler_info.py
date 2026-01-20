import joblib
import numpy as np
import argparse

def extract_scaler_info(file_path):
    try:
        # Cargar el objeto desde el archivo .joblib
        scaler = joblib.load(file_path)
        scaler_type = type(scaler).__name__
        
        print(f"--- Información del Escalador ---")
        print(f"Tipo detectado: {scaler_type}\n")

        # Diccionario de atributos comunes según el tipo de escalador
        # La mayoría terminan en '_' tras el proceso de .fit()
        attributes = {
            'StandardScaler': ['mean_', 'scale_', 'var_'],
            'MinMaxScaler': ['min_', 'scale_', 'data_min_', 'data_max_'],
            'RobustScaler': ['center_', 'scale_'],
            'MaxAbsScaler': ['max_abs_', 'scale_']
        }

        found_info = False

        # Intentar obtener los atributos específicos del tipo detectado
        target_attrs = attributes.get(scaler_type, [])
        
        # Si el tipo no está en el diccionario, buscamos atributos que terminen en '_'
        if not target_attrs:
            target_attrs = [attr for attr in dir(scaler) if attr.endswith('_') and not attr.startswith('__')]

        for attr in target_attrs:
            if hasattr(scaler, attr):
                value = getattr(scaler, attr)
                print(f"Atributo '{attr}':")
                print(f"{value}\n")
                found_info = True

        if not found_info:
            print("No se encontraron atributos de ajuste (¿Se ejecutó .fit() antes de guardarlo?)")

    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

if __name__ == "__main__":
    # Cambia 'mi_scaler.joblib' por la ruta de tu archivo
    parser = argparse.ArgumentParser(description='Obtiene informacion de un scaler.joblib')
    parser.add_argument('input', help='Ruta al archivo .joblib')
    args = parser.parse_args()
    extract_scaler_info(args.input)