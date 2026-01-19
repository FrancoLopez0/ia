
import argparse
import sys
import numpy as np

def representative_data_gen():
    """
    Generador de datos de ejemplo para calibrar la cuantización INT8.
    IMPORTANTE: Para un modelo real, deberías cargar ~100 imágenes o 
    muestras de tu set de entrenamiento real.
    """
    # Ajusta (1, 224, 224, 3) a la forma de entrada de TU modelo
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

def convert_model(model_path, output_path, quantize_q8=False):
    import tensorflow as tf
    print(f"--- Cargando modelo: {model_path} ---")
    model = tf.keras.models.load_model(model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize_q8:
        print("--- Aplicando cuantización Full INT8 (Q8) ---")
        # Optimizaciones básicas (Dynamic Range Quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configuración para Full Integer Quantization
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # Esto asegura que la entrada y salida también sean INT8
        # Útil para microcontroladores sin FPU
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        print("--- Exportando en Float32 (Sin cuantización) ---")

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"--- Modelo guardado con éxito en: {output_path} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convertidor de Keras a TFLite para Raspberry Pi 5 y MCUs")
    parser.add_argument("input", help="Ruta al modelo .h5 o SavedModel")
    parser.add_argument("output", help="Ruta de salida .tflite")
    parser.add_argument("--q8", action="store_true", help="Activar cuantización INT8 (Q8)")

    args = parser.parse_args()

    try:
        convert_model(args.input, args.output, args.q8)
    except Exception as e:
        print(f"Error durante la conversión: {e}")